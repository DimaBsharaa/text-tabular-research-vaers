"""Build a bounded plumbing fixture, never the final tennis research dataset.

Only previous *dates* update histories: source order within a day is not trusted.
Run from any directory: python research/scripts/prepare_tennis_smoke.py
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "research/data/tennis_smoke"
SOURCE = DEST / "kaggle_v1219_prefix.csv"
ORIENTATION_SEED = "tennis-smoke-v1"
STRUCTURED = ["surface", "court", "tournament_level", "round", "best_of",
              "a_rank", "b_rank", "a_prior_matches", "b_prior_matches",
              "a_prior_winrate", "b_prior_winrate", "a_surface_winrate", "b_surface_winrate"]
TEXT = ["player_a_text", "player_b_text"]
AUDIT = ["match_id", "match_date", "player_a", "player_b", "a_history_last_date", "b_history_last_date"]


def match_key(row):
    # Winner, score, odds and original slot order never participate in orientation.
    return json.dumps([row["Date"], row["Tournament"], row["Round"],
                       *sorted([row["Player_1"], row["Player_2"]])], ensure_ascii=False)


def history_features(player, surface, history):
    h = history[player]
    surface_h = [r for r in h if r["surface"] == surface]
    features = {"prior_matches": len(h),
                "prior_winrate": sum(r["won"] for r in h) / len(h) if h else 0.5,
                "surface_winrate": sum(r["won"] for r in surface_h) / len(surface_h) if surface_h else 0.5}
    if not h:
        text = f"{player} has no earlier dated matches in this limited archive. Prior performance and tactical characteristics are unknown."
    else:
        recent = h[-3:]
        details = " ".join(f"On {r['date']}, {player} {'won' if r['won'] else 'lost'} against {r['opponent']} on {r['surface']} at {r['tournament']}." for r in recent)
        text = (f"Earlier recorded results for {player}: {sum(r['won'] for r in h)} wins in {len(h)} matches. "
                f"On {surface}, {sum(r['won'] for r in surface_h)} wins in {len(surface_h)} earlier matches. "
                + details + " This statistical summary provides no injury or tactical assessment.")
    return features, text, h[-1]["date"] if h else ""


def build(raw):
    raw = raw.copy()
    required = {"Date", "Tournament", "Round", "Player_1", "Player_2", "Winner", "Surface", "Court", "Series", "Best of", "Rank_1", "Rank_2"}
    if not required <= set(raw.columns):
        raise ValueError(f"Missing source fields: {required - set(raw.columns)}")
    pd.to_datetime(raw["Date"], format="%Y-%m-%d", errors="raise")
    if not raw["Winner"].isin(pd.concat([raw["Player_1"], raw["Player_2"]])).all():
        raise ValueError("Unknown winner name")
    raw["_key"] = raw.apply(match_key, axis=1)
    if raw["_key"].duplicated().any():
        raise ValueError("Repeated match key needs manual resolution")
    rows, history = [], defaultdict(list)
    for day, matches in raw.sort_values(["Date", "_key"]).groupby("Date", sort=True):
        updates = []
        for _, r in matches.iterrows():
            names = sorted([r["Player_1"], r["Player_2"]])
            if names[0] == names[1] or r["Winner"] not in names:
                raise ValueError("Invalid player pair / winner")
            digest = hashlib.sha256((ORIENTATION_SEED + r["_key"]).encode()).hexdigest()
            a, b = names if int(digest, 16) % 2 == 0 else names[::-1]
            rank_by_name = {r["Player_1"]: r["Rank_1"], r["Player_2"]: r["Rank_2"]}
            row = {"match_id": digest[:24], "match_date": day, "player_a": a, "player_b": b,
                   "surface": r["Surface"], "court": r["Court"], "tournament_level": r["Series"],
                   "round": r["Round"], "best_of": int(r["Best of"]), "player_a_wins": int(r["Winner"] == a)}
            for side, player in [("a", a), ("b", b)]:
                features, text, last_date = history_features(player, r["Surface"], history)
                rank = float(rank_by_name[player])
                row[f"{side}_rank"] = rank if rank > 0 else float("nan")
                row.update({f"{side}_{k}": v for k, v in features.items()})
                row[f"player_{side}_text"] = text
                row[f"{side}_history_last_date"] = last_date
                updates.append((player, {"date": day, "opponent": b if player == a else a,
                                        "surface": r["Surface"], "tournament": r["Tournament"],
                                        "won": int(r["Winner"] == player)}))
            rows.append(row)
        # Never use a same-day result as a predictor for another row on that date.
        for player, event in updates:
            history[player].append(event)
    result = pd.DataFrame(rows)[AUDIT + STRUCTURED + TEXT + ["player_a_wins"]]
    for side in ["a", "b"]:
        dates = result[f"{side}_history_last_date"]
        assert ((dates == "") | (dates < result["match_date"])).all()
    assert result.match_id.is_unique
    return result


def main():
    raw = pd.read_csv(SOURCE)
    df = build(raw)
    DEST.mkdir(parents=True, exist_ok=True)
    output = DEST / "tennis_smoke.csv"
    df.to_csv(output, index=False)
    manifest = {
        "dataset_id": "BIN_TEXT_TENNIS_SMOKE", "purpose": "smoke_fixture", "task": "binary",
        "target": "player_a_wins", "structured_columns": STRUCTURED, "text_columns": TEXT, "audit_columns": AUDIT,
        "numeric_columns": [c for c in STRUCTURED if c not in {"surface", "court", "tournament_level", "round"}],
        "text_kind": "templated_past_results_not_expert_analysis", "orientation_seed": ORIENTATION_SEED,
        "csv_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "raw_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(), "rows": len(df),
        "date_min": df.match_date.min(), "date_max": df.match_date.max(),
        "source": "https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull",
        "source_version": 1219, "source_file": "atp_tennis.csv", "source_range": "bytes 0-65535/9165924",
        "source_license_declared": "CC0: Public Domain", "audit_date": "2026-10-14",
        "sampling": "All complete CSV rows from previously inspected 64 KiB file prefix; final truncated line removed.",
        "limitations": ["Nonrepresentative early-2000 prefix; no final dataset choice.",
                        "Historical features use only earlier dates within this prefix, not complete careers.",
                        "Reported match dates and historical ranking availability have not been independently audited.",
                        "Names are abbreviated source identifiers; not a globally verified player ID crosswalk.",
                        "Current score, current winner, and odds are excluded from predictors; same-day histories excluded.",
                        "Text restates past result numbers; cannot establish complementary expert language or project passage.",
                        "Official random holdouts are not a temporal forecasting evaluation."]}
    (DEST / "schema.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(df)} rows; labels: {df.player_a_wins.value_counts().to_dict()}")
    print(f"Distinct text values: {df[TEXT].nunique().to_dict()}")


if __name__ == "__main__":
    main()
