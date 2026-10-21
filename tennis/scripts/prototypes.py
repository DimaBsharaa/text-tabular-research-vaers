"""Deterministic joining, day-batched histories and three traceable prototypes.

Full/provisional text CSVs stay outside Git. No benchmark implementation is used
or changed. A's publication-time uncertainty is preserved, not converted to True.
"""
from collections import Counter, defaultdict
from datetime import date
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

from tennis.scripts.common import CACHE, OUT, PRIVATE, SNAPSHOT, SNAPSHOT_TIME, SEED, aliases, normalized, digest, write_json, text_is_detected

CONTEXT = ["gender", "surface", "court", "level", "round", "best_of"]
NUMERIC = ["a_rank", "b_rank", "a_prior_n", "b_prior_n", "a_form10", "b_form10", "a_surface_rate", "b_surface_rate", "a_rest_days", "b_rest_days"]


def load_matches():
    sys.path.insert(0, str(CACHE / "sources/xlrd-2.0.2.whl"))
    frames = []
    for path in sorted((CACHE / "sources").glob("[aw]t[pa]_20*.xls*")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            df = pd.read_excel(path)
        df["gender"] = "M" if path.name.startswith("atp") else "F"
        df["source_file"] = path.name; df["source_row"] = np.arange(len(df)) + 2
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    raw = raw.dropna(subset=["Date", "Winner", "Loser", "Round"])
    raw["winner_key"] = raw.Winner.map(normalized); raw["loser_key"] = raw.Loser.map(normalized)
    raw["match_id"] = raw.apply(lambda r: digest(json.dumps([r.gender, r.Date, r.Tournament, r.Round, *sorted([r.winner_key, r.loser_key])]))[:24], axis=1)
    duplicates = raw[raw.match_id.duplicated(keep=False)].copy()
    duplicates.drop(columns=[c for c in duplicates if c.startswith(("B365", "EX", "PS", "LB", "SJ", "Max", "Avg"))], errors="ignore").to_csv(OUT / "structured_duplicate_keys.csv", index=False)
    raw = raw.drop_duplicates("match_id").sort_values(["Date", "match_id"])
    raw.to_csv(PRIVATE / "structured_source_rows.csv", index=False)
    matches = derive_matches(raw)
    matches.to_csv(PRIVATE / "matches.csv", index=False)
    return raw, matches


def derive_matches(raw):
    """Day-batched features; callable on fixtures for future-label invariance tests."""
    history = defaultdict(list); records = []
    for day, group in raw.groupby("Date", sort=True):
        updates = []
        for r in group.itertuples():
            pair = sorted([r.winner_key, r.loser_key])
            a, b = pair if int(digest(SEED + r.match_id), 16) % 2 == 0 else pair[::-1]
            ranks = {r.winner_key: r.WRank, r.loser_key: r.LRank}
            names = {r.winner_key: r.Winner, r.loser_key: r.Loser}
            rec = {"match_id": r.match_id, "match_date": day, "gender": r.gender, "tournament": r.Tournament,
                   "surface": r.Surface, "court": r.Court, "round": r.Round, "best_of": r._asdict().get('_8', None),
                   "level": getattr(r, "Series", None) if r.gender == "M" else getattr(r, "Tier", None),
                   "player_a_key": a, "player_b_key": b, "player_a": names[a], "player_b": names[b],
                   "target": int(a == r.winner_key), "comment": r.Comment, "source_file": r.source_file, "source_row": r.source_row}
            # pandas namedtuple renames 'Best of'; retrieve it explicitly by index.
            rec["best_of"] = group.loc[group.match_id == r.match_id, "Best of"].iloc[0]
            for side, player in [("a", a), ("b", b)]:
                h = history[(r.gender, player)]; recent = h[-10:]; sh = [v for v in h if v["surface"] == r.Surface]
                rank = pd.to_numeric(ranks[player], errors="coerce")
                rec[f"{side}_rank"] = float(rank) if pd.notna(rank) and rank > 0 else np.nan
                rec[f"{side}_prior_n"] = len(h)
                rec[f"{side}_form10"] = sum(v["won"] for v in recent) / len(recent) if recent else np.nan
                rec[f"{side}_surface_rate"] = sum(v["won"] for v in sh) / len(sh) if sh else np.nan
                rec[f"{side}_rest_days"] = (date.fromisoformat(day) - date.fromisoformat(h[-1]["date"])).days if h else np.nan
                updates.append(((r.gender, player), {"date": day, "surface": r.Surface, "won": int(player == r.winner_key)}))
            records.append(rec)
        for player, event in updates: history[player].append(event)
    matches = pd.DataFrame(records)
    return matches


def join_interviews(raw, interviews):
    # Deterministic surname+initial aliases, then exact gender/date/round/opponent.
    known = {g: set(raw.loc[raw.gender == g, "winner_key"]) | set(raw.loc[raw.gender == g, "loser_key"]) for g in ["M", "F"]}
    crosswalk = []
    lookup = defaultdict(list)
    for r in raw.itertuples():
        lookup[(r.gender, r.Date, r.winner_key, r.Round)].append(r)
        lookup[(r.gender, r.Date, r.loser_key, r.Round)].append(r)
    links = []
    for r in interviews.itertuples():
        names = aliases(r.player) & known[r.gender]
        other = aliases(str(r.opponent)) & known[r.gender] if r.opponent_known else set()
        crosswalk.append({"player": r.player, "gender": r.gender, "possible_abbreviations": "|".join(sorted(names))})
        candidates = {m.match_id: m for alias in names for m in lookup[(r.gender, r.date, alias, r.stage)]}
        if r.opponent_known:
            candidates = {k:m for k,m in candidates.items() if bool({m.winner_key,m.loser_key} & other)}
        status = "matched" if len(candidates) == 1 else ("ambiguous" if candidates else "unmatched")
        row = {"interview_id": str(r.interview_id), "player": r.player, "gender": r.gender, "date": r.date,
               "opponent": r.opponent, "source_stage": r.stage, "source_tournament": r.tournament,
               "candidate_count": len(candidates), "join_status": status}
        if len(candidates) == 1:
            m = next(iter(candidates.values()))
            matched_player = next(iter({m.winner_key,m.loser_key} & names))
            outcome_agrees = int(matched_player == m.winner_key) == r.result
            row.update(match_id=m.match_id, player_key=matched_player, target_tournament=m.Tournament,
                       outcome_agrees=outcome_agrees, source_match_file=m.source_file, source_match_row=m.source_row)
            if not outcome_agrees: row["join_status"] = "outcome_conflict_excluded"
        links.append(row)
    joined = pd.DataFrame(links)
    # Human review found explicit loss statements against a source win label.
    # Do not repair the date/outcome by guessing a different match.
    joined.loc[joined.interview_id == "1499", "join_status"] = "manual_transcript_conflict_excluded"
    joined.to_csv(OUT / "cornell_match_join_audit.csv", index=False)
    pd.DataFrame(crosswalk).drop_duplicates().to_csv(OUT / "player_alias_crosswalk.csv", index=False)
    return joined


def variant_a(matches, interviews, links):
    good = links[links.join_status == "matched"]
    merged = interviews.merge(good[["interview_id", "match_id", "player_key"]], on="interview_id", validate="one_to_one")
    by_day = {day: group for day, group in merged.groupby("date")}
    latest = {}; records = []; coverage = Counter()
    for day, group in matches[matches.match_date <= "2015-12-31"].groupby("match_date", sort=True):
        for r in group.to_dict("records"):
            docs = [latest.get((r["gender"], r[f"player_{side}_key"])) for side in ["a", "b"]]
            docs = [d if d is not None and 0 < (date.fromisoformat(day) - date.fromisoformat(d["date"])).days <= 90 and d["answer_words"] >= 70 else None for d in docs]
            coverage[str(sum(d is not None for d in docs)) + "_sides"] += 1
            if not all(d is not None for d in docs) or r["comment"] != "Completed": continue
            for side, doc in zip(["a", "b"], docs):
                assert doc["date"] < day and doc["match_id"] != r["match_id"]
                r[f"player_{side}_text"] = " ".join(doc["answer_text"].split()[:300])
                r[f"{side}_interview_id"] = doc["interview_id"]; r[f"{side}_text_date"] = doc["date"]
                r[f"{side}_source_match_id"] = doc["match_id"]
            r["temporal_basis"] = "corroborated prior match-day metadata; publication time unverified"
            r["publication_verified"] = False
            records.append(r)
        # Source docs enter only AFTER processing every target match that day.
        for doc in by_day.get(day, pd.DataFrame()).to_dict("records"):
            latest[(doc["gender"], doc["player_key"])] = doc
    return pd.DataFrame(records), dict(coverage)


def variant_b(matches):
    source = CACHE / "sources/mcp_2015june_charting-m-stats-ServeDirection.csv"
    stats = pd.read_csv(source)
    directions = ["deuce_wide", "deuce_middle", "deuce_t", "ad_wide", "ad_middle", "ad_t"]
    meta = pd.read_csv(CACHE / "sources/mcp_2015june_charting-m-matches.csv", encoding="latin1")
    meta = meta.drop_duplicates("match_id").set_index("match_id")
    events = defaultdict(list)
    for r in stats[stats.row.isin(["1 Total", "2 Total"])].to_dict("records"):
        if r["match_id"] not in meta.index: continue
        m = meta.loc[r["match_id"]]
        side = r["row"][0]; player = str(m["Player " + side])
        parsed = pd.to_datetime(str(m.Date), format="%Y%m%d", errors="coerce")
        if pd.isna(parsed) or parsed.strftime("%Y-%m-%d") > "2015-06-11": continue
        for alias in aliases(player):
            events[alias].append({"date": parsed.strftime("%Y-%m-%d"), "source_id": r["match_id"], **{k: int(r[k]) for k in directions}})
    records = []; reasons = Counter()
    for row in matches[(matches.gender == "M") & (matches.match_date > "2015-06-11") & (matches.match_date <= "2015-12-31") & (matches.comment == "Completed")].to_dict("records"):
        valid = True
        for side in ["a", "b"]:
            prior = [e for e in events[row[f"player_{side}_key"]] if 0 < (date.fromisoformat(row["match_date"]) - date.fromisoformat(e["date"])).days <= 365]
            prior = list({e["source_id"]:e for e in prior}.values())
            total = sum(sum(e[k] for k in directions) for e in prior)
            if len(prior) < 2 or total < 50:
                valid = False; reasons[side + "_insufficient_history"] += 1; continue
            wide = sum(e["deuce_wide"] + e["ad_wide"] for e in prior) / total
            tee = sum(e["deuce_t"] + e["ad_t"] for e in prior) / total
            body = sum(e["deuce_middle"] + e["ad_middle"] for e in prior) / total
            row[f"player_{side}_text"] = (f"In {len(prior)} earlier charted matches within the preceding year, this player had {total} direction-coded serves. "
                    f"Wide deliveries accounted for {wide:.1%}, T deliveries for {tee:.1%}, and middle deliveries for {body:.1%}. "
                    "These are placement frequencies among direction-coded serves, not serve success rates. No injury or psychological assessment is available.")
            row[f"{side}_source_match_ids"] = "|".join(sorted(e["source_id"] for e in prior))
            row[f"{side}_latest_source_date"] = max(e["date"] for e in prior)
            row[f"{side}_wide_fraction"] = wide; row[f"{side}_t_fraction"] = tee; row[f"{side}_charted_n"] = len(prior)
            row[f"{side}_coded_serves"] = total
        if valid:
            row["source_snapshot"] = SNAPSHOT; row["source_available_at"] = SNAPSHOT_TIME
            row["temporal_basis"] = "Git snapshot committed before target day; all charted match days earlier"
            records.append(row)
    return pd.DataFrame(records), {"snapshot_commit": SNAPSHOT, "snapshot_committed_at": SNAPSHOT_TIME,
                                  "charted_matches_in_snapshot": int(stats.match_id.nunique()), "exclusion_sides": dict(reasons)}


def variant_c(matches):
    df = matches[(matches.comment == "Completed") & (matches.match_date <= "2015-12-31")].copy()
    df["context_text"] = df.apply(lambda r: f"This {'men' if r.gender == 'M' else 'women'}'s match is on {r.surface.lower()} in {r.court.lower()} conditions. "
           f"It is the {r['round']} at {r.level} level, played as best of {int(r.best_of)} sets.", axis=1)
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    interviews = pd.read_csv(PRIVATE / "interviews_with_answers.csv", dtype={"interview_id":str}, keep_default_na=False)
    if "--reuse-matches" in sys.argv:
        raw = pd.read_csv(PRIVATE / "structured_source_rows.csv", low_memory=False)
        matches = pd.read_csv(PRIVATE / "matches.csv")
    else:
        raw, matches = load_matches()
    joined = join_interviews(raw, interviews)
    a, coverage = variant_a(matches, interviews, joined)
    b, b_info = variant_b(matches)
    c = variant_c(matches)
    variants = {"A": a, "B": b, "C": c}; manifests = {}; summaries = []
    for name, df in variants.items():
        text = ["context_text"] if name == "C" else ["player_a_text", "player_b_text"]
        structured = NUMERIC if name == "C" else NUMERIC + CONTEXT
        path = PRIVATE / ("variant_" + name + ".csv"); df.to_csv(path, index=False)
        manifest = {"variant":name, "target":"target", "structured_columns":structured, "text_columns":text,
                    "numeric_columns":NUMERIC, "audit_columns":[v for v in df if v not in structured + text + ["target"]],
                    "csv_sha256":digest(path.read_bytes()), "purpose":"research_prototype", "rows":len(df),
                    "task":"binary", "dataset_id":"BIN_TEXT_TENNIS_VARIANT_" + name,
                    "strict_publication_temporal_gate_passed": name != "A", "redistribution_cleared":False,
                    "prediction_cutoff":"start of reported target-match day; same-day source evidence excluded"}
        write_json(PRIVATE / ("variant_" + name + "_schema.json"), manifest); manifests[name] = manifest
        row = {"variant":name, "rows":len(df), "unique_matches":df.match_id.nunique() if len(df) else 0,
               "players":len(set(df.player_a_key) | set(df.player_b_key)) if len(df) else 0,
               "structured_columns":len(structured), "text_columns":len(text),
               "date_min":df.match_date.min() if len(df) else None, "date_max":df.match_date.max() if len(df) else None,
               "a_win_fraction":float(df.target.mean()) if len(df) else None,
               "text_unique":{col:int(df[col].nunique()) for col in text} if len(df) else {},
               "text_detector_on_full_population":{col:text_is_detected(df[col]) for col in text} if len(df) else {},
               "median_words":{col:float(df[col].str.split().str.len().median()) for col in text} if len(df) else {},
               "missing_text_percent":{col:float(df[col].isna().mean()*100) for col in text} if len(df) else {},
               "duplicate_text_fraction":{col:float(df[col].duplicated().mean()) for col in text} if len(df) else {}}
        summaries.append(row)
        if len(df):
            meta_cols=[v for v in manifest["audit_columns"] if v in df] + ["target"]
            df[meta_cols].head(50).to_csv(OUT / f"variant_{name}_sample_metadata.csv", index=False)
            # Small real CSV examples, including text, remain in the private cache.
            df.head(50).to_csv(PRIVATE / f"variant_{name}_sample50.csv", index=False)
    write_json(OUT / "variant_statistics.json", summaries)
    write_json(OUT / "join_summary.json", {"source_match_rows":len(raw), "dated_unique_target_matches":len(matches),
         "join_status_counts":joined.join_status.value_counts().to_dict(), "joined_original_interview_fraction":float((joined.join_status == "matched").mean()),
         "unique_interview_source_matches":int(joined.loc[joined.join_status == "matched", "match_id"].nunique()),
         "a_interview_sides_coverage":coverage,"b":b_info,
         "common_ABC_rows":len(set(a.match_id)&set(b.match_id)&set(c.match_id)),
         "common_AB_rows":len(set(a.match_id)&set(b.match_id)),
         "a_strict_publication_verified_rows":0, "a_provisional_date_safe_rows":len(a)})
    print((OUT / "join_summary.json").read_text()); print(json.dumps(summaries, indent=2))


if __name__ == "__main__": main()
