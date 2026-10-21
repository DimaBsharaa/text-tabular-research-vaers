"""As-of interview association and explicit history lineage; no model imports."""
from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd

from tennis.scripts.common import digest, SEED
from tennis.scripts.prototypes import NUMERIC

PRIMARY_LAG_DAYS = 3
MAX_AGE_DAYS = 90


def ordinal(day):
    return date.fromisoformat(str(day)).toordinal()


def accepted_documents(interviews, joins, matches):
    good = joins[joins.join_status == "matched"].copy()
    good["interview_id"] = good.interview_id.astype(int)
    docs = interviews.merge(good[["interview_id", "match_id", "player_key"]], on="interview_id", validate="one_to_one")
    source = matches[["match_id", "match_date", "comment"]].rename(columns={"match_date": "linked_match_date", "comment": "source_comment"})
    docs = docs.merge(source, on="match_id", validate="many_to_one")
    docs["date_ordinal"] = docs.date.map(ordinal)
    docs["source_date_agrees"] = docs.date.eq(docs.linked_match_date)
    docs["source_played"] = docs.source_comment.isin(["Completed", "Retired"])
    return docs


def asof_interviews(targets, docs, lag=PRIMARY_LAG_DAYS, played_only=True):
    pool = docs[docs.source_date_agrees & (docs.answer_words >= 70)].copy()
    if played_only:
        pool = pool[pool.source_played]
    groups = {(gender, key): list(group.sort_values(["date", "interview_id"]).itertuples())
              for (gender, key), group in pool.groupby(["gender", "player_key"])}
    records, excluded = [], []
    for row in targets.sort_values(["match_date", "match_id"]).to_dict("records"):
        day = ordinal(row["match_date"]); chosen = []; reasons = []
        for side in ["a", "b"]:
            candidates = [v for v in groups.get((row["gender"], row[f"player_{side}_key"]), [])
                          if lag <= day - v.date_ordinal <= MAX_AGE_DAYS and v.match_id != row["match_id"]]
            if not candidates:
                chosen.append(None); reasons.append(side + "_no_eligible_prior_interview"); continue
            latest = candidates[-1]
            if sum(v.date == latest.date for v in candidates) > 1:
                chosen.append(None); reasons.append(side + "_ambiguous_latest_day"); continue
            chosen.append(latest)
        if any(v is None for v in chosen):
            excluded.append({"match_id": row["match_id"], "match_date": row["match_date"], "reason": "|".join(reasons)})
            continue
        for side, doc in zip(["a", "b"], chosen):
            row[f"{side}_interview_id"] = doc.interview_id
            row[f"{side}_text_date"] = doc.date
            row[f"{side}_source_match_id"] = doc.match_id
            row[f"{side}_source_comment"] = doc.source_comment
            row[f"{side}_calendar_age_days"] = day - doc.date_ordinal
            row[f"{side}_assumed_available_day"] = date.fromordinal(doc.date_ordinal + lag).isoformat()
        row["publication_verified"] = False
        row["availability_basis"] = f"ASSUMPTION: >= {lag} calendar days after interview event; no verified upload timestamp"
        records.append(row)
    return pd.DataFrame(records), pd.DataFrame(excluded)


def next_match_audit(docs, matches):
    lookup = defaultdict(list)
    for row in matches.sort_values(["match_date", "match_id"]).itertuples():
        for side in ["a", "b"]:
            lookup[(row.gender, getattr(row, f"player_{side}_key"))].append(row)
    records = []
    for doc in docs.itertuples():
        options = [r for r in lookup[(doc.gender, doc.player_key)] if r.match_date > doc.date]
        rec = {"interview_id": doc.interview_id, "source_match_id": doc.match_id, "source_date": doc.date,
               "player": doc.player, "source_played": doc.source_played, "gender": doc.gender}
        if not options:
            rec["status"] = "no_later_recorded_match"
        else:
            next_day = options[0].match_date
            same_day = [r for r in options if r.match_date == next_day]
            rec.update(next_date=next_day, next_day_candidate_count=len(same_day), calendar_gap=ordinal(next_day)-doc.date_ordinal)
            if len(same_day) != 1:
                rec["status"] = "ambiguous_next_day_multiple_matches"
            else:
                target = same_day[0]
                rec.update(next_match_id=target.match_id, next_comment=target.comment)
                rec["status"] = "next_recorded_match_resolved"
        records.append(rec)
    return pd.DataFrame(records)


def replay_histories(raw, targets, include_walkovers=False):
    """Rebuild predictors with a row-level list of every contributing match ID.

    Rankings stay attached to the player in the original target record; dictionary
    contract: entry ranking at tournament start. Histories update only after the
    whole day. Rest/form use played Completed/Retired matches by default.
    """
    wanted = {row["match_id"]: row for row in targets.to_dict("records")}
    history = defaultdict(list); records = {}; lineage = []
    for day, group in raw.sort_values(["Date", "match_id"]).groupby("Date", sort=True):
        updates = []
        for source in group.itertuples():
            if source.match_id in wanted:
                row = wanted[source.match_id].copy()
                rank_by_player = {source.winner_key: source.WRank, source.loser_key: source.LRank}
                for side in ["a", "b"]:
                    player = row[f"player_{side}_key"]
                    h = history[(source.gender, player)]; recent = h[-10:]
                    surface = [v for v in h if v["surface"] == source.Surface]
                    rank = pd.to_numeric(rank_by_player[player], errors="coerce")
                    row[f"{side}_rank"] = float(rank) if pd.notna(rank) and rank > 0 else np.nan
                    row[f"{side}_prior_n"] = len(h)
                    row[f"{side}_form10"] = sum(v["won"] for v in recent)/len(recent) if recent else np.nan
                    row[f"{side}_surface_rate"] = sum(v["won"] for v in surface)/len(surface) if surface else np.nan
                    row[f"{side}_rest_days"] = ordinal(day)-ordinal(h[-1]["date"]) if h else np.nan
                    lineage.append({"match_id": source.match_id, "side": side, "player_key": player,
                        "cutoff_day": day, "history_max_day": h[-1]["date"] if h else None,
                        "history_match_ids": [v["match_id"] for v in h],
                        "recent10_match_ids": [v["match_id"] for v in recent],
                        "surface_match_ids": [v["match_id"] for v in surface]})
                records[source.match_id] = row
            if include_walkovers or source.Comment in {"Completed", "Retired"}:
                for player, won in [(source.winner_key, 1), (source.loser_key, 0)]:
                    updates.append(((source.gender, player), {"date": day, "surface": source.Surface,
                        "won": won, "match_id": source.match_id}))
        for player, item in updates:
            history[player].append(item)
    return pd.DataFrame([records[mid] for mid in targets.match_id if mid in records]), lineage
