"""Bounded source, publication and split audits. No fitting or benchmark edits.

Raw pages remain in the external cache. The 2015 alternative-source probe is a
coverage calculation, not a replacement dataset or verified forecasting CSV.
"""
import datetime as dt
import html
import json
import re
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split

from tennis.scripts.common import CACHE, OUT, PRIVATE, digest, normalized, write_json
from tennis.scripts.diagnostics import chronological_split

RAW = CACHE / "sources/interview_followup"
REVISION = "8ac86f7c62889b5d5c4cfec4c6403edcb8c5d993"
CHECKS = [
    ("0", "110504", "corroboration"),
    ("497", "112348", "corroboration"),
    ("854", "45075", "corroboration"),
    ("2557", "61198", "corroboration"),
    ("3406", "69159", "corroboration"),
    ("5733", "104385", "corroboration"),
    ("76", "105578", "corroboration"),
    ("5019", "86673", "withdrawal_not_post_played_match"),
    ("1499", "51230", "known_corpus_result_conflict_excluded"),
    ("0", "110501", "deliberately_wrong_same_day_interview"),
]


def visible_text(value):
    value = re.sub(r"<!--.*?-->", " ", value, flags=re.S)
    value = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", value, flags=re.S | re.I)
    value = html.unescape(re.sub(r"<[^>]+>", " ", value))
    # These publisher pages contain mojibake before nonbreaking spaces. Repair
    # only this observed sequence; preserve ordinary letters and substantive text.
    value = value.replace("\u00c2\u00a0", " ")
    return " ".join(value.split())


def publisher_audit():
    originals = json.loads((CACHE / "cornell/transcripts_matchinfo.json").read_text())
    rows = []
    for iid, sid, purpose in CHECKS:
        page = (RAW / f"asap_{sid}.html").read_text(errors="replace")
        text = visible_text(page)
        item = originals[iid]
        h2 = re.search(r"<h2[^>]*>(.*?)</h2>", page, re.S | re.I)
        h3 = re.search(r"<h3[^>]*>(.*?)</h3>", page, re.S | re.I)
        event_date = dt.datetime.strptime(visible_text(h2[1]), "%B %d, %Y").date().isoformat()
        player = visible_text(h3[1])
        prefixes = [" ".join(pair[1].split()[:20]) for pair in item["QandA"] if len(pair[1].split()) >= 20]
        first = normalized(prefixes[0]) in normalized(text) if prefixes else False
        matched = sum(normalized(prefix) in normalized(text) for prefix in prefixes)
        row = {"interview_id": int(iid), "publisher_id": sid, "purpose": purpose,
               "player": item["player"], "corpus_event_date": item["date"],
               "publisher_event_date": event_date, "date_agrees": event_date == item["date"],
               "player_agrees": normalized(player) == normalized(item["player"]),
               "first_long_answer_prefix_matches": first, "matched_answer_prefixes": matched,
               "tested_answer_prefixes": len(prefixes),
               "publication_meta_tag_found": bool(re.search(r"datePublished|article:published_time|dateCreated", page, re.I)),
               "publisher_url": f"https://www.asapsports.com/show_interview.php?id={sid}",
               "raw_page_sha256": digest((RAW / f"asap_{sid}.html").read_bytes()),
               "historical_available_by_utc": None, "historical_availability_verified": False,
               "redistribution_permission_verified": False}
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "interview_publisher_validation.csv", index=False)
    return frame


def split_audit(train, test, name):
    def docs(rows):
        return set(rows.a_interview_id.astype(int)) | set(rows.b_interview_id.astype(int))
    train_docs, test_docs = docs(train), docs(test)
    train_refs = set(train.a_source_match_id) | set(train.b_source_match_id)
    test_targets = set(test.match_id)
    reference_mask = train.a_source_match_id.isin(test_targets) | train.b_source_match_id.isin(test_targets)
    reused = test.a_interview_id.isin(train_docs) | test.b_interview_id.isin(train_docs)
    train_players = set(train.player_a_key) | set(train.player_b_key)
    test_players = set(test.player_a_key) | set(test.player_b_key)
    out = {"split": name, "train_rows": len(train), "test_rows": len(test),
           "train_max_day": train.match_date.max(), "test_min_day": test.match_date.min(),
           "train_test_target_overlap": len(set(train.match_id) & test_targets),
           "shared_interview_documents": len(train_docs & test_docs),
           "test_rows_with_training_interview": int(reused.sum()),
           "test_fraction_with_training_interview": float(reused.mean()),
           "shared_players": len(train_players & test_players),
           "heldout_targets_referenced_in_training_source_matches": len(train_refs & test_targets),
           "training_rows_referencing_a_heldout_target": int(reference_mask.sum())}
    examples = []
    test_by_id = test.set_index("match_id")
    for row in train.loc[reference_mask].head(5).itertuples():
        for side in ["a", "b"]:
            ref = getattr(row, f"{side}_source_match_id")
            if ref in test_targets:
                examples.append({"split": name, "training_match_id": row.match_id,
                    "training_match_date": row.match_date, "text_side": side,
                    "training_interview_id": getattr(row, f"{side}_interview_id"),
                    "referenced_test_match_id": ref,
                    "referenced_test_match_date": test_by_id.loc[ref, "match_date"]})
    return out, pd.DataFrame(examples)


def provisional_sensitivity(frame):
    # A lag is a scenario, never evidence of when a historical page went online.
    ages = pd.concat([(pd.to_datetime(frame.match_date) - pd.to_datetime(frame[f"{side}_text_date"])).dt.days
                      for side in ["a", "b"]], axis=1)
    lag_rows = [{"minimum_calendar_gap_days": lag,
                 "candidate_rows": int((ages.min(axis=1) >= lag).sum()),
                 "publication_verified_rows": 0, "assumption_only": True}
                for lag in [1, 2, 3, 7, 14]]
    pd.DataFrame(lag_rows).to_csv(OUT / "interview_lag_sensitivity.csv", index=False)
    chron_train, chron_test, cutoff = chronological_split(frame)
    random_train, random_test = train_test_split(frame, test_size=.2, stratify=frame.target, random_state=42)
    rows, examples = [], []
    for train, test, name in [(chron_train, chron_test, "chronological_80_20"),
                              (random_train, random_test, "random_80_20_seed42_diagnostic")]:
        row, ex = split_audit(train, test, name)
        rows.append(row); examples.append(ex)
    pd.DataFrame(rows).to_csv(OUT / "interview_split_risks.csv", index=False)
    pd.concat(examples).to_csv(OUT / "interview_split_reference_examples.csv", index=False)
    return lag_rows, rows, cutoff


def latest_unique_before(documents, cutoff, max_age=90, age_reference=None):
    age_reference = cutoff if age_reference is None else age_reference
    eligible = documents[(documents.date < cutoff) &
                         (documents.date >= age_reference - pd.Timedelta(days=max_age))]
    if eligible.empty:
        return None
    latest = eligible[eligible.date == eligible.date.max()]
    return latest.iloc[0] if len(latest) == 1 else None


def alternative_coverage():
    """Count pair coverage on two small 2015 files, with no training CSV output.

    The source dictionary says 'usually' Monday: neither week anchor nor a
    seven-day margin is certified as an actual pre-match cutoff. They remain
    explicit coverage scenarios pending verified event/draw dates.
    """
    meta = pd.read_csv(OUT / "cornell_interview_metadata.csv")
    meta["date"] = pd.to_datetime(meta.date)
    # Known content conflict is not silently retained in the source probe.
    meta = meta[meta.interview_id != 1499].copy()
    meta["name_key"] = meta.player.map(normalized)
    lookup = {(gender, key): group for (gender, key), group in meta.groupby(["gender", "name_key"])}
    records, summary = [], []
    for gender, prefix in [("M", "atp"), ("F", "wta")]:
        df = pd.read_csv(RAW / f"sackmann_{prefix}_2015.csv")
        # Exclude team events / finals; their round/draw semantics differ.
        df = df[~df.tourney_level.isin(["D", "F", "T", "E"])].copy()
        df["anchor"] = pd.to_datetime(df.tourney_date.astype(str), format="%Y%m%d")
        df["round_size"] = pd.to_numeric(df["round"].str.extract(r"^R(\d+)$")[0], errors="coerce")
        df["first_round"] = df.round_size.eq(df.groupby("tourney_id").round_size.transform("max"))
        df["completed"] = df.score.notna() & ~df.score.str.contains(r"RET|W/O|DEF|ABD", case=False, na=False)
        for margin in [0, 7, 14]:
            for row in df.itertuples():
                cutoff = row.anchor - pd.Timedelta(days=margin)
                sides = []
                for name in [row.winner_name, row.loser_name]:
                    group = lookup.get((gender, normalized(name)))
                    doc = None if group is None else latest_unique_before(group, cutoff, age_reference=row.anchor)
                    sides.append(None if doc is None else int(doc.interview_id))
                records.append({"gender": gender, "margin_days_before_week_anchor": margin,
                    "tourney_id": row.tourney_id, "tourney_name": row.tourney_name,
                    "match_num": row.match_num, "first_round": row.first_round,
                    "completed": row.completed, "cutoff_scenario": cutoff.date().isoformat(),
                    "side_1_interview_id": sides[0], "side_2_interview_id": sides[1],
                    "both_recent_interviews": all(v is not None for v in sides),
                    "publication_verified": False, "cutoff_verified": False})
    frame = pd.DataFrame(records)
    for (gender, margin), group in frame.groupby(["gender", "margin_days_before_week_anchor"]):
        for scope, selected in [("all_completed_conditional_pairings", group[group.completed]),
                                 ("first_round_completed", group[group.completed & group.first_round]),
                                 ("grand_slam_first_round_completed", group[group.completed & group.first_round &
                                                                      group.tourney_name.isin(["Australian Open", "Roland Garros", "Wimbledon", "US Open"])])]:
            summary.append({"gender": gender, "margin_days_before_week_anchor": margin, "scope": scope,
                "source_matches": len(selected), "both_recent_interviews": int(selected.both_recent_interviews.sum()),
                "publication_verified_rows": 0, "forecasting_cutoff_verified_rows": 0})
    pd.DataFrame(summary).to_csv(OUT / "interview_alternative_source_coverage.csv", index=False)
    sample = frame[(frame.margin_days_before_week_anchor == 7) & frame.first_round & frame.completed & frame.both_recent_interviews]
    sample.head(12).to_csv(OUT / "interview_alternative_source_sample_metadata.csv", index=False)
    return summary


def source_manifest():
    records = []
    for item in json.loads((RAW / "requested_sources.json").read_text()):
        path = RAW / item["filename"]
        headers = (RAW / (item["name"] + ".headers")).read_text()
        codes = re.findall(r"^HTTP/[\d.]+ (\d+)", headers, re.M)
        observed = re.findall(r"^date: (.+)", headers, re.I | re.M)
        records.append(item | {"bytes": path.stat().st_size, "sha256": digest(path.read_bytes()),
            "http_status": int(codes[-1]), "response_date_header": observed[-1].strip() if observed else None,
            "response_date_is_not_historical_publication": True})
    for filename, source in [("archive_cdx_110504.json", "/tmp/tennis-publication-cdx.json"),
                              ("archive_available_110504.html", "/tmp/tennis-publication-wayback.json")]:
        path = RAW / filename
        if not path.exists() and __import__("pathlib").Path(source).exists():
            shutil.copyfile(source, path)
    write_json(OUT / "interview_followup_sources.json", {"retrieved_on": "2026-10-15", "mirror_revision": REVISION,
        "sources": records, "note": "Raw pages and 2015 files kept in external cache; no transcript text in this manifest."})
    return records


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    checks = publisher_audit()
    a = pd.read_csv(PRIVATE / "variant_A.csv")
    lags, splits, cutoff = provisional_sensitivity(a)
    coverage = alternative_coverage()
    sources = source_manifest()
    target_ids = set(a.match_id)
    disjoint = a[~a.a_source_match_id.isin(target_ids) & ~a.b_source_match_id.isin(target_ids)]
    policy = visible_text((RAW / "tennis_data_home.html").read_text(errors="replace"))
    restriction = "NOT commerical or data training products using automated bots/scrapers/AI" in policy
    assert restriction, "Re-read changed source terms rather than reuse a stale finding"
    summary = {"audited_on": "2026-10-15", "provisional_A_rows": len(a),
        "publisher_checks_including_negative_control": len(checks), "unique_cornell_interviews_checked": int(checks.interview_id.nunique()),
        "publisher_content_corroborations": int(checks.first_long_answer_prefix_matches.sum()),
        "publisher_publication_meta_tags_found": int(checks.publication_meta_tag_found.sum()),
        "strict_publication_verified_A_rows": 0, "raw_text_redistribution_cleared": False,
        "tennis_data_training_restriction_observed": restriction,
        "sackmann_upstream_github_http_status": {"atp": 404, "wta": 404},
        "sackmann_archive_license_notice": "CC BY-NC-SA 4.0; preserved upstream README and archive LICENSE",
        "sackmann_probe_scope": "2015 ATP and WTA only; coverage scenarios, no forecasting CSV or fitted model",
        "chronological_cutoff_day": cutoff, "lag_sensitivity": lags, "split_risks": splits,
        "source_target_disjoint_coverage_control": {
            "rule": "Retain rows only when neither source sporting-match ID belongs to the original 4,052 target IDs; fixed before any model fitting.",
            "remaining_rows": len(disjoint), "removed_rows": len(a) - len(disjoint),
            "limitations": "Coverage control only. Rank/form dependencies, repeated documents, publication and rights still need resolution. No replacement dataset saved or fitted."},
        "new_download_body_bytes": sum(r["bytes"] for r in sources),
        "archive_probe": {"interview_id": 0, "publisher_id": 110504,
            "exact_www_url_cdx_to_2016": json.loads((RAW / "archive_cdx_110504.json").read_text()),
            "availability_api_http_status": 429,
            "interpretation": "No usable pre-target capture recovered in this one-URL probe. This is not evidence of absence."},
        "decision": "Do not promote A to training or public text release yet. Resolve source terms and historical availability; retain conservative alternative as coverage probe."}
    write_json(OUT / "interview_followup_audit.json", summary)
    print(json.dumps(summary, indent=2, default=str))
    print(pd.DataFrame(coverage).to_string(index=False))


if __name__ == "__main__":
    main()
