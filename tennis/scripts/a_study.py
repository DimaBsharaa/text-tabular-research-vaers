"""Cornell A v1: fixed text policies, conservative dates and local diagnostics."""
from collections import Counter
import gzip
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from tennis.scripts.a_temporal import accepted_documents, asof_interviews, next_match_audit, replay_histories, PRIMARY_LAG_DAYS
from tennis.scripts.a_text import POLICIES, TOPICS, MODEL_REVISION, RULE_VERSION, TOKENIZER_DIR, select
from tennis.scripts.common import CACHE, OUT, PRIVATE, ROOT, digest, write_json
from tennis.scripts.diagnostics import preprocessor
from tennis.scripts.prototypes import NUMERIC, CONTEXT

STUDY = PRIVATE / "candidate_A_v1"
SELECTED_POLICY = "topical_answers"
FROZEN_CUTOFFS = {"development_start": "2012-10-13", "reserved_test_start": "2014-06-19"}


def assign_split(frame):
    split = pd.Series("train", index=frame.index)
    split.loc[frame.match_date >= FROZEN_CUTOFFS["development_start"]] = "development"
    split.loc[frame.match_date >= FROZEN_CUTOFFS["reserved_test_start"]] = "reserved_test"
    return split


def load_inputs():
    base = pd.read_csv(PRIVATE / "variant_A.csv")
    interviews = pd.read_csv(PRIVATE / "interviews_with_answers.csv")
    joins = pd.read_csv(OUT / "cornell_match_join_audit.csv")
    matches = pd.read_csv(PRIVATE / "matches.csv")
    raw = pd.read_csv(PRIVATE / "structured_source_rows.csv", low_memory=False)
    original = json.loads((CACHE / "cornell/transcripts_matchinfo.json").read_text())
    return base, interviews, joins, matches, raw, original


def freeze_config():
    recorded = OUT / "a_v1_artifact_manifest.json"
    if recorded.exists():
        inputs = {"variant_A.csv": PRIVATE / "variant_A.csv",
                  "structured_source_rows.csv": PRIVATE / "structured_source_rows.csv",
                  "transcripts_matchinfo.json": CACHE / "cornell/transcripts_matchinfo.json"}
        for entry in json.loads(recorded.read_text())["inputs"]:
            if digest(inputs[entry["filename"]].read_bytes()) != entry["sha256"]:
                raise ValueError("Inputs differ from the audited study; start a separately versioned experiment")
    config = {"study": "Cornell A v1", "selected_policy": SELECTED_POLICY,
        "rule_version": RULE_VERSION, "lag_days": PRIMARY_LAG_DAYS,
        "source_publication_time_verified": False,
        "availability_interpretation": "Assume online availability within a three-calendar-day buffer; local dates and upload times unresolved. Not proof of publication.",
        "history_inclusion": ["Completed", "Retired"], "walkovers_count_as_played": False,
        "cutoffs": FROZEN_CUTOFFS, "token_limit_including_prefix_and_special_tokens": 512,
        "tokenizer_revision": MODEL_REVISION, "text_policies": POLICIES,
        "selector_source_sha256": digest((ROOT / "tennis/scripts/a_text.py").read_bytes()),
        "temporal_source_sha256": digest((ROOT / "tennis/scripts/a_temporal.py").read_bytes()),
        "diagnostic": {"learner": "CPU LogisticRegression", "text": "TF-IDF, NOT E5",
            "C": 1, "solver": "liblinear", "max_iter": 500, "seed": 42,
            "development_only": True, "reserved_test_scoring": False,
            "policy_selected_before_scores": True, "official_curation_claim": False},
        "no_outreach": True, "no_push": True, "official_code_changes": False}
    encoded = json.dumps(config, sort_keys=True)
    config["config_sha256"] = digest(encoded)
    path = STUDY / "policy_frozen_before_diagnostics.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Study settings changed. Start an explicitly versioned study; don't silently tune this one.")
    write_json(path, config)
    write_json(OUT / "a_v1_policy.json", config)
    return config


def construct(base, docs, matches, raw):
    sensitivities = []
    for lag in [1, 2, 3, 7]:
        frame, excluded = asof_interviews(base, docs, lag)
        original_ids = base.set_index("match_id")
        changed = frame.a_interview_id.ne(frame.match_id.map(original_ids.a_interview_id)) | frame.b_interview_id.ne(frame.match_id.map(original_ids.b_interview_id))
        sensitivities.append({"lag_days": lag, "original_target_pool": len(base), "retained_rows": len(frame),
            "excluded_rows": len(excluded), "rows_using_older_eligible_interview": int(changed.sum()),
            "publication_verified_rows": 0, "rule_is_assumption": True})
        if lag == PRIMARY_LAG_DAYS:
            primary, reasons = frame, excluded
    pd.DataFrame(sensitivities).to_csv(OUT / "a_v1_date_sensitivity.csv", index=False)
    reasons.to_csv(OUT / "a_v1_excluded_targets.csv", index=False)
    rebuilt, lineage = replay_histories(raw, primary)
    # The as-of join may replace an interview with an older eligible one.
    # Never carry the old v0 passage beside a replacement source ID.
    rebuilt = rebuilt.drop(columns=["player_a_text", "player_b_text"], errors="ignore")
    with gzip.open(STUDY / "history_lineage.jsonl.gz", "wt") as stream:
        for item in lineage:
            stream.write(json.dumps(item) + "\n")
    rebuilt["split"] = assign_split(rebuilt)
    rebuilt.to_csv(STUDY / "metadata.csv", index=False)
    next_links = next_match_audit(docs, matches)
    next_links.to_csv(OUT / "a_v1_next_match_links.csv", index=False)
    next_counts = []
    resolved = next_links[next_links.status == "next_recorded_match_resolved"]
    for lag in [1, 2, 3, 7]:
        accepted = resolved[resolved.source_played & resolved.next_comment.eq("Completed") &
                            resolved.calendar_gap.between(lag, 90)]
        next_counts.append({"lag_days": lag, "interview_next_match_links": len(accepted),
            "unique_next_target_matches": accepted.next_match_id.nunique(),
            "links_in_original_4052_pool": int(accepted.next_match_id.isin(base.match_id).sum()),
            "not_a_two_sided_match_dataset_count": True})
    pd.DataFrame(next_counts).to_csv(OUT / "a_v1_next_match_counts.csv", index=False)
    next_map = resolved.set_index("interview_id").next_match_id
    for item in sensitivities:
        subset, _ = asof_interviews(base, docs, item["lag_days"])
        a_next = subset.a_interview_id.map(next_map).eq(subset.match_id)
        b_next = subset.b_interview_id.map(next_map).eq(subset.match_id)
        item["rows_both_sources_are_immediate_previous_match"] = int((a_next & b_next).sum())
        item["rows_any_source_is_immediate_previous_match"] = int((a_next | b_next).sum())
    pd.DataFrame(sensitivities).to_csv(OUT / "a_v1_date_sensitivity.csv", index=False)
    return rebuilt, lineage, sensitivities, next_links


def prepare_texts(frame, original):
    doc_ids = sorted(set(frame.a_interview_id.astype(int)) | set(frame.b_interview_id.astype(int)))
    train = frame[frame.split == "train"]
    train_ids = set(train.a_interview_id.astype(int)) | set(train.b_interview_id.astype(int))
    cache, metrics = {}, []
    output = STUDY / "passages_private.jsonl.gz"
    with gzip.open(output, "wt") as stream:
        for idx, iid in enumerate(doc_ids):
            qa = original[str(iid)]["QandA"]
            for policy in POLICIES:
                selection = select(qa, policy)
                cache[(iid, policy)] = selection
                stream.write(json.dumps({"interview_id": iid} | selection) + "\n")
                item = {"interview_id": iid, "policy": policy, "in_training_documents": iid in train_ids,
                    "formatted_tokens_a": selection["selected_tokens_a"], "formatted_tokens_b": selection["selected_tokens_b"],
                    "full_formatted_tokens": selection["full_formatted_tokens"],
                    "legacy_was_over_token_limit": selection["legacy_was_over_token_limit"],
                    "source_answer_sha256": selection["source_answer_sha256"], "selected_sha256": selection["selected_sha256"]}
                for topic in TOPICS:
                    item[topic + "_full"] = selection["full_topics"][topic]
                    item[topic + "_selected"] = selection["selected_topics"][topic]
                metrics.append(item)
            if (idx + 1) % 500 == 0:
                print("Selected policies for", idx + 1, "/", len(doc_ids), "unique interviews", flush=True)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(OUT / "a_v1_text_policy_metrics.csv", index=False)
    summary = []
    for policy, group in metrics[metrics.in_training_documents].groupby("policy", sort=False):
        item = {"policy": policy, "training_documents": len(group),
            "median_formatted_tokens": float(group.formatted_tokens_a.median()),
            "max_formatted_tokens": int(group[["formatted_tokens_a", "formatted_tokens_b"]].max().max()),
            "legacy_over_limit_docs": int(group.legacy_was_over_token_limit.sum()) if policy == "legacy300" else 0}
        for topic in TOPICS:
            full = group[topic + "_full"]
            item[topic + "_retention"] = float(group.loc[full, topic + "_selected"].mean()) if full.any() else None
        summary.append(item)
    pd.DataFrame(summary).to_csv(OUT / "a_v1_text_policy_summary.csv", index=False)
    for policy in POLICIES:
        exported = frame[["match_id"] + NUMERIC + CONTEXT + ["target"]].copy()
        for side in ["a", "b"]:
            exported[f"player_{side}_text"] = frame[f"{side}_interview_id"].map(lambda iid: cache[(int(iid), policy)]["text"])
        exported.to_csv(STUDY / f"features_{policy}.csv", index=False)
    audit_cols = [c for c in frame if c not in NUMERIC + CONTEXT + ["target", "player_a_text", "player_b_text"]]
    frame[audit_cols].to_csv(STUDY / "audit.csv", index=False)
    frame[["match_id", "match_date", "split"]].to_csv(STUDY / "splits.csv", index=False)
    manifest = {"dataset_id": "BIN_TEXT_TENNIS_CORNELL_A_V1", "target": "target", "task": "binary",
        "rows": len(frame), "structured_columns": NUMERIC + CONTEXT, "numeric_columns": NUMERIC,
        "text_columns": ["player_a_text", "player_b_text"], "audit_columns": ["match_id"],
        "csv_sha256": digest((STUDY / f"features_{SELECTED_POLICY}.csv").read_bytes()),
        "audit_file": "audit.csv", "audit_csv_sha256": digest((STUDY / "audit.csv").read_bytes()),
        "availability_lag_days": PRIMARY_LAG_DAYS, "split_file": "splits.csv",
        "selected_policy": SELECTED_POLICY, "availability_policy": "documented_calendar_delay_assumption",
        "strict_publication_temporal_gate_passed": False, "conditional_calendar_policy_passed": True,
        "redistribution_cleared": False, "prediction_cutoff": "Start of reported target-match local day; timezone and publication assumptions disclosed",
        "purpose": "private_retrospective_research_under_explicit_assumptions"}
    write_json(STUDY / "schema.json", manifest)
    return cache, metrics, summary


def model_fingerprint(pipeline):
    prep = pipeline.named_steps["columntransformer"]
    evidence = {}
    for name, transformer, cols in prep.transformers_:
        if name == "remainder":
            continue
        if name == "numeric":
            evidence[name] = {"medians": transformer.named_steps["simpleimputer"].statistics_.tolist(),
                "scaler_mean": transformer.named_steps["standardscaler"].mean_.tolist()}
        elif name.startswith("text"):
            evidence[name] = {"vocabulary_sha256": digest(json.dumps({k: int(v) for k, v in transformer.vocabulary_.items()}, sort_keys=True)),
                              "idf_sha256": digest(transformer.idf_.tobytes()),
                              "has_validation_sentinel": "zzvalidationonlysentinel" in transformer.vocabulary_}
        elif name == "categorical":
            evidence[name] = {"categories": [list(v) for v in transformer.named_steps["onehotencoder"].categories_]}
    return evidence, digest(json.dumps(evidence, sort_keys=True))


def diagnostics(frame, config):
    train_meta = frame[frame.split == "train"]; dev_meta = frame[frame.split == "development"]
    records, fit_records = [], []
    for policy in POLICIES:
        data = pd.read_csv(STUDY / f"features_{policy}.csv").set_index("match_id")
        train = data.loc[train_meta.match_id].copy(); dev = data.loc[dev_meta.match_id].copy()
        for d in [train, dev]:
            for col in CONTEXT:
                if col != "best_of":
                    d[col] = d[col].fillna("missing").astype(str)
        conditions = ["text_tfidf", "joint_tfidf"] + (["structured"] if policy == POLICIES[0] else [])
        for condition in conditions:
            structured = NUMERIC + CONTEXT if condition != "text_tfidf" else []
            text = ["player_a_text", "player_b_text"] if condition != "structured" else []
            prep = preprocessor(structured, NUMERIC + ["best_of"], text)
            model = make_pipeline(prep, LogisticRegression(C=1, solver="liblinear", max_iter=500, random_state=42))
            started = time.perf_counter()
            model.fit(train[structured + text], train.target)
            parameters_before = digest(model.named_steps["logisticregression"].coef_.tobytes())
            fit_evidence, prep_before = model_fingerprint(model)
            probabilities = model.predict_proba(dev[structured + text])[:, 1]
            prep_after = model_fingerprint(model)[1]
            pd.DataFrame({"match_id": dev.index, "target": dev.target,
                          "probability": probabilities}).to_csv(OUT / f"a_v1_dev_{policy}_{condition}_predictions.csv", index=False)
            records.append({"policy": policy, "condition": condition, "auroc": float(roc_auc_score(dev.target, probabilities)),
                "train_rows": len(train), "development_rows": len(dev), "reserved_test_rows_scored": 0,
                "runtime_seconds": time.perf_counter() - started, "official_result": False,
                "learner": "CPU LogisticRegression + TF-IDF; NOT E5/TAR", "config_sha256": config["config_sha256"]})
            # Injection stays exclusively in an evaluation copy; neither a .fit
            # nor target-aware passage-selection call occurs on that copy.
            changed_dev = dev.copy()
            for col in text:
                changed_dev[col] = "zzvalidationonlysentinel"
            for col in [v for v in structured if v in NUMERIC]:
                changed_dev[col] = 1e12
            model.predict_proba(changed_dev[structured + text])
            evidence_after = model_fingerprint(model)[1]
            fit_records.append({"policy": policy, "condition": condition, "fit_row_ids": train.index.tolist(),
                "development_row_ids": dev.index.tolist(), "fit_evidence": fit_evidence,
                "preprocessor_sha_before_evaluation": prep_before, "preprocessor_sha_after_evaluation": prep_after,
                "preprocessor_sha_after_evaluation_injection": evidence_after,
                "model_parameters_unchanged_by_evaluation": parameters_before == digest(model.named_steps["logisticregression"].coef_.tobytes()),
                "reserved_test_rows_used_for_fit_or_score": 0})
            print("Diagnostic", policy, condition, round(records[-1]["auroc"], 4), flush=True)
    pd.DataFrame(records).to_csv(OUT / "a_v1_development_scores.csv", index=False)
    write_json(OUT / "a_v1_preprocessing_fit_audit.json", fit_records)
    return records


def main():
    STUDY.mkdir(parents=True, exist_ok=True)
    config = freeze_config()
    base, interviews, joins, matches, raw, original = load_inputs()
    docs = accepted_documents(interviews, joins, matches)
    frame, lineage, sensitivities, next_links = construct(base, docs, matches, raw)
    print("A v1 rows:", len(frame), "from original", len(base), flush=True)
    cache, metrics, text_summary = prepare_texts(frame, original)
    scores = diagnostics(frame, config)
    summary = {"original_rows": len(base), "v1_rows": len(frame), "lag_days_assumed": PRIMARY_LAG_DAYS,
        "split_counts": frame.split.value_counts().to_dict(), "unique_source_interviews": metrics.interview_id.nunique(),
        "history_lineage_rows": len(lineage), "selected_policy": SELECTED_POLICY, "policy_selected_before_diagnostics": True,
        "actual_classifier_fits": len(scores), "E5_models_loaded": 0, "TAR_runs": 0,
        "reserved_test_model_scores_computed": 0, "publication_verified_rows": 0,
        "conditional_calendar_rule_passed": True, "dates_block_research_automatically": False,
        "config_sha256": config["config_sha256"], "private_directory": str(STUDY)}
    if (PRIVATE / "candidate_A_v1_superseded_debug" / "debug_attempts.json").exists():
        summary["superseded_debug_attempts"] = json.loads((PRIVATE / "candidate_A_v1_superseded_debug" / "debug_attempts.json").read_text())
    write_json(OUT / "a_v1_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
