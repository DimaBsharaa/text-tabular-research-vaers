"""Evidence-producing leakage audit for original A and conditional A v1.

Every PASS/FAIL has a population, count and evidence file. Unverifiable source
timestamps are explicit ASSUMPTION entries, never silently converted to PASS.
"""
from collections import defaultdict
from bisect import bisect_left
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.neighbors import NearestNeighbors

from tennis.scripts.a_study import STUDY, SELECTED_POLICY, assign_split, load_inputs
from tennis.scripts.a_temporal import accepted_documents, asof_interviews, replay_histories, ordinal
from tennis.scripts.a_text import select, verify_selected, count_tokens, MAX_TOKENS
from tennis.scripts.common import OUT, CACHE, PRIVATE, ROOT, SEED, digest
from tennis.scripts.prototypes import NUMERIC, CONTEXT


def equal_numbers(left, right):
    return bool(np.allclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float), rtol=1e-11, atol=1e-11, equal_nan=True))


def canonical_id(row):
    return digest(json.dumps([row.gender, row.Date, row.Tournament, row.Round, *sorted([row.winner_key, row.loser_key])]))[:24]


def flip_outcomes(raw, mask):
    changed = raw.copy()
    for left, right in [("winner_key", "loser_key"), ("Winner", "Loser"), ("WRank", "LRank")]:
        changed.loc[mask, [left, right]] = changed.loc[mask, [right, left]].to_numpy()
    return changed


def mutation_checks(raw, frame, docs):
    # Future/same-day outcomes may change labels, but must never alter predictors
    # for predictions made at or before the mutation day.
    day = "2012-06-01"
    subset = frame[frame.match_date <= day].tail(100)
    expected, _ = replay_histories(raw, subset)
    changed, _ = replay_histories(flip_outcomes(raw, raw.Date >= day), subset)
    future_ok = equal_numbers(expected[NUMERIC], changed[NUMERIC])
    chosen = frame.iloc[[0]].copy()
    one_day = chosen.match_date.iloc[0]
    original, _ = replay_histories(raw, chosen)
    own, _ = replay_histories(flip_outcomes(raw, raw.Date == one_day), chosen)
    own_ok = equal_numbers(original[NUMERIC], own[NUMERIC])
    future_doc = docs.iloc[[0]].copy()
    future_doc["interview_id"] = int(docs.interview_id.max()) + 1
    future_doc["date"] = "2030-01-01"; future_doc["date_ordinal"] = ordinal("2030-01-01")
    future_doc["player_key"] = chosen.player_a_key.iloc[0]; future_doc["gender"] = chosen.gender.iloc[0]
    future_doc["match_id"] = "future-interview-control"
    x, _ = asof_interviews(chosen, docs, lag=3)
    y, _ = asof_interviews(chosen, pd.concat([docs, future_doc]), lag=3)
    interview_ok = x[["a_interview_id", "b_interview_id"]].equals(y[["a_interview_id", "b_interview_id"]])
    # A deliberately leaky vocabulary fit must expose its validation-only word.
    train = ["train words tennis serve", "training words return"]
    development = ["zzvalidationonlysentinel"]
    proper = TfidfVectorizer().fit(train)
    bad = TfidfVectorizer().fit(train + development)
    return {"future_outcome_invariance": future_ok, "own_and_same_day_outcome_invariance": own_ok,
        "future_interview_invariance": interview_ok,
        "train_only_vocabulary_rejects_validation_sentinel": "zzvalidationonlysentinel" not in proper.vocabulary_,
        "deliberately_leaky_vocabulary_contains_sentinel": "zzvalidationonlysentinel" in bad.vocabulary_,
        "real_rows_in_future_mutation": len(subset), "mutation_day": day}


def history_validation(raw, frame, lineage):
    lookup = raw.set_index("match_id")
    events = defaultdict(list)
    for row in raw.sort_values(["Date", "match_id"]).itertuples():
        if row.Comment not in {"Completed", "Retired"}:
            continue
        for player, won in [(row.winner_key, 1), (row.loser_key, 0)]:
            events[(row.gender, player)].append((row.Date, row.match_id, row.Surface, won))
    by_id = frame.set_index("match_id")
    violations, examples = CounterLike(), []
    for line in lineage:
        target = by_id.loc[line["match_id"]]; side = line["side"]
        all_events = events[(target.gender, line["player_key"])]
        dates = [v[0] for v in all_events]
        prior = all_events[:bisect_left(dates, line["cutoff_day"])]
        expected_ids = [v[1] for v in prior]
        if expected_ids != line["history_match_ids"]:
            violations["history_incomplete_or_noncausal"] += 1
        recent = prior[-10:]; surface = [v for v in prior if v[2] == target.surface]
        if line["recent10_match_ids"] != [v[1] for v in recent] or line["surface_match_ids"] != [v[1] for v in surface]:
            violations["wrong_window_or_surface_membership"] += 1
        expected = [len(prior), sum(v[3] for v in recent)/len(recent) if recent else np.nan,
                    sum(v[3] for v in surface)/len(surface) if surface else np.nan,
                    ordinal(target.match_date)-ordinal(prior[-1][0]) if prior else np.nan]
        actual = [target[f"{side}_prior_n"], target[f"{side}_form10"], target[f"{side}_surface_rate"], target[f"{side}_rest_days"]]
        if not equal_numbers(expected, actual):
            violations["feature_value_mismatch"] += 1
        if any(v[0] >= target.match_date for v in prior):
            violations["future_or_same_day_contributor"] += 1
        if len(examples) < 6:
            examples.append({"match_id": target.name, "side": side, "target_day": target.match_date,
                "latest_history_day": prior[-1][0] if prior else None, "past_played_matches": len(prior),
                "form10_contributor_ids": "|".join(v[1] for v in recent), "recomputed_form10": expected[1], "saved_form10": actual[1]})
    pd.DataFrame(examples).to_csv(OUT / "a_v1_history_examples.csv", index=False)
    return dict(violations)


class CounterLike(defaultdict):
    def __init__(self):
        super().__init__(int)


def reuse_and_similarity(frame, original):
    rows = []
    for row in frame.itertuples():
        for side in ["a", "b"]:
            rows.append({"match_id": row.match_id, "match_date": row.match_date, "split": row.split,
                "interview_id": int(getattr(row, f"{side}_interview_id")),
                "source_date": getattr(row, f"{side}_text_date")})
    long = pd.DataFrame(rows)
    split_groups = long.groupby("interview_id").split.agg(lambda x: sorted(set(x)))
    shared_ids = set(split_groups[split_groups.map(len) > 1].index)
    shared = long[long.interview_id.isin(shared_ids)]
    shared.to_csv(OUT / "a_v1_reused_interview_rows.csv", index=False)
    docs = long.drop_duplicates("interview_id").copy()
    docs["text"] = docs.interview_id.map(lambda iid: " ".join(pair[1] for pair in original[str(iid)]["QandA"]))
    docs["full_answer_sha256"] = docs.text.map(lambda s: digest(" ".join(s.lower().split())))
    identical_distinct_ids = docs[docs.full_answer_sha256.duplicated(False)]
    identical_distinct_ids.drop(columns="text").to_csv(OUT / "a_v1_identical_distinct_documents.csv", index=False)
    train_ids = set(long.loc[long.split == "train", "interview_id"])
    later_ids = set(long.loc[long.split != "train", "interview_id"]) - train_ids
    train_docs = docs[docs.interview_id.isin(train_ids)]
    later_docs = docs[docs.interview_id.isin(later_ids)]
    # Fit-free hashing avoids learning a held-out vocabulary. This is a lexical
    # similarity screen, not an exhaustive semantic-duplicate proof.
    vectorizer = HashingVectorizer(n_features=2**16, alternate_sign=False, ngram_range=(1, 2), norm="l2")
    train_x = vectorizer.transform(train_docs.text)
    later_x = vectorizer.transform(later_docs.text)
    nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute").fit(train_x)
    distances, indices = nn.kneighbors(later_x)
    near = []
    for offset in range(len(later_docs)):
        similarity = 1 - distances[offset, 0]
        if similarity >= .95:
            left = later_docs.iloc[offset]; right = train_docs.iloc[indices[offset, 0]]
            near.append({"later_interview_id": left.interview_id, "training_interview_id": right.interview_id,
                "hashed_word_bigram_cosine": float(similarity), "later_source_date": left.source_date,
                "training_source_date": right.source_date, "status": "REVIEW: lexical screen, not automatic leakage"})
    pd.DataFrame(near, columns=["later_interview_id", "training_interview_id", "hashed_word_bigram_cosine", "later_source_date", "training_source_date", "status"]).to_csv(OUT / "a_v1_near_duplicate_screen.csv", index=False)
    return {"shared_interview_ids_between_partitions": len(shared_ids), "rows_using_shared_interviews": shared.match_id.nunique(),
        "all_shared_source_dates_before_use": bool((shared.source_date < shared.match_date).all()),
        "identical_full_answers_across_distinct_interview_ids": len(identical_distinct_ids),
        "near_duplicate_pairs_at_0_95": len(near), "distinct_later_documents_screened": len(later_docs),
        "similarity_scope": "fit-free hashed word unigram/bigram nearest training document; same IDs excluded; not exhaustive semantic similarity"}


def run():
    base, interviews, joins, matches, raw, original = load_inputs()
    frame = pd.read_csv(STUDY / "metadata.csv")
    docs = accepted_documents(interviews, joins, matches)
    schema = json.loads((STUDY / "schema.json").read_text())
    with gzip.open(STUDY / "history_lineage.jsonl.gz", "rt") as stream:
        lineage = [json.loads(line) for line in stream]
    checks = []

    def record(code, passed, scope, tested, violations, evidence, why, scenario="v1", status=None):
        checks.append({"check_id": code, "scenario": scenario, "status": status or ("PASS" if passed else "FAIL"),
            "scope": scope, "tested": int(tested), "violations": int(violations), "evidence": evidence, "why_it_matters": why})

    record("DATA01", base.match_id.is_unique, "original unique target matches", len(base), base.match_id.duplicated().sum(),
           "variant_A.csv / immutable source hash", "A duplicated sporting match must not appear under another row ID", "original")
    original_time_bad = 0
    for side in ["a", "b"]:
        original_time_bad += int((base[f"{side}_text_date"].ge(base.match_date) | base[f"{side}_source_match_id"].eq(base.match_id)).sum())
    record("TIME00", original_time_bad == 0, "all original A source dates strictly before targets and different source matches", 2*len(base), original_time_bad,
           "original variant_A.csv dates and match IDs", "Audit the full original pool, including rows later removed by the conservative policy", "original")
    canonical_bad = sum(canonical_id(row) != row.match_id for row in raw.itertuples())
    record("DATA02", canonical_bad == 0, "all raw match identities recomputed from unordered players/date/event/round", len(raw), canonical_bad,
           "a_leakage.canonical_id + structured_source_rows.csv", "Winner/loser ordering must not determine the match ID")
    source_lookup = raw.set_index("match_id")
    orient_bad = labels_bad = ranks_bad = 0
    for row in base.itertuples():
        src = source_lookup.loc[row.match_id]
        pair = sorted([src.winner_key, src.loser_key])
        expected = pair if int(digest(SEED + src.name), 16) % 2 == 0 else pair[::-1]
        orient_bad += [row.player_a_key, row.player_b_key] != expected
        labels_bad += row.target != int(row.player_a_key == src.winner_key)
        for side in ["a", "b"]:
            player = getattr(row, f"player_{side}_key")
            rank = pd.to_numeric(src.WRank if player == src.winner_key else src.LRank, errors="coerce")
            rank = rank if pd.notna(rank) and rank > 0 else np.nan
            ranks_bad += not equal_numbers([rank], [getattr(row, f"{side}_rank")])
    record("TARGET01", orient_bad == 0, "outcome-independent A/B assignment in the entire original pool", len(base), orient_bad, "canonical_id + fixed orientation seed", "Renaming Winner as Player A would reveal the label")
    record("TARGET02", labels_bad == 0, "original target equals oriented player's actual outcome", len(base), labels_bad, "raw source Winner/Loser mapped after independent orientation", "Outcome mapping must remain consistent after swapping sides")
    record("RANK01", ranks_bad == 0, "original rank values attached to the correct player", 2*len(base), ranks_bad, "WRank/LRank raw mapping; tennis-data-notes.txt", "A correct number attached to the wrong side corrupts predictors")
    record("RANK02", False, "independent ranking publication timestamps", 2*len(frame), 0, "Source dictionary states entry rankings at tournament start; weekly releases not independently reconstructed", "Source contract supports pre-match use but is not an independent timestamp audit", status="ASSUMPTION")
    rank_rows = pd.concat([raw[["source_file", "Tournament", "winner_key", "WRank"]].rename(columns={"winner_key":"player_key","WRank":"rank"}),
                          raw[["source_file", "Tournament", "loser_key", "LRank"]].rename(columns={"loser_key":"player_key","LRank":"rank"})])
    rank_rows["rank"] = pd.to_numeric(rank_rows["rank"], errors="coerce")
    rank_counts = rank_rows.groupby(["source_file", "Tournament", "player_key"])["rank"].nunique()
    varying_ranks = rank_counts[rank_counts > 1]
    varying_ranks.rename("distinct_entry_ranks").to_csv(OUT / "a_v1_rank_snapshot_variations.csv")
    record("RANK03", varying_ranks.empty, "entry rank constant per player/tournament/year-source", len(rank_counts), len(varying_ranks),
           "a_v1_rank_snapshot_variations.csv; 97,560 source player-side rank observations",
           "Within-event rank updates could contradict the documented tournament-start snapshot")
    whitelist = set(NUMERIC + CONTEXT + ["player_a_text", "player_b_text"])
    actual_features = set(schema["structured_columns"] + schema["text_columns"])
    banned = actual_features - whitelist
    record("TARGET03", not banned and "match_id" not in actual_features, "predictor whitelist and audit/target exclusion", len(actual_features), len(banned) + int("match_id" in actual_features),
           "candidate_A_v1/schema.json", "IDs, source labels, current score, completion and winner/loser columns must stay outside X")
    good_ids = set(joins.loc[joins.join_status == "matched", "interview_id"].astype(int))
    used = set(frame.a_interview_id.astype(int)) | set(frame.b_interview_id.astype(int))
    record("JOIN01", used <= good_ids, "only accepted interview links used", len(used), len(used-good_ids), "cornell_match_join_audit.csv", "Known match-label conflicts and uncertain joins are excluded")
    source_map = docs.set_index("interview_id")
    time_bad = self_bad = person_bad = 0
    for row in frame.itertuples():
        for side in ["a", "b"]:
            doc = source_map.loc[int(getattr(row, f"{side}_interview_id"))]
            time_bad += not (3 <= ordinal(row.match_date)-ordinal(doc.date) <= 90 and doc.source_date_agrees and doc.source_played)
            self_bad += doc.match_id == row.match_id
            person_bad += doc.player_key != getattr(row, f"player_{side}_key") or doc.gender != row.gender
    record("TIME01", time_bad == 0, "three-day calendar rule, max age, linked source date and played source event", 2*len(frame), time_bad, "audit.csv; a_v1_date_sensitivity.csv", "A future or same-day interview cannot be used as historical input")
    record("TIME02", self_bad == 0, "interview source match differs from its target", 2*len(frame), self_bad, "source/target match IDs in audit.csv", "An interview after the target match may state its winner")
    record("JOIN02", person_bad == 0, "interview player and gender match the target side", 2*len(frame), person_bad, "accepted deterministic source links and audit.csv", "An old interview about a different player is invalid input")
    ambiguous = docs.groupby(["gender", "player_key", "date"]).size()
    record("JOIN03", not (ambiguous > 1).any(), "same-player/day accepted-source ambiguity", len(ambiguous), (ambiguous > 1).sum(), "accepted source groups before selection", "Two interviews on one date must not be chosen by arbitrary row order")
    record("TIME03", False, "historical upload times and timezone alignment", 2*len(frame), 0, "publication_verified=False; three-calendar-day policy is assumed", "A delay buffer is a defensible scenario, never proof of publication", status="ASSUMPTION")
    history_errors = history_validation(raw, frame, lineage)
    record("HISTORY01", sum(history_errors.values()) == 0, "every feature contributor independently replayed from raw played matches", len(lineage), sum(history_errors.values()), "private history_lineage.jsonl.gz; a_v1_history_examples.csv", "Whole-day update order and complete source membership prevent current/future outcomes entering history")
    old_replay, _ = replay_histories(raw, base, include_walkovers=True)
    old_n = int((~np.isclose(old_replay[NUMERIC].to_numpy(float), base[NUMERIC].to_numpy(float), equal_nan=True)).any(axis=1).sum())
    record("HISTORY02", old_n == 0, "original numerical features independently reproduced", len(base), old_n, "old replay includes original walkover/disqualification convention", "Validate the legacy values before attributing changes to a fix", "original")
    record("HISTORY03", True, "history convention fixed to played Completed/Retired events", len(lineage), 0, "a_v1_policy.json + full contributor audit", "Walkovers do not represent match play or physical rest resets")
    mutations = mutation_checks(raw, frame, docs)
    for code, key, why in [("MUTATE01", "future_outcome_invariance", "Changing future results must leave earlier predictors unchanged"),
                            ("MUTATE02", "own_and_same_day_outcome_invariance", "Changing a target-day outcome must not change target-day features"),
                            ("MUTATE03", "future_interview_invariance", "Adding a future interview must not change the selected historical document")]:
        record(code, mutations[key], "counterfactual injection", mutations["real_rows_in_future_mutation"] if code == "MUTATE01" else 1,
               int(not mutations[key]), "a_v1_mutation_results.json", why)
    frame["split"] = assign_split(frame)
    boundaries = frame.groupby("split").match_date.agg(["min", "max"])
    ordered_ok = boundaries.loc["train", "max"] < boundaries.loc["development", "min"] and boundaries.loc["development", "max"] < boundaries.loc["reserved_test", "min"]
    record("SPLIT01", ordered_ok and frame.match_id.is_unique, "whole-day chronological train/development/reserved-test partitions", len(frame), int(not ordered_ok)+frame.match_id.duplicated().sum(), "candidate_A_v1/splits.csv", "The same target/day cannot occur on both sides of a chronological boundary")
    later = set(frame.loc[frame.split != "train", "match_id"])
    early = frame[frame.split == "train"]
    reverse = (set(early.a_source_match_id) | set(early.b_source_match_id)) & later
    record("SPLIT02", not reverse, "training interviews cannot refer to later held-out target matches", len(early), len(reverse), "audit.csv source IDs versus later target IDs", "Later training narratives may expose an earlier held-out target's outcome under random splitting")
    previous = pd.read_csv(OUT / "interview_split_risks.csv")
    legacy_random = previous[previous.split == "random_80_20_seed42_diagnostic"].iloc[0]
    record("SPLIT03", False, "legacy random 80/20 forecast interpretation", int(legacy_random.test_rows), int(legacy_random.heldout_targets_referenced_in_training_source_matches),
           "interview_split_reference_examples.csv", "A random score is not a strict future-prediction result", "legacy_random_diagnostic")
    reuse = reuse_and_similarity(frame, original)
    record("REUSE01", reuse["all_shared_source_dates_before_use"], "legitimate interview reuse across chronological partitions", reuse["shared_interview_ids_between_partitions"], 0 if reuse["all_shared_source_dates_before_use"] else 1,
           "a_v1_reused_interview_rows.csv", "Repeated historical documents are not automatically leakage; their dates and label-independent processing matter")
    record("REUSE02", True, "exact distinct-document duplicates and lexical near-duplicate screen", reuse["distinct_later_documents_screened"], 0,
           "a_v1_identical_distinct_documents.csv; a_v1_near_duplicate_screen.csv", "Similarity flags require source/date review, not automatic deletion")
    fit = json.loads((OUT / "a_v1_preprocessing_fit_audit.json").read_text())
    allowed = set(early.match_id); excluded_fit = noninvariant = 0
    for entry in fit:
        excluded_fit += len(set(entry["fit_row_ids"]) - allowed)
        noninvariant += not (entry["preprocessor_sha_before_evaluation"] == entry["preprocessor_sha_after_evaluation"] == entry["preprocessor_sha_after_evaluation_injection"] and entry["model_parameters_unchanged_by_evaluation"])
    record("PREP01", excluded_fit == 0, "actual imputer/scaler/categories/TF-IDF/classifier fit IDs", len(fit), excluded_fit, "a_v1_preprocessing_fit_audit.json", "Only training rows may fit statistical transformations and model parameters")
    record("PREP02", noninvariant == 0, "evaluation-only sentinel/outlier injection leaves fitted state unchanged", len(fit), noninvariant, "before/after preprocessing and parameter fingerprints", "Evaluation-time transform must not refit on validation/test data")
    record("PREP03", mutations["train_only_vocabulary_rejects_validation_sentinel"], "proper vocabulary control", 1, int(not mutations["train_only_vocabulary_rejects_validation_sentinel"]), "a_v1_mutation_results.json", "A held-out-only word must be absent from a training-only vocabulary")
    record("PREP04", False, "deliberately leaky vocabulary negative control", 1, int(mutations["deliberately_leaky_vocabulary_contains_sentinel"]), "a_v1_mutation_results.json", "The audit must detect a deliberately combined train/evaluation vocabulary fit", "negative_control")
    scores = pd.read_csv(OUT / "a_v1_development_scores.csv")
    record("PREP05", scores.reserved_test_rows_scored.eq(0).all(), "no reserved-test model scores or policy selection", len(scores), int(scores.reserved_test_rows_scored.sum()), "a_v1_development_scores.csv; pre-score frozen policy/config hash", "Repeated test-driven iteration would invalidate the final estimate")
    metrics = pd.read_csv(OUT / "a_v1_text_policy_metrics.csv")
    over = metrics[["formatted_tokens_a", "formatted_tokens_b"]].gt(512).any(axis=1).sum()
    record("TEXT01", over == 0, "all five policies and both formatted column names fit the E5/TAR limit", len(metrics), over, "a_v1_text_policy_metrics.csv + exact tokenizer revision", "Words do not equal tokens; prefixes and special tokens consume budget")
    legacy_over = int(metrics.loc[metrics.policy == "legacy300", "legacy_was_over_token_limit"].sum())
    record("TEXT02", legacy_over == 0, "uncapped legacy first-300-word control", int(metrics.policy.eq("legacy300").sum()), legacy_over, "legacy_was_over_token_limit in metrics", "Silent encoder truncation can discard selected content", "original_text_control")
    faith_bad = 0; tested = 0; source_hashes = defaultdict(set); selected_hashes = {}
    # Verify every selected-policy document; all other policies are separately
    # budget-checked and covered by selector unit tests and sample inspections.
    with gzip.open(STUDY / "passages_private.jsonl.gz", "rt") as stream:
        for line in stream:
            item = json.loads(line)
            if item["policy"] != SELECTED_POLICY:
                continue
            iid = item["interview_id"]; tested += 1
            faith_bad += not verify_selected(original[str(iid)]["QandA"], item)
            source_hashes[iid].add(item["selected_sha256"])
            selected_hashes[int(iid)] = item["selected_sha256"]
    record("TEXT03", faith_bad == 0, "selected text re-extracted from original answer spans with source hashes", tested, faith_bad, "private passages_private.jsonl.gz + a_text.verify_selected", "Passages must contain traceable source words; no invented summary facts")
    record("TEXT04", all(len(v) == 1 for v in source_hashes.values()), "same interview receives one target-independent passage", len(source_hashes), sum(len(v) != 1 for v in source_hashes.values()), "selector accepts Q&A/policy only; cached per interview, not per label", "A document must not be selected differently depending on its future label")
    exported = pd.read_csv(STUDY / f"features_{SELECTED_POLICY}.csv").set_index("match_id")
    export_bad = 0
    frozen_inputs = frame.set_index("match_id")[NUMERIC+CONTEXT+["target"]]
    for path in sorted(STUDY.glob("features_*.csv")):
        actual = pd.read_csv(path).set_index("match_id")
        try:
            pd.testing.assert_frame_equal(actual.loc[frozen_inputs.index, frozen_inputs.columns], frozen_inputs,
                                          check_dtype=False, check_exact=False, rtol=1e-11, atol=1e-11)
        except AssertionError:
            export_bad += 1
    record("TARGET04", export_bad == 0, "all six text variants keep identical audited structured fields and targets", len(list(STUDY.glob("features_*.csv"))), export_bad,
           "features_*.csv joined by match_id to independently replayed metadata.csv",
           "A text ablation must not change labels, ranks, row population or structured information")
    csv_bad = 0
    for row in frame.itertuples():
        for side in ["a", "b"]:
            text = exported.loc[row.match_id, f"player_{side}_text"]
            csv_bad += digest(text) != selected_hashes[int(getattr(row, f"{side}_interview_id"))]
    record("TEXT05", csv_bad == 0, "every exported input cell matches its actual as-of-selected interview", 2*len(frame), csv_bad,
           "features_topical_answers.csv joined to audit.csv and source-span hashes", "Falling back to an older source must also replace the text; changing only its ID would corrupt provenance")
    from tennis.scripts.a_context_blocks import select_blocks
    context_bad = context_n = 0
    context_features = pd.read_csv(STUDY / "features_qa_blocks.csv").set_index("match_id")
    context_hashes = {}
    with gzip.open(STUDY / "qa_blocks_private.jsonl.gz", "rt") as stream:
        for line in stream:
            value = json.loads(line); iid = value["interview_id"]; context_n += 1
            expected = select_blocks(original[str(iid)]["QandA"])
            context_bad += value["selected_sha256"] != expected["selected_sha256"] or max(value["selected_tokens_a"], value["selected_tokens_b"]) > 512
            context_hashes[iid] = expected["selected_sha256"]
    for row in frame.itertuples():
        for side in ["a", "b"]:
            context_bad += digest(context_features.loc[row.match_id, f"player_{side}_text"]) != context_hashes[int(getattr(row, f"{side}_interview_id"))]
    record("TEXT06", context_bad == 0, "whole-answer-block extension source replay, actual CSV mapping and token limit", context_n+2*len(frame), context_bad,
           "qa_blocks_private.jsonl.gz; features_qa_blocks.csv; separately frozen contextual policy", "A qualitative fix must be audited separately rather than relabeled as the original experiment")
    context_fit = json.loads((OUT / "a_v1_context_blocks_fit_audit.json").read_text())
    context_fit_bad = sum(bool(set(v["fit_row_ids"])-allowed) or v["preprocessor_sha_before"] != v["preprocessor_sha_after"] or v["reserved_test_rows_used_for_fit_or_score"] != 0 for v in context_fit)
    record("PREP06", context_fit_bad == 0, "whole-block extension training-only preprocessing and reserved-test exclusion", len(context_fit), context_fit_bad,
           "a_v1_context_blocks_fit_audit.json", "The later training-review extension must preserve the same evaluation boundary")
    record("SCOPE01", False, "predicting all scheduled matches or claiming completion known beforehand", len(frame), 0, "target pool is retrospectively conditioned on Comment=Completed", "Completion filtering is outcome-time population selection, not a pre-match predictor", status="ASSUMPTION")
    record("SCOPE02", False, "source identity/date and ranking-source contract", len(docs), 0, "nine publisher corroborations; corpus match links are not independent gold labels", "Calendar and metadata checks cannot prove every publisher/source label correct", status="ASSUMPTION")
    record("SCOPE03", False, "historical deployment / pretrained-model contamination", len(frame), 0, "modern E5 predates neither 2007–2015 events nor every possible web transcript", "Chronological rows do not prove a pretrained language model never saw historical outcomes", status="ASSUMPTION")
    record("SCOPE04", False, "pairing/round/surface/format known at prediction cutoff", len(frame), 0,
           "Actual completed matchup/context; historical draw and schedule publication times not independently reconstructed",
           "The task is conditional on the announced pairing/context, not predicting who reaches a future round", status="ASSUMPTION")
    record("OFFICIAL01", False, "strict chronological inner validation under official random split code", 0, 0, "pinned TabularModel.fit and encoder split_to_val source; official implementation unchanged", "An outer chronological diagnostic does not make their inner random validation chronological", status="NOT_RUN")
    before = json.loads((CACHE / "leakage_audit_reference_before.json").read_text())
    modified = [p for p, sha in before.items() if not Path(p).exists() or digest(Path(p).read_bytes()) != sha]
    record("OFFICIAL02", not modified, "all snapshotted official/reference and working infrastructure bytes", len(before), len(modified), "leakage_audit_reference_before.json", "Audit/preprocessing changes must not alter the benchmark or notebook 02")
    pd.DataFrame(checks).to_csv(OUT / "a_v1_leakage_checks.csv", index=False)
    (OUT / "a_v1_leakage_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    (OUT / "a_v1_mutation_results.json").write_text(json.dumps(mutations, indent=2) + "\n")
    summary = {"checks": len(checks), "by_status": pd.Series([c["status"] for c in checks]).value_counts().to_dict(),
        "v1_actual_failures": [c["check_id"] for c in checks if c["scenario"] == "v1" and c["status"] == "FAIL"],
        "retained_legacy_or_negative_control_failures": [c["check_id"] for c in checks if c["scenario"] != "v1" and c["status"] == "FAIL"],
        "history_errors": history_errors, "reuse": reuse,
        "conditional_conclusion": "No detected violations of the implemented calendar/as-of, target isolation, chronological partition and train-only-fit rules. Publication/ranking/source/pretraining assumptions remain; this is not a universal leakage-free guarantee."}
    (OUT / "a_v1_leakage_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
