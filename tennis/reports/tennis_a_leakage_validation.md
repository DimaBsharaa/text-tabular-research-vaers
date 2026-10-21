# Cornell A leakage validation

October 15, 2026. Evidence is in [a_v1_leakage_checks.csv](../data/prototypes/a_v1_leakage_checks.csv), its JSON counterpart, [notebook 05](../notebooks/03_tennis_interview_validation.ipynb), the mutation results and the private row-level provenance files. This covers the **entire original 4,052-row pool**, all 48,780 source match records, and the refined **3,627-row** conditional research candidate.

**Conclusion:** No detected violations of the implemented calendar/as-of, target isolation, chronological partition and train-only-fit rules. Publication/ranking/source/pretraining assumptions remain; this is not a universal leakage-free guarantee. In particular, a three-day assumed delay is not a verified publication timestamp. No current-v1 check returned FAIL. Historical random-split and deliberately leaky negative-control failures remain visible. ASSUMPTION and NOT_RUN are not counted as verified PASS.

## Dates: yes, an interview can predict a later match

The original Cornell field is the source match day, not an unknown/missing date. Accepted deterministic links provide **6,366 interviews with the same date as their linked source record**; **6,319** follow Completed/Retired recorded events, while 47 attach to unplayed/other events. There are no ambiguous accepted same-player/day groups. Original A has 8,104 interview uses with source day strictly earlier than target day and different source/target match IDs. Nine publisher-content checks corroborate specific transcripts, while known corpus conflicts remain excluded. These checks support event association, not universal publisher-date accuracy or online first-publication time.

We can therefore associate a valid interview with the event just ended and examine the player's next recorded match. We explicitly distinguish that event-time relation from an upload-time claim. A withdrawal interview can contain useful condition information, but is not silently called a post-played-match interview in the current construction.

**Working conservative rule:** target day minus interview day must be **at least three calendar days and at most 90**. Assume that buffer is sufficient for online availability; select the most recent qualifying interview for each player, falling back to an older qualifying one. Exclude same-day/future sources, self-match sources, known bad/ambiguous links and source events without recorded play. The pairing and context are conditional inputs presumed announced by prediction time. Exact local timestamps and cross-timezone alignment are unresolved. This is an explicitly conditional retrospective experiment, not proof of a historical deployment.

| lag_days | retained_rows | excluded_rows | rows_using_older_eligible_interview | rows_both_sources_are_immediate_previous_match |
| --- | --- | --- | --- | --- |
| 1 | 4037 | 15 | 46 | 836.0000 |
| 2 | 3857 | 195 | 661 | 439.0000 |
| 3 | 3627 | 425 | 1219 | 70.0000 |
| 7 | 3360 | 692 | 1335 | 17.0000 |

The working three-day rule leaves **3,627** targets, removing **425** from the original pool; **1,219** retained targets use an older eligible document on at least one side. Median source ages are **28/28 days** for A/B. A delay buffer is a policy sensitivity, never proof that publication occurred.

### Immediate next match versus latest available interview

The next-match audit does not skip an inconvenient retirement/walkover to find a later completed outcome. It resolves the first strictly later recorded day, rejects ambiguous multiple matches that day, and only then applies completion, age and delay eligibility. At three days it yields **2,166 interview-to-next-match links over 2,095 distinct target matches across the source collection**. That is a one-player document count, not a ready two-sided dataset.

Among our 3,627 retained two-sided rows, only **70** have both source interviews attached to each player's immediate previous recorded match; 385 have at least one. Most rows therefore use older interviews. The stricter rule preserves volume by accepting stale text, which may weaken fatigue/condition relevance. Calling every row a fresh previous-match interview would be wrong. A two-day sensitivity retains 3,857 rows and 439 two-sided immediate-next cases; it is a different declared assumption, not an automatically better dataset. We did not choose delay based on model AUROC.

## Structured features and target isolation

The match ID and A/B orientation are recomputed from date/event/round and an **unordered** player pair with a fixed hash seed. Winner status is used only after orientation to construct the label and attach each player's source rank. All 4,052 original orientations, labels and 8,104 rank assignments are checked against raw records. The actual feature whitelist contains ten historical numerical fields, six categorical/numerical context fields and two text fields. Target, match/source IDs, names, source files, scores, odds, completion flags and winner/loser roles are audit-only or excluded.

The dictionary defines entry rank at tournament start. All **97,560 source player-side observations** were also checked for within-player/tournament/year-source consistency: no varying entry-rank groups were found. This corroborates the source contract without independently dating every weekly ranking release.

Historical predictors are replayed in whole-day batches: a target sees only strictly earlier days. Every A/B history has a complete ordered source-ID list, the last-ten IDs and surface-specific IDs. The independent auditor reconstructs those memberships and values from raw outcomes. Original numeric features are also reproduced under their old convention before comparing the refinement.

**Definition correction:** walkovers and the single disqualified event no longer count as played matches or reset rest days. Completed and retired past matches do count, since both involved recorded play. This is a data-definition improvement, not discovery that every previous walkover was temporal leakage. The change affects at least one history field in 3,140 retained rows, largely through accumulated prior-match counts. Same-day past matches have deterministic hash ordering; true within-day order is unavailable, so a last-ten boundary through a tied past day remains an approximation. Current-day results are always excluded.

## Chronological and preprocessing boundaries

Original date cutoffs are frozen, so eligibility filtering does not move the final boundary:

| partition | rows | first_day | last_day |
| --- | --- | --- | --- |
| train | 2162 | 2007-01-19 | 2012-10-12 |
| development | 716 | 2012-10-13 | 2014-06-18 |
| reserved_test | 749 | 2014-06-19 | 2015-11-22 |

The test protocol is **rolling historical information with a fixed trained model**: an earlier test-period match outcome may inform a later test-period row after it is observed. It is not batch prediction of the entire future block at the start date. Interview documents can be reused when genuinely older than every use. Here **42 interview IDs** cross partition boundaries, used in **155 rows**; every source date predates use. Same documents receive the same target-independent passage. These dependencies reduce effective sample size and do not represent unseen-player/document generalization.

There are zero exact full-answer duplicates across distinct used interview IDs. A fit-free hashed unigram/bigram nearest-training-document screen checks **1313 distinct later documents**, excluding already shared IDs, and finds **0** pairs at cosine ≥0.95. This is a lexical screen, not proof that no semantic near-duplicate exists.

A secondary name-plus-score screen flags **0 selected text cells** mentioning both sufficiently long target surnames and a score-like string. A flag can describe a previous encounter and is not automatically target leakage; no flagged cell uses the target itself as its source match. The screen excludes short/ambiguous surnames and does not exhaust semantic outcome detection. Its metadata are in `a_v1_target_reference_text_flags.csv`. The source/target tournament-name crosswalk is also saved; sponsor-name differences are not treated as mismatches without event review. Independent publisher-to-match gold validation remains a source-quality limitation.

All 13 completed local classifier conditions fit median/scaler/category/vocabulary/IDF/model parameters on the 2,162 training rows only. Fit row IDs and fitted-state hashes are saved; validation-only word/outlier injection leaves the state unchanged. A deliberately combined train/validation vocabulary fit exposes its sentinel and is correctly marked FAIL. Passage selectors use source Q&A and fixed rules, with no labels or future target statistics. Manual passage-policy review uses training interviews; whole blocks are a separately recorded exploratory extension. No reserved-test model score was computed. Corpus metadata/EDA had been inspected earlier, so the claim is an unscored final model holdout, not a never-viewed corpus.

## Check results

The saved automated boundary/adversarial test run contains **33 tests**, successful=True. Individual test IDs and PASS/FAIL results are in [a_v1_automated_test_results.json](../data/prototypes/a_v1_automated_test_results.json). These regression tests are separate from the 45 dataset/protocol audit entries below.

| check_id | scenario | status | tested | violations | scope |
| --- | --- | --- | --- | --- | --- |
| DATA01 | original | PASS | 4052 | 0 | original unique target matches |
| TIME00 | original | PASS | 8104 | 0 | all original A source dates strictly before targets and different source matches |
| DATA02 | v1 | PASS | 48780 | 0 | all raw match identities recomputed from unordered players/date/event/round |
| TARGET01 | v1 | PASS | 4052 | 0 | outcome-independent A/B assignment in the entire original pool |
| TARGET02 | v1 | PASS | 4052 | 0 | original target equals oriented player's actual outcome |
| RANK01 | v1 | PASS | 8104 | 0 | original rank values attached to the correct player |
| RANK02 | v1 | ASSUMPTION | 7254 | 0 | independent ranking publication timestamps |
| RANK03 | v1 | PASS | 49745 | 0 | entry rank constant per player/tournament/year-source |
| TARGET03 | v1 | PASS | 18 | 0 | predictor whitelist and audit/target exclusion |
| JOIN01 | v1 | PASS | 3324 | 0 | only accepted interview links used |
| TIME01 | v1 | PASS | 7254 | 0 | three-day calendar rule, max age, linked source date and played source event |
| TIME02 | v1 | PASS | 7254 | 0 | interview source match differs from its target |
| JOIN02 | v1 | PASS | 7254 | 0 | interview player and gender match the target side |
| JOIN03 | v1 | PASS | 6366 | 0 | same-player/day accepted-source ambiguity |
| TIME03 | v1 | ASSUMPTION | 7254 | 0 | historical upload times and timezone alignment |
| HISTORY01 | v1 | PASS | 7254 | 0 | every feature contributor independently replayed from raw played matches |
| HISTORY02 | original | PASS | 4052 | 0 | original numerical features independently reproduced |
| HISTORY03 | v1 | PASS | 7254 | 0 | history convention fixed to played Completed/Retired events |
| MUTATE01 | v1 | PASS | 100 | 0 | counterfactual injection |
| MUTATE02 | v1 | PASS | 1 | 0 | counterfactual injection |
| MUTATE03 | v1 | PASS | 1 | 0 | counterfactual injection |
| SPLIT01 | v1 | PASS | 3627 | 0 | whole-day chronological train/development/reserved-test partitions |
| SPLIT02 | v1 | PASS | 2162 | 0 | training interviews cannot refer to later held-out target matches |
| SPLIT03 | legacy_random_diagnostic | FAIL | 811 | 242 | legacy random 80/20 forecast interpretation |
| REUSE01 | v1 | PASS | 42 | 0 | legitimate interview reuse across chronological partitions |
| REUSE02 | v1 | PASS | 1313 | 0 | exact distinct-document duplicates and lexical near-duplicate screen |
| PREP01 | v1 | PASS | 11 | 0 | actual imputer/scaler/categories/TF-IDF/classifier fit IDs |
| PREP02 | v1 | PASS | 11 | 0 | evaluation-only sentinel/outlier injection leaves fitted state unchanged |
| PREP03 | v1 | PASS | 1 | 0 | proper vocabulary control |
| PREP04 | negative_control | FAIL | 1 | 1 | deliberately leaky vocabulary negative control |
| PREP05 | v1 | PASS | 11 | 0 | no reserved-test model scores or policy selection |
| TEXT01 | v1 | PASS | 16620 | 0 | all five policies and both formatted column names fit the E5/TAR limit |
| TEXT02 | original_text_control | PASS | 3324 | 0 | uncapped legacy first-300-word control |
| TEXT03 | v1 | PASS | 3324 | 0 | selected text re-extracted from original answer spans with source hashes |
| TEXT04 | v1 | PASS | 3324 | 0 | same interview receives one target-independent passage |
| TARGET04 | v1 | PASS | 6 | 0 | all six text variants keep identical audited structured fields and targets |
| TEXT05 | v1 | PASS | 7254 | 0 | every exported input cell matches its actual as-of-selected interview |
| TEXT06 | v1 | PASS | 10578 | 0 | whole-answer-block extension source replay, actual CSV mapping and token limit |
| PREP06 | v1 | PASS | 2 | 0 | whole-block extension training-only preprocessing and reserved-test exclusion |
| SCOPE01 | v1 | ASSUMPTION | 3627 | 0 | predicting all scheduled matches or claiming completion known beforehand |
| SCOPE02 | v1 | ASSUMPTION | 6366 | 0 | source identity/date and ranking-source contract |
| SCOPE03 | v1 | ASSUMPTION | 3627 | 0 | historical deployment / pretrained-model contamination |
| SCOPE04 | v1 | ASSUMPTION | 3627 | 0 | pairing/round/surface/format known at prediction cutoff |
| OFFICIAL01 | v1 | NOT_RUN | 0 | 0 | strict chronological inner validation under official random split code |
| OFFICIAL02 | v1 | PASS | 149 | 0 | all snapshotted official/reference and working infrastructure bytes |

`SPLIT03` retains the original random 80/20 diagnostic failure: interviews used by later training rows refer to **242 held-out target matches**. Such a reference is an outcome-exposure route; it does not prove every clipped answer explicitly reveals the result or quantify score inflation. The current chronological partitions have zero reverse-time source references. `PREP04` is an intentionally leaky negative control that the audit detects. The complete evidence/why-it-matters explanations are in the CSV/JSON, not only this summary table.

## What the checks do not establish

- Publication times and timezones remain assumed; the three-day delay does not certify them.
- Source match identity and entry-rank timing rely partly on documented source contracts. Automated joins share a source with the original corpus; high join coverage is not independent gold accuracy.
- Completion filtering defines a retrospective completed-match population. It does not prove that completion, later-round pairings or every context field was known for every scheduled match at cutoff.
- Modern pretrained E5 could have encountered historical results or transcripts. This study does not audit its pretraining corpus or establish a 2007-era deployable system.
- The official benchmark's random outer/inner holdouts remain unchanged. Official curation scores must be labeled separately from this chronological diagnostic; no E5/TAR evaluation was run here.
- Source-use/redistribution questions remain recorded separately. No email/contact step or public transcript release is part of this work.

The audit snapshots **149 reference/cache/working-infrastructure files** and verifies unchanged bytes afterward, including the official source cache and working notebook 02. New code resides in our research modules and optional temporal adapter. Input/output hashes and source/span lineage make the result reproducible and inspectable, without claiming cryptographic proof of execution history or universal absence of leakage.

Reproduce: `python3 -m tennis.scripts.a_leakage`; run boundary/adversarial checks with `python3 -m unittest tennis.scripts.test_research tennis.scripts.test_interview_followup tennis.scripts.test_a_validation -v`. The actual CSV/JSON evidence, fit IDs, negative controls and mutation outputs should accompany any later dataset claim.
