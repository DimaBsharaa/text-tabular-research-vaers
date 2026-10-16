# Independent review: VAERS v8 and v9

October 17, 2026. Reviewed commit `60001ca` after safely pulling main. All ten previously modified/untracked local files retained their pre-pull SHA-256 hashes. No existing builder, benchmark or running experiment was changed during this review.

## Recommendation

Keep the current runs as development experiments. **V8 is a well-motivated representation experiment; v9 is a separate case-enriched sampling experiment.** Neither is guaranteed to improve TAR. Do not select a final dataset solely by the largest observed gain on these repeatedly inspected development splits. First complete the v7 content audit, inspect v8's text retention and compare the finished runs with their exact data hashes and populations documented.

The official final registered evaluation at `89a1854` remains necessary. Earlier positive screening results are a useful checkpoint, not a completed course acceptance result.

## Verified implementation and tests

`python3 -m unittest research.scripts.vaers.test_build_vaers evaluation.test_smoke_infrastructure -q` passed **48 tests** (29 builder tests and 19 infrastructure tests). This checks synthetic behavior and infrastructure; it does not establish the quality of real reports.

V8 replaces `history`, `current_illness`, `other_meds` and `allergies` with a merged, field-prefixed `patient_background`, while keeping `narrative_onset`. In an additional synthetic check with a 500-row cap, v7 and v8 had identical report-ID order, labels and excerpts. The existing test also verifies the merged content.

V9 uses the same two text fields and retains every eligible serious report, then chooses non-serious reports in deterministic hash order to target 25% serious. This is explicitly label-dependent cohort sampling; it does not append the target to predictors. It uses the entire eligible cohort before the ordinary 60,000-row cap.

**Implementation detail:** v9 ignores the requested `n_rows` cap. In the synthetic 500-row-cap probe, v8 contained 500 rows and v9 contained **1,412 rows at exactly 25% serious**, including **912 IDs absent from sampled v8**. This follows the current deliberate case-enrichment code, rather than demonstrating a random-sampling bug. It must be documented: v9 is not necessarily a subset of the specific 60,000-row v8 file. Actual real-data v9 counts must come from its schema, not this fixture.

## Why v8 is worth testing

The official TAR routine builds equally weighted training examples from every text cell. V7 has five text fields, so background fields account for 80% of cell examples and the excerpt for 20%. V8 has two fields, allocating half to each. Consolidating fragmented history may give E5 more useful context within a single passage and reduce repeated filler examples.

These are hypotheses, not an established diagnosis of the encoder. Four columns being 52–84% filler does **not** mean 80% of all training examples are filler. Eighty percent are background cells, some of which contain information. A loss near the prior and modest cell-level AUROC may reflect weak individual fields, repeated inputs with different outcomes, class imbalance or limited information; it does not prove underfitting or that overfitting is absent.

V8 changes more than cell weighting:

- Text PCA dimensions fall from 5 × 30 = **150** to 2 × 30 = **60**; Joint input dimensions fall from **162 to 72**. Downstream model capacity and regularization may therefore change.
- The same text moves under different column names and contexts.
- Four separate token budgets become one shared budget for background text. Long merged backgrounds may be truncated, despite retaining the same information in the CSV.
- At a fixed epoch maximum, fewer cells produce fewer optimization steps. Early-stopping behavior and the validation objective can change.

Therefore a v8 gain would establish that this complete representation construction helped, not that one hypothesized cause was isolated. Compare Structured, Text, Joint Frozen and Joint TAR absolute scores alongside both gaps. A larger TAR gap caused only by a lower Frozen score is a tradeoff to report, not automatically a better dataset.

## How to interpret v9

Case enrichment can supply more positive training examples under the fixed 10,000-row benchmark cap. That is a plausible reason to improve learning. Random within-class sampling is not automatically target leakage, provided the sampling design and selected population are explicit.

However:

1. V9 changes both training and test populations. It does not isolate an effect of changing only training class balance.
2. For a fixed scoring rule and unchanged class-conditional distributions, AUROC does not increase merely because prevalence increases. Training with a different balance may change the scoring rule; finite samples and class coverage also change.
3. The enriched 25% rate is not a natural VAERS reporting rate. Precision, calibration and prevalence-dependent metrics cannot be interpreted as natural-cohort performance without additional adjustment/validation.
4. Its smaller row pool means repeated randomized evaluations can share more reports. Use actual split-ID overlap; do not assume seed labels define independent or disjoint folds.
5. Repeated/secondary clinical events remain a concern even with the current heuristic deduplication. This matters particularly when retaining all eligible positives.

Thus v9 is a legitimate candidate experiment when honestly described, but should not be called a straightforward improvement of the same population.

## Statistical corrections

The new `screen_stats.py` is useful for descriptive per-fold gains. Its uncertainty calculations have limits:

- Hanley–McNeil standard error for **one AUROC** is not the standard error of **TAR minus Frozen on the same test reports**. The latter depends on paired prediction covariance. The illustrative 0.020 → 0.011 calculation assumes AUROC 0.85 and approximate class counts; it is not a measured confidence interval for these gains.
- `sd / sqrt(n_folds)` does not establish a valid standard error when repeated train/test samples overlap. The v7 archive audit found 65, 71 and 84 shared test IDs between its fold pairs.
- The mean of per-fold Joint gains need not equal the course/dashboard quantity: mean(Frozen) minus max(mean(Structured), mean(Text)). They coincide for the supplied v7 runs because Text is stronger in every observed fold. Report both appropriately if that changes.
- A positive three-fold mean with one negative fold is encouraging, not proof that remaining folds will change scores only slightly.

Do not use these descriptive calculations to promise qualification or a grade.

## Concrete next improvements, in order

1. **Complete the content audit before inventing more variants.** Obtain the exact v7 CSV/schema/audit matching the archived hashes. Check retained outcome statements, repeated long narratives, suspicious clinical-event clusters and reporting-style associations. The separate archive audit found no recorded execution inconsistency but could not inspect these contents.
2. **Measure what E5 actually retains in v8.** Use the pinned tokenizer and actual official passage formatting to measure background/excerpt truncation and retained field coverage, by class. Read before/after examples with negation and qualifications preserved. Do not interpret a combined CSV cell as proof of identical encoded information.
3. **Compare v7 and v8 on their matched rows and recorded folds.** Save absolute condition scores, both gains, class counts, text-feature detection and split IDs. Preserve the already-running experiments and their CSV/schema hashes.
4. **Report v9 separately with its enrichment ledger.** Compare natural and enriched cohorts transparently. A same-test-population auxiliary study could later help distinguish training benefits from population changes, but would be an additional diagnostic protocol, not a replacement for the official final run.
5. **Prepare final-version compatibility early.** A small native integration preflight at `89a1854` should precede the expensive final sweep. Keep the official model implementations unchanged and preserve the older screening reference.
6. **Freeze one candidate and document selection history.** Improve task validity and text quality; do not remove useful columns or add filters solely because they lower a baseline or increase a development gap. No additional GPU work is necessary to perform steps 1–2.

The previously confirmed broad `product` regex matching `reproductive` remains present. This is a known over-filtering issue, not evidence of target leakage. Do not silently change it under the same experiment version or overwrite the inputs of running jobs.

## Files relevant to this review

- [Builder](../scripts/vaers/build_vaers.py): current v8/v9 construction.
- [Decision log](vaers_decision_log.md): the author's observations and hypotheses; the qualifications above distinguish inference from verification.
- [V7 archive audit](vaers_v7_archive_audit.md): completed execution evidence and its remaining limitations.

This review adds independent documentation only. No training, dependency installation, external messaging or official benchmark changes occurred.
