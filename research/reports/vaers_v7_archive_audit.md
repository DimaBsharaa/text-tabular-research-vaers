# VAERS v7: independent audit of diagnostic result archives

October 17, 2026. This report covers the two result ZIPs supplied by the user. It does not modify the dataset builder, official benchmark or completed experiments. No models were trained and nothing was pushed.

**Conclusion:** no inconsistency was found in the recorded per-run split boundaries, predictor selection, condition matching, source provenance or score aggregation. The archives provide substantially stronger evidence of a correctly configured diagnostic experiment than the summary CSVs alone. They do not establish that the dataset contents are free of target leakage or duplicate clinical events.

## Inputs and reproducible checks

- `tennis_diagnostic_721a59225d840c64_results.zip`: 12 evaluations, TabM/TabPFN 2.0/TabPFN 2.5, fold 0.
- `tennis_diagnostic_a12d3b1c04dbbdd7_results.zip`: 24 evaluations, LightGBM/CatBoost, folds 0–2.
- Both plans select all four conditions. The ZIP filenames retain an old tennis prefix; the run summaries identify `BIN_TEXT_VAERS_SERIOUS_V7`.
- All 36 runs record research commit `f8dbdd4ff417a16c81f2c91671be3102f54c49d6` and benchmark commit `3bb95079a6146ed35d45187e98bcad5b063273a1`.
- Dataset SHA-256: `9ffdfad811ef6d0910fdf4e69af9d1e44a69870f31f308e80fd59750259a8922`.
- Schema SHA-256: `3294eed03d7634fe319323caa9a55938dce2508fe5c25900d34eb60dbe721051`.

The new, separate [audit script](../scripts/vaers/audit_result_archives.py) reads ZIP members without extracting them. It completed **894 individual check instances: 894 PASS, 0 FAIL**. Many checks repeat across runs; this is not 894 independent demonstrations of absence of leakage. Full results, input archive hashes, per-run counts and limitations are in [the JSON evidence](vaers_v7_archive_audit.json).

Three deliberate mutations were also detected by their corresponding checks: adding a test ID to encoder training, adding `serious` to predictors, and changing a CSV score without changing its source JSON. These are checks of the auditor, not model permutation experiments. Original archives were unchanged; mutated copies were temporary.

## What passed

| Check | Actual evidence | Scope of conclusion |
|---|---|---|
| Completion and arithmetic | All 36 JSON statuses are `ok`; both error lists are empty; exported AUROCs match their original JSON values | Scores were not manufactured by averaging errors or treating missing conditions as zero |
| Condition identity | 12 structured fields, 5 text fields, or their 17-field union; target and report ID absent from configured predictors | No direct target/ID column was requested as a predictor; says nothing about indirect outcome statements inside text |
| Matched comparison | Exactly the same ordered outer-training and test IDs for every condition and learner at each shared fold | Conditions were not compared on different recorded test samples |
| Outer partition | 9,998 train IDs and 2,000 test IDs per run; unique within partitions; zero within-run intersections | No report-ID overlap in the recorded outer train/test split |
| Learner partition | Learner fit/validation sets are disjoint and exactly partition outer training | Recorded learner-validation rows are held out from learner fitting |
| Encoder partition | Encoder train/validation sets are disjoint and exactly partition learner fitting; neither intersects outer test or learner validation | No recorded test ID was supplied to the encoder-training or encoder-validation partitions |
| Text detection | The same five expected v7 fields are recorded as text in every text-bearing run | E5 was not silently skipped because the columns were classified as categorical |
| Actual text features | Logs report 12 features for structured-only, 150 for text-only and 162 for joint conditions | Five 30-component text blocks were present in the learner inputs |
| TAR configuration and logs | Tuning is enabled only for Joint TAR; all nine TAR jobs have training logs with the expected passage counts | The TAR label is backed by fine-tuning activity, not just a renamed frozen run |
| Provenance | Config/run/plan hashes recompute; environment bytes match their saved hashes; no duplicate logical jobs or mixed dataset/configuration identities | The recorded results form one compatible diagnostic experiment |
| Reference source | Each run's 470 recorded Git blob IDs match the independently saved official pinned Git tree; imported locations/hashes agree with the tracked-file records | Recorded execution provenance points to the official reference |

### Encoder sizes explain the large training-example counts

| Learners | Outer train | Learner fit | Learner validation | Encoder train | Encoder validation | TAR training/validation passages |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM, CatBoost, TabM | 9,998 | 8,997 | 1,001 | 8,096 | 901 | 40,480 / 4,505 |
| TabPFN 2.0, TabPFN 2.5 | 9,998 | 9,998 | 0 | 8,997 | 1,001 | 44,985 / 5,005 |

Each report supplies five text passages. Thus 8,096 × 5 = 40,480; these are not 40,480 independent reports or evidence that the full 60,000-row dataset was fine-tuned on. The pinned official implementation splits rows **before** expanding their text columns into passages. This avoids placing different fields of one recorded report on opposite sides of the encoder split.

TAR configuration is E5-small-v2, LoRA rank 16 on the last three layers, learning rate 0.0001, maximum 50 epochs, patience 3, weight decay 0.001, batch size 256. Last logged training epochs are 10/15/10 for folds 0/1/2 of LightGBM and CatBoost, 10 for TabM, and 9 for both TabPFNs. A 50-epoch maximum with earlier stopping is consistent with this configuration.

### Independent inspection of the pinned code

Four selected official files were downloaded to a separate temporary directory, not copied into this repository. Their actual SHA-256 hashes match the archive proofs, and their Git blob hashes match the pinned tree:

- [Official evaluator](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/baselines/benchmarks/evaluate.py): calls model fitting on training data and scoring on test data.
- [Official abstract model](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/baselines/abstract_model.py): constructs learner validation before fitting numerical/categorical/text preprocessors on learner-fitting rows.
- [Official text encoder](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/baselines/preprocessing/text_embeddings.py): splits rows before expanding column passages; fits PCA on learner-fitting embeddings.
- [Official E5 tuning](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/e5/e5_finetune.py): creates a fresh backbone for tuning, uses supplied training/validation datasets and validation loss for checkpoint selection. Temporary tuning directories do not establish persistent cross-fold checkpoint sharing.

**Evidence limitation:** `splits.json` is a reconstruction produced by the adapter before the evaluator call, rather than an independently instrumented runtime trace. The evaluator's sizes are checked against it, and the inspected code/logs corroborate the intended pathway. This supports consistency; it is not an external execution attestation.

## Fold overlap and result stability

The test sets for folds 0/1, 0/2 and 1/2 share 65, 71 and 84 report IDs respectively. They are repeated randomized evaluations, not disjoint K-fold test partitions. Such overlap is not by itself leakage between independently trained runs. Do not treat fold estimates as independent replications or claim that the word “fold” establishes disjointness. Whether the same *clinical event* crosses training/test within any run remains unverified.

| Learner | Folds | Joint Signal from condition means | TAR Gain from condition means |
|---|---:|---:|---:|
| LightGBM | 3 | +0.036248 | +0.003081 |
| CatBoost | 3 | +0.037857 | +0.009606 |
| TabM | 1 | +0.009846 | +0.010593 |
| TabPFN 2.0 | 1 | +0.037908 | +0.017838 |
| TabPFN 2.5 | 1 | +0.027531 | +0.010201 |

All available mean gains are positive. TAR is negative on fold 1 for LightGBM (−0.005350) and CatBoost (−0.001835). One-fold results for the remaining learners do not establish stability. The archives use the older diagnostic route, not the required final registered evaluation at `89a1854`.

## Not verified by these archives

Neither ZIP includes the actual dataset CSV, `schema.json`, construction `audit.txt`, raw source records, event-cluster lineage or saved row-level predictions. Consequently:

1. The common dataset/schema hashes identify the intended inputs but cannot be recomputed from their contents here.
2. No audit of direct/indirect outcome statements in actual retained text is possible from these ZIPs.
3. Different report IDs may describe the same event. Unique recorded IDs do not resolve clinical-event duplication.
4. Label derivation and source joins cannot be independently reconstructed.
5. Cohort selection, filtering false positives, background missingness, reporting-style shortcuts and full-cohort category grouping remain construction concerns.
6. Confidence intervals from paired predictions cannot be calculated from the saved aggregate AUROCs alone.

**Next evidence:** the exact `vaers_v7.csv`, its `schema.json` and construction `audit.txt`, matching the hashes above. Those support a bounded text/missingness/duplicate audit. Original yearly source tables and row-level construction provenance would additionally support checking label reconstruction and suspected-event clusters. No new GPU experiment is required for this archive audit.

## Reproduction

With the four pinned source files placed at their original relative paths inside an external `SOURCE_DIR`, run:

```bash
python3 research/scripts/vaers/audit_result_archives.py \
  '/Users/dimabshara/Downloads/tennis_diagnostic_721a59225d840c64_results.zip' \
  '/Users/dimabshara/Downloads/tennis_diagnostic_a12d3b1c04dbbdd7_results.zip' \
  --tree research/data/sources/paper_tree.json \
  --source-dir "$SOURCE_DIR" \
  --output /private/tmp/vaers_v7_archive_audit_rerun.json
```

The source-directory argument is required. The saved audit used `/private/tmp/vaers-archive-audit-official`; those four downloaded files remain separate from the project. The script reproduces the 894 archive checks. The deliberate mutation checks are recorded in the saved evidence; they were run separately on temporary copies.
