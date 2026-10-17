# VAERS candidate: decision log

Each entry follows the same pattern: what we **saw**, what we **inferred**, what we **did**, and how we **checked** it. The detailed reports are in `vaers_candidate.md` (builds) and `vaers_codex_review.md` (independent reviews).

## A. Choosing the source

1. **Saw.** Tennis failed: its text was weaker than its table (text − structured −0.17). In notebook 01's data, passing datasets are large, mostly classification, and have text at least as strong as the table: TAR gain correlates with text − structured (Spearman +0.54), with row count (+0.41), and negatively with the repeated-text rate (−0.41).
   **Inferred.** We needed a large classification task whose free text carries signal the table lacks.
   **Did.** Chose VAERS: label = recorded seriousness, text = the report, table = patient and vaccine facts.

## B. Making the label honest (leakage)

2. **Saw.** The coded MedDRA terms include outcome and care words ("Death", "Hospitalisation", "Intensive care").
   **Inferred.** These are the label in other words.
   **Did.** Removed 93 outcome, 745 care-process and 150 administration terms. Flags only build the label. Post-event fields (ER, HOSPDAYS, RECOVD, LAB_DATA, dates) are excluded.
   **Checked.** Leak counter: 0 in every column except kept clinical terms. Twenty-plus unit tests.

3. **Saw.** The full narrative's strongest serious-class words were `remdesivir`, `iv`, `o2`, `tx`, `unspecified` and `pmhx`.
   **Inferred.** The narrative carries the hospital chart, which reveals hospitalisation without naming it.
   **Did.** Rejected v4/v5 (full narrative). Kept only the opening of the narrative, cut at the first care, testing, treatment or outcome sentence (v6/v7).

4. **Saw.** The column audit showed `onset_days` alone at AUROC 0.877. The serious rate was 0.42 in 2023 and 0.54 for COVID reports.
   **Inferred.** A selection artifact: hospitalised breakthrough-COVID cases had to be reported, so long onset plus a COVID vaccine meant "serious".
   **Did.** Scoped the cohort to reactions with onset of 0–42 days and excluded infection/failure reports (build 4).
   **Checked.** `onset_days` fell to 0.717, the serious rate in 2023 fell to 0.16, and `state` fell from 0.77 to 0.59.

5. **Saw.** Top excerpt words included `unspecified` and `init` (ICD "init encntr").
   **Inferred.** Hospital staff paste diagnosis lists into the narrative.
   **Did.** Cut at ICD and EHR wording (build 5).

6. **Saw.** Empty excerpts: 29% of serious reports vs 12% of non-serious; the flag alone scored AUROC 0.588.
   **Inferred.** An empty excerpt means "the report starts with hospital text", so emptiness itself leaked the label.
   **Did.** Restricted the cohort to reports with a non-empty excerpt in every variant (build 6), and documented it as a cohort definition.

7. **Saw.** Some problem lists survived: "ACUTE HYPOXEMIC RESPIRATORY FAILURE ANEMIA OF CHRONIC DISEASE…", and vital signs ("BP 137/82").
   **Inferred.** The rules kept missing new formats. Following the reviewer, we needed a written policy rather than more word lists.
   **Did.** Wrote the excerpt policy. It cuts at:
   - runs of 4+ capitalised words
   - numeric vital signs only
   - disability statements

   We also added event-level deduplication, provenance and per-build folders (build 8). The policy was frozen before any GPU screen.

## C. Making TAR work

8. **Saw.** v1 (coded symptoms + background) on the official code, fold 0: Joint Signal positive for 4/4 learners, TAR gain about 0.
   **Inferred.** Coded MedDRA terms are a fixed vocabulary that the frozen E5 already embeds well, so fine-tuning has nothing to add.
   **Did.** v7: background + the reporter's own words, without coded terms.

9. **Saw.** v7, build 8, official code, LightGBM + CatBoost, folds 0–2:

   | | Joint Signal (mean, sd) | TAR Gain (mean, sd) | Positive folds |
   |---|---|---|---|
   | LightGBM | +0.036 (0.005) | +0.003 (0.009) | JS 3/3, TAR 2/3 |
   | CatBoost | +0.038 (0.007) | +0.010 (0.011) | JS 3/3, TAR 2/3 |

   **Inferred.**
   - Free reporter language gives TAR room. CatBoost's +0.010 is above the median of accepted datasets (+0.004).
   - Joint Signal is about twice the accepted median (+0.016) and very stable.
   - TAR gain is still noisy: its fold sd (~0.01) is as large as its mean.

10. **Saw (TAR log).**
    - The encoder is fine-tuned on (column, cell) pairs from all 5 text columns: 8,096 rows × 5 = 40,480 examples.
    - Four of the five columns are 52–84% "none reported".
    - The validation loss starts at 0.276, exactly the entropy of a 7.9% base rate (= predicting the prior), and ends at about 0.259 (−6%). The best cell-level validation AUROC is 0.61–0.65.
    - Early stopping triggers at epoch 10–15 of 50, and train and validation loss stay close.
    - PCA-30 keeps only 59% of the excerpt embedding's variance, against 69–77% for the background columns.
    - The fine-tuning log is identical for LightGBM and CatBoost in each fold. Fine-tuning does not depend on the learner, so the learner differences come from how each learner uses the same embeddings.

    **Inferred.** Fine-tuning is **under-fed, not over-fitted**. About 80% of its examples are near-empty background cells, and with 7.9% positives the cross-entropy barely moves from the prior. The MulTaBench paper also notes that the shared encoder "might harm representations, especially as feature size grows".

    **Did.**
    - **v8**: the four background fields merged into one cell. Same rows and information; 2 text columns, so half of the fine-tuning examples are the reaction description.
    - **v9**: v8 case-enriched to 25% serious. Every serious report is kept, plus a deterministic sample of non-serious ones; about 20k rows.
      - Training rows per fold: 10k cap, serious ~700 → ~2,500.
      - Test rows per fold: 2k, serious ~157 → ~500.
      - The standard error of one test AUROC drops from about 0.020 to 0.011, which makes the gains measurable as well as larger.
      - The class ratio is a stated design choice of the benchmark population, not a feature change.

    **Will check.** v8 and v9 against v7 with the same learners and folds (LightGBM + CatBoost, folds 0–2), then the remaining three learners on the winner.

## D. Process rules we keep

- One frozen extraction policy per build. A rule change means a new build and a new screen.
- Every build is written to its own folder with commit and input hashes, plus `audit.txt`.
- Screens use the paper-version code (exploratory). The final result is produced once, at 89a1854, by `benchmark.py`, with the dataset registered natively.
- A high single-feature score is a reason to investigate, not proof of leakage. Zero hits on a removal rule is a consistency check, not validation.

## E. Status and the open audit (fifth review)

**Status.** Every result so far comes from a *development screen*. The dataset is **not** yet established as leakage-free, and every report and the paper keep that distinction. The fact that five learners agree does not settle it: they share one dataset and one evaluation pipeline, so a single common mistake would help all five.

11. **Saw.** The reviewer asked for evidence on six open questions:
    - text that states the outcome indirectly
    - the same event split across train and test
    - TAR seeing held-out rows
    - conditions evaluated on different test rows
    - gains coming from reporting style
    - over-tuning of the curation

    **Did.** Wrote `run_audit.py` (with tests) and notebook 07 (CPU), which audit the exact CSV and run folder that produced a result.

    **Checked so far, on the v7 run `a12d3b1c04dbbdd7` (24 runs), from its `splits.json` and `config.json`:**
    - All runs used CSV sha `9ffdfad8…`.
    - In every fold, all 8 runs share one outer test set (2,000 rows) and one outer training set (9,998 rows), with no overlap between them.
    - The TAR encoder's 8,096 + 901 rows lie inside `learner_fit`, with none in the test set or the learner validation set.
    - Joint Frozen and Joint TAR use the identical 17 predictor columns.
    - The detected text columns match the schema, and neither the target nor the ID column is a predictor.
    - The encoder starts from the public `intfloat/e5-small-v2` checkpoint (log: "Doing LoRA on last 3 layers of intfloat/e5-small-v2").

    **Pending (notebook 07, on the real CSV):**
    - near-duplicate events across train/test, and AUROC with those test rows removed
    - the text signal within year, vaccine-group and facility strata
    - style-only AUROC compared with the full excerpt
    - 12 + 12 random reports, each showing the original opening next to the retained text

12. **Rule.** Curation is frozen at build 8 + variant choice. v7, v8 and v9 come from one frozen extraction policy. We pick one variant from development screens, run the audit on it, and only then run the single official evaluation.

## F. Auditor corrections and local validation

October 17, 2026. Codex integrated update 15 locally at the user's request and corrected the auditor without changing the builder, dataset policy or official benchmark.

- Encoder train/validation partitions must be unique, disjoint and exactly cover learner fitting. Missing conditions, missing evidence and unsuccessful runs are flagged; failed input identity/mechanics stops proxy fitting.
- Duplicate sensitivity now excludes only candidate test members whose cluster also intersects that fold's training set. Exact token matches in large blocks are checked, and unsearched approximate comparisons are reported.
- Conditional AUROCs are reported per fold, including year × vaccine type, with both class counts and insufficient-support notices. Overlapping folds are not pooled.
- Style diagnostics include structured+style versus structured+style+excerpt; no causal or percentage attribution is claimed.
- Original-text examples require source-ID matches and check source hashes when supplied. Full matched raw/retained examples are saved locally; absent raw files remain NOT_VERIFIED.
- Notebook 07 runs auditor regression tests first, prints the auditor version, requires a fresh separate CPU runtime and saves progressive output.

See `vaers_audit_local_validation.md`. The actual CSV/schema/raw sources were not found locally at this stage, so an end-to-end real-data content audit is **not** claimed. Tests and archive checks do not substitute for that missing audit.

## G. Actual v7/source-data audit

October 17, 2026. The exact screened CSV/schema and 2019/2023/2024 raw ZIPs were subsequently supplied. The corrected audit and a separate all-row source check were run locally; see [vaers_v7_data_audit.md](vaers_v7_data_audit.md) and its compact JSON evidence.

- Both candidate fingerprints and all source-member fingerprints match. All 60,000 labels and 17 predictors reproduce from the supplied source records, with zero mismatches.
- The initial source comparison exposed a leading-zero ID issue in the auditor, not a bad dataset join. This was corrected with collision checks and two regression cases; the rerun matches all 24 sampled original reports.
- Recorded mechanics pass. The bounded duplicate detectors find 1/1/2 candidate test members across folds 0/1/2 and no cross-split complete raw-event-key clusters. This is not a global proof against secondary reports.
- Conditional and style proxies show signal beyond the measured year/vaccine/style features, but cannot establish that the remaining signal is free of care/outcome proxies.
- **Confirmed policy gap:** onset excerpts retain care such as blood transfusion, surgery and medical leave despite zero hits on the implemented care patterns. Some background fields explicitly describe subsequent care/testing. Therefore neither “no care information” nor “all background is pre-event” is justified.
- Preserve v7 and completed results. Review these examples and agree on one information boundary before a separately versioned policy correction. v8/v9 inherit this extraction policy; merging or resampling does not itself fix these gaps. No builder or benchmark change has been made and nothing from this task has been pushed.

## H. Supplied v8/v9 LightGBM/CatBoost screens

October 17, 2026. Reviewed both complete 24-run archives plus their actual CSV/schema ZIPs. See [vaers_v8_v9_results_comparison.md](vaers_v8_v9_results_comparison.md) and JSON evidence: 756 recorded check instances pass, source labels/excerpts/merged backgrounds reproduce across every candidate row, and the means recompute from original result JSONs.

V8 is a matched merge experiment on the exact v7 rows/folds. Its LightGBM TAR gap grows largely because Frozen weakens; final TAR is slightly lower than v7. CatBoost final TAR improves only +0.000916. V9 has positive TAR gains on all three folds for both learners, with mean gains about +0.010, but uses a different 25%-serious cohort and different test rows. V9 is promising for consistency, not a demonstrated improvement on a common population or acceptance across five learners. Both variants retain the confirmed care-policy gaps; no extraction correction or official evaluation was performed.

## I. Implemented correction: build 9 / v10

Codex now owns the implementation. Claude's original update-15 ZIP was committed as `eee71ac`; corrected audits followed in `57cd0db`. Legacy v9 and the official benchmark remain unchanged. `refine_v9.py` preserves reaction prefixes before care, quarantines explicit post-event background, and excludes uninformative excerpts. The first prerelease policy retained 19,834 rows; its source and scores remain saved locally. Three outputs preserve the comparison: v10, all-row v10_matched, and original-text v9_control on exactly the v10 rows.

All 79 tests passed. Thirty-two CPU diagnostic fits are saved in `research/data/screens/v10_care_boundary1_cpu.json`; notebook 08 is executed with tables, examples and a plot. On identical eligible rows, mean joint TF-IDF/logistic AUROC is 0.862415 for original text and 0.862321 for corrected text. This supports preserved proxy discrimination, not a measured E5/TAR improvement. Fold-0 label/text permutations score 0.5255/0.7785. New onset care-pattern hits and explicit post-event-background hits are zero within the implemented checks; universal semantic absence is not claimed. Next GPU test: v9_control versus v10, LightGBM/CatBoost fold 0, 16 evaluations total in notebook 02. Local candidate/prediction files stay outside Git; no score-based policy search was performed.

**Final prerelease scan / care-boundary-2:** an independent word scan found remaining “surgically,” generic “underwent,” pathology and outpatient-appointment cases; these were fixed from text evidence, with regression cases, rather than selected by score. Final build 9 / v10 has **19,832 rows, 4,903 serious (24.7227%)**, excluding 304 parent rows. All **80 tests pass**. The final 32-fit CPU run and executed notebook 08 use `v10_care_boundary2_cpu.json`: matched eligible-row joint means **0.862532 original / 0.862416 corrected**, label/text permutations **0.4966 / 0.7732**. No surgery/procedure/document-pointer excerpt matches remain in the independent scan; its one care-action hit describes outpatient vaccine administration before symptoms, which is permitted. First GPU step is now **v10_matched only, LightGBM/CatBoost fold 0 (8 runs)** versus the already-completed v9 baseline; the final eligible-cohort control pair follows if useful. These remain development diagnostics and do not certify universal semantic leakage absence or improved official scores.
