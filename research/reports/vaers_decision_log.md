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
