# VAERS severity triage: next Track 2 candidate

October 15, 2026. Replaces tennis as the lead candidate. Tennis stays documented as a negative result.

## 1. What passing datasets have in common (from notebook 01 data)

Computed from `research/data/learner_curation_results.csv` (56 original candidates; median over the 5 learners) and `dataset_profiles.csv` (34 profiled):

| Property | Pass rate / median |
| --- | --- |
| Rows ≤ 5k / 5–10k / 10–20k / > 20k | 25% / 67% / 56% / **91%** |
| Task: regression / binary / multiclass | 26% / 67% / 75% |
| Text-only score minus structured-only score ≤ −0.1 | **≤ 13%** (tennis A: −0.17) |
| Text-only minus structured-only in (+0.1, +0.3] | **83%** |
| Median structured / text columns (passing) | 6 / 3 |
| Median longest text column (passing vs failing) | 121 vs 43 chars |
| Spearman with TAR gain | text−structured +0.54, rows +0.41, repeated-text rate −0.41 |

Missing values and the number of structured columns hardly matter. What matters is many rows, classification, text that is **at least as strong as the structure** and not repetitive, and structure that still adds something.

## 2. Candidate

- **Source:** [VAERS yearly files](https://vaers.hhs.gov/data/datasets.html), U.S. government public data (CDC/FDA). It is downloaded manually, because the download asks for a verification code.
- **Unit:** one domestic, non-manufacturer report.
- **Target `serious`:** any of DIED, L_THREAT, HOSPITAL, X_STAY, DISABLE or BIRTH_DEFECT = Y.
- **Text (5 columns):**
  - coded MedDRA symptoms, with outcome and care-setting terms removed
  - history
  - current illness
  - other medications
  - allergies
- **v2 only:** an extra column with the narrative, after removing every sentence that mentions an outcome, a care setting or an emergency.
- **Structured (12 columns):** age, sex, state, onset days, vaccine type, manufacturer, dose series, route and site, number of vaccines, administering facility, report year.

## 3. Leakage rules (enforced in code, tested)

- Outcome flags only build the label. All post-event fields are excluded: ER and office visits, HOSPDAYS, RECOVD, death date, LAB_DATA. Raw dates are excluded too.
- MedDRA terms such as Death, Hospitalisation, Intensive care, Endotracheal intubation, Resuscitation, Disability and Dialysis are removed from the symptom text. The removed list is saved in `schema.json`.
- Manufacturer reports are dropped. They are mandated mostly for serious events and use templated text, so the model would otherwise learn the reporter type.
- Likely secondary reports of the same event are collapsed to the earliest ID. Since May 2025 the public files include them, and there is no flag to identify them.
- The text-only proxy model prints its most trusted n-grams, so we can check by eye that no outcome words remain.

Tests: `python3 -m unittest research.scripts.vaers.test_build_vaers -v` (12 tests). They cover:
- the label definition
- the predictor whitelist
- term removal
- manufacturer filtering and deduplication
- normalisation of "None"/"N/A" to `none reported`
- that flipping outcome flags changes only the label
- narrative masking
- adapter compatibility
- deterministic sampling
- the categorical cap staying below the official text detector

## 4. Risks to watch in the screen

- **The symptom terms may make the text so strong that the joint score adds nothing.** That is Yelp's failure mode. The structured signal (age, vaccine, onset) has to add to it. If it doesn't, try a version without the symptoms column.
- **VAERS reports are unverified.** Seriousness is reported, not adjudicated. The paper must say this, and must not make any causal claim about vaccines.
- **Medical-domain TAR gain is a hypothesis**, not yet a measured result.

## 5. How to run

1. Download `2019VAERSData.zip`, `2023VAERSData.zip` and `2024VAERSData.zip` into `MyDrive/text-tabular-research-2/vaers_raw/`.
2. Run notebook 06 (CPU). It runs the tests, builds v1 and v2, profiles them, runs the proxy screen and prints the paths.
3. Run notebook 02 with these settings. It uses the unchanged official code, fold 0.
   - `MODE="diagnostic"`
   - `MODELS_OVERRIDE=["light","cat"]`
   - `CSV_PATH` / `SCHEMA_PATH` set to the paths notebook 06 printed
4. If Joint Signal and TAR Gain are clearly positive for both learners, run `MODE="full"` (all five learners × 5 folds).

## Other sources checked

- **Israeli disability-benefit files and bariatric follow-up records:** there is no public version with free text. They are private health and benefits data, and research use would need ethics approval.
- **ClinicalTrials.gov early termination (backup):** about 154k trials, 18.5% stopped. A 2026 leakage audit shows that actual enrollment, `why_stopped` and the summary or eligibility text (edited after a trial stops) leak the label ([NeurIPS 2026](https://neurips.cc/virtual/2026/163735)). A clean version needs the first-registered version of each record.

## Update, October 16: first official screen and build version 3

**Official-code screen, VAERS v1 (build 2), fold 0** (`paper_version` code through our CSV loader; AUROC):

| Learner | Structured | Text | Joint Frozen | Joint TAR | Joint Signal | TAR Gain |
|---|---|---|---|---|---|---|
| LightGBM | 0.919 | 0.948 | 0.968 | 0.961 | +0.020 | −0.007 |
| CatBoost | 0.928 | 0.950 | 0.962 | 0.964 | +0.012 | +0.002 |
| TabM | 0.921 | 0.949 | 0.963 | 0.962 | +0.014 | −0.001 |
| TabPFN-v2 | 0.927 | 0.953 | 0.964 | 0.963 | +0.011 | −0.001 |
| TabPFN-2.5 | did not run (gated checkpoint: accept the licence on Hugging Face, add `HF_TOKEN`) | | | | | |

- **Joint Signal** is positive for 4 of 4 learners. That is in the range of the accepted classification datasets: their median is +0.016.
- **TAR Gain** is about zero (median +0.004 for the accepted ones). One fold moves TAR gain by roughly ±0.007 to ±0.011, but three of four learners slightly negative is a pattern.
- **Hypothesis, not a result:** the coded MedDRA terms are a controlled vocabulary that frozen E5 already embeds well. TAR helps most on free language.

**Build version 3** responds to the independent review (`vaers_codex_review.md`):

- **Outcome masking everywhere:** every sentence or clause naming an outcome or care setting is removed from the four background fields as well as the narrative. This also removes some legitimate past history; per-field counts are in `counts.background_cells_with_outcome_text_removed`.
- **Symptom filter gaps closed:** "Life-threatening …", "Congenital anomaly" and "birth defect" are now removed.
- **Safer secondary-report merging:** the merge key now also requires the same coded terms. Clusters, removed reports and clusters with conflicting labels are counted.
- **Duplicate-input guard:** passing a yearly ZIP together with its extracted CSVs now raises an error.
- **Leak counter:** `counts.outcome_mentions_left` reports, per text column, the cells still matching the outcome pattern. It should be 0 for every column except `symptoms`, whose kept clinical terms ("Hypertensive emergency") can match by design.
- **Frozen category vocabulary:** the category vocabulary is saved in `schema.json`. The grouping is label-free and fixed when the dataset is built.
- **Task framing:** retrospective classification of recorded report seriousness. It is not a forecast and says nothing about vaccine causation.
- **New variants:**
  - **v4** = v1 + the reporter's free-text narrative, after removing boilerplate, lot and ID numbers, and outcome sentences.
  - **v5** = the narrative instead of the coded terms.

  Both use the same population as v1.
- **Screening plan:** LightGBM + CatBoost, folds 0–2 (`FOLDS_OVERRIDE`), one variant per Colab session.
