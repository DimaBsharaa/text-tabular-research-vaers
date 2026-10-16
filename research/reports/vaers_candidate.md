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

## Update: v4/v5 rejected, v6 (onset narrative)

The build-3 CPU screen of v4 and v5 passed the keyword leak counter: 0 outcome words were left in every masked column. Even so, the text-only model's strongest narrative n-grams were:

- `remdesivir`, `iv`, `o2`, `tx`: inpatient treatment
- `unspecified`: ICD diagnosis text
- `pmhx`: chart shorthand
- `21`, `2021`, `2022`: dates of the hospital course

The full narrative therefore carries **hospital-chart language that reveals hospital care without naming it**. Sentence masking cannot remove this reliably, so v4 and v5 are rejected. That is a documented negative finding.

**v6** keeps v1 and adds only the **opening** of the narrative: the reporter's own account of the reaction.

- Boilerplate, identifiers, dates, ICD-like codes and chart abbreviations are removed first.
- The text is then cut at the first sentence that mentions care, treatment, testing, a clinician or an outcome, and limited to at most 3 sentences.
- The leak counter adds `narrative_onset_care_words` (expected 0).
- The top-n-gram audit must show no care or chart language before any GPU screen.

## Update: column leak audit found a selection artifact, so build 4 narrows the scope

A per-column audit of v1 (each column alone, cross-fitted, 60,000 rows) gave:

- `symptoms` 0.957
- **`onset_days` 0.877**
- `vax_manu` 0.818
- `vax_type` 0.817
- `state` 0.768
- `admin_by` 0.714

The serious rate was **0.420 in 2023 against 0.059 in 2019**, and **0.536 for COVID19** reports.

**Diagnosis:** VAERS also holds breakthrough-infection reports: COVID disease weeks or months after vaccination. During the pandemic, hospitalised and fatal breakthrough cases were the ones required to be reported. Seriousness in those reports therefore follows from how they were collected, and long onset plus a COVID vaccine identifies them.

**Build 4** restricts the population to **reactions to the vaccine**:

- Reports coded with the infection or failure terms in `OUT_OF_SCOPE_TERM` are excluded (vaccine breakthrough infection, COVID-19, SARS-CoV-2 test positive or infection, vaccination failure, drug ineffective).
- Reports with **onset more than 42 days** after vaccination are excluded; unknown onset is kept.
- Removal counts are in `schema.json`.
- Notebook 06 now runs the column audit automatically. A high single-column score is a reason to investigate that column, not an acceptance rule. `state`/`admin_by` are re-checked after the fix.
- **Status of this explanation:** the breakthrough-reporting story is a hypothesis. It is checked after the rebuild by comparing serious rates by year and vaccine type before and after the scope change. The scope rule is a cohort restriction (which reports the dataset covers), not a causal claim.

### Information boundary (why v7 exists)

Every VAERS field is recorded when the report is filed, which is after the event. No column is guaranteed to have been *recorded* before the outcome. The fields differ in what they *describe*:

- **Pre-existing by nature:** age, sex, state, the vaccine given (type, manufacturer, dose, route, site, facility), history, current illness at vaccination, medications and allergies.
- **Early event:** onset days, and the reporter's opening account of the reaction (v6/v7 `narrative_onset`, cut before any care).
- **Whole-report:** the coded MedDRA terms, coded from the full report, including findings made in hospital.

**v7 drops the whole-report column.** It is classification using restricted narrative excerpts and background. It is still retrospective: the excerpt was written after the event, and the extractor's cut is a heuristic. For example, "Two weeks later she could no longer walk." can survive the cut. v1/v6/v7 are all described as retrospective report classification, not as forecasting.

### Follow-up to the second review (same build 4)

- **Identical rows across variants.** Exact duplicates are now judged on all source fields, and the 60,000-row sample is a fixed hash of the report ID. v1/v3/v6/v7 therefore contain the same report IDs; a test checks this. This is needed for fair comparisons.
- **Wider masking.** "ward", "overnight", "stay(ed)" and "bedside" are now masked. "An overnight stay on the ward was required." is removed. A zero keyword count is still only a screen, not proof; the n-gram audit and real examples complement it.
- **Masking by label.** `counts.background_masked_share_by_label` reports how often background text was altered for serious vs non-serious reports.
- **Non-monotonic audit.** For numeric columns the column audit now also scores an out-of-fold rate per value bin. A year that is much more serious than its neighbours is caught; the linear 0.562 for `report_year` understated this.
- **Unchanged, documented.** Category grouping is a label-free dataset definition fixed at build time, not model preprocessing. VAERS seriousness rates are rates among *reported* events, not vaccine risk.

### Follow-up to the third review

- **Negative onset removed.** Reports where onset is before vaccination (NUMDAYS < 0) are now excluded (`counts.removed_negative_onset`). Before, they became a missing value.
- **Exact-duplicate key widened.** It now also includes the vaccination date, onset date, lot and number of vaccines, so two different patients with identical text are no longer merged.
- **Onset extractor.** "given" now cuts the excerpt only when it refers to treatment ("given epinephrine", "given IV fluids"). "Vaccine was given at noon." is kept. Notebook 06 prints real excerpts per label and `counts.narrative_onset_empty_share_by_label`. A large gap between labels in the empty share would itself be a signal to investigate.
- **Category grouping.** It now applies only to a column with more than 99 distinct values, so none of the current columns are grouped.
- **Before any GPU screen:** inspect the rebuilt CSV's real excerpts, the empty share per label, the duplicate counts and the column audit.

## Update: build 4 results (CPU screen) and build 5

**The scope change worked as predicted.** Compared with build 3 (same years, same 60,000-row size):

| Check | Build 3 | Build 4 |
|---|---|---|
| Serious rate, 2023 reports | 0.420 | 0.164 |
| Serious rate, COVID19 reports | 0.536 | 0.225 |
| `onset_days` alone (AUROC) | 0.877 | 0.717 |
| `vax_type` alone | 0.817 | 0.715 |
| `state` alone | 0.768 | 0.586 |
| `admin_by` alone | 0.714 | 0.628 |
| `symptoms` alone | 0.957 | 0.923 |

Build 4 removed 8,054 infection or failure reports and 10,592 reports with onset over 42 days. This supports the breakthrough-reporting hypothesis, but does not prove it. COVID reports remain more often serious (0.225 against 0.03–0.08 for other vaccines). This is consistent with the COVID emergency-use authorisations, which required providers to report serious adverse events; we read it as a reporting rule, not as vaccine risk. Unknown `state` has a serious rate of 0.294 (6.7% of rows), likely a reporter-type effect; it is documented, not removed.

**CPU proxy (TF-IDF + logistic regression, one split; not the benchmark):**

| Variant | Structured | Text | Joint | Joint − best single | Text − structured |
|---|---|---|---|---|---|
| v1 | 0.807 | 0.887 | 0.900 | +0.013 | +0.080 |
| v6 | 0.807 | 0.898 | 0.907 | +0.009 | +0.091 |
| v7 | 0.807 | 0.825 | 0.857 | +0.032 | +0.019 |

**New finding: chart language in the onset excerpt.** Among the strongest serious-class words in `narrative_onset` were:

- `unspecified` and `init`: ICD-10 wording, as in "…, unspecified, init encntr"
- `acute`, `hypoxia`, `respiratory failure`: discharge-diagnosis wording
- `chronic` and `obstructive sleep apnea`: comorbidity lists

So some excerpts are pasted hospital diagnosis lists. They reveal the hospital course without naming it, the same problem as v4/v5 at a smaller scale. The keyword counter also found 85 excerpts with care words. These came from the sentence splitter breaking inside "Dr. Lee".

**Build 5:**

- The excerpt is now also cut at chart and diagnosis-list language: ICD descriptors (unspecified, init, encntr, sequela, NOS), EHR markers (HCC) and clinical shorthand (s/p, c/o, h/o, HPI, assessment, impression).
- The joined excerpt is re-checked and shortened until no care, chart or outcome pattern matches, so both counters are 0 by construction.
- Notebook 06 prints three serious-row examples for each of the strongest onset words, so we can read where each word comes from.
- Clinical event words such as "stroke" or "seizure" in a first-hand account are kept. They describe the reaction itself, as the coded symptoms do.
- Expected cost: more empty excerpts among serious reports. `narrative_onset_empty_share_by_label` and the missing-flag AUROC (build 4: 0.556) are watched for this.

## Update: build 5 results and build 6

**Build 5 (CPU screen, notebook 06 at f7878a6):**

- Chart words left in the excerpt: 0.
- Care words left: 3. Cause: removing a date left a double space ("went 12/3 to" became "went  to"), which the single-space care pattern missed until the final clean-up joined it back. Fixed in build 6, with a test.
- Strongest onset words in v7: stroke, hypoxia, my, hypotension, acute, neuropathy, paralysis, seizure, within. These are event descriptions; the ICD wording (unspecified, init) is gone.
- **New issue: the empty excerpt carries the label.** 28.9% of serious reports had an empty excerpt, against 11.8% of non-serious ones; the missing-flag AUROC rose to 0.588. An excerpt is empty when the narrative *starts* with care, chart or outcome text, which is typical of hospital-written serious reports.

| Variant | Structured | Text | Joint | Joint − best single |
|---|---|---|---|---|
| v1 | 0.818 | 0.902 | 0.912 | +0.010 |
| v6 | 0.818 | 0.914 | 0.919 | +0.004 |
| v7 | 0.818 | 0.844 | 0.875 | +0.031 |

**Build 6:**

- **Population:** reports whose narrative opens with an account of the reaction (non-empty excerpt). This applies to every variant, so v1, v6 and v7 keep identical report IDs and the empty-excerpt signal no longer exists. It is a cohort definition and is stated in the paper. It lowers the share of serious reports, because hospital-written reports are more often excluded.
- Whitespace is collapsed after dates and codes are removed (the care-word fix above).
- The notebook's audit printing moved into `research/scripts/vaers/audit_report.py`, so an older notebook copy still prints the current checks after `git pull`.
- **GPU screen priority:** v7 (largest joint gain, most free text), then v1. v6 adds little over v1 in the proxy and goes last.

## Update: build 6 results and build 7

**Build 6 (notebook 06 at f1ed4a6):**

- Population: 75,944 → 66,760 reports with an onset excerpt; 60,000 sampled; serious rate 8.3%.
- Care and chart words left: 0. No table column alone is above 0.716.

| Variant | Structured | Text | Joint | Joint − best single |
|---|---|---|---|---|
| v1 | 0.813 | 0.904 | 0.913 | +0.009 |
| v6 | 0.813 | 0.914 | 0.918 | +0.003 |
| v7 | 0.813 | 0.852 | 0.868 | +0.016 |

**The real examples behind the top onset words** were mostly first-hand accounts:

- "Hours after my 4 month old received …"
- "Stroke within a few days of vaccination."
- "Loss of speech and memory."

Two leftover chart patterns appeared:

- **Capitalised EHR problem lists:** "HYPOTENSION HYPERKALEMIA ACUTE HYPERCAPNIC RESPIRATORY FAILURE HYPERKALEMIA HYPOXIA", "HYPOXIA BENIGN NEOPLASM, LEFT CHOROID".
- **Measured vital signs:** "Chest pain and hypoxia, BP 137/82, pulse 102."

**Build 7** also cuts the excerpt at:

- vital-sign patterns: BP, mmHg, SpO2, sats, pulse ox, "pulse 102"
- sentences that are almost all capitals and contain no ordinary narrative word, i.e. problem lists. "MY ARM WAS SWOLLEN" is kept.

Lay wording such as "I sat down" or "my pulse was racing" is kept, and tests cover both cases. New counters: `narrative_onset_vitals` and `narrative_onset_problem_lists`, both 0 by construction.

**Known, documented limitations:**

- Some excerpts are only a diagnosis name ("Acute kidney injury."). This carries the same information as the coded MedDRA terms in v1. We do not try to remove it.
- First-person reports ("my", 706 serious rows) are more often serious. This is a reporter-type effect present at filing time, not outcome text.

## Update: build 7 results, fourth review, build 8

**Build 7 (notebook 06 at 6791e60):**

- 65,575 reports with an excerpt → 60,000 sampled; serious rate 7.8%.
- Empty-excerpt share before the filter: 34.7% of serious reports vs 12.7% of non-serious.
- Max single table column AUROC 0.705.

| Variant | Structured | Text | Joint | Joint − best single |
|---|---|---|---|---|
| v1 | 0.780 | 0.870 | 0.882 | +0.013 |
| v6 | 0.780 | 0.887 | 0.891 | +0.005 |
| v7 | 0.780 | 0.833 | 0.850 | +0.017 |

**A real failure of the build-7 problem-list rule:** "ACUTE HYPOXEMIC RESPIRATORY FAILURE ANEMIA OF CHRONIC DISEASE …" survived, because "OF" counted as a narrative word. The fourth review showed the same with "AND".

**Fourth review (Codex), and what changed:**

- **Policy before rules.** The excerpt now follows a written policy (`ONSET_POLICY`, saved in `schema.json` as `narrative_onset_policy`):
  - It may contain what the reporter says happened, including medical words for the reaction.
  - It stops at the first sentence that states an outcome category or care setting, describes care, treatment, testing or a clinician, or carries clinical-record content.
  - Rules match the text only, never the label. We do not add filters only because a word predicts seriousness.
- **Clinical-record content, redefined:**
  - **Runs of 4+ capitalised words** (`CAPS_RUN`) replace the build-7 rule. The prose before such a run is kept ("Moderna vaccine was administered - reported by patient"); short capitalised symptom lists ("FEVER CHILLS HEADACHE") are kept.
  - **Vital signs only with a numeric reading:** "BP 137/82", "HR 120, RR 28, T 39.1", "SpO2 88%". "My home blood pressure monitor read low" is kept.
- **Disability statements are outcome text:** wheelchair, bed-bound, bedridden, and "cannot/unable to work". A disability is one of the label's outcome categories.
- **Event duplicates, stage 2.** Reports that agree on vaccination date, onset date, age, sex, state, vaccine type and a non-blank lot, and whose coded terms overlap strongly (one set contains the other, or Jaccard ≥ 0.5), are one event; the earliest ID is kept. Same-key reports with unrelated terms stay, as the second review asked. Counts: `event_dedup_*`.
- **Cohort reporting.**
  - `schema.population` now states the excerpt restriction and that it disproportionately excludes serious reports.
  - `counts.cohort_filter` gives retained/excluded counts and serious rates by year and by vaccine type.
- **Provenance.**
  - Each build is written to its own folder (`candidates/build8/vaers_vX/`).
  - `schema.provenance` records the commit and the SHA-256 of every input ZIP.
  - `audit.txt` (the full printed audit) is saved next to the CSV.
- **Benchmark version.** Notebook 02 runs the paper-version code (3bb9507) through our CSV loader. That is an exploratory development screen only. The final result must be produced at 89a1854 by `benchmark.py`, with the dataset registered natively. That step is planned after the screen.

**Policy frozen at build 8.** The GPU screen (v7 first, then v1; LightGBM + CatBoost; folds 0–2) uses build 8 unchanged. Later rule changes would need a new build and a new screen.
