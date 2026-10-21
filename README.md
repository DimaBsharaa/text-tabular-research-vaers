# VAERS-Serious: a new text + table dataset for MulTaBench

A curated dataset of **18,859 vaccine adverse-event reports** where neither the structured fields nor the free text is
enough on its own: **combining them helps, and fine-tuning the text encoder helps further**. It passes both
[MulTaBench](https://github.com/alanarazi7/MulTaBench) curation checks on **all 5 learners**, evaluated with the
official benchmark code, unchanged.

**Task:** given what a report contains when it is filed, predict whether it describes a **serious** outcome
(death, life-threatening event, hospitalisation, disability or birth defect).

## Results

Official MulTaBench `benchmark.py` at commit [`c955466`](https://github.com/alanarazi7/MulTaBench/tree/c95546638be9bac472a2ee38afa7171060ab80bc)
(`master` as of 26 Jul 2026), 5 learners × 4 conditions × 5 folds = 100 runs. AUROC, mean of 5 folds.

| Learner | Structured only | Text only | Joint (frozen E5) | Joint (fine-tuned E5, TAR) | Joint gain | TAR gain |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM | 0.721 | 0.778 | 0.809 | 0.839 | +0.031 | +0.030 |
| CatBoost | 0.739 | 0.787 | 0.823 | 0.843 | +0.036 | +0.020 |
| TabM | 0.725 | 0.799 | 0.832 | 0.847 | +0.033 | +0.015 |
| TabPFN-v2 | 0.735 | 0.795 | 0.821 | 0.843 | +0.027 | +0.022 |
| TabPFN-2.5 | 0.740 | 0.805 | 0.835 | 0.851 | +0.029 | +0.016 |

- **Joint gain** = Joint frozen − max(Structured, Text). **TAR gain** = Joint TAR − Joint frozen. Curation needs each gain > 0 on at least 3 of 5 learners, counted separately for the two checks.
- **Both checks pass on 5/5 learners**, and on **all 25 learner × fold runs** (smallest TAR gain +0.0065).
- Median gains **+0.031 (Joint)** and **+0.020 (TAR)**, against +0.023 and +0.010 for the 20 accepted MulTaBench text datasets. The TAR gain ranks **3rd of 21**.
- Structured-only ≈ 0.72 is real signal: with shuffled labels the same model scores 0.49 on average (max 0.51).

Raw scores: [`results/official_c955466/`](results/official_c955466/) · analysis and charts: [notebook 08](research/notebooks/08_vaers_results.ipynb).

## The dataset

Source: [VAERS](https://vaers.hhs.gov/data/datasets.html), the U.S. vaccine adverse-event reporting system (CDC/FDA, public data).
Yearly files **2019, 2023 and 2024**: 2019 covers the usual mix of vaccines before COVID-19, and 2023–2024 are recent years
after the 2021–2022 COVID-19 vaccine surge, which would otherwise dominate the data.

| Column | Type | What it holds |
|---|---|---|
| `age_years`, `onset_days`, `n_vaccines`, `report_year` | numeric | age, days from vaccination to onset, number of vaccines given, year the report was received |
| `sex`, `state`, `vax_type`, `vax_manu`, `vax_dose_series`, `vax_route`, `vax_site`, `admin_by` | categorical | patient, vaccine and administering-facility facts |
| `patient_background` | text | history, allergies, medications and illness at vaccination |
| `narrative_onset` | text | the opening of the symptom narrative (up to 3 sentences), cut before any care or outcome |
| `serious` | target | 1 if the report's outcome flags record death, life threat, hospitalisation, prolonged stay, disability or birth defect |
| `report_id` | audit | VAERS ID, dropped by the benchmark |

How 208,309 raw reports became 18,859 (full detail in [notebook 02](research/notebooks/02_vaers_eda_and_preprocessing.ipynb)):

| Step | Reports |
|---|---:|
| Unique reports in the three yearly files | 208,309 |
| Drop manufacturer-submitted reports (templated, a different serious rate) | 131,372 |
| Keep reactions with onset 0–42 days (or unknown); drop infection / vaccine-failure reports | 112,726 |
| Merge duplicate reports of the same event | 111,127 |
| At least one reaction term after removing outcome and care terms | 75,925 |
| Non-empty narrative excerpt (7.9% serious) | 63,564 |
| Remove care text from excerpts and post-event background | 62,754 |
| Matched controls: every serious report + up to 3 non-serious per serious, same vaccine × year × age band × sex | 18,863 |
| Final care-text cleanup (503 excerpts shortened, 4 reports removed) | **18,859 (26.0% serious)** |

**Leakage controls.** The label flags, dates, hospital stay, lab data, recovery status and coded symptom terms are never inputs.
The narrative is cut before the first sentence about treatment or outcome, because "admitted", "ICU" or "IVIG" give the label away.
Controls are matched on structured fields only, never on text. Each rule and the evidence behind it is in notebook 02.

## Reproduce

```bash
pip install -r research/requirements.txt

# 1. Raw files -> dataset (CPU, about 3 minutes). Stops if the result differs from the published sha256.
#    The three raw VAERS ZIPs are in research/data/vaers_raw/ (exact copies of the files we used).
python -m research.scripts.vaers.build_v11b research/data/vaers_raw/2019VAERSData.zip \
    research/data/vaers_raw/2023VAERSData.zip research/data/vaers_raw/2024VAERSData.zip --out-dir out/

# 2. Dataset -> official scores (Colab GPU, about 8 hours on an A100)
#    Open research/notebooks/07_vaers_official_run_colab.ipynb in Colab and Run all.

# 3. Scores -> tables and charts (CPU, a few minutes)
#    Open research/notebooks/08_vaers_results.ipynb in Jupyter and Run all.
```

**How we know it is the official benchmark.** Notebook 07 clones the MulTaBench repository, checks out `c955466` and asserts
the commit hash, installs its pinned `requirements.txt`, and calls `benchmark.py` unchanged. The only addition is registering the
dataset: one enum line, one file-path line, one curation recipe and the CSV, saved as
[`registration.diff`](results/official_c955466/registration.diff). Every row of `results.csv` records the commit and the GPU.

## Notebooks

| # | Notebook | What it answers |
|---|---|---|
| 01 | [MulTaBench benchmark study](research/notebooks/01_multabench_benchmark_analysis.ipynb) | Why do the 56 original candidate datasets pass or fail? What does a passing dataset look like? |
| 02 | [VAERS EDA and preprocessing](research/notebooks/02_vaers_eda_and_preprocessing.ipynb) | What is in the raw data, which problems it has, and how each rule fixes one — run end to end from the raw files |
| 07 | [Official run (Colab)](research/notebooks/07_vaers_official_run_colab.ipynb) | The 100 official runs, with their real outputs |
| 08 | [Results](research/notebooks/08_vaers_results.ipynb) | Scores, permutation test, dataset versions, comparison with the benchmark |

Notebooks 03–06 in [`research/notebooks/development/`](research/notebooks/development/) record how the dataset was built and screened.

## How we got here

1. **Study the benchmark** (notebook 01). Only 23 of the 56 original candidates passed, and the most common failure is the TAR check. The subject area does not predict success. What matters is that each modality carries information the other lacks, and that the text uses task-specific language a generic encoder only partly captures.
2. **First candidate: tennis** — predict the match winner from statistics plus the players' press-conference interviews. The text added nothing over the statistics (joint 0.766 vs structured 0.768) and the sources could not be redistributed, so it was dropped. Full story: [`tennis/`](tennis/README.md).
3. **VAERS**, version by version: remove leaking fields, keep only the opening of the narrative, fix the cohort, match the controls. Each version was screened with LightGBM + CatBoost on folds 0–2 before choosing **v11b**; folds 3–4 were first seen in the official run, and stay positive. Step-by-step log: [`research/reports/vaers_progress.md`](research/reports/vaers_progress.md).

## Limitations

- VAERS reports are unverified and self-selected. The label is what the reporter recorded, not a clinical judgement, and nothing here says anything about whether vaccines cause these events.
- The 26% serious rate comes from the matched-control design (about 8% across all eligible reports). AUROC does not depend on the class ratio, but absolute probabilities do not transfer to the full population.
- The narrative excerpt is defined by written rules. Hidden care language can never be ruled out completely; the rules, audits and edit ledger are published so they can be checked.
- Each test fold has about 1,885 reports, so a single AUROC has a standard error of about ±0.01. The conclusions rely on all 25 runs, not one.

## Repository layout

```
results/official_c955466/   official scores (100 runs), per-fold and summary tables, registration diff
research/
  data/vaers_v11b/           the dataset: CSV, schema, care-cleanup edit ledger
  data/vaers_raw/            the three raw VAERS yearly files the dataset is built from
  data/screens/              development screen results per version
  notebooks/                 01, 02, 07, 08 (+ development/03–06)
  scripts/vaers/             build pipeline (build_v11b.py is the entry point) and tests
  reports/                   development log and audits
evaluation/                  Colab adapter used for the development screens
tennis/                      the earlier, dropped candidate (self-contained)
```
