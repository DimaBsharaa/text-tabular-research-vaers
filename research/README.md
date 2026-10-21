# Research guide

Everything behind the VAERS dataset and the benchmark study. The [project README](../README.md) has the summary and results.
The earlier tennis candidate lives in [`../tennis/`](../tennis/README.md).

## Notebooks — read in this order

| # | Notebook | What it shows |
|---|---|---|
| 01 | [MulTaBench benchmark study](notebooks/01_multabench_benchmark_analysis.ipynb) · [HTML](notebooks/01_multabench_benchmark_analysis.html) | The 56 original candidate datasets: which pass, which fail and why. 13 charts and tables, each followed by an insight. |
| 02 | [VAERS EDA and preprocessing](notebooks/02_vaers_eda_and_preprocessing.ipynb) | **A** raw data and the problems it exposes · **B** the pipeline run end to end from the raw files, with the funnel, nulls, duplicates and before/after text · **C** the final dataset: per-column signal, correlations, leakage checks, words that drive the label. |
| 07 | [Official run (Colab)](notebooks/07_vaers_official_run_colab.ipynb) | The official `benchmark.py` at MulTaBench `c955466`: 100 runs, with their real outputs. |
| 08 | [Results](notebooks/08_vaers_results.ipynb) | Scores and both checks, permutation test, dataset versions, comparison with the 20 core text datasets, class balance. |

### Development notebooks ([`notebooks/development/`](notebooks/development/))

Kept as a record of how the dataset was built; numbered in the order they were used.

| # | Notebook | Role |
|---|---|---|
| 03 | [Build from raw](notebooks/development/03_vaers_build_from_raw.ipynb) | Builds VAERS versions v1–v9 from the raw files (`build_vaers.py`) and runs a cheap CPU screen. |
| 04 | [Care-boundary cleanup](notebooks/development/04_vaers_care_cleanup.ipynb) | v10: cuts care text from excerpts and background (`refine_v9.py`), tested on matched rows. |
| 05 | [Benchmark screen runner](notebooks/development/05_benchmark_screen_colab.ipynb) | Colab runner for the development screens (paper-version MulTaBench code `3bb9507` through [`../evaluation/`](../evaluation/README.md)). Began as a tennis smoke test. |
| 06 | [Screen audit](notebooks/development/06_vaers_screen_audit.ipynb) | CPU audit of a finished screen: identical test rows across conditions, TAR trained on training rows only, near-duplicates, style vs content. |

## Data

| Path | Contents |
|---|---|
| [`data/vaers_v11b/`](data/vaers_v11b/README.md) | **The dataset**: `vaers_v11b.csv`, `schema.json`, and the care-cleanup edit ledger |
| [`data/vaers_raw/`](data/vaers_raw/README.md) | The three raw VAERS yearly files (2019, 2023, 2024) with their sha256 |
| [`data/screens/`](data/screens/) | Development screen results per version; `version_screens_light_cat_folds012.csv` feeds notebook 08 |
| `data/dataset_inventory.csv`, `learner_curation_results.csv`, `historical_fold_scores.csv` | The 56 original candidates, their per-learner gains and per-fold scores (notebook 01) |
| `data/*_profiles.csv`, `sample_target_distribution.csv`, `analog_comparisons.csv`, … | Release-preview profiles and comparisons used by notebook 01 |
| [`data/sources/`](data/sources/) | Pinned official files, manifests and small release previews (not a benchmark clone) |

## Scripts

| Path | Responsibility |
|---|---|
| [`scripts/vaers/build_v11b.py`](scripts/vaers/build_v11b.py) | **Entry point**: raw ZIPs → `vaers_v11b.csv`, asserting both intermediate and final sha256 |
| `scripts/vaers/build_vaers.py` | Cohort, field selection, excerpt policy, term removal (versions v1–v9) |
| `scripts/vaers/refine_v9.py` | Care-boundary-2 cleanup (v10) |
| `scripts/vaers/proxy_ideas/` | CPU proxy used to compare dataset designs (matched controls won) |
| `scripts/vaers/run_audit.py`, `audit_*.py`, `compare_variant_runs.py`, `screen_stats.py` | Audits of screen runs and data |
| `scripts/vaers/test_*.py` | 61 unit tests for the pipeline and audits (command below) |
| `scripts/analyze.py`, `build_notebook.py`, `execute_notebook.py`, `write_reports.py` | Notebook 01 and the benchmark-study reports |

## Reports

| Report | Question |
|---|---|
| [`vaers_progress.md`](reports/vaers_progress.md) | **Start here.** Every VAERS version as problem → fix → result, and the final status |
| [`vaers_decision_log.md`](reports/vaers_decision_log.md) | Each decision as saw → inferred → did → checked |
| [`vaers_candidate.md`](reports/vaers_candidate.md) | Detailed build notes, builds 1–9 |
| `vaers_*_review.md`, `vaers_*_audit.md`, `vaers_audit_local_validation.md` | Independent reviews and audits of runs and data |
| [`research_findings.md`](reports/research_findings.md), [`dataset_comparison.md`](reports/dataset_comparison.md), [`validation.md`](reports/validation.md) | Benchmark study findings, domain comparisons, how the historical reconstruction was checked |
| [`evaluation_infrastructure_plan.md`](reports/evaluation_infrastructure_plan.md) | How the official code splits, trains and scores |
| `project_context.md`, `source_availability.md` | Early planning notes |

## Reproduce

From the repository root, with `pip install -r research/requirements.txt`:

```bash
# Dataset from the raw files (asserts sha256)
python -m research.scripts.vaers.build_v11b research/data/vaers_raw/2019VAERSData.zip \
    research/data/vaers_raw/2023VAERSData.zip research/data/vaers_raw/2024VAERSData.zip --out-dir out/

# Pipeline tests
python -m unittest research.scripts.vaers.test_build_vaers research.scripts.vaers.test_refine_v9 \
    research.scripts.vaers.test_run_audit research.scripts.vaers.test_audit_source_rows

# Benchmark study (notebook 01) from the cached sources
python3 research/scripts/analyze.py && python3 research/scripts/build_notebook.py && python3 research/scripts/execute_notebook.py
```

Notebook 02 runs the same build inside the notebook, from the ZIPs in `research/data/vaers_raw/`.
