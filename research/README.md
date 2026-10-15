# Research guide

The [project README](../README.md) explains our goal, process and current status. **No final dataset has qualified.** For current Cornell A findings, open [notebook 05](notebooks/05_tennis_interview_validation.ipynb), the [text study](reports/tennis_a_text_preprocessing.md) and the [leakage report](reports/tennis_a_leakage_validation.md).

## Notebooks

| File | Scope and saved outputs |
| --- | --- |
| [01_multabench_dataset_analysis.ipynb](notebooks/01_multabench_dataset_analysis.ipynb) · [HTML](notebooks/01_multabench_dataset_analysis.html) | Historical 56-candidate analysis: 22 executed code cells, 13 charts and tables; an insight follows each chart. Includes per-dataset text/target inspection. |
| [02_tennis_official_benchmark_colab.ipynb](notebooks/02_tennis_official_benchmark_colab.ipynb) | A100 notebook: reference verification, installation, smoke run, dashboard and export. The successful user-run results are in [observed_run](data/tennis_smoke/observed_run/README.md); this notebook is not a saved copy of that Colab session. |
| [03_tennis_interviews_eda.ipynb](notebooks/03_tennis_interviews_eda.ipynb) | Corpus hierarchy, missingness, lengths, content review and source/match linkage; executed tables and three plots. |
| [04_tennis_dataset_prototypes.ipynb](notebooks/04_tennis_dataset_prototypes.ipynb) | Original A/B/C definitions, coverage, CPU diagnostics and earlier A v0 review; four plots. |
| [05_tennis_interview_validation.ipynb](notebooks/05_tennis_interview_validation.ipynb) | Current A v1: dates, exact token budgets, six text methods, real examples, 45 audit entries and 13 completed CPU condition fits; three plots. |

Rerunning 03–05 needs the local cache and analysis environment; inspecting saved outputs needs neither GPU nor raw downloads. The [evaluation guide](../evaluation/README.md) explains notebook 02 and its exported results.

## Reports

| Report | Question it answers |
| --- | --- |
| [project_context.md](reports/project_context.md) | What did the course instructions, supplied context and earlier discussion establish? |
| [research_findings.md](reports/research_findings.md) | What do historical accepted/rejected datasets teach us, and which observations remain hypotheses? |
| [source_availability.md](reports/source_availability.md) | Which historical/released sources were actually inspectable? |
| [dataset_comparison.md](reports/dataset_comparison.md) | Which existing domains offer comparisons for a new dataset? |
| [validation.md](reports/validation.md) | How was the historical reconstruction checked? |
| [evaluation_infrastructure_plan.md](reports/evaluation_infrastructure_plan.md) | How does the pinned official implementation split, train, score and integrate datasets? What are its hardware and leakage concerns? |
| [tennis_dataset_feasibility.md](reports/tennis_dataset_feasibility.md) | Initial tennis source survey, proposed schema, possible coverage and risks. |
| [tennis_cornell_eda.md](reports/tennis_cornell_eda.md) | What is actually inside Cornell's interview corpus? |
| [tennis_variant_comparison.md](reports/tennis_variant_comparison.md) | How do A interviews, B tactical templates and C ordinary-context templates compare? Original A counts are provisional v0. |
| [tennis_pilot_results.md](reports/tennis_pilot_results.md) | What happened in the original B/C CPU pilots? Not E5/TAR results. |
| [tennis_interview_followup.md](reports/tennis_interview_followup.md) | Earlier publisher checks, source terms, mirror coverage and random-split risks. That stage did not fit models. |
| [tennis_a_text_preprocessing.md](reports/tennis_a_text_preprocessing.md) | **Current:** which passages fit E5, what is lost, and how did six methods compare on development data? |
| [tennis_a_leakage_validation.md](reports/tennis_a_leakage_validation.md) | **Current:** what passed/failed, what assumptions remain, and where is the evidence? |
| [tennis_source_clarification_drafts.md](reports/tennis_source_clarification_drafts.md) | Historical unsent reference drafts only. No outreach is part of the workflow. |

`reports/analysis_summary.json` supplies historical summary numbers; `reports/figures/` contains historical figures. Current v1 reports and notebook 05 supersede earlier statements that A was untrained or that publication verification automatically blocked local research.

## Data and evidence

| File or directory | Contents |
| --- | --- |
| [data/dataset_inventory.csv](data/dataset_inventory.csv) | All 56 original candidates and acceptance status |
| [data/learner_curation_results.csv](data/learner_curation_results.csv) | Historical four-condition means, gains and learner outcomes |
| [data/historical_fold_scores.csv](data/historical_fold_scores.csv) | Underlying per-fold scores |
| [data/dataset_profiles.csv](data/dataset_profiles.csv), [data/column_profiles.csv](data/column_profiles.csv), [data/text_column_profiles.csv](data/text_column_profiles.csv) | Release-preview properties at dataset, column and text-column level |
| [data/sample_target_distribution.csv](data/sample_target_distribution.csv) | Preview target distributions, not full-population estimates |
| [data/published_reproduction_disagreements.csv](data/published_reproduction_disagreements.csv) | Published versus reconstructed learner flags |
| [data/leakage_audit_candidates.csv](data/leakage_audit_candidates.csv) | Columns flagged for review, not confirmed violations |
| [data/analog_comparisons.csv](data/analog_comparisons.csv) | Comparable historical datasets/domains |
| [data/sources/](data/sources/) | Pinned selected official files, manifests and small release previews; not a full benchmark clone |
| [data/tennis_smoke/](data/tennis_smoke/README.md) | 488-row Kaggle engineering fixture, schema and construction audit; generated history text |
| [data/tennis_smoke/observed_run/](data/tennis_smoke/observed_run/README.md) | Four successful user-run conditions, summary and archive-derived verification metadata |
| [data/tennis_prototypes/](data/tennis_prototypes/README.md) | A/B/C counts, metadata, predictions, checks and current `a_v1_*` evidence; no bulk interviews |

The [prototype guide](data/tennis_prototypes/README.md) maps individual evidence files. Full sources and text-bearing candidates live outside Git in `../tennis-research-cache/`, overridable by `TENNIS_RESEARCH_CACHE`. A fresh clone contains results/code, but not all inputs needed to reconstruct tennis CSVs. No public transcript redistribution grant has been established. Both course PDFs remain in the parent `NLP/` folder.

## Scripts

| Module group under `scripts/` | Responsibility |
| --- | --- |
| `analyze.py`, `build_notebook.py`, `execute_notebook.py`, `write_reports.py` | Historical tables, notebook 01, HTML and reports |
| `fetch_sources.py`, `fetch_previews.py` | Optional bounded historical source/preview acquisition |
| `tennis/common.py`, `fetch.py`, `audit.py`, `prototypes.py` | Cache/provenance helpers, acquisition, corpus audit and original A/B/C construction |
| `tennis/additional.py`, `diagnostics.py`, `compare.py`, `notebooks.py` | Initial extra checks, CPU pilots, comparison tables and notebooks 03/04 |
| `tennis/interview_followup.py`, `prepare_interview_review.py`, `followup_notebook.py` | Earlier publisher/source/split checks, private v0 review and notebook dispatch |
| `tennis/a_text.py`, `a_temporal.py` | Exact token-budget extraction and as-of interview/history rules |
| `tennis/a_study.py`, `a_context_blocks.py` | Frozen v1 construction and actual CPU diagnostics; answer-block extension has a separate freeze |
| `tennis/a_leakage.py`, `a_additional_checks.py` | Independent replay, counterfactual/negative controls, duplicate/source checks and secondary evidence |
| `tennis/a_review_outputs.py` | Render notebook 05, current A reports and artifact/tokenizer/environment manifests |
| `tennis/test_research.py`, `test_interview_followup.py`, `test_a_validation.py` | 33 regression/adversarial tests for construction, dates, token limits, provenance and preprocessing isolation |

`../evaluation/` contains our adapters/results display, not replacement model implementations. Its optional `tennis_temporal_pilot.py` is checks-only by default; no official Cornell E5/TAR run has occurred.

## Reproduce

Run from the repository root. Saved outputs are the quickest review route; these commands can overwrite generated artifacts.

Historical analysis, with `research/requirements.txt` and cached sources:

```bash
python3 research/scripts/analyze.py
python3 research/scripts/build_notebook.py
python3 research/scripts/execute_notebook.py
python3 research/scripts/write_reports.py
```

The original environment used Python 3.14. Direct requirements are not a complete transitive lock. Its notebook runner captures ordinary Python, display/stdout and matplotlib; it does not support arbitrary magics or kernel isolation.

Current tennis audit/render/tests, with the **existing private cache**, analysis packages, scikit-learn/scipy/openpyxl/psutil and isolated tokenizer assets:

```bash
python3 -m research.scripts.tennis.a_leakage
python3 -m research.scripts.tennis.a_additional_checks
python3 -m research.scripts.tennis.a_review_outputs
python3 -m unittest research.scripts.tennis.test_research research.scripts.tennis.test_interview_followup research.scripts.tennis.test_a_validation -v
```

These commands do not fit E5/TAR. Full construction and CPU fitting commands are separate in the [prototype guide](data/tennis_prototypes/README.md). Hashes guard frozen studies against drift. Source acquisition is not needed to inspect outputs; source-use findings are not permission to fetch or redistribute.

## Interpretation boundaries

- All 56 historical candidates have scores; 40 have release profiles and 34 preserve the original task prefix. Six extended releases change the task.
- Two reconstructed learner flags differ from published flags; overall candidate acceptance agrees. Rounded legacy thresholds differ from the paper's strict equation.
- Profiles cover contiguous 50-row previews, not full populations. They do not establish global balance, missingness or duplication.
- The current three-day delay is an interview-availability assumption. Event dates/content corroboration do not prove publication timestamps.
- Chronological diagnostics and official random holdouts are different protocols. Legitimate source reuse and future-outcome exposure are distinct.
- No CPU joint gain, official Cornell E5/TAR score or reserved-test model score is claimed. Public accessibility does not establish redistribution permission.
