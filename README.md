# Text-tabular dataset research

Technion NLP Final Project, **Standard Track 2**, due **October 26, 2026**. We are curating one new dataset in which structured features and natural-language text provide complementary predictive information. VAERS v9 is our leading development candidate; v10 applies the care-boundary correction. Final qualification remains unproven. Earlier tennis work is preserved as research history.

## Current VAERS workflow (October 17, 2026)

**Start here for the story:** [VAERS progress in one page](research/reports/vaers_progress.md) (v1 → v10: problem, fix, did it work, results).

**For the current run: open 08 on CPU, then 02 on A100. Skip 06 if frozen v9 already exists.**

| Step | Notebook | Input → output |
| --- | --- | --- |
| Raw-data construction | [06 VAERS build](research/notebooks/06_vaers_candidate_build.ipynb), CPU | Official 2019/2023/2024 VAERS ZIPs → frozen build-8 v9 CSV/schema |
| Care-boundary correction | [08 v10 refinement](research/notebooks/08_vaers_v10_refinement.ipynb), CPU | Verified raw ZIPs + frozen v9 → build-9 v10, all-row v10_matched, and matched v9_control; tests, text examples and CPU diagnostics |
| Evaluation | [02 benchmark](research/notebooks/02_tennis_official_benchmark_colab.ipynb), GPU | Selected CSV/schema → Structured, Text, Joint Frozen and Joint TAR scores using unchanged official code |

Raw sources are `2019VAERSData.zip`, `2023VAERSData.zip`, `2024VAERSData.zip`. In Colab they live under `MyDrive/text-tabular-research-2/vaers_raw/`; code is in Git, raw and generated text-bearing datasets are outside Git. Notebook 08 writes `candidates/build9/vaers_<variant>/` on Drive. Start the GPU check with **v10_matched, light + cat, fold 0** (8 evaluations), comparing with the completed v9 baseline; v10_matched is diagnostic-only. The final v10 candidate contains 19,832 rows.

Construction is explicit in [build_vaers.py](research/scripts/vaers/build_vaers.py) and [refine_v9.py](research/scripts/vaers/refine_v9.py). Source/input hashes and row-lineage records link the stages. Frozen v9 and running v8/v9 experiments are preserved. Notebook 08 clones our code from GitHub automatically; no tools ZIP upload is needed.

## What are we trying to demonstrate?

For each learner—LightGBM, CatBoost, TabM, TabPFN 2.0 and TabPFN 2.5—we compare four conditions on the same dataset:

| Condition | Inputs |
| --- | --- |
| Structured Only | Numerical and categorical features |
| Text Only | Frozen E5 embeddings of the text |
| Joint Frozen | Structured features plus frozen E5 embeddings |
| Joint TAR | Structured features plus E5 embeddings adapted using training targets |

**Joint Signal** is Joint Frozen minus the stronger single modality. **TAR Gain** is Joint TAR minus Joint Frozen. The course clarification requires **at least three of five learners for each criterion separately**; the learner sets may differ. A high prediction score alone is insufficient. Exact threshold conventions and official settings are in the [evaluation plan](research/reports/evaluation_infrastructure_plan.md).

## Earlier tennis snapshot

Updated October 15, 2026:

- **Infrastructure works:** the official LightGBM smoke test completed all four conditions on a 488-row Kaggle fixture. This establishes execution, not dataset suitability. See the [observed run](research/data/tennis_smoke/observed_run/README.md).
- **Cornell A is our leading tennis candidate:** genuine player interviews plus historical statistics. The original 4,052 provisional rows remain preserved locally; the refined version has **3,627 rows** after date/link/played-match rules.
- **Text curation is explicit:** six extractive approaches fit the actual E5 512-token budget, including prefixes and special tokens. Notebook 05 shows real before/after examples and what each method loses.
- **Small CPU results are negative so far:** structured-only development AUROC is **0.7684**; the best joint TF-IDF diagnostic is **0.7660**. None of the six text variants improves on structured alone. These are logistic-regression diagnostics, not official E5/TAR results.
- **Leakage evidence is available:** 45 audit entries comprise 36 PASS, six ASSUMPTION, two FAIL and one NOT_RUN. The failures expose an older random-split design and a deliberately leaky negative control; no implemented current-v1 rule failed. The 749 reserved-test rows have **not been model-scored**.
- **Uncertainty remains:** the three-calendar-day interview delay is an availability assumption, not proof of publication. Source-use/redistribution terms remain unresolved; transcript-bearing CSVs stay outside Git.

## Open these notebooks

The research notebooks contain saved tables and visualizations. Notebook 02 is the executable Colab runtime; its successful user-run export is stored separately.

| Notebook | What it contains | When to use it |
| --- | --- | --- |
| [01 — Historical benchmark EDA](research/notebooks/01_multabench_dataset_analysis.ipynb) · [HTML](research/notebooks/01_multabench_dataset_analysis.html) | Original candidates, pass/fail patterns, modality gains and text/column profiles; insights after each chart | Learn from previous dataset outcomes |
| [02 — Official benchmark in Colab](research/notebooks/02_tennis_official_benchmark_colab.ipynb) · [Open in Colab](https://colab.research.google.com/github/DimaBsharaa/text-tabular-research-vaers/blob/main/research/notebooks/02_tennis_official_benchmark_colab.ipynb) | Setup, source-integrity proof, four-condition smoke run, dashboard and Drive export | Check the runtime on A100; its default CSV is an engineering fixture |
| [03 — Cornell interviews EDA](research/notebooks/03_tennis_interviews_eda.ipynb) | Actual corpus hierarchy, lengths, content review, match-link quality and coverage | Understand the interview source |
| [04 — Tennis dataset prototypes](research/notebooks/04_tennis_dataset_prototypes.ipynb) | A/B/C constructions, coverage and preliminary CPU comparisons; earlier A review | Understand why we prioritized real interviews over generated templates |
| [05 — Current Cornell A validation](research/notebooks/05_tennis_interview_validation.ipynb) | Date sensitivity, six text policies, before/after examples, leakage checks and development scores | **Start here for the latest dataset findings** |

Rerunning 03–05 needs the existing local source cache; their saved outputs are readable without it. Notebook 02 uses a separate Python environment and GPU runtime, and still defaults to the Kaggle fixture, not Cornell A.

## Our process

1. **Learn from the existing benchmark.** Reconstruct historical outcomes and inspect available previews: 23 accepted and 33 rejected candidates. TAR gain is a frequent bottleneck. Observed associations guide hypotheses, not guarantees.
2. **Prepare evaluation.** Inspect the pinned official implementation, build our CSV adapter and results display around it, then verify execution with a bounded smoke test.
3. **Compare candidate datasets.** Investigate tennis sources and three constructions: A, real interviews; B, charted tactical-statistic templates; C, verbalized ordinary match context. B/C did not show complementary gains in their CPU pilots.
4. **Construct auditable rows.** Assign Player A/B independently of the winner; join earlier interviews; compute historical statistics using strictly earlier days; exclude target and provenance from predictors.
5. **Curate text and check leakage.** Select original passages within the tokenizer limit, declare publication assumptions, use chronological partitions, and fit transformations only on training rows. Retain source IDs, passage spans, histories, hashes and automated evidence.
6. **Run cheap diagnostics, then freeze a candidate.** We are here: useful language exists, but complementary predictive value remains unproven.
7. **Evaluate the chosen candidate officially.** A bounded E5/TAR diagnostic should precede a full five-learner run. Keep the reserved test for final use. Official random holdouts and temporal forecasting diagnostics answer different questions and must be reported separately.

## File guide

| Location | Contents |
| --- | --- |
| [research/README.md](research/README.md) | Detailed notebook/report/data/script guide and reproduction commands |
| [research/notebooks/](research/notebooks/) | The five notebooks above; notebook 01 also has standalone HTML |
| [research/reports/](research/reports/) | Findings, source comparisons, evaluation plan, text study and leakage report |
| [research/data/](research/data/) | Historical benchmark tables, source manifests, bounded previews and tennis evidence |
| [research/data/tennis_prototypes/README.md](research/data/tennis_prototypes/README.md) | Individual A/B/C and `a_v1_*` evidence files explained |
| [research/scripts/](research/scripts/) | Historical EDA/report scripts; `tennis/` contains our construction, text, audit, diagnostic and test modules |
| [evaluation/README.md](evaluation/README.md) | Colab instructions, presets, output files and the official-code boundary |
| [evaluation/colab_runner.py](evaluation/colab_runner.py) | Our CSV/schema adapter, reference verification and per-job orchestration |
| [evaluation/results.py](evaluation/results.py) | Our aggregation of four scores, gains, completeness and qualification flags |
| [evaluation/colab_session.py](evaluation/colab_session.py) | Colab bootstrap/logging, export and runtime shutdown support |
| [evaluation/tennis_temporal_pilot.py](evaluation/tennis_temporal_pilot.py) | Optional chronological adapter; validates provenance, defaults to checks only, and uses unchanged official classes if explicitly run |

For conclusions, read the [current text study](research/reports/tennis_a_text_preprocessing.md), [leakage validation](research/reports/tennis_a_leakage_validation.md), [original A/B/C comparison](research/reports/tennis_variant_comparison.md) and [evaluation plan](research/reports/evaluation_infrastructure_plan.md). Older reports describe earlier stages; notebook 05 and the two current A reports take precedence for v1 counts/results.

## Dataset files: public evidence and local inputs

GitHub contains our code, executed notebooks, reports, counts, metadata, audit outcomes and prediction tables. Full tennis sources and text-bearing CSVs are in the **sibling** directory `../tennis-research-cache/`, configurable with `TENNIS_RESEARCH_CACHE`. They are not published because redistribution permission has not been established.

| Local path under `../tennis-research-cache/derived/` | Meaning |
| --- | --- |
| `variant_A.csv`, `variant_B.csv`, `variant_C.csv` | Original exploratory constructions; A has 4,052 provisional rows |
| `candidate_A_v0_review/` | Earlier A review, preserved for comparison |
| `candidate_A_v1/features_<policy>.csv` | Current 3,627 rows: one CSV per text policy, identical structured inputs and labels |
| `candidate_A_v1/schema*.json` | Feature/target definitions, policy and explicit unresolved assumptions |
| `candidate_A_v1/audit.csv`, `splits.csv` | Source dates/identities and fixed train/development/reserved-test membership |
| `candidate_A_v1/*private*.jsonl.gz`, `history_lineage.jsonl.gz` | Exact passage provenance and contributing historical match IDs |

Current A has ten numerical historical predictors, six match-context fields, and two text columns. Structured data can be categorical as well as numerical. Target, match ID, source IDs and winner/loser roles are excluded from predictors. The two course PDFs remain in the parent `NLP/` folder.

## Official benchmark boundary

The reference is [alanarazi7/MulTaBench](https://github.com/alanarazi7/MulTaBench), pinned to `paper_version` commit [`3bb9507`](https://github.com/alanarazi7/MulTaBench/tree/3bb95079a6146ed35d45187e98bcad5b063273a1). Our repository contains a **selected read-only source cache for inspection**, not a full benchmark clone. Colab clones it separately into `/content/MulTaBench-reference` and verifies tracked-file bytes and imported module origins.

Our notebooks, adapters, construction and dashboard are our code. The official model implementations are unchanged. Smoke parameters are shorter than the official evaluation; smoke success is not a passing Track 2 result.
