# Colab execution with the unchanged official evaluator

Open [02_tennis_official_benchmark_colab.ipynb](../research/notebooks/02_tennis_official_benchmark_colab.ipynb) in Colab:

[Open in Colab](https://colab.research.google.com/github/DimaBsharaa/text-tabular-research-vaers/blob/main/research/notebooks/02_tennis_official_benchmark_colab.ipynb)

Select **Runtime → Change runtime type → A100 GPU**, high RAM if offered, then run all cells. Authorize Drive mounting when prompted if you retain the default `SAVE_TO_DRIVE=True`. Hardware is subject to Colab availability. EDA and the tabular LightGBM/CatBoost fits use CPU; E5 encoding/fine-tuning and the neural learners use GPU through their official wrappers.

`AUTO_STOP_RUNTIME=True` now ends the Colab session in the final cell, after saving a ZIP in the Drive results folder and flushing/unmounting Drive. Google's `runtime.unassign()` releases the VM; a disconnect is expected. Download the saved ZIP from Drive afterwards. Set this option to `False` to keep the session for debugging and use the browser ZIP download. Automatic shutdown requires Drive saving and does not run if an earlier cell raises and prevents Run all from reaching the final cell.

Commands explicitly stream stdout and stderr into the notebook. Preflight output is saved to `preflight.log`, and Python failures also write `preflight_error.json`, before training begins. A `CalledProcessError` alone reports only that the child failed; use the worker traceback above it or these files to diagnose the cause. Existing Colab project checkouts are refreshed with `git pull --ff-only`; the official reference remains unchanged. The existing benchmark environment can be reused.

Benchmark subprocesses explicitly use `MPLBACKEND=Agg`, set before any official/Matplotlib imports. Colab exports `module://matplotlib_inline.backend_inline`, which is unavailable in the isolated environment and caused the observed `skrub` → Matplotlib preflight import error. The worker overrides that inherited setting even when launched by an older notebook copy. Current notebook calls also pass a separate worker environment dictionary, preserving inline plots in the notebook itself. No official source or package pins change. This follows [Matplotlib's documented per-process backend selection](https://matplotlib.org/stable/users/explain/figure/backends.html).

The notebook uses a separate reference checkout in the Colab VM, pinned to `paper_version` commit `3bb95079a6146ed35d45187e98bcad5b063273a1`, and an external Python 3.11 environment with the reference `requirements.txt` unchanged. It does not execute the historical source snapshots in our repository. The complete reference is neither copied into nor modified inside our project.

Notebook section **1a** explains which parts are official and which are ours, links the official source files, and verifies every tracked file's bytes against the pinned Git blobs. It checks the remote origin and commit too. The preflight worker then verifies that the actually imported `multabench` modules came from those checked files, showing their paths in the notebook. `reference_proof.json` stores this evidence in the exported results. These checks stop on a source/location mismatch; source identity alone does not prove a successful run or full paper-equivalent experiment settings.

**The whole notebook is not a 1:1 official artifact.** Our CSV integration, job scheduling, result display and local persistence surround their unchanged evaluation/model code. Smoke mode deliberately changes TAR parameters. Also, current upstream `master` has newer `sweep.py` and `collect_results.py` tools; the [official README](https://github.com/alanarazi7/MulTaBench) directs paper reproduction to `paper_version` / `official-paper-version`. Our pinned paper-version route should not be confused with that newer leaderboard pipeline.

The official [`benchmark.py`](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/benchmark.py) runs one registered dataset/model/condition/fold and logs to Weights & Biases. It has no arbitrary-CSV option or complete four-condition notebook dashboard. Our `colab_runner.py` supplies a `MultimodalDataset` to the official [`evaluate_on_loaded_dataset`](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/baselines/benchmarks/evaluate.py), which performs the actual sampling, splitting, training, encoding and scoring. No W&B credentials are needed for this returned-dictionary route.

| Preset | Learners | Holdouts | TAR | Runs |
| --- | --- | --- | --- | ---: |
| `smoke` (default) | LightGBM | fold 0 | 2 epochs, batch 32 | 4 |
| `diagnostic` | all five | fold 0 | official maximum 50 epochs, batch 256 | 20 |
| `full` | all five | folds 0–4 | official maximum 50 epochs, batch 256 | 100 |

All use E5-small-v2 and 30 PCA components per detected text column. Smoke is intentionally a shortened runtime check with changed *parameters*, not changed implementation. Diagnostic is a one-holdout measurement, not official qualification. Full is blocked for the supplied fixture and requires a genuinely curated candidate manifest. The current adapter supports binary targets and nonempty text per row. For TabPFN, checkpoint terms/access may require a permitted Hugging Face token in Colab Secrets named `HF_TOKEN`; it is never printed or saved.

The manifest selects columns **before** the call: the lower-level evaluator does not apply `multimodal_state` to an already-loaded dataframe. Target/provenance never enter the predictors. Before fitting, we reproduce splits with the same official helpers to audit row IDs and check that actual training-pool text detection agrees with the manifest. The official evaluator then repeats those helpers for its actual fit; model code and split behavior are not patched. All four conditions use the same CSV, target/order and outer fold. This preserves the historical binary extra-two splitting behavior documented in the evaluation report.

Each job has its own subprocess and outputs, so its GPU memory is released before the next. A completed `result.json` is written atomically. Resume requires identical data, manifest, adapter, reference, dependencies and job parameters. A changed run cannot silently reuse an incompatible result. The reference tree is checked before and after; outputs and Python environments live outside it.

Outputs persist under `MyDrive/text-tabular-research-2/benchmark-results/<plan-id>` by default:

- `results.csv`: learner/condition/fold, AUROC, actual train/test sizes, timing, GPU/memory, status/error, hashes and raw-result location.
- `summary.csv`: five learner rows, four means and standard deviations, Joint Signal and TAR gains, completeness and three qualification definitions.
- Per job: `result.json` with the unchanged official summary, `config.json`, `splits.json`, `pip_freeze.txt`, `reference_proof.json`.
- Per requested job: console `.log`; plan and process-error files cover failures before a model result can be returned.
- The final cell downloads a ZIP containing these artifacts. Save a copy of the notebook to Drive to retain displayed outputs too.

The dashboard uses mean Frozen − max(mean Structured, mean Text), and mean TAR − mean Frozen. It displays scores on a 0–1 AUROC scale, not percent. Missing conditions remain missing, never zero; incompatible experiments and duplicate successful folds are rejected. Qualification flags are blank for this toy fixture, all smoke/diagnostic runs and incomplete learners. For a complete real candidate, `assignment_pass` uses both gains >0, `paper_pass` uses both >0.001 unrounded, and `legacy_pass` uses rounded scores and ≥0.001 as in the historical curation. Confirm the course's preferred definition before the final submission. Three learners must meet both criteria individually.

**Verification status:** offline CSV construction, preview execution, leakage-invariance and result-aggregation tests passed (19 tests, including isolated Matplotlib backend recovery, bootstrap/worker error logging, source-byte/import-origin checks, and mocked export-before-shutdown checks). A bounded probe of the inspected official split function definitions, without installing/importing model packages, gave fold-0 counts of 438 outer training / 50 test, 393 learner fitting / 45 validation, and 352 encoder training / 41 validation. Both fitting text columns have over 100 distinct values (298/292), and all label splits contain both classes. The adapter call's argument names were checked against the official evaluator signature. No benchmark environment was installed locally. The user subsequently completed all four LightGBM conditions in Colab, including frozen E5 and shortened TAR, on 438 training / 50 held-out fixture rows. The preserved [observed run](../research/data/tennis_smoke/observed_run/README.md) contains the exact scores and 470-file reference proof. Structured-only AUROC was 0.669872; Joint Frozen was 0.487179 and Joint TAR 0.448718, so this is engineering success without a modality gain. The other four learners, full TAR settings and official Cornell A evaluation remain unrun. Separately, the local tennis research now includes CPU TF-IDF diagnostics; those are not official benchmark results.

Offline checks and notebook regeneration:

```bash
python3 -m unittest evaluation.test_smoke_infrastructure -v
python3 research/scripts/build_colab_smoke_notebook.py
python3 research/scripts/preview_colab_smoke.py
```


## Cornell A research adapter

[tennis_temporal_pilot.py](tennis_temporal_pilot.py) is a separate optional adapter for the private Cornell candidate. It validates schema/CSV/audit hashes, source-before-target rules and chronological partitions, accepts an explicitly declared calendar-delay assumption, and defaults to checks only. Its default evaluation partition is development; reserved-test use requires an explicit choice. No official E5/TAR candidate run has occurred. It does not change notebook 02 or the official learner implementations. The reference still uses random inner validation; the [leakage report](../research/reports/tennis_a_leakage_validation.md) records that unrun protocol limitation.
