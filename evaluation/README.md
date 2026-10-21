# Screening adapter (development)

The code behind the development screens in
[notebook 05](../research/notebooks/development/05_benchmark_screen_colab.ipynb): it let us score a candidate CSV with
MulTaBench's own models and splits **before** registering anything in the benchmark, and compare dataset versions cheaply
(for example LightGBM + CatBoost on folds 0–2).

> The final results do **not** use this adapter. They come from the official `benchmark.py` at `c955466`, run unchanged in
> [notebook 07](../research/notebooks/07_vaers_official_run_colab.ipynb). On the same learners and folds, the official run
> reproduced the v11b screen exactly (Joint +0.0309, TAR +0.0292).

## How it works

- The notebook checks out MulTaBench's **paper version** (commit `3bb95079`) in a separate folder and installs its pinned
  `requirements.txt` in a Python 3.11 environment. Before any run it verifies every tracked file against the pinned Git
  blobs and checks that the imported `multabench` modules come from that checkout (`reference_proof.json`).
- [`colab_runner.py`](colab_runner.py) loads the CSV and its `schema.json`, selects the columns for one condition
  (Structured, Text, Joint frozen, Joint TAR) and calls MulTaBench's `evaluate_on_loaded_dataset`, which does the sampling,
  splitting, training, E5 encoding and scoring. Model and split code are not modified.
- Each learner × condition × fold runs in its own subprocess and writes `result.json`; completed jobs are skipped on resume
  only if data, schema, code and settings are identical.
- [`results.py`](results.py) collects the runs into `results.csv` and `summary.csv` (four condition means, Joint gain = Frozen −
  max(Structured, Text), TAR gain = TAR − Frozen, pass flags only for complete runs).

| Preset | Learners | Folds | TAR epochs | Runs |
|---|---|---|---|---:|
| `smoke` | LightGBM | 0 | 2, batch 32 | 4 |
| `diagnostic` | all five, or a subset | 0, or a list | official maximum 50, batch 256 | 4 per learner × fold |
| `full` | all five | 0–4 | official maximum 50, batch 256 | 100 |

The default input is a 488-row tennis fixture (`tennis/data/smoke/`) used to test the pipeline end to end; set `CSV_PATH` and
`SCHEMA_PATH` in the notebook to screen another dataset.

## Tests

```bash
python3 -m unittest evaluation.test_smoke_infrastructure -v   # 19 offline tests: CSV checks, result aggregation, notebook cells
python3 research/scripts/build_colab_smoke_notebook.py       # regenerate notebook 05
python3 research/scripts/preview_colab_smoke.py              # run its offline preview cells
```
