# Official MulTaBench results — VAERS v11b

Produced by [notebook 07](../../research/notebooks/07_vaers_official_run_colab.ipynb) on Colab (NVIDIA A100-SXM4-80GB),
20 Oct 2026, with the official MulTaBench `benchmark.py` at commit
[`c95546638be9`](https://github.com/alanarazi7/MulTaBench/tree/c95546638be9bac472a2ee38afa7171060ab80bc)
(`master` as of 26 Jul 2026), unchanged. Dataset: `research/data/vaers_v11b/vaers_v11b.csv`, sha256 `8e59f09b…e16690`.

| File | Contents |
|---|---|
| `results.csv` | One row per attempt: learner, condition, fold, status, test AUROC, runtime, GPU, MulTaBench commit, finish time. **100 `ok` rows** (5 learners × 4 conditions × 5 folds). |
| `per_fold.csv` | The 100 scores as a learner × fold table, with the Joint and TAR gain of each fold |
| `summary.csv` | Mean over the 5 folds per learner and condition, both gains and pass flags (the table in the project README) |
| `registration.diff` | The only change made to the MulTaBench checkout: the dataset enum line, the local file mapping and the curation recipe |

**Conditions:** `no_text` = structured only · `text_only` = text only · `all` = joint with frozen E5-small-v2 embeddings (PCA to 30 per text column) · `ft` = joint after fine-tuning E5 (TAR).

**The 10 `failed` rows** are setup failures from the first session, all re-run successfully afterwards:
the first 5 attempts stopped within seconds on Colab configuration (the inline plotting backend, fixed with
`MPLBACKEND=Agg`, and the W&B account name), and the 5 fold-0 `ft` attempts stopped because the fine-tuning code requires
`CUDA_VISIBLE_DEVICES` to be set (fixed by setting it to `0`). None of these fixes touches benchmark code. Summaries use only `ok` rows.

**A note on the recipe text.** The docstring inside the registered recipe says "report years 1990-2024", describing VAERS as a
whole. The dataset itself uses the 2019, 2023 and 2024 yearly files. The docstring does not affect the run.

Recompute the summary:

```python
import pandas as pd
r = pd.read_csv("results/official_c955466/results.csv").query("status == 'ok'")
m = r.groupby(["model", "state"]).test_score.mean().unstack()
m["joint_gain"] = m["all"] - m[["no_text", "text_only"]].max(axis=1)
m["tar_gain"] = m["ft"] - m["all"]
print(m.round(4))
```
