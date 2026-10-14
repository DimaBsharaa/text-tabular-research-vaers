# Track 2 evaluation infrastructure plan

**Investigated October 14, 2026. Status: source-verified plan; implementation and model execution have not started.**

We can evaluate a new local CSV with the official model classes without editing MulTaBench. The practical route is an adapter around `evaluate_on_loaded_dataset`, plus our own job orchestration and result collection. This requires careful feature selection, dependency setup, split auditing, and CUDA access. The current historical EDA notebook does not execute this pipeline.

The authoritative reproduction reference is [`paper_version`, commit `3bb95079a6146ed35d45187e98bcad5b063273a1`](https://github.com/alanarazi7/MulTaBench/tree/3bb95079a6146ed35d45187e98bcad5b063273a1). The associated [paper, Appendix A.2–A.3](https://arxiv.org/html/2605.10616v1#A1.SS2), specifies five learners, four conditions, five random repetitions, a 10,000-training-example cap, and AUROC/R². Do not substitute the newer `master` leaderboard: its split, data, metric, and logging conventions differ.

## 1. What was verified, and what remains untested

Selected official source files were read in a separate temporary audit directory, `/tmp/nlp-track2-audit/official/`. The 151 KB **TabSTAR 1.1.15 wheel was inspected as a ZIP archive, not installed**: this matters because the evaluator imports its split functions. No full MulTaBench checkout, fork, package installation, learner import/execution, encoder download, training, or fine-tuning was performed. No source was modified and nothing was pushed.

The following are source-level findings. They are not claims that a complete environment or custom dataset has already passed a runtime smoke test.

| Integration point | Verified behavior |
| --- | --- |
| [`benchmark.py`](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/benchmark.py) | CLI executes one model × registered dataset × condition × fold. No arbitrary `--csv` option. |
| [`evaluate_on_dataset`](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/baselines/benchmarks/evaluate.py) | Downloads/loads a registered dataset, applies the loader route, then calls the loaded-dataset evaluator. |
| `evaluate_on_loaded_dataset` in the same file | Accepts a `MultimodalDataset`, subsamples, creates the outer split, instantiates the official model class, fits it, scores it, and returns a summary dictionary. |
| [`MultimodalDataset`](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/datasets/curation.py) | Holds `x`, `y`, `task_type`, `dataset_id`, and optional `image_folder`. |
| [`TabularModel`](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/baselines/abstract_model.py) | Fits training-only preprocessing and the selected learner; computes validation where enabled. |
| [`text_embeddings.py`](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/baselines/preprocessing/text_embeddings.py) | Frozen or LoRA-adapted E5, with a separate training-fitted PCA for each text column. |
| [`logging.py`](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/utils/logging.py) | CLI requires Weights & Biases credentials; the lower-level evaluator returns a dictionary that our runner can save locally. |

**Important corrections to a naive adapter:** `multimodal_state` passed to `evaluate_on_loaded_dataset` is recorded in the returned summary; it does not filter the supplied dataframe. `target_override` is likewise recorded there, not applied to an already-loaded dataset. Our adapter must extract the target and select the correct predictor columns **before** calling this function. Merely changing those argument values while retaining all columns would produce mislabeled experiments.

## 2. Exact four-condition execution

For a text-only modality dataset, with a frozen schema identifying structured and text columns:

| Condition | Official CLI state | Columns supplied by local adapter | `tune_e5` |
| --- | --- | --- | --- |
| Structured Only | `no_text` | Structured features; no raw text columns | `False` |
| Text Only | `text_only` | Text columns only | `False` |
| Joint Frozen | `all` | Structured + text columns | `False` |
| Joint TAR | `ft` | Structured + text columns | `True` |

Use the same eligible rows, target, ordering, sampling budget, and fold IDs in all conditions. Missing text must not silently cause row deletion in one condition only. Audit identifiers, source URLs, provenance timestamps, and split markers belong outside `x` unless a field is explicitly justified as a predictor.

Verified encoding defaults are `intfloat/e5-small-v2`, normalized 384-dimensional embeddings, `passage: <column>: <value>` inputs, maximum length 512 tokens, and PCA with **30 components per text column**. Two text columns therefore usually add 60 numerical embedding features. These are not 30 components for the entire dataset. Long articles are truncated; any passage selection must be fixed without test-label feedback. [Encoding source](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/e5/e5_finetune.py).

### Commands verified against the CLI source

These are examples for an **existing registered dataset**, to run only after an approved separate checkout, environment setup, credentials, and model-download access. They were not executed here and do not load our proposed tennis CSV.

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark.py --model light --dataset_name BIN_TEXT_FAKE_JOB_POSTING --fold 0 --train_examples 10000 --multimodal_state no_text --e5_model e5-small
CUDA_VISIBLE_DEVICES=0 python benchmark.py --model light --dataset_name BIN_TEXT_FAKE_JOB_POSTING --fold 0 --train_examples 10000 --multimodal_state text_only --e5_model e5-small
CUDA_VISIBLE_DEVICES=0 python benchmark.py --model light --dataset_name BIN_TEXT_FAKE_JOB_POSTING --fold 0 --train_examples 10000 --multimodal_state all --e5_model e5-small
CUDA_VISIBLE_DEVICES=0 python benchmark.py --model light --dataset_name BIN_TEXT_FAKE_JOB_POSTING --fold 0 --train_examples 10000 --multimodal_state ft --e5_model e5-small
```

The CLI turns on E5 tuning automatically for `ft` on a text dataset. Replace `light` with `cat`, `tabm`, `tabpfnv2`, or `tabpfnv2p5`, and repeat folds `0`–`4`. Neither a loop nor an experiment scheduler is supplied by the above commands. **There is no root `sweep.py` in the inspected paper-version tree**; a newer `master` sweep script must not be presented as this version's interface.

## 3. Actual split and scoring protocol

### Outer repetitions

Five paper-curation folds are **five seeded random holdouts**, not five disjoint cross-validation test partitions. The inspected TabSTAR dependency has `SEED = 42`, so outer split seeds are `42 + fold`, or 42–46. Most learner initialization uses the fixed seed 42, not a different initialization seed for every fold.

[`evaluate.py`](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/baselines/benchmarks/evaluate.py) and [`sampling.py`](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/baselines/preprocessing/sampling.py) call `tabstar.preprocessing.splits` from the pinned [TabSTAR 1.1.15 distribution](https://pypi.org/project/tabstar/1.1.15/). Its nominal test size is 10% of rows, capped at 2,000; large datasets are first sampled toward 10,000 training rows plus the nominal test count. If the loaded sample exceeds the training budget, the outer splitter uses the excess as its nominal test size. For multiclass problems, rare singleton classes receive special treatment.

**Binary off-by-two behavior, verified in the dependency source:** the helper first reserves one test example of each class, then draws the requested `test_size` from the remaining data, and appends the two reserved examples. The realized binary test size is therefore the requested size plus two. The subsampler also uses this helper. For a sufficiently large, otherwise ordinary binary dataset with the default cap, the combined sequence produces 9,998 outer training rows and 2,000 test rows rather than exactly 10,000 and 2,000. For 5,000 binary rows without subsampling, the outer split is 4,498 training / 502 test. We should preserve this behavior for faithful reproduction and record actual row IDs and counts, not silently repair it.

A bounded, **non-training synthetic split probe** executed only the inspected split/subsampling function definitions with the existing numerical libraries. It confirmed both examples and disjoint train/test row IDs. The TabSTAR wheel was not installed, no model classes were instantiated, and no tennis dataset was generated by this check. It does not validate the complete environment or CSV adapter.

Although `evaluate.py` contains `FOLDS = 10`, it does not run a ten-fold loop. The paper and historical curation exports use **five** repetitions. Our complete preset should explicitly list `[0, 1, 2, 3, 4]`.

### Inner validation and TAR

LightGBM, CatBoost, and TabM reserve nominally 10% of outer training rows, capped at 1,000, for learner validation. Their `split_to_val` call does not receive the outer fold, so it uses default fold −1, hence seed **41**. Binary inner validation inherits the same extra-two behavior.

TabPFN 2.0 and 2.5 have `USE_VAL_SPLIT = False` and use the full outer training portion. For TAR, all learners additionally create an encoder train/validation split inside the data available to their preprocessor. Thus the three validation-using learners and the two TabPFN learners do **not** tune their encoders on identical row sets. PCA uses the preprocessor's training pool, never the outer test. [Base-model source](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/baselines/abstract_model.py), [TAR source](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/baselines/preprocessing/text_embeddings.py).

The CLI's `E5TrainArgs` settings are rank 16, last 3 transformer layers, learning rate `1e-4`, maximum 50 epochs, early-stopping patience 3, weight decay `0.001`, and batch size 256. LoRA alpha is 32 and dropout 0.1; the best encoder is selected by validation loss. The encoder is shared across text columns: each row/column text becomes a labeled training example. For regression, the auxiliary encoder target is discretized into 20 bins, while the final learner still predicts the continuous target. [Train-args source](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/finetune/train_args.py), [fine-tuning source](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/e5/e5_finetune.py).

**An adapter must pass `E5TrainArgs().to_dict()` explicitly for TAR.** The lower-level fine-tuning function has different fallback defaults—40 epochs, batch 32, weight decay 0.01. Omitting the CLI's configuration would silently change the experiment.

### Metrics and returned results

The official [baseline metrics](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/baselines/training/metrics.py) use:

- Binary classification: AUROC on probabilities.
- Multiclass: macro one-versus-rest AUROC, with a fallback per-class calculation; log loss is also recorded when calculable.
- Regression: R² as the primary score; MSE and RMSE as auxiliary metrics.

The evaluator returns `test_score`, `metrics_dict`, fold, actual `n_train`/`n_test`, runtime, encoder settings, train/inference timing and memory, and hardware. `n_train` refers to the outer training count, not the smaller learner-fitting or encoder-training pools. Its `git` field is obtained from the process working directory: an external runner launched in our repository could log our commit instead of the benchmark commit. Record **both** explicitly.

## 4. Required learner configurations

| Learner / CLI name | Verified wrapper configuration | Consequence |
| --- | --- | --- |
| [LightGBM / `light`](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/baselines/lgbm.py) | Default LGBM classifier/regressor, `verbose=-1`, seed 42; categorical encoding; validation set passed to fit | No early-stopping callback is supplied in this wrapper. Do not invent tuned settings. |
| [CatBoost / `cat`](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/baselines/catboost.py) | 2,000 iterations, early stopping 50, `od_pval=0.001`, seed 42; native categorical features and best validation model | `thread_count` is an unannotated class attribute, so it is not included by `vars(dataclass_instance)`; do not assume that the wrapper passes eight threads to CatBoost. |
| [TabM / `tabm`](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/baselines/tabm.py) | `pytabkit.TabM_D_Classifier/Regressor` defaults, device, seed 42, `n_threads` from TabSTAR; training median fill and categorical encoding | Inner validation is used. |
| [TabPFN 2.0 / `tabpfnv2`](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/baselines/tabpfnv2.py) | `create_default_for_version(ModelVersion.V2)`, seed 42, categorical indices, `ignore_pretraining_limits=True`; wrapper row limit 10,000 | No learner validation split. Package device choice is automatic: this factory call does not forward `self.device`. |
| TabPFN 2.5 / `tabpfnv2p5` in the same file | Same mechanism with `ModelVersion.V2_5`; wrapper limits 50,000 rows / warning threshold 2,000 columns | Standard comparison should still use the common 10,000 training budget. |

All five use the official text preprocessing and label encoding. Wine Review and Spotify Genres historically lack the two TabPFN evaluations because of their many classes. A binary tennis target avoids that particular obstacle, but checkpoint access and runtime compatibility remain untested.

## 5. Integrating an arbitrary CSV without editing the reference

**Recommended design, not yet executed:**

1. Maintain a local schema manifest declaring target, task, structured columns, text columns, excluded audit columns, dataset hash, and eligibility policy.
2. Read the CSV in our adapter; create matching `X`/`y` indices and stable audit row IDs. Restore declared numerical/categorical/date types. Remove only predefined invalid rows and audit-only fields.
3. Represent the local dataset with the official `MultimodalDataset` and `SupervisedTask`. A small local enum can provide a text-prefixed `.name` and `.value`; the inspected loaded evaluator uses this identity for metadata rather than downloading. This route still needs a runtime compatibility test—type annotations referring to the official ID union are not proof of custom-ID support everywhere.
4. Select the condition's features before constructing the dataset object. Joint Frozen and Joint TAR must receive the identical raw predictors.
5. Invoke the official learner class through `evaluate_on_loaded_dataset`, with explicit CLI-equivalent encoder settings and the requested fold.
6. Save the returned summary locally; catch errors in our job runner and mark them explicitly.

No Kaggle or Hugging Face dataset upload is necessary for this route. It bypasses registered-source downloading while retaining the official learners and evaluation function. It does not introduce a fake official dataset registry entry or monkey-patch the benchmark.

### Feature-detection obstacles

The official [semantic classifier](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/preprocessing/feat_types.py) calls a non-numerical column text when it has at least 100 distinct valid values, or a distinct-value ratio of at least 0.8. A small set of repeated player profiles can therefore become **categorical**, meaning E5/TAR will not actually process it. Column selection in our manifest does not override this downstream detection.

Log the actual detected text/categorical columns for every training pool and condition. Do not append row identifiers to force artificial uniqueness. Also reject entirely empty predictor columns: the inspected [image detector](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/baselines/preprocessing/feature_types.py) uses `all(...)` on nonempty values and can classify an empty series as image-like. PCA requires sufficient training samples and dimensionality. Inconsistent detection across conditions/folds must be surfaced, not hidden.

The returned summary does not include every fitted preprocessor attribute. Auditing actual text recognition and inner row counts may require a small companion driver using the same official split functions and public model `fit`/`score_all_metrics` methods; verify equivalence against the official evaluator before using it for final results. This is orchestration, not a reimplementation of learners.

## 6. Proposed repository layout and runner

Only this report is being created now. The following files are **proposed**, not existing runnable commands:

```text
NLP/
├── MulTaBench-reference/       # separate pinned checkout, only after approval
└── text-tabular-research-2/
    ├── evaluation/
    │   ├── load_local_csv.py   # schema, target, condition subsets
    │   ├── run.py             # subprocess job orchestration
    │   ├── collect.py         # validate, aggregate, compute criteria
    │   └── audit_splits.py    # row/time/article overlap checks
    ├── configs/
    │   ├── candidate.yaml
    │   └── presets.yaml
    ├── data/candidates/       # original and curated files with provenance
    ├── runs/<experiment_id>/  # JSON + stdout/stderr per logical run
    ├── results/              # long run CSV + learner summary CSV
    └── research/notebooks/
        ├── 02_candidate_eda.ipynb
        └── 03_curation_results.ipynb
```

The notebook should be the inspection/dashboard surface; independent subprocesses should perform the potentially long model runs. One process per run isolates GPU state and allows restart after failure. Schedule one TAR job at a time per GPU. The runner should validate the reference commit and clean working tree before/after execution, never write outputs into its source tree, and resume only exact matching experiment keys.

| Preset | Proposed scope | What it establishes |
| --- | --- | --- |
| Smoke | Non-training schema/feature/split checks first; after approval, small fixed subset, one fold, all five learners × four conditions, very short explicitly nonofficial TAR schedule | Integration, weights access, feature detection, dimensions and result capture; **no acceptance claim** |
| Diagnostic | One fold × all 20 combinations on a bounded development dataset, with official learner/encoder settings; optionally repeat a second development fold | Estimates bottlenecks, runtime and memory; still not the final curation verdict |
| Complete historical reproduction | Five folds × five learners × four conditions, fixed data/configuration and common cap | 100 logical evaluations, including 25 TAR fits in the unmodified flow |

Smoke/diagnostic/full jobs with different subsets or training schedules must have different keys and separate output folders. A shortened TAR fit must never be reused as a full result. For binary data, all five learners should be present; missing runs leave the dashboard **incomplete**, not a fabricated score or silent non-pass.

## 7. Results CSVs and notebook dashboard

### Long CSV: one row per model–condition–fold

```text
experiment_id,dataset_version,dataset_sha256,benchmark_commit,runner_commit,
model,condition,fold,outer_split_seed,learner_seed,protocol,preset,status,
metric,test_score,auc,r2,logloss,rmse,n_train,n_test,n_learner_fit,n_encoder_train,
train_row_hash,test_row_hash,learner_val_row_hash,encoder_val_row_hash,
structured_columns,text_columns,detected_text_columns,e5_model,e5_revision,
pca_components,tune_e5,lora_rank,lora_layers,learning_rate,batch_size,epochs,
runtime_s,train_time_s,inference_time_s,peak_ram_gb,peak_gpu_gb,hardware,
checkpoint_hash,config_hash,source_json,error_type,error_message
```

Store original per-run JSONs alongside the CSV. Additional audit fields not returned natively require explicit instrumentation; leave unavailable fields null rather than inventing values. Save row-level predictions only through a verified companion driver, since the loaded evaluator does not return them.

### Learner summary CSV

One row per learner and dataset/protocol, showing all four raw mean scores, fold standard deviations, fold counts, three-decimal scores used by legacy reproduction, both gains, criterion flags, and overall paired pass status. The main table is:

```text
Learner | Structured | Text | Joint Frozen | Joint TAR | Joint gain | TAR gain | Both pass?
```

Compute gains from **condition means**:

```text
joint_gain = mean(Joint Frozen) - max(mean(Structured), mean(Text))
tar_gain   = mean(Joint TAR) - mean(Joint Frozen)
accepted   = count(learner passes BOTH conditions) >= 3
```

Do not count three learners passing Joint Signal and a different three passing TAR as sufficient. Do not average a per-fold maximum instead of taking the maximum of condition means.

The official [historical acceptance code](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/leaderboard/analysis/pass_matrix.py) rounds means to three decimals, rounds subtraction to avoid floating-point tie artifacts, and accepts both gains **>=0.001**. The paper's equation says **>0.001**; the assignment describes strict improvement. Record separate flags for (a) legacy rounded reproduction, (b) the paper formula on unrounded means, and (c) assignment-style positive unrounded gains. The earlier EDA's strict test on rounded scores is a sensitivity view, not an exact replacement for the unrounded paper equation. Resolve which criterion governs submission before treating a boundary result as final.

Dashboard views should include the 5×4 score matrix, paired gain bars with the rule line, fold-level variability, pass/fail/N/A badges, a completion counter out of 100, detected-feature counts, timing/memory, and the leakage audit. Separate IID reproduction and temporal robustness results. Add a short interpretation after each plot, as in our historical notebook.

## 8. Dependencies, hardware and time

### Verified version pins and setup obstacles

The inspected [requirements](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/requirements.txt) include LightGBM 4.6.0, CatBoost 1.2.8, pytabkit 1.7.0, TabPFN 7.1.1, TabSTAR 1.1.15, torch 2.7.1, transformers 4.57.3, peft 0.18.0, scikit-learn 1.6.1, pandas 2.3.3, numpy 2.3.5, and wandb 0.23.0. It also includes many unrelated model/GPU packages and a Git dependency. This is separate from our Python 3.14 research environment.

[`init.sh`](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/init.sh) targets Python 3.11 and can install Homebrew/pyenv, change shell configuration or global Python selection, and upgrade tools. **Do not run it blindly.** After approval, create a separate Python 3.11 environment and determine compatible dependencies using the official pins. A reduced text-only dependency set is plausible but has not been installation-tested. The root CLI eagerly imports all baseline wrappers, so installing only five learner packages may not suffice for that entry point.

The CLI requires `WANDB_API_KEY` and `WANDB_ENTITY`; registered dataset downloads may require Kaggle credentials. A local adapter can log JSON without calling `wandb_run`, and can avoid Kaggle. E5/TabPFN weights still need download/access. TabPFN checkpoint gating/license acceptance should be checked explicitly. Keep credentials outside tracked files.

### Hardware

The paper's cost measurements used one **A100-SXM4 40 GB GPU and eight AMD EPYC CPU cores**. A Linux CUDA machine is the preferred reproduction environment. The TAR code asserts that `CUDA_VISIBLE_DEVICES` exists; it is not a verified drop-in training path on this Mac. A 24 GB GPU may be workable for small E5 but is an untested planning option, not a guarantee. Provision approximately 32 GB system RAM initially and record actual peaks.

`--memory 32G` is logged as a request; it does not itself allocate or enforce a RAM limit. Hardware logging records available devices/resources, not a guarantee that jobs run on identical hardware.

### Timing anchors, not promises

From [paper Appendix F.3, Table 11](https://arxiv.org/html/2605.10616v1#A6.SS3), the published **median text E5-small** costs per dataset/fold are:

| Learner | Frozen seconds | TAR seconds | TAR peak GPU GB |
| --- | ---: | ---: | ---: |
| LightGBM | 223 | 2,417 | 17.3 |
| CatBoost | 323 | 2,471 | 17.3 |
| TabM | 225 | 2,408 | 17.3 |
| TabPFN 2.0 | 253 | 2,430 | 18.6 |
| TabPFN 2.5 | 242 | 2,396 | 18.6 |

A rough uncached sequential budget for 25 TAR fits plus 50 frozen-text evaluations is **20.35 hours**, obtained as `5 × (sum(TAR medians) + 2 × sum(Frozen medians)) / 3600`. This extrapolates Joint Frozen medians to Text Only and sums medians across different datasets; it is **not a measured tennis runtime**. Add 25 structured fits, installation/downloads, diagnostic runs, failures, and GPU queue time. Actual text lengths, column counts, matched sample size, and early stopping can change the cost substantially. Measure a diagnostic run before committing a full schedule; reserve one to two compute days if an appropriate GPU is available.

### Safe caching

- Reuse downloaded immutable pretrained weights and frozen model instances. The paper code caches the vanilla E5 model in-process, not a persistent table of embeddings.
- Frozen **raw** embeddings could be cached by exact formatted input, column name, tokenizer/model revision, truncation and normalization settings. These are label-free, but PCA remains training-only. A persistent embedding cache is a proposed optimization requiring equivalence checks, not a native feature we have verified in this paper version.
- Resume completed logical runs using hashes of data, row splits, model/configuration, source/dependency revisions, and checkpoints.
- TAR checkpoints must be isolated by training/encoder-validation row IDs and labels, seed, text columns, model revision and settings. Never reuse across test boundaries or across different fitting pools. The three validation-using learners and the two TabPFNs use different pools, so blanket sharing across five learners is invalid.
- The unmodified TAR helper saves temporary Trainer checkpoints and deletes that directory on return. Cross-run fitted-encoder persistence/reuse is not implemented by the inspected evaluator. Do not budget unverified savings or silently change the flow to achieve them.

## 9. Leakage audit and tennis protocol decision

| Stage | Risk | Required control |
| --- | --- | --- |
| Target construction | Winner-first A/B slots reveal the label | Outcome-independent orientation from an unordered player pair, fixed before computing `player_a_wins`; remove winner/loser labels from predictors. |
| Match fields | Target-match score, duration, aces, statistics or eventual tournament progress enter `x` | Only pre-match information and strictly prior-match aggregates; retain outcome fields solely for labels/audits. |
| Text selection | Article published or updated after target match | Publication/version evidence strictly before cutoff; post-previous-match text is allowed only for a later target. Masking a result in a target-match recap does not make it valid. |
| History features | Future matches, full-season totals, current live ratings, target row in rolling calculations | Chronological as-of joins, explicit shifted histories and documented update times. |
| Article/profile reuse | Same passage or near-duplicate appears in training and evaluation | Audit passage hashes and connected match/article groups; evaluate temporal generalization separately. Reusing a legitimate known profile is not automatically illegal leakage, but random splits can reward memorization. |
| Learned preprocessing | Imputation, category mapping, scaling, vocabulary, PCA fitted on the whole dataset | Fit on the correct training pool, transform validation/test only; frozen pretrained encoding itself may be label-free cached. |
| Encoder adaptation | Test labels or test-selected checkpoints affect TAR | Separate training, encoder validation, and outer test; split-aware cache keys; no global fine-tuning before fold evaluation. |
| Dataset selection | Repeatedly improve the CSV based on final test results | Develop on designated development data; freeze eligibility, target, features, passage policy and evaluation rule before final runs. |
| Runner/results | Conditions use different rows, stale cache, missing results averaged away | Split hashes and run completeness checks; errors remain visible; never mix presets/protocols. |

**Temporal features and temporal evaluation are different safeguards.** Even if every row's predictors predate its match, the official IID evaluator may train on later match labels while testing earlier matches. It also has no article-group or entity-group splitting. That does not establish future-match forecasting.

For tennis, report (1) faithful historical IID curation as required by the course, and (2) a separately labeled forward-in-time robustness evaluation using the same official model classes, with externally fixed chronological outer splits. The latter cannot be obtained simply by passing dates or another fold ID to `evaluate_on_loaded_dataset`, because that function recreates random splits. A companion driver can call `model.fit` and the official scoring methods on externally supplied data without changing learner implementations. Its inner validation remains the official random validation unless a further protocol change is explicitly agreed and disclosed. Confirm the submission protocol with course staff before replacing IID curation or claiming strict chronological validation at every level.

## 10. Recommendation and approval boundary

Proceed to infrastructure implementation **after review**, with an isolated pinned reference, a Linux/CUDA environment, local CSV adapter, strict feature/split audits, and resumable JSON-first execution. First validate one known dataset and the custom-ID route; then run smoke and diagnostic presets. The full protocol is 100 logical evaluations, not just 20 once.

Known obstacles are the heavyweight/unverified installation, checkpoint access, repeated-text detection, actual binary split counts, CUDA-only assumptions in TAR, two different E5 default paths, externally supplied temporal splits, and the threshold discrepancy. None was hidden or resolved by editing the official code.

**Current stopping point:** this investigation is complete. No reference checkout, installation, dataset integration implementation, training, or push has been performed. The tennis choice remains open.
