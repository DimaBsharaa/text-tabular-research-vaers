# Tennis prototype pilot results

Executed locally October 15, 2026. **These are CPU TF-IDF/logistic-regression diagnostics, not MulTaBench results.** No official model source, E5 checkpoint or TAR training was used locally. LightGBM is unavailable in the existing local Python environment; rather than install a training stack, we used one accessible learner consistently. The working notebook 02 and its official A100 smoke test remain unchanged.

**Later source/split audit:** [the interview follow-up](tennis_interview_followup.md) records a current Tennis-Data training-use restriction and additional A evaluation risks. Earlier diagnostic results remain as recorded; no additional models were fitted during that follow-up. Do not infer source-use clearance from a completed engineering test.

**New, separate A v1 study:** [the text comparison](tennis_a_text_preprocessing.md) now records 13 completed local conditions on 3,627 refined A rows under a declared conservative calendar rule. It does not modify the B/C results below or constitute an E5/TAR benchmark. The reserved-test rows remain unscored; see the [45-check leakage validation](tennis_a_leakage_validation.md).

## Fixed procedure

`diagnostics.py` fits logistic regression with `C=1`, `solver=liblinear`, maximum 500 iterations and seed 42. Numerical median imputation/scaling, categorical one-hot encoding and per-text-column word/bigram TF-IDF are fitted **only on training rows**. Each text field has at most 4,000 TF-IDF features. No token vocabulary, scaler, imputer or feature selector sees test data during fitting.

The split is chronological: sort by actual reported match day and stable match ID, choose the 80% row-position day, then put the **entire cutoff day in test**. No date appears on both sides. Current-match outcomes/stats, betting odds, source IDs and winner/loser roles are absent from model inputs. Histories are computed using strictly earlier dates. Final qualification flags are not computed. There is no hyperparameter search, no held-out-test-driven data revision and no improvised TAR.

## Comparable B/C results: same 415 target matches

Both use **328 training / 87 test** matches. Latest training day: **2015-10-18**. First test day/cutoff: **2015-10-20**. There are 46 players shared between train and test; this evaluates later matches involving previously observed players, not unseen-player generalization. Shared histories/text can correlate observations and reduce effective sample size.

| Condition | B: tactical templates | C: context verbalization |
| --- | ---: | ---: |
| Structured only | **0.8128** | **0.8330** |
| Text, TF-IDF | 0.6026 | 0.4584 |
| Joint, TF-IDF | 0.7990 | 0.8208 |
| Joint gain over stronger unimodal score | **−0.0138** | **−0.0122** |
| Extra raw-numeric tactical features, no text | 0.7980 | N/A |
| All original context retained as structured features | N/A | 0.8128 |
| Joint with training text shuffled | 0.8012 | 0.8256 |
| Joint TAR | **Not run** | **Not run** |

B's structured condition has general-strength numbers **and** context. C's structured condition has only the numbers because context is allocated to text. C's full-context control exactly reproduces B's structured score, as expected on identical inputs and splits. This is a useful implementation check.

The B numeric-tactics control adds charted-match count, coded-serve count and wide/T fractions directly to the general table; middle fraction is inferable. It supplies at least the information in the rounded text. Its similarity to the verbalized joint score does not establish a uniquely linguistic advantage. Training-text shuffling is one fixed negative control; small score differences are not a significance test or a tuning target.

## Separate C pilot: latest 1,500 eligible matches

This deliberately bounded cohort uses **1,190 training / 310 test** rows, with a **2015-10-16** test cutoff and 170 shared players. It is a different population from the 415-match comparison.

| Condition | AUROC |
| --- | ---: |
| Structured only | **0.7424** |
| Text, TF-IDF | 0.4701 |
| Joint, TF-IDF | 0.7312 |
| Joint gain | **−0.0112** |
| Original context kept structured | 0.7266 |
| Joint with training text shuffled | 0.7323 |
| Joint TAR | **Not run** |

The negative gains are point estimates on small holdouts. They do not prove the constructions can never work, and text AUROC below 0.5 is not a justification to flip predictions after seeing the test. These results provide no positive evidence for committing to the current B/C designs.

## Why A has no model score

A has **4,052 real, provisional calendar-safe rows**, not 4,052 fully certified forecasting observations. Neither release supplies historical first-publication timestamps or a transcript license granting our public dataset redistribution. The metadata/source-content checks establish date consistency and some transcript identity, while leaving historical online availability unresolved. We therefore did **not** run A models under an unlabeled availability assumption. Its Structured/Text/Joint/TAR scores are **unavailable**, not zero and not inferred from B/C. The 104-match three-way intersection is recorded for a possible later comparison if that gate is resolved.

## Important official-encoder diagnostic

The pinned MulTaBench detector treats a nonnumeric nonempty string column as text if it has at least **100 unique values** or uniqueness ratio at least **0.8**. B and C pass this check on their full constructed populations, but **fail it on all measured chronological fitting pools**. A column named `player_a_text` is not sufficient.

Our TF-IDF diagnostic explicitly encodes these columns as text and is labeled accordingly. It does not pretend the unmodified official pipeline would necessarily do the same. The prepared official-class chronological worker checks actual inner-fitting-pool detection and **refuses to run** if it does not agree with the declared text schema. We do not patch the semantic detector or add IDs to satisfy it.

## Runtime, reproducibility and prepared GPU work

Exact per-condition runtime and resident memory after fitting are in [diagnostic_scores.csv](../data/prototypes/diagnostic_scores.csv). Resident memory is a process snapshot, not peak memory or a benchmark-wide resource estimate. Predictions, targets, cutoffs and split IDs are saved, so scores can be recomputed. The 415-match conditions share identical train/test IDs. Nine research sanity tests check temporal boundaries, outcome-independent orientation, rank mapping, field separation, matching controls and manual-review coverage. No expensive model installation occurred.

`tennis/scripts/tennis_temporal_pilot.py` is a separate optional Colab worker using the **unchanged official LightGBM, preprocessing, E5/TAR and metric classes**. It reuses the existing source verification and environment helpers. Its outer split is chronological, so it is explicitly a diagnostic departure from official random holdouts; inner validation remains the official implementation. If later authorized and viable, `--run` fits four conditions with two TAR epochs/batch 32 and saves actual returned metrics. Without `--run`, it performs checks only. It refuses A's current timing manifest and B/C's current training-pool text mismatch. No GPU experiment was launched and there is no reason to pay for these known-blocked runs now.

Notebook 04 contains the preparation steps and commands for the existing A100 environment. Do not overwrite notebook 02 or run a full five-model sweep yet. The next useful expenditure is resolving A's source gates, not increasing TAR epochs on the current templates.

## Research decision

The infrastructure is operational, as established by the earlier A100 smoke test. This investigation instead finds **a promising real-language source with unresolved release/timing gates (A)**, and **two currently unconvincing generated-language constructions (B/C)**. No dataset has passed official curation, no frozen-E5/TAR score is claimed here, and no result has been pushed to GitHub.
