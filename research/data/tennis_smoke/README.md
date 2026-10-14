# Bounded Kaggle tennis smoke fixture

This directory is a pipeline test fixture, not the selected Track 2 dataset.

- `kaggle_v1219_prefix.csv`: 488 complete source rows from bytes 0–65,535 of `atp_tennis.csv`, dataset version 1219, retrieved during the October 14, 2026 feasibility audit. The final incomplete line was discarded. No full dataset download was used.
- `tennis_smoke.csv`: 488 matches, 13 structured features, two temporary text columns, a binary target and six audit columns. The reported date range is January 3–February 21, 2000. A wins 237 matches, B wins 251.
- `schema.json`: exact hashes, predictor/audit separation, source, sampling, license declaration and construction limitations.

Source and attribution: [dissfya, ATP Tennis 2000–2026 Daily update](https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull). The publisher declares **CC0: Public Domain** in the inspected Kaggle metadata. This records that declaration; it does not resolve upstream ownership for a final dataset. No copyrighted article corpus is included.

The raw file retains `Winner`, `Score`, and odds for audit/reproduction. **They are never loaded as predictors by the evaluation adapter.** The curated CSV contains only `player_a_wins` as the target; its audit identifiers/dates are also excluded from `X`.

A/B orientation is a fixed hash of date, tournament, round and the unordered abbreviated player pair. Neither winner nor score enters the hash. Every paired field follows the orientation. Histories use only rows with a strictly earlier reported date, updating after each whole day. Unknown historical rates use the explicit neutral value 0.5, alongside a zero-history count. Rankings ≤0 become missing and are handled by the official learner preprocessing.

Text is **programmatically templated summaries of previous results**, with the last three previous matches and earlier surface counts. It is not expert writing, tactical interpretation, independently sourced contextual information, or evidence of complementary text signal. Early rows explicitly state that no earlier history exists in this limited archive. Missing-history statements are not fabricated injury/playing-style claims.

The small prefix is not representative, and its dates/ranking availability have not been independently validated. Do not use this fixture for research claims or complete official qualification. The Colab runner rejects full evaluation on a manifest with `purpose: smoke_fixture`.

Rebuild from the checked-in small source slice:

```bash
python3 research/scripts/prepare_tennis_smoke.py
python3 -m unittest evaluation.test_smoke_infrastructure -v
```
