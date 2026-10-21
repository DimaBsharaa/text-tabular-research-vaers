# Tennis: an earlier candidate dataset (dropped)

Before VAERS, we built and tested a tennis dataset. It did not work: the text added nothing over the match statistics,
and the sources could not be republished. This folder keeps the full investigation so it can be checked or reused.

## The idea

Match statistics — rankings, recent form, rest — describe a player from the outside. They miss what players say
about themselves: an injury, fatigue, confidence, how they handle a surface. Players talk about exactly this in their
press-conference interviews. **Question:** do a player's earlier interviews add information about who wins the next match,
beyond the statistics?

**Task:** binary, `player_a_wins`, one row per match. Which player is "A" is fixed by a hash of the unordered player pair,
so the label does not leak through the column order.

## Raw data

| Source | What we used | Link |
|---|---|---|
| Cornell tennis interviews (ConvoKit) | Post-match press-conference transcripts, 2007–2015 (originally from ASAP Sports) | [ConvoKit corpus](https://convokit.cornell.edu/documentation/tennis.html) · [original release](https://www.cs.cornell.edu/~liye/tennis.html) (Fu, Danescu-Niculescu-Mizil and Lee) |
| Tennis-Data | Dated match results and rankings (ATP and WTA) | [tennis-data.co.uk](https://www.tennis-data.co.uk/alldata.php) |
| Match Charting Project | Point-by-point serve-placement charts, snapshot of 11 Jun 2015 (CC BY-NC-SA 4.0) | [JeffSackmann/tennis_MatchChartingProject](https://github.com/JeffSackmann/tennis_MatchChartingProject/tree/dc55cebd77fffc5f418540c47f9febda500fc342) |

## Three constructions

| | A: earlier interviews | B: serve-placement descriptions | C: match context in words |
|---|---|---|---|
| Text | Each player's most recent interviews before the match | Template sentences from the charting data | The structured context written as a sentence |
| Matches | 4,052 provisional → **3,627** after date rules | 415 | 44,359 (415 used for a fair comparison with B) |
| Structured / text columns | 16 / 2 | 16 / 2 | 10 / 1 |

## Preprocessing (construction A)

1. Parse the interview corpus; link each interview to its match through dated Tennis-Data records and a player-name alias table (`cornell_match_join_audit.csv`, `player_alias_crosswalk.csv`).
2. Keep only interviews that existed **before** the target match. Publication times are not recorded, so we assume an interview is available three calendar days after its match and check how sensitive the row count is to that assumption (`a_v1_date_sensitivity.csv`).
3. Build the structured features from the same dated records, using only earlier matches: both players' rankings, number of earlier matches, form over the last 10, win rate on the surface and rest days, plus the match context (tour, surface, court, level, round, best-of).
4. Fit long interviews into E5's 512-token limit: six passage-selection policies were compared; `topical_answers` was frozen **before** any scores were computed (`a_v1_policy.json`).
5. Split chronologically: 2,162 train / 716 development / 749 reserved test. The reserved test was never scored.
6. Leakage audit: 45 checks with negative controls and counterfactual mutations, for example renaming the winner or shuffling the training text ([leakage report](reports/tennis_a_leakage_validation.md)).

## Results

CPU diagnostics (TF-IDF + logistic regression), AUROC on the development split. No E5 or TAR run was made, because the
cheap test already showed no joint gain.

| Construction | Structured only | Text only | Joint | Joint gain |
|---|---:|---:|---:|---:|
| A — selected policy (`topical_answers`) | 0.768 | 0.557 | 0.761 | −0.007 |
| A — best of six policies (`token_head`) | 0.768 | 0.604 | 0.766 | −0.002 |
| B (415 matches) | 0.813 | 0.603 | 0.799 | −0.014 |
| C (same 415 matches) | 0.833 | 0.458 | 0.821 | −0.012 |

All six text policies for A have a negative joint gain (−0.002 to −0.010). Source files:
`data/prototypes/a_v1_development_comparison.csv` and `data/prototypes/diagnostic_scores.csv`.

## Why we dropped it

1. **No joint signal.** The statistics are strong (AUROC 0.77–0.83) and the text is weak (0.46–0.60). Adding the text made every model slightly worse.
2. **Text weaker than the table.** In the benchmark study (notebook 01 in `research/`), the datasets that pass have text at least as informative as the table. Here text-only trails structured-only by about 0.17.
3. **Timing could not be verified.** No interview has a verified publication time, so "before the match" rests on an assumption.
4. **Data use.** The Tennis-Data site limits use to private individuals and excludes data-training products, and the interview transcripts have no redistribution licence. A public dataset could not include them.

In B and C the "text" was also too repetitive: on the training folds, the benchmark's text-detection rule did not even treat it as free text.

## Reproduce

Raw sources and the text-bearing CSVs are kept **outside Git**, in a sibling folder `../tennis-research-cache/`
(override with `TENNIS_RESEARCH_CACHE`). Run from the repository root with `research/requirements.txt` plus
scikit-learn, scipy, openpyxl and psutil.

**Raw data → candidate CSVs**

```bash
python3 -m tennis.scripts.fetch          # bounded download (~104 MB), hash-checked against data/prototypes/source_manifest.json
python3 -m tennis.scripts.audit          # interview corpus audit
python3 -m tennis.scripts.prototypes     # constructions A, B, C (v0)
python3 -m tennis.scripts.additional     # extra coverage checks
python3 -m tennis.scripts.a_study        # construction A v1: date rules, six text policies, 3,627 rows
```

The A v1 CSVs are written to `../tennis-research-cache/derived/candidate_A_v1/` (`features_<policy>.csv`, `metadata.csv`, `schema.json`).

**Candidate CSVs → results**

```bash
python3 -m tennis.scripts.diagnostics          # B and C CPU diagnostics
python3 -m tennis.scripts.compare              # A/B/C comparison table
python3 -m tennis.scripts.a_context_blocks     # A: answer-block extension
python3 -m tennis.scripts.a_leakage            # A: leakage audit
python3 -m tennis.scripts.a_additional_checks
python3 -m tennis.scripts.a_review_outputs     # reports and notebook 03
python3 -m tennis.scripts.notebooks            # notebooks 01 and 02
python3 -m unittest tennis.scripts.test_research tennis.scripts.test_interview_followup tennis.scripts.test_a_validation -v
```

## Contents

| Path | Contents |
|---|---|
| [`notebooks/01_tennis_interviews_eda.ipynb`](notebooks/01_tennis_interviews_eda.ipynb) | What is inside the interview corpus: hierarchy, missingness, lengths, links to matches |
| [`notebooks/02_tennis_dataset_prototypes.ipynb`](notebooks/02_tennis_dataset_prototypes.ipynb) | Constructions A/B/C, coverage and CPU diagnostics |
| [`notebooks/03_tennis_interview_validation.ipynb`](notebooks/03_tennis_interview_validation.ipynb) | A v1: dates, token budgets, six text policies, examples, leakage audit |
| [`reports/`](reports/) | Feasibility survey, corpus EDA, A/B/C comparison, pilot results, follow-up, text and leakage reports |
| [`data/prototypes/`](data/prototypes/README.md) | Counts, metadata, predictions and audit evidence (no interview text) |
| [`data/smoke/`](data/smoke/README.md) | A 488-row fixture used to test the Colab screening runner end to end |
| `scripts/` | All tennis code and its 33 tests; `tennis_temporal_pilot.py` is an optional checks-only adapter |

Notes: the reports and notebooks were written before this folder existed, so their text uses the old notebook numbers
(03 → 01, 04 → 02, 05 → 03; "notebook 02" there is the Colab runner, now `research/notebooks/development/05_benchmark_screen_colab.ipynb`).
The archived hash manifests in `data/prototypes/` record the original file paths.
