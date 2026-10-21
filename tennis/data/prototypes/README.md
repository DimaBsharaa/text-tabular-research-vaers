# Tennis prototype research artifacts

**Latest A study:** read [notebook 05](../../notebooks/03_tennis_interview_validation.ipynb), [text comparison](../../reports/tennis_a_text_preprocessing.md) and [leakage validation](../../reports/tennis_a_leakage_validation.md). The original 4,052-row files remain unchanged. Refined A v1 has 3,627 rows under an explicit three-calendar-day assumption, with six text variants and 13 completed local TF-IDF conditions; this is not an official E5/TAR result.

Open the executed [Cornell audit](../../notebooks/01_tennis_interviews_eda.ipynb) and [prototype/diagnostic notebook](../../notebooks/02_tennis_dataset_prototypes.ipynb). Reports explain limitations and the recommendation. No final dataset is chosen.

This directory holds **counts, metadata, predictions and our review annotations**, not the raw interview corpus. Full source downloads and text-bearing candidate CSVs are in the external sibling folder `tennis-research-cache/`, configurable with `TENNIS_RESEARCH_CACHE`.

- `cornell_audit.json` and `cornell_by_*.csv`: hierarchy counts, missingness, exact duplication and distributions from actual native JSON.
- `cornell_interview_metadata.csv`: one row per interview, with source fields and length/hash/lexical indicators. Lexical flags are not validated substantive-content labels.
- `manual_review_sample.csv` / `manual_content_review.csv`: 36 purposively sampled 70–350-word answer sets and our subjective review notes. No raw answers; not a representative prevalence sample.
- `cornell_match_join_audit.csv` / `player_alias_crosswalk.csv`: deterministic name candidates, dated match links, source file/row and explicit exclusions. A recoverable-link rate is not independently verified match accuracy.
- `variant_*_sample_metadata.csv`: up to 50 real target rows with identity/provenance, without transcript text.
- `variant_statistics.json`, `join_summary.json`, `additional_audit.json`: full bounded coverage enumeration and remaining gates. A's 4,052 rows are provisional; strict publication verification is zero.
- `variant_comparison.csv`: flat source/size/timing/score/recommendation comparison. Its C scores refer to the separate 1,500-row pilot; the fair B/C 415-match comparison remains in `diagnostic_scores.csv`.
- `publisher_spot_checks.json`: two positive source-content checks and a deliberately wrong same-day semifinal check. This does not establish original upload timestamps.
- `diagnostic_scores.csv`, `diagnostic_*_predictions.csv`, `diagnostic_*_split.json`: actual CPU TF-IDF/logistic scores, out-of-sample predictions and chronological split IDs. No E5 or TAR results are present.
- `diagnostic_status.json`: explicit untested/blocked cases and the calendar-candidate three-way intersection.
- `source_manifest.json`: exact URLs, sizes and hashes of the bounded downloaded sources.
- `interview_followup_audit.json` / `interview_publisher_validation.csv`: nine publisher content corroborations, a wrong same-day negative control, unresolved historical publication and current source-use findings. Read the [follow-up report](../../reports/tennis_interview_followup.md).
- `interview_lag_sensitivity.csv` / `interview_split_risks.csv` / `interview_split_reference_examples.csv`: measured delay coverage and chronological/random source-reference audits, without fitting any learner.
- `interview_alternative_source_coverage.csv` / `interview_alternative_source_sample_metadata.csv`: a bounded 2015 Sackmann archive coverage probe; scenario cutoffs are not verified forecasting dates and no replacement dataset was selected.
- `interview_followup_sources.json`: exact pinned mirror URLs, current policy/source checks, response statuses, body sizes and hashes. Raw pages and the two small year files remain in the external cache.
- `interview_candidate_review.json` / `interview_candidate_split_plan.csv`: a private review export of the existing 4,052 A rows; 2,430 training / 809 development / 813 reserved-test rows. This v0 export predates the later v1 CPU diagnostics.
- `interview_candidate_clipping.csv` / `interview_candidate_clipping_summary.csv`: per-document word-length/cue metadata and **training-only** summaries of cues lost by clipping. Keyword flags are not verified condition/tactic annotations.
- `interview_candidate_missingness.csv`: missingness retained in the review CSV; no globally fitted imputer or scaler. Full inputs, audit sidecar, splits, schema and executed workbench stay in external `derived/candidate_A_v0_review/`.

The private `derived/` cache contains `variant_A.csv`, `variant_B.csv`, `variant_C.csv`, their schema manifests, 50-row samples, full reconstructed answers and raw match records. The original A schema records unverified historical publication; the v1 schema permits only the explicitly declared calendar-delay assumption, without claiming verification. No public transcript redistribution right has been established. MCP data requires [CC BY-NC-SA 4.0 attribution](https://github.com/JeffSackmann/tennis_MatchChartingProject/blob/dc55cebd77fffc5f418540c47f9febda500fc342/README.md). The [Cornell project](https://www.cs.cornell.edu/~liye/tennis.html) credits Fu, Danescu-Niculescu-Mizil and Lee; underlying transcripts are from ASAP Sports and match metadata from Tennis-Data. Public accessibility is not represented as a redistribution grant.

Reproduce from the repository root using the existing analysis packages plus scikit-learn, scipy, openpyxl and psutil:

```bash
python3 -m tennis.scripts.fetch
python3 -m tennis.scripts.audit
python3 -m tennis.scripts.prototypes
python3 -m tennis.scripts.additional
python3 -m tennis.scripts.diagnostics
python3 -m tennis.scripts.compare
python3 -m tennis.scripts.notebooks
python3 -m unittest tennis.scripts.test_research -v
```

The fetch step is bounded at roughly 104 MB of source downloads and validates cached hashes against the recorded manifest. It reads the current historical-file links from the Tennis-Data index rather than assuming obsolete CSV URLs. The 97 KB xlrd 2.0.2 wheel is imported as a ZIP for old `.xls` files; no global package installation occurs. Extracted parsed SpaCy data, model weights and official benchmark clones are not needed. Model/environment versions are recorded in `diagnostic_environment.json`.

The optional `tennis/scripts/tennis_temporal_pilot.py` reuses official source checks and learner classes in the existing Colab environment. It is checks-only unless `--run` is explicitly provided. The current version accepts the explicitly documented conditional calendar rule and rejects inconsistent source dates or text-detection failures. Do not run it merely to work around a failed gate. Notebook 02 is unchanged. GitHub publication contains code and metadata evidence; transcript-bearing CSVs remain in the external cache.

**October 15 follow-up:** the current Tennis-Data homepage explicitly restricts certain data-training uses. Do not treat the earlier fetch/diagnostic commands as source-use clearance. No more Tennis-Data match files or models were requested in the follow-up. The original Sackmann ATP/WTA GitHub endpoints now return 404; the small mirror probe is revision-pinned and preserves its CC BY-NC-SA notice. See the follow-up report for this earlier source audit; notebook 05 now presents the newer v1 study.

With the existing local cache, reproduce the new analysis and checks separately:

```bash
python3 -m tennis.scripts.interview_followup
python3 -m tennis.scripts.followup_notebook
python3 -m unittest tennis.scripts.test_interview_followup -v
```

For the current local-only candidate review, using the existing cache:

```bash
python3 -m tennis.scripts.prepare_interview_review
python3 -m tennis.scripts.notebooks
```

The first command writes the text-bearing CSV and notebook outside Git, without fetching data or training. The second refreshes the metadata-only review section in notebook 04. No outreach is part of this task; the earlier clarification drafts remain unsent reference material.

## A v1 evidence and reproduction

- `a_v1_policy.json`: original five-policy/settings freeze with selector and temporal-source hashes, created before the diagnostic scores. The whole-answer-block extension has its own frozen policy.
- `a_v1_date_sensitivity.csv` / `a_v1_next_match_*.csv`: actual conservative-delay and strict-next-recorded-match coverage, without treating one-player links as two-sided rows.
- `a_v1_text_policy_metrics.csv`, `a_v1_all_text_policies.csv` and `a_v1_training_review_notes.csv`: exact tokenizer measurements and training-only lexical/manual diagnostics. Dictionary cues are not condition annotations.
- `a_v1_leakage_checks.csv` / `.json`, `a_v1_leakage_summary.json`, `a_v1_mutation_results.json`: PASS/FAIL/ASSUMPTION/NOT_RUN evidence, negative controls, counterfactual mutations and remaining limitations.
- `a_v1_preprocessing_fit_audit.json` / `a_v1_context_blocks_fit_audit.json`: actual training/evaluation row IDs and fitted-state fingerprints. No reserved-test model scoring.
- `a_v1_development_comparison.csv` and per-condition predictions: 13 completed CPU logistic/TF-IDF conditions. All joint gains are negative; neither E5 nor TAR has been run.
- `a_v1_reused_interview_rows.csv` / `a_v1_near_duplicate_screen.csv`: legitimate shared-document uses and the bounded lexical similarity screen, with no automatic equivalence between reuse and leakage.
- `a_v1_artifact_manifest.json`: input and output file hashes, including private derived files. Transcript-bearing feature CSVs and source/span/history lineages remain in external `derived/candidate_A_v1/`.

Using the existing cache and isolated tokenizer assets:

```bash
python3 -m tennis.scripts.a_study
python3 -m tennis.scripts.a_context_blocks
python3 -m tennis.scripts.a_leakage
python3 -m tennis.scripts.a_additional_checks
python3 -m tennis.scripts.a_review_outputs
python3 -m unittest tennis.scripts.test_research tennis.scripts.test_interview_followup tennis.scripts.test_a_validation -v
```

The first two commands run the explicitly labeled local training/development diagnostics. Later commands audit/render saved evidence. The original study freeze refuses changed selector/temporal source code; use a new version for a new experiment. `tennis/scripts/tennis_temporal_pilot.py` now accepts the documented conditional calendar rule, validates the hashed audit sidecar and defaults to development rather than consuming the reserved test. It still uses unchanged official classes if explicitly run in Colab. Notebook 02 remains unchanged.
