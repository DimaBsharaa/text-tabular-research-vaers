"""Generate concise reports from the same CSVs used by the executed notebook."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
def table(df):
    df = df.copy().fillna('unknown')
    rows = [list(df.columns), *df.astype(str).values.tolist()]
    escape = lambda value: str(value).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(escape,rows[0]))+' |', '| '+' | '.join(['---']*len(rows[0]))+' |'] + ['| '+' | '.join(map(escape,row))+' |' for row in rows[1:]])

def main():
    inv = pd.read_csv(ROOT/'data/dataset_inventory.csv')
    profiles = pd.read_csv(ROOT/'data/dataset_profiles.csv')
    results = pd.read_csv(ROOT/'data/learner_curation_results.csv')
    summary = json.loads((ROOT/'reports/analysis_summary.json').read_text())
    analysis = inv.merge(profiles[[c for c in profiles if c=='dataset_id' or c not in inv]],on='dataset_id',validate='one_to_one')
    compatible = analysis[analysis.historical_task_matches_release.fillna(False)]
    release_cols = ['dataset_name','released_collection','task_type','released_target_column','number_of_passing_learners']
    available = inv[inv.released_dataset_available][release_cols]
    unreleased = inv[~inv.released_dataset_available][['dataset_name','task_type','passed_curation','original_dataset_url']]
    changed = inv[inv.released_dataset_id.notna() & inv.dataset_id.str[:3].ne(inv.released_dataset_id.str[:3])]
    source_manifest = json.loads((ROOT/'data/sources/manifest.json').read_text())
    hf_manifest = json.loads((ROOT/'data/sources/hf_previews/manifest.json').read_text())
    source_bytes = sum(r['bytes'] for r in source_manifest['files'])
    preview_bytes = sum(f.get('bytes',0) for row in hf_manifest for f in row['files'])
    availability = f"""# Source availability · October 14, 2026

The pasted context and both course PDFs were read. The shared ChatGPT conversation was retrieved after a direct retry and its recent project decisions were read. NFL drafting and tennis were exploratory examples, not a chosen dataset. Scope: Standard Track 2, research and EDA before choosing a new dataset. No new learner runs, encoder inference, training, or fine-tuning were performed. No push, fork, submodule, or full MulTaBench clone was made. The two original PDFs remain in the parent NLP folder.

## Evidence found

- All **56** original names and published five-learner outcome flags: paper v1, Appendix C.4.
- **23 passed, 33 failed** historically; **20** became core. The other successes are Consumer Complaints, News Channel, and Hearthstone Cards. The first two have extended releases; Hearthstone Cards is unreleased.
- Original registered source URLs for **56/56** candidates. They are provenance references; their download endpoints were not tested.
- Current public, ungated Hugging Face repositories with `data.parquet` listed: **40/56** (20 core, 20 extended). These artifacts were not downloaded in full.
- Four-condition historical scores for **276/280** learner–dataset pairs: **5,519** fold rows. Four missing pairs are both TabPFN variants on Wine Review and Spotify Genres. One CarDekho/TabPFN 2.0 TAR fold is missing.
- Historical released summaries: N, total structured columns, text columns, task, classes for **40** datasets. They are release properties, not verified untouched original-candidate properties.
- Current typed cards and metadata: target, numeric/categorical/boolean/date/text columns, row counts, artifact size, source and revision. All 40 viewer previews returned 50 rows, yielding **2,000** rows and **214** text-column profiles. No reported cell truncation.
- Individual original benchmark membership is verified for **{int(inv.original_benchmark_source.notna().sum())}/56** candidates from the paper's overlap table and TextTabBench exports. Membership lists remain partial; blanks are unknown, never fabricated.
- Original-source target column names were not verified. `target_column` remains blank; `released_target_column` records the verified current-release name. A matching task prefix alone is insufficient to equate them.

## Exact official artifacts

Historical reference: [paper_version commit 3bb95079](https://github.com/alanarazi7/MulTaBench/tree/3bb95079a6146ed35d45187e98bcad5b063273a1). `official-paper-version` was not separately fetched or assumed identical. `master` is a newer leaderboard protocol and was used only to find current release identities.

- `multabench/leaderboard/results/analysis_curation_sensitivity/pool_scores_long.csv`: canonical historical folds. Official `committee_pool.py` traces the original five-learner scores to `tabstar_corpus/text_50_datasets.csv` and `texttabench_datasets.csv`.
- `analysis_curation_sensitivity/pass_matrix.csv` and `leaderboard/analysis/pass_matrix.py`: reproduction flags and exact arithmetic.
- `datasets_summary.csv` / `datasets_summary_extra.csv`: structure of core and extended releases.
- `leaderboard/analysis/pool_names.py`: explicit mappings between candidate IDs and release IDs.
- `datasets/all_datasets.py`: original sources and release IDs; targeted `benchmark/datasets/*_TEXT_*.py` recipes describe preparation.
- [Paper v1](https://arxiv.org/html/2605.10616v1): published acceptance flags and definitions.
- [Hugging Face organization](https://huggingface.co/multabench): current dataset cards, `metadata.json`, API sizes, revision IDs, and public viewer rows.

The bounded cache is approximately {(source_bytes+preview_bytes)/1e6:.2f} MB, plus the small repository-tree listing. Full currently released datasets total about {profiles.dataset_bytes.sum()/1e6:.1f} MB and were not downloaded. SHA-256 hashes, exact source URLs, per-release revisions, and offsets are recorded in `data/sources/manifest.json` and `data/sources/hf_previews/manifest.json`.

## Accessible released artifacts

Public accessibility here means an ungated repository, listed parquet file, fetched metadata, and a successful bounded preview—not full original source availability. `data/dataset_inventory.csv` contains clickable source and release URLs.

{table(available)}

## Candidates requiring original-source retrieval for row-level EDA

{table(unreleased)}

## Version and task differences

These six releases change the original task prefix. Their current rows must not be treated as profiles of the original evaluation task:

{table(changed[['dataset_name','task_type','released_task_type','released_target_column']])}

The reproduction code rounds each condition mean to three decimals and uses gain **>=0.001**. Paper Appendix A.3 uses **>0.001**. Both are shown in the notebook. Two published model cells disagree with reconstruction (CarDekho/LightGBM and AI Job Salaries/CatBoost); the candidate-level accepted set remains unchanged. These are retained separately, not overwritten.

## Analysis plan carried out

1. Preserve published inventory and reconstruct learner-level gains from pinned scores.
2. Compare failures, learners, thresholds, quorum, and fold signs on the complete candidate pool.
3. Compare historical release structure and current preview text only with explicit coverage and task compatibility.
4. Keep core/extended analysis separate, inspect analog domains, and present a decision worksheet.
5. Save executed notebook outputs, standalone HTML, tables, figures, and these reports for review. Stop before new dataset construction or benchmark training.
"""
    (ROOT/'reports/source_availability.md').write_text(availability)
    fields = ['historical_released_n_rows','historical_released_structured_features','historical_released_text_features','sample_mean_text_chars','sample_longest_column_mean_chars']
    medians = compatible.groupby('passed_curation')[fields].median().round(2)
    medians.insert(0,'n',compatible.groupby('passed_curation').size())
    medians = medians.reset_index()
    failure = results[~results.passed_curation].failure_type.value_counts().rename_axis('learner_pair_outcome').reset_index(name='n')
    findings = f"""# Research findings

## Confirmed observations

The original pool has 56 candidates: 23 published successes and 33 failures. Reproduction agrees with this accepted set, while two learner cells differ from the publication. There are 11 published five-learner successes and 12 published zero-learner successes. Release membership is not an outcome label: two historical successes are extended and one is unreleased.

Across the 165 original learner pairs of rejected candidates:

{table(failure)}

TAR-only failure is the most common learner-level failure category (79/165). This does not mean TAR fails for every learner of every rejected dataset. Joint-only failure occurs 32 times and simultaneous failure 23 times. The remaining 31 pairs pass both criteria, but the dataset lacks a three-learner quorum.

With three learners required, acceptance drops from 23 at a rounded minimum gain of 0.001 to 19 at 0.002, 15 at 0.005, and 9 at 0.01. These are descriptive margin checks, not tests of statistical significance. Applying the paper's strict >0.001 rule to rounded means accepts 19 rather than the reproduction code's 23.

Structure and text comparisons cover **21 passing and 13 failing** compatible released profiles, not all 23 and 33. Their medians are:

{table(medians)}

Passing profiles in this restricted subset have more rows at the median (18,843 versus 8,035) and a longer richest text column (121 versus 42.9 characters in the preview). Median structured feature counts are 6 versus 7, and median text-column counts are 3 in both groups. This is an association in a selected subset, not evidence that changing row counts or text lengths causes passage. Some successful datasets rely on short strings, so requiring long narratives would discard useful candidates.

The domain tables and analog comparisons show successes and failures within broad food, consumer, professional, and social domains. Housing candidates in this particular historical pool do not pass, but that small selected pool cannot establish a universal claim about housing data.

## Hypotheses for later development-set experiments

- A target-sensitive distinction in the text may provide room for TAR that generic embeddings miss.
- A narrative can be useful, but short names, titles, and other strings may encode task-relevant semantics too.
- A strong unimodal predictor can leave little room for frozen fusion. Complementarity should be checked empirically, not assumed from a schema.
- More training rows could help adaptation; release coverage, task changes, and domain confounding prevent attributing the observed row-count association to that mechanism.

## Limitations

There are no row-level profiles for 16 unreleased candidates. Six release tasks differ from original tasks. Even unchanged task prefixes do not prove identical rows, transformations, or prediction targets. The 50-row contiguous previews can be affected by ordering and are too small for global missingness, duplicate rates, class balance, rare-event prevalence, or representative text statistics. Viewer/category conversion differs from historical CSVs. Original benchmark membership is only partially verified; direct original download endpoints are untested. One fold and four learner pairs are missing. Mean-fold thresholding is not significance testing, and reused fold rows can be dependent. Outcome and profile selection induce bias. No new experiments or causal claims were made.

## Implications for choosing an idea

Seek a defensible task with complementary structured and textual evidence and a plausible task-specific language signal, rather than selecting a domain from its historical pass rate. The shared conversation treats NFL drafting and tennis as inspiration only. Count rows surviving the join of statistics, dated pre-outcome text, and targets; a large statistics table alone does not establish usable dataset size. Audit target definition, prediction timing, leakage, entities, license, and availability before investing compute. Use cheap development-only screens before official evaluation; freeze the target, processing, and holdout protocol before final testing. A very small improvement near the threshold is fragile.

No new dataset or prediction task has been selected. The notebook's existing-dataset selector and analog groups are research aids. The next decision belongs to the user after reviewing the evidence; this phase ends here.
"""
    (ROOT/'reports/research_findings.md').write_text(findings)
    analog_path = ROOT/'data/analog_comparisons.csv'
    analog = pd.read_csv(analog_path)
    comparisons = "# Comparable domains and different outcomes\n\nThese are analog groups, not matched causal comparisons. Scores are median original learner gains. Profile values are absent when the released task differs or the candidate was not released. Numeric targets and units differ, so compare signs and criteria rather than ranking raw AUC and R² together.\n\n"
    cols = ['dataset_name','task_type','number_of_passing_learners','historical_released_n_rows','historical_released_structured_features','sample_longest_column_mean_chars','joint_signal_gain','tar_gain']
    for group, frame in analog.groupby('analog_group',sort=False):
        comparisons += f"## {group} ({len(frame)} candidates)\n\n"+table(frame[cols].round(4))+"\n\n"
    comparisons += "## What is supported\n\nSalary examples include both successful datasets and AI Job Salaries, which fails the quorum. Product-price examples similarly span successful Baby Products/Book Price/Mercari and rejected products. TAR and joint gains, rather than the domain name, reveal different bottlenecks. The Notebook contains all five learners' scores and preview column profiles so these analogies can be inspected.\n\n## What remains unknown\n\nWe did not run feature-level predictive analyses or new models, and several failed candidates lack row previews. Therefore we cannot explain outcomes by specific strong structured columns or establish that a particular text feature causes improvement. Housing releases with changed targets are excluded from original-task profile comparisons. Review source target processing before asserting that targets within an analog group are comparable.\n"
    (ROOT/'reports/dataset_comparison.md').write_text(comparisons)
    print('Wrote availability, comparison, and findings reports from verified CSVs.')

if __name__=='__main__':main()
