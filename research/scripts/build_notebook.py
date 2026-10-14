"""Create a standard Jupyter notebook with visible, editable offline analysis cells."""
from pathlib import Path
import json
import re
import textwrap

ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(text):
    cells.append({"cell_type":"markdown","metadata":{},"source":textwrap.dedent(text).strip()+"\n"})
def code(text):
    cells.append({"cell_type":"code","metadata":{},"source":textwrap.dedent(text).strip()+"\n","execution_count":None,"outputs":[]})

md("""
# Why do text-tabular datasets pass or fail MulTaBench?

**Track 2 research notebook · October 14, 2026 · no new dataset chosen**

We want evidence to guide a new dataset idea. This notebook studies the 56 original candidates, not a new benchmark experiment. All tables and plots below have saved outputs. Run the notebook offline with the cached sources. No learners, encoders, or fine-tuning run here.

Three evidence levels must stay separate:

- **Historical outcomes and scores:** all 56 candidates; five original learners and five folds, except documented missing evaluations.
- **Historical released structure summaries:** 40 text datasets. Six extended releases change the original task, leaving 34 task-compatible profiles for outcome comparisons.
- **Current-release previews:** 50 rows per released dataset, 2,000 rows total. A seeded contiguous block is deliberately small, potentially ordered, and **not a representative random sample**. Length, missingness, duplicates, and target frequencies refer only to these previews.

The assignment PDF is the deliverable authority. Historical reconstruction follows the pinned official reproduction code; differences from the paper's wording are shown explicitly. No conclusions here certify that a new dataset will pass.
""")
code("""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'data/dataset_inventory.csv').exists()) if (Path.cwd() / 'data/dataset_inventory.csv').exists() or any((p / 'data/dataset_inventory.csv').exists() for p in Path.cwd().parents) else Path.cwd() / 'research'
assert (ROOT / 'data/dataset_inventory.csv').exists(), 'Open from the repository, research directory, or notebooks directory.'
FIGURES = ROOT / 'reports/figures'
FIGURES.mkdir(parents=True, exist_ok=True)
inventory = pd.read_csv(ROOT / 'data/dataset_inventory.csv')
profiles = pd.read_csv(ROOT / 'data/dataset_profiles.csv')
results = pd.read_csv(ROOT / 'data/learner_curation_results.csv')
folds = pd.read_csv(ROOT / 'data/historical_fold_scores.csv')
texts = pd.read_csv(ROOT / 'data/text_column_profiles.csv')
columns = pd.read_csv(ROOT / 'data/column_profiles.csv')
risks = pd.read_csv(ROOT / 'data/leakage_audit_candidates.csv')
disagreements = pd.read_csv(ROOT / 'data/published_reproduction_disagreements.csv')
summary = json.loads((ROOT / 'reports/analysis_summary.json').read_text())
models = ['LightGBM','CatBoost','TabM','TabPFNv2','TabPFN-2.5']
sns.set_theme(style='whitegrid', palette='colorblind', font_scale=.95)
np.random.seed(20261014)
pd.set_option('display.max_rows', 60)
pd.set_option('display.max_columns', 16)
pd.set_option('display.float_format', lambda x: f'{x:,.3f}')
def showplot(name):
    plt.tight_layout()
    plt.savefig(FIGURES / f'{name}.png', dpi=150, bbox_inches='tight')
    plt.show()
def showtable(df):
    display(df)

profile_cols = [c for c in profiles if c == 'dataset_id' or c not in inventory.columns]
analysis = inventory.merge(profiles[profile_cols], on='dataset_id', validate='one_to_one')
comparable = analysis[analysis.historical_task_matches_release.fillna(False)].copy()
comparable['outcome'] = comparable.passed_curation.map({True:'Pass',False:'Fail'})
assert len(inventory)==56 and inventory.passed_curation.sum()==23
assert len(comparable)==34 and profiles.sample_n_rows.sum()==2000
showtable(pd.DataFrame([summary]).T.rename(columns={0:'Verified value'}))
""")
md("""
## 1. Availability before analysis

All 56 names, original task prefixes, published model decisions, and historical four-condition scores are available. The 40 releases are publicly listed, ungated, and contain `data.parquet`; full data has not been downloaded. The other 16 need their original sources for row-level profiling. Registered original URLs are provenance, not a guarantee that a download still works. Source-benchmark memberships are only partially verified; blank membership means unknown.

The canonical score source is `pool_scores_long.csv` from paper-version commit `3bb95079a6146ed35d45187e98bcad5b063273a1`. Its original curation scores come from `tabstar_corpus`, not the later changed-target release runs. Current release identities and per-repository revisions are cached separately.
""")
code("""
coverage = pd.DataFrame({
    'Evidence':['Original candidate names','Published learner decisions','Historical four-condition model pairs','Released full-data artifacts listed','Task-compatible release profiles','Preview rows per released dataset'],
    'Available':[56,56,276,40,34,50], 'Expected / scope':[56,56,280,56,56,50]})
showtable(coverage)
showtable(inventory.groupby(['released_collection','passed_curation']).agg(n=('dataset_id','size')).reset_index())
showtable(inventory.loc[inventory.passed_curation & ~inventory.selected_for_core_benchmark,
                        ['dataset_name','released_collection','number_of_passing_learners']])
fig, ax = plt.subplots(figsize=(8,4))
inventory.groupby(['released_collection','passed_curation']).size().unstack(fill_value=0).rename(columns={False:'Fail',True:'Pass'}).plot.bar(stacked=True,ax=ax)
ax.set(title='Release collection and historical outcome are different variables',ylabel='Original candidates',xlabel='Release collection')
showplot('01_coverage')
""")
code("""
availability_cols = ['dataset_name','task_type','released_collection','passed_curation','number_of_passing_learners','released_dataset_available','original_benchmark_source']
showtable(inventory[availability_cols].sort_values(['passed_curation','number_of_passing_learners','dataset_name'],ascending=[False,False,True]).reset_index(drop=True))
""")
md("""
## 2. Published outcomes versus reconstruction

**Reproduction code:** average each condition's five fold scores, round each mean to three decimals, then calculate joint gain and TAR gain. A gain **≥ 0.001** passes. At least three of five learners must pass **both** criteria. The paper's equation instead says **> 0.001**. This notebook preserves published flags and shows both interpretations; a threshold is not a statistical significance test.

TabPFN 2.0 and 2.5 have no evaluations for Wine Review and Spotify Genres (too many classes). Missing evaluations remain N/A and do not count toward the fixed three-learner quorum. CarDekho/TabPFN 2.0 has only four TAR folds; the official rule explicitly permits that missing fold.
""")
code("""
showtable(disagreements)
showtable(inventory[['dataset_name','number_of_passing_learners','reproduced_number_of_passing_learners','passed_curation','reproduced_passed_curation']].query('number_of_passing_learners != reproduced_number_of_passing_learners'))
count_table = inventory.number_of_passing_learners.value_counts().reindex(range(6),fill_value=0).rename_axis('Passing learners').to_frame('Published candidate count')
showtable(count_table)
fig, ax = plt.subplots(figsize=(7,3))
count_table.plot.bar(ax=ax,legend=False,color=['#b9655c']*3+['#377a98']*3)
ax.set(title='Published original pool: 23 pass, 33 fail',xlabel='Learners satisfying both criteria',ylabel='Datasets')
showplot('02_pass_counts')
""")
code("""
pass_cols = ['lightgbm_pass','catboost_pass','tabm_pass','tabpfn2_pass','tabpfn25_pass']
ordered = inventory.sort_values(['passed_curation','number_of_passing_learners','dataset_name'],ascending=[False,False,True])
matrix = ordered.set_index('dataset_name')[pass_cols].astype(float)
matrix.columns = models
fig, ax = plt.subplots(figsize=(8,17))
ax.set_facecolor('#dddddd')
sns.heatmap(matrix, vmin=0,vmax=1,cmap=sns.color_palette(['#cb7770','#4586a0'],as_cmap=True),cbar=False,linewidths=.5,ax=ax)
ax.axhline(23,color='black',linewidth=2)
ax.set(title='Published 56 × 5 outcomes: red fail, blue pass, gray N/A',xlabel='Original curation learner',ylabel='Original candidate (approved first)')
showplot('03_published_matrix')
""")
md("""
## 3. Which curation condition fails?

Joint gain = Joint Frozen − max(Structured, Unstructured). TAR gain = Joint TAR − Joint Frozen. Failure types below come from actual scores, not guessed from final flags. A failed dataset may still have one or two successful learners. Each dataset receives equal weight per learner; these are descriptive differences in AUC or R², not interchangeable clinical or economic effects.
""")
code("""
failed = results[~results.passed_curation & results.evaluation_available].copy()
failure_order = ['joint signal','TAR gain','both','pass']
failure_counts = pd.crosstab(failed.model_name,failed.failure_type).reindex(index=models,columns=failure_order,fill_value=0)
showtable(failure_counts)
showtable(failed.failure_type.value_counts().rename_axis('Learner-level outcome').to_frame('Pairs among 33 rejected datasets'))
fig, ax = plt.subplots(figsize=(9,4))
failure_counts.plot.bar(stacked=True,ax=ax,color=['#c07463','#dba95c','#895f8f','#44859e'])
ax.set(title='Historical failures across 33 rejected datasets (165 model–dataset pairs)',ylabel='Pairs',xlabel='Learner')
ax.legend(title='Condition',bbox_to_anchor=(1.02,1),loc='upper left')
showplot('04_failure_types')
""")
code("""
dataset_failures = failed.groupby(['dataset_id','dataset_name']).agg(
    learners_joint_fails=('joint_signal_pass',lambda x:(~x.astype(bool)).sum()),
    learners_tar_fails=('tar_pass',lambda x:(~x.astype(bool)).sum()),
    min_joint_gain=('joint_signal_gain','min'),median_joint_gain=('joint_signal_gain','median'),
    min_tar_gain=('tar_gain','min'),median_tar_gain=('tar_gain','median'),
    mixed_failure_types=('failure_type',lambda x:', '.join(sorted(set(x)))))
showtable(dataset_failures.reset_index().drop(columns='dataset_id').sort_values('median_tar_gain'))
fig, axes = plt.subplots(1,2,figsize=(12,5),sharey=True)
for ax, task in zip(axes,['classification','regression']):
    sub = results[results.evaluation_available & (results.task_type.eq('regression') if task=='regression' else results.task_type.ne('regression'))]
    sns.scatterplot(data=sub,x='joint_signal_gain',y='tar_gain',hue='passed_curation',style='model_name',alpha=.75,ax=ax)
    ax.axvline(.001,color='black',linestyle='--');ax.axhline(.001,color='black',linestyle='--')
    ax.set(title=f'{task.title()}: {len(sub)} pairs',xlabel='Joint gain (rounded score units)',ylabel='TAR gain')
    if ax is axes[0]: ax.legend_.remove()
axes[1].legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=8)
showplot('05_joint_tar_gains')
""")
md("""
## 4. Thresholds, quorum, and fold stability

These sensitivity views ask whether success relies on a tiny margin or a particular learner. They are not new runs. Fold signs and standard deviations are descriptive: five splits can share rows, so do not treat them as five independent statistical samples. A weak boundary pass deserves validation on an untouched future holdout.
""")
code("""
threshold_rows = []
for threshold in [0,.001,.002,.005,.01,.02]:
    keep = (results.joint_signal_gain > 0) & (results.tar_gain > 0) if threshold==0 else (results.joint_signal_gain >= threshold) & (results.tar_gain >= threshold)
    per_dataset = results.assign(keep=keep.fillna(False)).groupby('dataset_id').keep.sum()
    for quorum in [1,2,3,4,5]:
        threshold_rows.append({'Minimum gain':threshold,'Quorum':quorum,'Accepted datasets':int((per_dataset>=quorum).sum())})
sensitivity = pd.DataFrame(threshold_rows)
showtable(sensitivity[sensitivity.Quorum.eq(3)].reset_index(drop=True))
strict_count = int((results.assign(strict=results.paper_strict_margin_pass.fillna(False)).groupby('dataset_id').strict.sum()>=3).sum())
print(f'Paper strict >0.001 after rounding: {strict_count}/56; implementation >=0.001: 23/56.')
fig, ax = plt.subplots(figsize=(8,4))
sns.heatmap(sensitivity.pivot(index='Minimum gain',columns='Quorum',values='Accepted datasets'),annot=True,fmt='d',cmap='Blues',ax=ax)
ax.set(title='Acceptance sensitivity on the original pool (56 datasets)')
showplot('06_threshold_quorum')
showtable(results.loc[results.evaluation_available,['dataset_name','model_name','joint_signal_gain','tar_gain','paired_joint_gain_sd','paired_tar_gain_sd','paired_joint_positive_fold_count','paired_tar_positive_fold_count','paired_tar_n_folds']].sort_values('tar_gain').head(15))
""")
md("""
## 5. Domain and original target type

Domain labels are the broad tags in official source-corpus identifiers (consumer, food, houses, professional, social, etc.). They are neither hand-selected success explanations nor mutually comparable applications. Counts are small and uneven. The task variable here is the **original task**, not the task of a changed release.
""")
code("""
domain_counts = inventory.groupby('domain').agg(n=('dataset_id','size'),passing=('passed_curation','sum'))
domain_counts['pass_rate'] = domain_counts.passing/domain_counts.n
task_counts = inventory.groupby('task_type').agg(n=('dataset_id','size'),passing=('passed_curation','sum'))
task_counts['pass_rate'] = task_counts.passing/task_counts.n
showtable(domain_counts);showtable(task_counts)
fig, axes = plt.subplots(1,2,figsize=(12,4))
for ax, table, title in [(axes[0],domain_counts,'Source domain'),(axes[1],task_counts,'Original task')]:
    table.pass_rate.plot.bar(ax=ax,color='#44859e')
    for idx, row in enumerate(table.itertuples()):ax.text(idx,row.pass_rate+.03,f'{row.passing}/{row.n}',ha='center',fontsize=9)
    ax.set(ylim=(0,1.2),ylabel='Historical pass fraction',xlabel='',title=title)
showplot('07_domain_task')
""")
md("""
## 6. Structural properties: compatible profiles only

Rows and total structured/text feature counts use historical release summaries. We exclude six changed-task releases, and 16 unreleased candidates have no row-level profile. This leaves **21 passing versus 13 failing** candidates. Missingness of profiles is not random; comparisons below do not represent all 23 versus 33 outcomes. Even unchanged task prefixes do not prove identical source/release rows or preprocessing.

Numeric, categorical, boolean, and date counts in the next table come from current typed dataset cards; they are not retroactively inferred original feature types. Dates and typed-category conversion can alter the breakdown.
""")
code("""
showtable(analysis.loc[analysis.released_collection.ne('unreleased') & ~analysis.historical_task_matches_release.fillna(False),
                       ['dataset_name','task_type','released_task_type','released_target_column','historical_released_n_classes']])
features = ['historical_released_n_rows','historical_released_structured_features','historical_released_text_features']
showtable(comparable.groupby('outcome')[features].agg(['count','median','mean','min','max']))
fig, axes = plt.subplots(1,3,figsize=(12,4))
for ax, feature, title in zip(axes,features,['Historical released rows','Structured features','Text columns']):
    sns.boxplot(data=comparable,x='outcome',y=feature,order=['Fail','Pass'],ax=ax,color='#b8d4e0')
    sns.stripplot(data=comparable,x='outcome',y=feature,order=['Fail','Pass'],ax=ax,color='#345469',jitter=.12,alpha=.65)
    if feature.endswith('n_rows'):ax.set_yscale('log')
    ax.set(title=title,xlabel='Fail n=13; pass n=21',ylabel=title)
showplot('08_structure_outcomes')
showtable(analysis.loc[analysis.released_collection.ne('unreleased'),['dataset_name','hf_n_rows','hf_numeric_features','hf_categorical_features','hf_boolean_features','hf_date_features','hf_text_features','dataset_bytes','historical_task_matches_release']].sort_values('hf_n_rows'))
""")
md("""
## 7. Core versus extended: a separate question

This is a comparison of two released collections, **20 versus 20**, not passing versus failing candidates. Extended contains two historical successes (Consumer Complaints and News Channel), and some targets were changed before release. Raw size comparisons remain useful descriptively, but cannot explain the original curation decisions.
""")
code("""
released = analysis[analysis.released_collection.ne('unreleased')].copy()
showtable(released.groupby('released_collection')[features].agg(['count','median','mean']))
fig, axes = plt.subplots(1,3,figsize=(12,4))
for ax, feature in zip(axes,features):
    sns.boxplot(data=released,x='released_collection',y=feature,ax=ax,color='#b8d4e0')
    sns.stripplot(data=released,x='released_collection',y=feature,ax=ax,color='#345469',alpha=.5)
    if feature.endswith('n_rows'):ax.set_yscale('log')
    ax.set(xlabel='20 datasets per collection',ylabel=feature.replace('historical_released_','').replace('_',' '))
showplot('09_core_extended')
""")
md("""
## 8. Text: length, diversity, and differences between columns

Measurements use stripped, nonempty string cells. **Characters:** Python `len`, in Unicode code points. **Words:** whitespace splitting, not model-tokenizer tokens. Missing and whitespace-only cells count as empty. Repetition means exact duplicate nonempty strings within a column. Unique ratio = distinct nonempty strings / nonempty cells. Short = at most three words; long = at least twenty. These thresholds are descriptive proxies, not verified semantic categories.

Pooling all text cells can dilute a long narrative when a dataset also has many short names. Therefore we report both pooled mean length and the **longest column's mean length**. Individual column profiles show that heterogeneity. The viewer reported no truncated cells in these cached previews.
""")
code("""
text_fields = ['sample_mean_text_chars','sample_longest_column_mean_chars','sample_median_text_words','sample_empty_text_rate','sample_repeated_text_rate']
showtable(comparable.groupby('outcome')[text_fields].agg(['count','median','mean']))
fig, axes = plt.subplots(1,2,figsize=(12,4))
for ax, field, title in zip(axes,['sample_mean_text_chars','sample_longest_column_mean_chars'],['Pooled text mean','Longest column mean']):
    sns.boxplot(data=comparable,x='outcome',y=field,order=['Fail','Pass'],ax=ax,color='#b8d4e0')
    sns.stripplot(data=comparable,x='outcome',y=field,order=['Fail','Pass'],ax=ax,color='#345469',alpha=.6)
    ax.set_yscale('log');ax.set(title=title,ylabel='Characters (50-row preview)',xlabel='Fail n=13; pass n=21')
showplot('10_text_lengths')
longest = texts.sort_values('sample_mean_chars').groupby('dataset_id',sort=False).tail(1)
showtable(longest[['dataset_name','column_name','sample_mean_chars','sample_median_chars','sample_mean_words','sample_unique_nonempty_ratio','sample_empty_rate']].sort_values('sample_mean_chars',ascending=False))
""")
code("""
fig, axes = plt.subplots(1,2,figsize=(12,4))
sns.histplot(data=texts,x='sample_short_string_rate',bins=10,ax=axes[0],color='#44859e')
axes[0].set(title=f'{len(texts)} current text-column previews',xlabel='Fraction of nonempty cells with ≤3 words',ylabel='Columns')
sns.scatterplot(data=comparable,x='sample_longest_column_mean_chars',y='number_of_passing_learners',hue='outcome',style='task_type',s=80,ax=axes[1])
axes[1].set_xscale('log');axes[1].set(title='Long text does not guarantee passage',xlabel='Longest text column mean characters (preview)',ylabel='Published passing learners',yticks=range(6))
showplot('11_text_form_passage')
structure_bins = pd.cut(comparable.historical_released_structured_features,[0,5,15,np.inf],labels=['1–5','6–15','16+'])
length_bins = pd.cut(comparable.sample_longest_column_mean_chars,[0,50,200,np.inf],labels=['≤50','51–200','>200'])
for label, groups in [('Structured feature count',structure_bins),('Longest text column mean characters',length_bins)]:
    table = comparable.assign(bucket=groups).groupby('bucket',observed=False).agg(n=('dataset_id','size'),passing=('passed_curation','sum'))
    table['pass_rate'] = table.passing/table.n
    print(label, '(descriptive bins; task-compatible released subset)');showtable(table)
""")
md("""
## 9. Data quality and targets

These rates are **preview-only**, not full-dataset duplicate rates or global class balance. Regression variance is in each target's own units and cannot be compared meaningfully across prices, ratings, and salaries. Low observed class counts in 50 rows do not establish the full class count.

Outcome-like column names and high correlations are prompts for manual leakage review, not proof. Establish whether each feature would be known at the intended prediction time. We make no feature-removal or training decisions here, and these old benchmark previews must never become a new project's test set.
""")
code("""
quality = ['dataset_name','sample_n_rows','sample_feature_missing_rate','sample_duplicate_predictor_row_rate','sample_duplicate_full_row_rate','sample_empty_text_rate','sample_repeated_text_rate','sample_target_n_unique','sample_majority_class_fraction','sample_target_variance']
showtable(profiles.loc[profiles.sample_n_rows.notna(),quality].sort_values('sample_feature_missing_rate',ascending=False))
showtable(risks)
""")
md("""
## 10. Inspect one existing dataset

Change `SELECTED_DATASET` and rerun these two cells to inspect any released dataset. This selector inspects benchmark evidence; its default is not a proposed new dataset. The source offset and sample size are printed so the scope of every number is clear.
""")
code("""
SELECTED_DATASET = 'Book Price'
selected = inventory[inventory.dataset_name.eq(SELECTED_DATASET)].iloc[0]
assert pd.notna(selected.released_dataset_location), 'No current preview for this unreleased candidate.'
sample_folder = ROOT / 'data/sources/hf_previews' / selected.released_dataset_location.rsplit('/',1)[1]
payload = json.loads((sample_folder/'rows.json').read_text())
sample = pd.DataFrame([row['row'] for row in payload['rows']])
meta = json.loads((sample_folder/'metadata.json').read_text())
print(f"{SELECTED_DATASET}: {len(sample)} preview rows starting at row {payload['rows'][0]['row_idx']}; target={meta['target']}; release task={meta['task_type']}")
showtable(texts[texts.dataset_name.eq(SELECTED_DATASET)].drop(columns=['dataset_id','measurement','sample_scope']))
showtable(columns[columns.dataset_name.eq(SELECTED_DATASET)].drop(columns='dataset_id'))
showtable(results[results.dataset_name.eq(SELECTED_DATASET)][['model_name','structured_score','unstructured_score','joint_frozen_score','joint_tar_score','joint_signal_gain','tar_gain','failure_type']])
""")
code("""
fig, axes = plt.subplots(1,2,figsize=(12,4))
selected_text = texts[texts.dataset_name.eq(SELECTED_DATASET)]
for c in selected_text.column_name:
    values = sample[c].dropna().astype(str).str.strip()
    values = values[values.ne('')]
    if len(values): sns.ecdfplot(x=values.str.len(),label=c,ax=axes[0])
axes[0].set_xscale('symlog',linthresh=1)
axes[0].set(title=f'{SELECTED_DATASET}: text-length distributions',xlabel='Nonempty characters',ylabel='Cumulative fraction')
axes[0].legend(fontsize=8)
target = sample[meta['target']]
if meta['task_type']=='reg':
    sns.histplot(pd.to_numeric(target,errors='coerce'),bins=12,ax=axes[1],color='#44859e')
else:
    target.value_counts().plot.bar(ax=axes[1],color='#44859e')
axes[1].set(title=f'{SELECTED_DATASET}: target preview',xlabel=meta['target'],ylabel='Preview rows')
showplot('12_selected_dataset')
""")
md("""
## 11. Similar domains, different outcomes

These are analog groups, not matched causal pairs. Historical gains come from the original scores. Structural/text profiles are shown only when the task prefix is unchanged; missing values stay missing. Source/release preprocessing can still differ. Check target definitions and text contents before transferring lessons.

Salary datasets, product-price datasets, review/rating datasets, and housing examples let us inspect why a simple domain heuristic would be unreliable. We have not measured the predictive strength of individual structured features; strongest-unimodal scores only indicate modality-level strength.
""")
code("""
groups = {
    'Salary':['Data Scientist Salary','Montgomery Salaries','Vancouver Salaries','AI Job Salaries'],
    'Product price':['Baby Products','Book Price','Mercari Marketplace','JC Penney','Indian Laptops','CarDekho Cars','BikeWale Bikes'],
    'Review / rating':['Wine Review',"Women's Clothing",'Beer Ratings','Coffee Reviews','Ramen Ratings','Anime Planet'],
    'Real estate':['Airbnb Seattle','Airbnb Melbourne','California Prices','SF Permits'],
    'Sports (historical analogs only)':['FIFA22 Wages','NBA Draft Value']}
median_gains = results.groupby('dataset_id')[['joint_signal_gain','tar_gain','structured_score','unstructured_score']].median()
pair_rows = []
for group, names in groups.items():
    table = analysis[analysis.dataset_name.isin(names)].copy().merge(median_gains,on='dataset_id')
    fields = ['historical_released_n_rows','historical_released_structured_features','historical_released_text_features','sample_longest_column_mean_chars']
    table.loc[~table.historical_task_matches_release.fillna(False),fields] = np.nan
    table['analog_group']=group
    pair_rows.append(table)
    print(group, f'({len(table)} original candidates)')
    showtable(table[['dataset_name','task_type','passed_curation','number_of_passing_learners',*fields,'joint_signal_gain','tar_gain']].sort_values('passed_curation',ascending=False))
analog = pd.concat(pair_rows,ignore_index=True)
analog.to_csv(ROOT/'data/analog_comparisons.csv',index=False)
fig, axes = plt.subplots(1,2,figsize=(13,9),sharey=True)
ordered_analog = analog.sort_values(['analog_group','passed_curation','dataset_name'],ascending=[True,False,True])
for ax, gain, title in [(axes[0],'joint_signal_gain','Median joint gain across available learners'),(axes[1],'tar_gain','Median TAR gain across available learners')]:
    sns.barplot(data=ordered_analog,x=gain,y='dataset_name',hue='passed_curation',dodge=False,errorbar=None,ax=ax)
    ax.axvline(.001,color='black',linestyle='--');ax.set(title=title,xlabel='Original rounded score difference',ylabel='')
    if ax is axes[0]:ax.legend_.remove()
showplot('13_analog_gains')
""")
md("""
## 12. Five-learner successes versus zero-learner successes

The outcome comparison covers the original pool (11 unanimous successes versus 12 published zero-learner successes). The profile comparison covers fewer datasets because original-task profiles are missing for some candidates. Do not silently change the denominator. The published and reconstructed zero-learner sets differ by CarDekho, but their final failure decisions agree.
""")
code("""
extremes = inventory[inventory.number_of_passing_learners.isin([0,5])]
showtable(extremes[['dataset_name','task_type','released_collection','number_of_passing_learners']].sort_values(['number_of_passing_learners','dataset_name'],ascending=[False,True]))
showtable(comparable[comparable.number_of_passing_learners.isin([0,5])].groupby('number_of_passing_learners')[features+['sample_longest_column_mean_chars']].agg(['count','median']))
""")
md("""
## 13. What this changes about selecting an idea

**Confirmed:** 23/56 candidates pass historically. Domain alone does not separate outcomes. Among the 165 learner pairs belonging to rejected candidates, TAR-only failure occurs 79 times, joint-only failure 32 times, both 23 times, and both criteria pass 31 times. Stronger thresholds shrink acceptance: 23 at 0.001, 19 at 0.002, 15 at 0.005, and 9 at 0.01 (quorum three). The compatible-profile subset has different size/text distributions, but release and profile selection prevent a causal explanation.

**Hypotheses to test later:** task-specific language may leave room for TAR; repeated names or identifiers may behave differently from narratives; strong structured predictors or text-dominated tasks may leave too little room for fusion. None of these is a guarantee. Short names can pass, and long text can fail.

**Decision worksheet:** when you propose ideas, compare these questions before downloading or training:

- Is the intended target and every feature available at prediction time? Can entity/time leakage be prevented?
- What real information can structured fields add beyond text, and vice versa?
- Does the language contain distinctions relevant to the target that a generic encoder may miss?
- How many complete, matched examples remain after joining structured data, dated pre-outcome text, and targets? Count matched rows, not just the largest source table.
- Is one row an entity, an entity-season, or a matchup, and how will repeated entities and time be handled?
- Are the source, license, target definition, sample size, and evaluation splits reproducible?
- Can we first screen on development splits and then run the official five-learner/four-condition evaluation on a locked holdout?

The shared conversation treats NFL drafting and tennis as inspiration for combining expert qualitative assessments with statistics. They remain unselected options; their source availability and joint coverage have not been verified in this phase.

No new dataset, domain, or prediction target has been selected. No new benchmark experiments have run. The next discussion should use this evidence to compare your candidate ideas, not assume that recreating a successful domain will pass.

## Sources and limitations

- [Paper v1](https://arxiv.org/html/2605.10616v1), Appendix A.3 and C.4: definitions and 56-candidate published flags.
- [Pinned paper-version source](https://github.com/alanarazi7/MulTaBench/tree/3bb95079a6146ed35d45187e98bcad5b063273a1): historical exports, summary CSVs, explicit pool/release mapping, and acceptance implementation.
- [Current public releases](https://huggingface.co/multabench): metadata, cards, sizes, and bounded viewer previews; per-dataset revisions and URLs are in the local manifests.

Limitations: source-benchmark membership incomplete; original source downloads untested; 16 candidate profiles missing; six task changes; current HF typed conversions differ from historical CSVs; preview ordering and small sample bias; no global missingness, duplication, balance, or text-length statistics; two published/reproduced model-flag discrepancies; one known missing fold; selection bias; correlated fold splits; no causal or statistical-significance claims. Dataset names and outcomes are historical, not newly generated results. The recent decisions in the shared ChatGPT conversation were read after a direct retry, alongside the supplied context and both local PDFs. Prior conversation suggestions are not treated as verified dataset availability.
""")

GRAPH_INSIGHTS = {
    "01_coverage": """
### Insight — release membership is not the curation outcome

**What we see:** the core contains 20 historical successes. The extended collection contains 2 successes and 18 failures; the unreleased group contains 1 success and 15 failures. Therefore, comparing core with extended is not the same as comparing all passing with all failing candidates.

**For our decision:** use the historical outcome flags when studying passage. A dataset's release location is an availability clue, not evidence that its original task passed or failed. Profile coverage is selected, so later structure comparisons cannot explain the full pool.
""",
    "02_pass_counts": """
### Insight — passing is demanding, and success has different strengths

**What we see:** 23/56 candidates meet the three-learner quorum, while 33 do not. Only 11 pass all five learners. A few favorable learners are insufficient for the assignment.

**For our decision:** favor an idea with a clear reason for both modalities to contribute, rather than assuming that any table with a text column will pass. Later, check consistency across learners and aim for a meaningful margin instead of relying on one favorable result.
""",
    "03_published_matrix": """
### Insight — one learner's result does not represent the dataset

**What we see:** some datasets have unanimous success, while others depend on a particular combination of learners. Gray cells are unavailable evaluations, not measured failures. Wine Review and Spotify Genres reach the quorum through the three non-TabPFN learners despite the two unavailable evaluations.

**For our decision:** inspect learner-level evidence and the feasibility of the chosen target's class count. Do not select an idea because one preferred learner succeeds, or quietly reduce the required quorum when a model cannot run. This grid shows agreement, not the magnitude or statistical reliability of gains.
""",
    "04_failure_types": """
### Insight — TAR gain is the most common isolated bottleneck

**What we see:** among the 165 learner pairs belonging to rejected datasets, 79 fail only TAR gain, 32 fail only joint signal, and 23 fail both. The other 31 pairs pass both, but their datasets do not reach the three-learner quorum.

**For our decision:** it is not enough to explain why text and numbers should combine well. We also need a plausible reason that adapting text representations to the target will help. Task-specific terminology, negation, or contextual assessments are hypotheses worth checking—not evidence of guaranteed TAR gain. These are pair counts, not 79 failed datasets.
""",
    "05_joint_tar_gains": """
### Insight — fusion gains and adaptation gains are separate requirements

**What we see:** the scatter contains learners with positive joint gain but insufficient TAR gain, and learners with TAR gain but insufficient frozen joint gain. Only points satisfying both margins count toward acceptance. A positive value on just one axis is not enough.

**For our decision:** ask two separate questions: what does each modality add beyond the other, and what target-specific information might a frozen encoder miss? If development results fail, identify the relevant condition before changing the data. Classification and regression panels use different metrics; their raw gain sizes should not be ranked together.
""",
    "06_threshold_quorum": """
### Insight — boundary passes can be fragile

**What we see:** with a quorum of three, acceptance is 23/56 at the implementation's 0.001 margin, 19 at 0.002, 15 at 0.005, and 9 at 0.01. The paper's strict >0.001 wording and the reproduction code's >=0.001 comparison yield different counts after rounding.

**For our decision:** seek improvements that remain useful across folds and learners, not merely a tiny threshold crossing. Freeze the evaluation rule before final testing. This sensitivity analysis reuses historical means; it is neither a statistical significance test nor a new experiment.
""",
    "07_domain_task": """
### Insight — subject matter alone is a weak selection rule

**What we see:** broad domains contain both successes and failures, and their sample counts differ substantially. Original task types also have different observed passage fractions. These groups reflect a curated historical candidate pool, not a representative sample of all possible tasks.

**For our decision:** choose a concrete prediction problem and its information sources, rather than choosing 'products,' 'reviews,' or 'sports' from a pass-rate chart. Neither domain nor classification versus regression establishes why a candidate passes. Small groups and confounding prevent causal explanations.
""",
    "08_structure_outcomes": """
### Insight — more usable rows looks promising; fewer columns is not a rule

**What we see:** in this restricted subset of 21 passing and 13 failing profiles, median row counts are 18,843 versus 8,035. Median structured feature counts are 6 versus 7, and median text-column counts are 3 in both groups. The distributions overlap.

**For our decision:** check how many complete examples remain after joining structured records, text, and targets. More usable data may help, but this analysis establishes no minimum required size. Do not delete columns or demand a small schema just to resemble passing datasets. Release coverage, task differences, and other confounders limit these associations.
""",
    "09_core_extended": """
### Insight — collection differences are descriptive, not failure explanations

**What we see:** the released core and extended groups differ in their row and feature-count distributions. However, extended includes historical successes, and six extended releases change the original task.

**For our decision:** use this chart to understand the released collections' practical sizes and schemas. Do not interpret it as proving that large feature counts cause failure or that the core schema is an optimal recipe. The compatible pass/fail subset in the previous chart addresses a different question.
""",
    "10_text_lengths": """
### Insight — inspect the actual text columns, not just average length

**What we see:** passing compatible profiles have a longer richest text column at the median (121 versus 42.9 characters in the preview), but there is substantial overlap. Pooling short names with a narrative can obscure that narrative's length, which is why both summaries are shown.

**For our decision:** inspect whether the text supplies relevant evidence that structured features lack. Longer text may simply contain irrelevant material; short titles or job descriptions can still be useful. These measurements come from contiguous 50-row current-release previews, not representative full-dataset samples, so this is an association to investigate rather than a selection threshold.
""",
    "11_text_form_passage": """
### Insight — neither long text nor 'free-form' appearance guarantees success

**What we see:** text columns vary from mostly short strings to longer narratives, and passing-learner counts do not separate cleanly by the richest column's length. Length does not directly measure semantic usefulness or whether text behaves like a categorical identifier.

**For our decision:** read examples and explain what a generic encoder might miss about the target. Do not exclude an idea just because its text is short, or approve it just because its text is long. The short/long proxies and the following bins are descriptive choices, not validated acceptance rules.
""",
    "12_selected_dataset": """
### Insight — check within-dataset heterogeneity and the target definition

**What to inspect:** the left panel compares individual text columns, so one rich narrative need not make every text feature rich. The right panel shows the observed target values for this small preview. The initial saved example is Book Price; changing the selector changes the plots and tables.

**For our decision:** inspect individual fields and target preprocessing before interpreting a dataset-level average. Ask whether each field is available at prediction time and whether the target's units or transformations are documented. Fifty rows cannot establish full class balance, rare-event coverage, or the population target distribution. This illustrative existing dataset is not a selected new project.
""",
    "13_analog_gains": """
### Insight — similar subjects can fail for different reasons

**What we see:** Vancouver Salaries has positive median joint and TAR gains, while AI Job Salaries has positive median joint gain but negative median TAR gain. BikeWale Bikes shows the opposite pattern: positive median TAR gain but negative median joint gain. Ramen Ratings and Anime Planet combine modalities better at the median while their TAR gains are negative. These medians are descriptive and do not replace the learner-by-learner quorum rule.

**For our decision:** transfer the reasoning about complementary information and task-sensitive language, not the domain label. Sports remains open: FIFA22 Wages has 2 published passing learners and NBA Draft Value has 0. Neither is a direct test of pre-outcome scouting reports or tennis previews. Before selecting such an idea, verify matched row coverage, dated text, target leakage, and implementation effort. These are analog groups with different targets and metrics, not matched causal comparisons.
""",
}

def with_graph_insights(original_cells):
    """Split trailing table code so each interpretation directly follows its chart output."""
    annotated = []
    seen = []
    for cell in original_cells:
        if cell['cell_type'] != 'code':
            annotated.append(cell)
            continue
        source = cell['source']
        position = 0
        for match in re.finditer(r"^showplot\('([^']+)'\)\n?", source, re.M):
            name = match.group(1)
            chunk = source[position:match.end()].strip()
            annotated.append({**cell, 'source':chunk+'\n', 'outputs':[], 'execution_count':None})
            annotated.append({'cell_type':'markdown', 'metadata':{'research_insight_for':name},
                              'source':textwrap.dedent(GRAPH_INSIGHTS[name]).strip()+'\n'})
            seen.append(name)
            position = match.end()
        remainder = source[position:].strip()
        if remainder:
            annotated.append({**cell, 'source':remainder+'\n', 'outputs':[], 'execution_count':None})
    assert len(seen) == 13 and set(seen) == set(GRAPH_INSIGHTS)
    return annotated

def main():
    path = ROOT / "notebooks/01_multabench_dataset_analysis.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    annotated = with_graph_insights(cells)
    notebook = {"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.14"}},"cells":annotated}
    for index, cell in enumerate(annotated):cell['id']=f'cell-{index:03d}'
    path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False))
    print(f"Created {path.name}: {len(annotated)} cells, including 13 interpretations directly after graphs.")

if __name__ == "__main__":main()
