# Research findings

## Confirmed observations

The original pool has 56 candidates: 23 published successes and 33 failures. Reproduction agrees with this accepted set, while two learner cells differ from the publication. There are 11 published five-learner successes and 12 published zero-learner successes. Release membership is not an outcome label: two historical successes are extended and one is unreleased.

Across the 165 original learner pairs of rejected candidates:

| learner_pair_outcome | n |
| --- | --- |
| TAR gain | 79 |
| joint signal | 32 |
| pass | 31 |
| both | 23 |

TAR-only failure is the most common learner-level failure category (79/165). This does not mean TAR fails for every learner of every rejected dataset. Joint-only failure occurs 32 times and simultaneous failure 23 times. The remaining 31 pairs pass both criteria, but the dataset lacks a three-learner quorum.

With three learners required, acceptance drops from 23 at a rounded minimum gain of 0.001 to 19 at 0.002, 15 at 0.005, and 9 at 0.01. These are descriptive margin checks, not tests of statistical significance. Applying the paper's strict >0.001 rule to rounded means accepts 19 rather than the reproduction code's 23.

Structure and text comparisons cover **21 passing and 13 failing** compatible released profiles, not all 23 and 33. Their medians are:

| passed_curation | n | historical_released_n_rows | historical_released_structured_features | historical_released_text_features | sample_mean_text_chars | sample_longest_column_mean_chars |
| --- | --- | --- | --- | --- | --- | --- |
| False | 13 | 8035.0 | 7.0 | 3.0 | 21.63 | 42.9 |
| True | 21 | 18843.0 | 6.0 | 3.0 | 55.06 | 121.0 |

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
