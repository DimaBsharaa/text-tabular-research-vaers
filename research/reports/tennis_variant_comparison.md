# Three tennis dataset constructions: evidence and recommendation

October 15, 2026. Notebook [04](../notebooks/04_tennis_dataset_prototypes.ipynb) displays real examples, counts and diagnostic plots. **Recommend A for the next bounded research gate; deprioritize B and keep C as a representation control. No final dataset is selected.**

**Follow-up completed:** [notebook 05](../notebooks/05_tennis_interview_validation.ipynb) and [the follow-up report](tennis_interview_followup.md) add publisher corroboration, a current Tennis-Data training-use restriction, a small pinned alternative-source probe and a random-split outcome-exposure audit. A is still provisional; read those findings before training or releasing data.

**Current A curation study:** the updated [text report](tennis_a_text_preprocessing.md) and [leakage report](tennis_a_leakage_validation.md) supersede treating missing upload timestamps as an automatic local-research blocker. An explicit three-day calendar assumption yields 3,627 rows with six passage variants and local training/development diagnostics; no joint gain is observed. The earlier A/B/C numbers below remain the original v0 results. No source permission or verified publication timestamp is invented.

## Changes to the proposed plan

1. Recover actual legacy metadata and compare ConvoKit with the original release before assuming dates are missing or trustworthy. Distinguish interview IDs from sporting-match IDs.
2. Use dated Tennis-Data records for this join rather than inventing exact days from Sackmann tournament dates. This source change is motivated by timing and source compatibility, not test performance.
3. For B, require an **actually published pre-target chart snapshot**, not just an earlier date in today's chart database. We found a small June 2015 snapshot and used it. We did not reinterpret recent retrospective charting as historically available.
4. Add a numerical-tactical control for B and a full-structured-context control for C. Splitting measurements between representations can manufacture an apparent information advantage unless the original feature set is also tested.
5. Inspect text detection on actual training pools. Full-data uniqueness is insufficient: a natural-language column can become categorical after splitting. Never append IDs or artificial variation to force the E5 detector.
6. Keep interview publication uncertainty explicit. A has real calendar-safe candidate rows, but has not passed the stricter availability gate; no A model was trained on a silent assumption.

## Measured constructions

| Property | A: previous player interviews | B: historical serve-placement descriptions | C: context verbalization |
| --- | --- | --- | --- |
| Sources | Cornell originals + ConvoKit; Tennis-Data | June 11, 2015 MCP Git snapshot; Tennis-Data | Tennis-Data |
| Constructed unique target matches | **4,052 provisional** | **415** | **44,359** |
| Strict interview publication-verified rows | **0** | Not an interview task; pre-target snapshot verified | Not an interview task; match-context availability assumed as specified |
| Players in constructed set | 288 | 64, men only | 1,556, men and women |
| Structured / text columns | 16 / 2 | 16 / 2 | 10 / 1 |
| Unique text values, by column | 2,616 / 2,637 | 110 / 106 | 342 |
| Median words per text column | 300 / 300 after fixed clipping | 47 / 47 | 23 |
| Duplicate occurrences beyond first, by text column | 35.4% / 34.9% | 73.5% / 74.5% | 99.23% |
| Missing text among retained rows | 0%; missingness is handled by eligibility, not hidden imputation | 0% | 0% |
| Target-date coverage | Jan 19, 2007–Nov 22, 2015 | Jun 12–Nov 22, 2015 | Dec 31, 2006–Nov 22, 2015 |
| Calendar cutoff | Prior reported day; interview age ≤90 days | Snapshot committed before target day; prior charts within 365 days | Start of target day; previous-day historical features |
| NLP character | Genuine interview answers | Deterministic rendering of extra tactical measurements | Deterministic rendering of ordinary context |
| Publication / redistribution limitation | Event dates verified against source metadata; online availability and redistribution remain unresolved | Historical Git timing established; MCP CC BY-NC-SA attribution required; combined release rights still need review | No explicit general redistribution grant established for source tables |
| E5/TAR training-pool detector | A not trained while timing gate unresolved | Fails on the measured chronological pilot | Fails on measured pilots despite passing globally |

Counts are actual outputs, not corpus-size projections. The initial 50-row examples, full schemas and pilots with text are saved **outside Git** under `../tennis-research-cache/derived/`. Tracked metadata samples retain match/player identities, source IDs and cutoffs without redistributing transcripts. A count is not a claim that every gate has passed.

The structured download contains 48,780 distinct rows across ATP/WTA 2007–2015 plus ATP 2016. The primary three constructions stop at 2015. Completed-match eligibility excludes walkovers/retirements, so this is a retrospective completed-match population; completion is not something known before an advertised future match. It should not be sold as a deployment guarantee for every scheduled match.

The original match dictionary explicitly warns that pre-2003 dates were tournament-start dates. This is another reason to keep the old early-2000 Kaggle smoke CSV as an engineering fixture rather than promote it to a historically verified forecasting dataset. We did not change that working smoke test; the new research uses the later dated source files.

## Shared schema and matching

The common label is outcome-independent `target = player_a_wins`. A fixed hash orients an unordered abbreviated player pair and match identity; the winner never enters that orientation. Both sides' rankings and histories follow the same orientation. Inputs exclude winner/loser role fields, current score, odds, source IDs and target-match performance statistics.

Common numerical predictors: each player's ranking, earlier match count, last-ten-match win rate, earlier surface win rate and rest days. General context is gender, surface, court, level, round and best-of format. Histories update **after an entire date**, preventing target and ambiguous same-day results from entering predictors. Missing rates/ranks remain missing for training-only preprocessing. Early-history incompleteness is visible, not repaired using future careers.

A/B keep the ten numerical and six context fields. C verbalizes the six context fields and removes them from structured input. That division is fixed before diagnostics. It is a representation allocation, not creation of new information. Dates/identities/source links remain sidecar audit fields.

## A: why it merits a bounded next gate

After link exclusions, prior interviews exist on both sides for 4,255 calendar-candidate target rows; restricting to completed targets leaves **4,052**. Across all 46,154 dated pre-2016 matches, the current eligibility audit finds **26,932 with no recent qualifying side, 14,967 with one side, 4,255 with two**. These counts precede the completed-target filter.

The two-sided A set has 2,232 men's and 1,820 women's matches. The median interview age is **19 days**; 90th percentiles are 66 and 65 days. One source document is reused at most nine times in a given A/B text column. Its full answer set is retained locally; a fixed first-300-answer-word policy makes the pilot compact. This clipping is not tokenizer-aware and may omit useful later material; it is a preliminary policy, not an optimized segmentation strategy.

A secondary player-centric audit finds **5,913 source documents linked to a next strictly later completed match, representing 5,067 distinct target matches**. It excludes ambiguous multiple matches on the next date and applies a 90-day limit. This alternative changes the target population and would need source-player role information; it must not silently replace the two-sided design or duplicate the same target from two perspectives. Publication verification is still zero, so it is not a shortcut around that gate.

Why it could work: interviews explicitly discuss preparation, condition, tactical adjustments and surface adaptation, which can be absent from rankings and historical win rates. Why Joint Signal might fail: much text restates results, is generic or off-topic, or describes a previous opponent. Why TAR might fail: long answers are truncated, profiles are reused under different opponent-dependent labels, and each text column is adapted separately rather than conditioned jointly on both players' statistics. Real language is a necessary opportunity, not a guarantee of gains.

What we learned: volume and language are substantially better than the old generated-history smoke fixture. Reliable publication availability and redistribution remain unresolved, and metadata agreement can hide transcript-match errors. **Priority 1: investigate those gates, not model tuning.**

## B: temporal construction works, but the current NLP case is weak

We used commit [`dc55cebd77fffc5f418540c47f9febda500fc342`](https://github.com/JeffSackmann/tennis_MatchChartingProject/tree/dc55cebd77fffc5f418540c47f9febda500fc342), committed **2015-06-11 16:38:31 UTC**, which contains **458 charted matches** in its serve-direction file. Current MCP coverage is much larger but cannot be substituted into a 2015 forecast without first-seen evidence. This fixes an important temporal weakness in the original proposal.

For targets starting June 12, we require each player to have at least two charted matches in the previous 365 days and at least 50 direction-coded serves. This yields **415 / 1,185 completed men's target matches (35.0%)** in the bounded period. It is heavily selected coverage, not an estimate for all ATP years or WTA.

Each sentence gives charted-match count, coded-serve denominator, and proportions wide/T/middle. All values trace to `1 Total` / `2 Total` rows in [`ServeDirection.csv`](https://github.com/JeffSackmann/tennis_MatchChartingProject/blob/dc55cebd77fffc5f418540c47f9febda500fc342/charting-m-stats-ServeDirection.csv). These are **placement frequencies, not success rates**. We did not invent injuries, mentality, forehand aggression, net approaches or rally tendencies. The richer `Overview` file was inspected for scope but not silently turned into unverified shot-decoder claims. Expanding to those constructs needs a separate dictionary/measurement audit.

A version-pinned 64 KiB point-file range returned **292 complete point rows from two matches**, with fields for first/second shot sequences, rally length and multiple point outcomes. Its 8 nonempty `Notes` are terse challenge annotations; 4 have at least five whitespace tokens but remain coded notes. Shot strings such as `6d` are not natural-language descriptions. This bounded, nonrepresentative prefix does not establish corpus-wide Notes prevalence or justify treating those notes as a tactical article collection. The 16.6 MB full point file was not downloaded.

The [MCP README](https://github.com/JeffSackmann/tennis_MatchChartingProject/blob/dc55cebd77fffc5f418540c47f9febda500fc342/README.md) requires attribution and noncommercial/share-alike treatment. Charted-match dates alone do not establish chart-publication dates; our fixed pre-target Git snapshot supplies a conservative availability bound.

Why it could work: the text conveys finer tactical variables absent from the compact general-strength table. Why Joint Signal might fail: sparse/star-biased coverage, stale summaries and correlated strength information. Why TAR might fail: repetitive templates, rounded measurements and too few unique profiles, with no linguistic ambiguity that obviously needs adaptation. Numerical controls give the downstream learner those same fine measurements directly.

What we learned: current CPU text has some standalone association but no joint gain; raw numeric tactics perform similarly to the verbalized joint condition. Actual training pools do not trigger official text encoding. **Priority 3: deprioritize this construction as the main NLP project.** It could be a tactical representation study, but that is not evidence it will meet Track 2.

## C: a useful control, not our recommended dataset

The sentence conveys gender, surface, indoor/outdoor context, event level, round and best-of format, with no made-up historical experience. These fields are removed from structured input. The remaining table retains genuine ranking/form information. A full-structured-context baseline restores the original fields to test whether verbalization helps compared with the same underlying data in tabular form.

The full population has 342 unique sentences, but small chronological training subsets have fewer than 100 and a low uniqueness ratio. They therefore fail the pinned semantic detector. Adding a row ID or meaningless text variation to force E5 would change the experiment artificially; we do not do it. A larger training pool might trigger E5, but does not create richer language or a stronger TAR hypothesis.

Why it could work: contextual categories can interact with player strength, and a pretrained encoder may encode meaningful category relationships. Why Joint Signal might fail: categories are weak alone and direct tabular encoding is already appropriate. Why TAR might fail: little lexical diversity, repeated templates and information content wholly determined by a handful of fields. The CPU diagnostics show no advantage here. **Priority 2 as a control only; reject the current C design as the main dataset direction.**

## Fairness, results and decision

B and C were compared on the **same 415 matches**, with 328 earlier training and 87 later test rows and the same cutoff. There are **104 shared calendar-candidate matches across A/B/C**, but no honest three-way model comparison because A's publication gate is unresolved. C also has a separate latest-1,500-row pilot; its score must not be compared with B as though construction were the only difference. [Pilot report](tennis_pilot_results.md) contains actual AUROCs, controls and limits. No E5 or TAR result is invented.

**Next direction:** pursue the interview idea only far enough to resolve rights and historical availability, with a fixed small validation audit of transcript identity. If that cannot be resolved promptly, abandon tennis for this deadline. Do not escalate B/C into expensive searches for a passing score. Preserve a development/final-evaluation boundary; we have not chosen representations by optimizing a final MulTaBench holdout.
