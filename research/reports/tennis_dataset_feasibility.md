# Tennis text-tabular dataset feasibility

**Investigated October 14, 2026. Recommendation: INVESTIGATE FURTHER, with a short feasibility gate; do not commit to tennis yet.**

The structured data is adequate for a serious candidate. Actual dated prose contains information about injuries, movement, tactics, surface suitability and risk-taking that could complement historical numbers. However, this audit has not established a sufficiently large, **redistributable and temporally valid** joined corpus. Neither Joint Signal nor TAR Gain has been tested. With 12 calendar days until October 26, text permissions and coverage—not acquiring more match statistics—are the decision bottleneck.

## 1. Audit boundaries and evidence levels

**Verified:** public file listings, data dictionaries, licenses/terms where available, bounded Kaggle file samples, two small annual ATP files, charting match metadata, selected actual articles, publication metadata, and a few manual match anchors.

**Hypotheses:** complementary predictive value, benefits of TAR, larger joined-row counts, GPU runtime on our future dataset, and whether the corpus can be assembled within the deadline.

**Not performed:** a full archive download, bulk text scraping, dataset construction, feature engineering at scale, new preprocessing pipeline, model fitting, fine-tuning, or pushing changes. Source inspection stayed in `/tmp/nlp-track2-audit/`; only this report is added to our repository. Annual-file and match-index inspection is a bounded source audit, not a curated tennis dataset.

No tennis dataset appears in our 56-original-candidate inventory. That is not an exhaustive novelty search across all published tennis research. A new text/statistics join must still be distinguished from an existing equivalent dataset before submission.

## 2. Structured source comparison

| Source | Counts and coverage verified or explicitly unknown | Actual inspected schema / identifiers | Pre-match usefulness | Availability and license |
| --- | --- | --- | --- | --- |
| [Kaggle `sijovm/atpdata`](https://www.kaggle.com/datasets/sijovm/atpdata) | Version **3**, updated **2023-01-31**. Files are named `atp_matches_till_2022.csv`, `atp_players_till_2022.csv`, `atp_rankings_till_2022.csv`. **Full match count and exact last event date unknown**; 2,844 complete prefix rows were inspected after partial ZIP decompression, with tournament dates in 1968. Filename/description suggest 1968–2022, but the 2022 maximum was not independently checked. | Match header has **49 columns**, including `tourney_id`, `match_num`, numeric winner/loser IDs, names, surface, level, seeds, entry codes, height/age/hand, ranks/points, outcome and match statistics. Composite event ID + match number is the natural match key. | Demographic fields, seed, event context and historically dated ranks may be usable after independent A/B orientation. Current-match serve/return statistics, score and duration are forbidden predictors. | Public listing and 64 KB range download succeeded. Match CSV is 40,104,093 bytes uncompressed; other files about 2.4 MB and 79.7 MB. Kaggle says **Other** license and credits Sackmann; original-license/provenance compliance must be established rather than treating “Other” as an open grant. |
| [Kaggle `dissfya/atp-tennis-2000-2023daily-pull`](https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull) | Version **1219**, updated **2026-10-14 04:49 UTC**. **60k+ is the publisher's claim; full row count was not counted.** Actual first/last 64 KB yielded 488/475 complete sample rows. Inspected dates start **2000-01-03** and reach **2026-10-06** in the tail. The update timestamp does not imply matches through October 14. | Actual CSV has **17 columns**: tournament, date, series, indoor/outdoor court, surface, round, best-of, `Player_1`, `Player_2`, `Winner`, paired ranks/points/odds and score. Names are abbreviated; there are no Sackmann player IDs or stable tournament/match IDs in the inspected header. | Convenient match-day dates, rankings, court and surface. Historical performance must be derived from prior results. Odds lack an observed timestamp and can absorb much of the predictive signal; initially exclude them pending provenance/timing justification. | Public single CSV, **9,165,924 bytes**. Range responses verified actual contents. Kaggle declares **CC0**; upstream provenance and authority for that declaration were not verified. Do not assume it settles all upstream data rights. |
| [Hugging Face `Aneeshers/tennis-sackmann-archive`](https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive) | Pinned revision **`8ac86f7c62889b5d5c4cfec4c6403edcb8c5d993`**, last modified **2026-06-25**. ATP main-draw annual files are listed from 1968 through 2026; qualifying/challenger, futures, doubles, rankings and player tables are separate. **Full archive match count unknown.** Actual `atp_matches_2024.csv`: **3,076 rows**. Actual `atp_matches_2026.csv`: **1,449 rows**, with maximum `tourney_date` **2026-05-25**; that is not an exact last-match timestamp. | Both annual files have the same **49-column** Sackmann schema, stable numeric player IDs and event/match keys. The 2024 keys are unique in the inspected file. Rankings are dated at/near the tournament's reference date. | Preferred initial backbone for IDs, historical statistics and a documented dictionary. Need a reliable match-day/date source for within-tournament text joins. Archive is not current through October. | Public, ungated, explicit **CC BY-NC-SA 4.0**, with original attribution, README and license retained. The general HF viewer currently fails with `ArrowInvalid: Failed to parse string: 'R' as ... double`; do not infer full-file types from its partial preview or blindly load all heterogeneous folders as one table. |
| [Jeff Sackmann Match Charting Project](https://github.com/JeffSackmann/tennis_MatchChartingProject) | Pinned tree **`1813a1309b7ed7ebf1c7e884b32bf675d00e4edf`**. Match indexes contain **7,566 men's rows / 7,565 distinct IDs** and **4,264 women's rows / 4,257 distinct IDs**: **11,830 rows / 11,822 distinct IDs** overall. Men's valid `Date` values span **1960-05-29–2026-05-21**; women's reach **2026-09-09**. The README's “over 5,000” is not an accurate current total. | Match metadata has 15 fields, including `match_id`, player names/hands, date, tournament, round, court, surface, best-of, time, umpire, contributor. No Sackmann numeric player IDs in that file. Shot-level codes and aggregate-stat tables are separate. | Prior-match tactical statistics could supplement history, but current-target charted data is post-match. Match IDs/date fields assist manual matching. **Chart creation/publication time is not supplied by match date.** | Public, **CC BY-NC-SA 4.0**. Match metadata files were inspected; the large point-by-point files were not downloaded. Charted matches are a selected volunteer sample, not all ATP matches. |

File-list and version evidence: [sijovm API](https://www.kaggle.com/api/v1/datasets/list/sijovm/atpdata), [daily API](https://www.kaggle.com/api/v1/datasets/list/dissfya/atp-tennis-2000-2023daily-pull), [HF pinned ATP tree](https://huggingface.co/api/datasets/Aneeshers/tennis-sackmann-archive/tree/8ac86f7c62889b5d5c4cfec4c6403edcb8c5d993/atp?recursive=false&expand=false&limit=1000), [MCP pinned tree](https://api.github.com/repos/JeffSackmann/tennis_MatchChartingProject/git/trees/1813a1309b7ed7ebf1c7e884b32bf675d00e4edf?recursive=1).

### Types and specific data-quality findings

CSV has no intrinsic typed schema. The following are **inferences from inspected rows**, not a guarantee for every season:

- Sackmann: string event IDs/names/codes; integer date codes and player IDs; numerical ranks, ages and performance totals; nullable seeds, heights and statistics often load as floats because of missing values. Dates need explicit `YYYYMMDD` parsing.
- Daily Kaggle: string names/context/date/score; numerical paired ranks, points and odds; integer best-of in the prefix. `-1` appears as a missing-value sentinel in early points/odds, not a legitimate negative measurement. It does not contain the detailed serve statistics or stable IDs of Sackmann's schema.
- MCP: date and nominally numerical metadata load as strings because the file contains malformed values. Its `Time` field is not a verified timezone-normalized target start timestamp.

The 2024 Sackmann sample has rank missingness of approximately **0.55% for winner ranks and 1.30% for loser ranks**, and `w_ace` missingness of **1.95%**. It contains 81 score strings with `RET` and 20 with `W/O`. These are source-audit measurements for this annual file only. Retirement/walkover handling changes the task population; define it once and report selection consequences rather than filtering based on favorable model results.

The men's MCP index duplicates `20240915-M-Davis_Cup_World_Group-RR-Botic_Van_De_Zandschulp-Matteo_Berrettini`: one row has `Date='RR'`, another has `20240915`. This demonstrates the need for validation; no cleaning was applied to construct a new dataset here.

The inspected sijovm match header **already contains ranking columns**, despite its description's note that rankings are in other CSVs. Its description also calls several serve totals percentages, while the original [Sackmann dictionary](https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive/blob/8ac86f7c62889b5d5c4cfec4c6403edcb8c5d993/atp/matches_data_dictionary.txt) describes integer match totals. Prefer actual headers and authoritative dictionaries to metadata paraphrases.

The original `JeffSackmann/tennis_atp` repository/API returned 404 during this audit. That establishes unavailability at the checked address, not the reason for it. The HF archive remains accessible. Explicitly pin the archival snapshot rather than assuming the original live repository can be cloned later.

## 3. Which structured fields existed before the target?

| Feature family | Timing assessment |
| --- | --- |
| Surface, event level, announced round and best-of, known indoor/outdoor setting | Plausibly known beforehand; validate event context and any mid-event changes. |
| Historical rankings/points, seed, entry status, age and handedness | Usable if the observation/update precedes the match; current live or retrospectively corrected values are not automatically historical. |
| Rolling win rate, surface win rate, opponent-adjusted strength / Elo | Must be calculated from completed **prior** matches only, with the target row excluded and update order documented. No whole-season aggregates. |
| Target-match aces, double faults, first-serve percentages, break-point totals, minutes, final score, winner/loser roles | Post-outcome information. Exclude from predictors. Prior-match versions may enter shifted historical features. |
| Betting odds | Potential pre-match forecast, but observation time/provider definition is absent in the inspected file. They also risk making text redundant. No assumption of safe timing. |
| Player/event IDs, source URL, article date, join flags | Keep in audit metadata initially. Do not let accidental IDs or label-bearing source fields become model features. |

**Critical date distinction:** Sackmann's `tourney_date` is usually a tournament reference/start-week date, not the date of each match. It is often a Monday near the start, so it should not be assumed to be a strict lower bound for every match without checking the event calendar. Sorting by `match_num` does not supply a reliable chronological order either. [Upstream documentation](https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive/blob/8ac86f7c62889b5d5c4cfec4c6403edcb8c5d993/atp/UPSTREAM_README.md).

For within-event text, use verified match dates/times. When exact timing is unavailable, exclude ambiguous same-day articles or use a separately verified conservative event cutoff. Do not invent match timestamps from `tourney_date`.

## 4. Natural-language source assessment

| Text source | Availability / timing / identification | Is it the desired language? | Reuse, coverage and bias |
| --- | --- | --- | --- |
| [ATP Tour news](https://www.atptour.com/en/news) | Historical dated report/preview URLs are indexed. Two October 1, 2024 Beijing reports were located; direct article fetches in the web reader returned 403. Names, event and opponent support matching. No stable public article-to-Sackmann ID mapping was found. | Yes, reports include prose and player quotations; richness varies. Indexed snippets alone do not establish the whole article's tactical detail or extractability. | No open prose license verified. The inspected [ATP terms](https://www.atptour.com/en/terms-and-conditions-app) prohibit systematic content retrieval without express permission. A production corpus is not cleared. Strong bias toward prominent players, main draws and later rounds; usable count unknown. |
| [Tennis.com](https://www.tennis.com/news) | Full text of selected 2024 articles was accessible, with visible publication dates, player names and tournament context. Article version history / exact timezone metadata was not verified. | Strong positive example: tactical comparisons, injury context, movement, fatigue/rest and matchup reasoning. Some previews also include explicit expert picks. | [Sinclair terms linked by Tennis.com](https://sbgi.net/terms-conditions/) allow personal display/printing but require permission for other reproduction/distribution. Public visibility is not a dataset redistribution grant. Archive accessible in examples; corpus count and permissions unknown; stars/major-event bias. |
| [Tennis Abstract / Heavy Topspin](https://www.tennisabstract.com/blog/) | Dated articles and a public WordPress metadata API. Verified **1,347 total posts**, earliest API post **2011-02-20**, and **51 posts in calendar 2024**. These totals mix ATP, WTA, general studies and other formats; they are not 1,347 eligible player reports. Names are identifiable; aliases still need mapping. | Several genuine tactical analyses, blending numerical evidence with explanations of how a player constructs points. These directly resemble the intended qualitative context. | No blanket open license for the prose was verified. The search-result Creative Commons reference on a podcast page refers to **background music**, not the tennis articles. The CC BY-NC-SA license on Sackmann's data cannot be transferred to his prose by assumption. Useful ATP coverage likely much smaller than all post counts. |
| MCP `Notes` | The [dictionary](https://github.com/JeffSackmann/tennis_MatchChartingProject/blob/1813a1309b7ed7ebf1c7e884b32bf675d00e4edf/data_dictionary.txt) describes a free-text point-note field. Point logs have match IDs, but no verified original-publication time for each note. No bulk point data was read, so meaningful-note frequency is unknown. | `1st` and `2nd` are structured shot codes, **not natural language**. Notes can describe challenges or individual points; they have not been shown to form substantial player-performance assessments. | Data license is documented. Quantity, quality, historical availability and player-level assignment remain unresolved. Do not treat 7,565 charted men's matches as 7,565 informative prose documents. |
| Live Tennis Abstract player/charting pages | Match tables and numeric analysis are available. They incorporate current accumulated information and were not verified as historical snapshots. | Mostly tables/codes, not automatically a textual assessment corpus. | Not an appropriate shortcut to dated historical prose. Verbalizing current statistics would not demonstrate the desired independent natural-language signal. |

Archive-count evidence: [2024 metadata endpoint](https://www.tennisabstract.com/blog/wp-json/wp/v2/posts?per_page=1&after=2024-01-01T00%3A00%3A00&before=2025-01-01T00%3A00%3A00&_fields=id,date,modified,link,title) returned `X-WP-Total: 51`; [ascending metadata endpoint](https://www.tennisabstract.com/blog/wp-json/wp/v2/posts?per_page=1&order=asc&_fields=id,date,modified,link,title) returned `X-WP-Total: 1347`. These are counts of published post records at retrieval, not estimates of usable joined matches. No text corpus was crawled.

## 5. Actual dated text examples and small mapping audit

Only short paraphrases are recorded below. Article bodies were not copied into our repository.

| Article and verified publication evidence | Relevant content / player mapping | Potential use and limitation |
| --- | --- | --- |
| [ATP: Sinner sets Alcaraz blockbuster in Beijing final](https://www.atptour.com/en/news/sinner-buyunchaokete-beijing-2024-final), **2024-10-01** in indexed publisher text | Sinner's completed semifinal against Buyunchaokete and discussion of his forthcoming Alcaraz matchup. | Candidate history for the **October 2 final**, never for the semifinal it reports. Direct body access failed; only indexed availability/date was verified. |
| [ATP: Alcaraz defeats Medvedev in Beijing](https://www.atptour.com/en/news/alcaraz-medvedev-beijing-2024-sf), **2024-10-01** in indexed publisher text | Alcaraz's completed semifinal against Medvedev. | Potential complementary player-side history for that same later final. No corpus permissions or full extraction validation. |
| [Tennis.com: Sinner or Alcaraz? Roland Garros semifinal preview](https://www.tennis.com/news/articles/jannik-sinner-carlos-alcaraz-preview-pick-semifinal-roland-garros-french-open), **2024-06-06**, Steve Tignor | Hip/forearm recovery, surface/style differences and net play; a separate section analyzes Zverev/Ruud and fatigue/rest. | Identifiable **two June 7 semifinals**. Names and event match existing rows. Pre-match expert picks are forecasts, not retrospectively observed labels; they may nevertheless make text-only prediction dominate. Redistribution not cleared. |
| [Tennis.com: Alcaraz and Sinner's ninth meeting](https://www.tennis.com/news/articles/carlos-alcaraz-jannik-sinner-roland-garros-french-open-semifinal), **2024-06-07** | Post-semifinal discussion of performance and physical adversity. | Forbidden for that June 7 target, but candidate Alcaraz history for the **June 9 final**; Zverev context could come from the earlier preview. Exact article version/provenance still needs checking. |
| [Heavy Topspin: Joao Fonseca and Damage Down the Line](https://www.tennisabstract.com/blog/2024/12/27/joao-fonseca-and-damage-down-the-line/), API publication/modification **2024-12-27 10:37:26 UTC** | Forehand power, precision and tactical interpretation, with cautions about opposition and charted sample selection. | Candidate profile for later matches, not for the NextGen performances discussed. Later eligible matches were not joined in this audit. Prose license not cleared. |
| [Heavy Topspin: Felix Auger-Aliassime on indoor hard courts](https://www.tennisabstract.com/blog/2026/02/11/what-does-felix-auger-aliassime-do-so-right-on-indoor-hard-courts/), API publication **2026-02-11 21:19:12 UTC**, modification **21:58:19 UTC** | Serve placement, plus-one play, rally selection and risk management. | Player is identifiable in the inspected 2026 file as `Felix Auger Aliassime`; later Dubai/Indian Wells event rows exist. Exact target-day alignment and the opponent's text were not completed. Prose license not cleared. |

The [Fonseca metadata](https://www.tennisabstract.com/blog/wp-json/wp/v2/posts?slug=joao-fonseca-and-damage-down-the-line&_fields=id,date,date_gmt,modified,modified_gmt,link,title) and [Auger-Aliassime metadata](https://www.tennisabstract.com/blog/wp-json/wp/v2/posts?slug=what-does-felix-auger-aliassime-do-so-right-on-indoor-hard-courts&_fields=id,date,date_gmt,modified,modified_gmt,link,title) independently establish creation/modification timestamps. Metadata cannot guarantee an unchanged historical body unless its version is verified. A current page with an old publication date and later edits must not be automatically treated as an original pre-match document.

### Manually verified target anchors

These are **four plausible target matches**, not four finalized dataset rows. Three use accessible Tennis.com examples; the fourth uses indexed ATP reports. No article corpus or training-ready CSV was constructed.

| Target | Sackmann event / match number | Match-day evidence from MCP | Why matching is possible |
| --- | --- | --- | --- |
| Alcaraz–Sinner, Roland Garros SF | `2024-520` / `400` | `20240607-M-Roland_Garros-SF-Carlos_Alcaraz-Jannik_Sinner`, Date `20240607` | June 6 preview names both players and the exact event/round. |
| Zverev–Ruud, Roland Garros SF | `2024-520` / `399` | `20240607-M-Roland_Garros-SF-Casper_Ruud-Alexander_Zverev`, Date `20240607` | The same preview's second section names this matchup. |
| Alcaraz–Zverev, Roland Garros F | `2024-520` / `401` | `20240609-M-Roland_Garros-F-Alexander_Zverev-Carlos_Alcaraz`, Date `20240609` | Earlier preview plus post-Alcaraz-semifinal prose can precede this later target. |
| Alcaraz–Sinner, Beijing F | `2024-0747` / `400` | `20241002-M-Beijing-F-Carlos_Alcaraz-Jannik_Sinner`, Date `20241002` | October 1 semifinal reports name the subsequent finalists. |

This audit also demonstrates a concrete timing trap: the Sackmann Roland Garros rows have `tourney_date=20240527`, and Beijing's final has `20240926`. Those are not June 7/9 or October 2. A join using the tournament reference date as the match date would wrongly reject legitimate prior-round articles and can misorder historical features.

The corresponding Sackmann player IDs are Alcaraz `207989`, Sinner `206173`, Zverev `100644`, and Ruud `134770`. Date + event + round + an **unordered player pair** maps these examples reliably. It does not demonstrate automatic matching for all abbreviated names or articles discussing many players. No gold-standard accuracy rate was estimated from this selected famous-player sample.

## 6. Proposed compact schema and temporal rules

**Schema proposal only; no dataset has been selected or generated.**

| Fields | Purpose / timing requirement |
| --- | --- |
| `match_id`, `player_a_id`, `player_b_id`, `match_start_utc` or audited match-day cutoff | Audit identity and strict as-of boundary. Names/IDs need not be learner inputs. |
| `surface`, `tournament_level`, `indoor`, `best_of` | Known match context; availability verified before the cutoff. |
| `a_rank`, `b_rank`, `a_rank_points`, `b_rank_points` | Ranking observation dated before the match. |
| `a_form_winrate`, `b_form_winrate`, `a_surface_winrate`, `b_surface_winrate` | Shifted histories with explicit windows and minimum-history handling. |
| `a_strength`, `b_strength` | Optional opponent-adjusted historical rating, updated only after earlier completed matches. |
| `player_a_text`, `player_b_text` | Informative player passages published before the cutoff; fixed selection/recency policy. |
| `player_a_wins` | Binary label computed **after** outcome-independent A/B assignment. |
| Sidecar: article URLs/IDs, author, publication/modification/as-of timestamps, license, passage hashes, source snapshot and confidence | Provenance, legal review and overlap auditing; excluded from predictors. |

### A/B assignment

Never rename winner to A and loser to B. Start from the unordered pair of stable player IDs and assign orientation by a reproducible hash/coin rule independent of the result, using a fixed seed chosen before inspecting labels. Swap **all** paired statistics and texts consistently, then derive `player_a_wins` from the actual outcome. Do not select the orientation seed to maximize class balance or model performance.

In the inspected daily Kaggle prefix, Player 1 wins 245 of 488 matches, so it is not simply a winner-first sample. However, the source's orientation rule was not established for the whole file; use our own audited assignment rather than assuming its slots satisfy every requirement. Do not manufacture two mirrored copies of each match and place them in different splits.

### Text/history rules

- `text_published_at < target_cutoff`, and any content version used must also be valid as of that cutoff. Where only dates are available, ambiguous same-day text cannot be accepted as strictly pre-match.
- A previous-match recap can predict a later match. The **target-match recap is excluded entirely**, even if winner names/scores are masked; it still describes how the target unfolded.
- Past outcomes appearing legitimately in prior reports are not automatically leakage. The task is to distinguish past evidence from current-target facts.
- Fix a recency window and passage selection rule on development data. Do not use the future outcome to select the most relevant article or paragraph.
- Every performance aggregate excludes the target row and later matches. Exact within-tournament ordering needs actual match timing; row order and `match_num` are not substitutes.
- Freeze article versions and audit repeated passages. Current season/career summary pages contain future information relative to many historical targets.

## 7. How many complete rows are realistically available?

**What is measured:** 3,076 main-draw rows in the inspected 2024 ATP file; 51 total blog posts in 2024 across topics/genders; six located prose examples; four plausible famous-player match anchors. **What is not measured:** the archive-wide proportion of matches with suitable text for both players, legal redistribution coverage, matching recall, or a final joined-row count.

At this stage there are **zero fully validated training-ready rows** with all of identity, historical version/timing, informative text on both sides, and redistribution permission established. This does not mean no such rows exist; it means they have not been cleared by this audit. Four hand-selected anchors are not an empirical estimate of population join coverage.

### Illustrative coverage scenarios, explicitly assumptions

If `M=3,076` annual matches and each player-side independently has suitable recent text with probability `p`, then complete two-sided text coverage is approximately `M × p²`:

| Assumed side coverage | Illustrative complete matches/year |
| ---: | ---: |
| 10% | 31 |
| 20% | 123 |
| 40% | 492 |
| 60% | 1,107 |

These are **not observed yields or forecasts**. Opponents' coverage is correlated: major semifinals can have both players covered, while early-round low-ranked matches have neither. Licensing, historical versions, article age and quality filters further reduce yield. One article can legitimately cover both sides, so article count is not a one-to-one bound on match count; repeated use also does not create independent textual evidence.

The 51-post 2024 blog count includes WTA/general posts and undermines any assumption of thousands of fresh ATP tactical reports per year from that source alone. Its 1,347-post multi-year archive is finite, and current permission for prose redistribution is unknown. ATP/Tennis.com may have greater coverage, but this audit does not support a numeric corpus-size claim for them.

A plausible **planning target**, not an official minimum, is roughly 1,500–3,000 unique matched matches with substantial fresh text coverage and both outcome classes across years. Hundreds may support a pilot, but cannot be assumed sufficient for stable five-learner gains or fine-tuning. A dataset with many rows but only a few repeatedly reused profiles can be small in effective information content. The official semantic detector may also classify highly repeated prose as categorical rather than text; the evaluation report explains its 100-distinct-value / 0.8-unique-ratio rule.

After review, a useful gate would sample 50–100 matches from predetermined event/round/year strata, **not from known available articles**. Record the fraction with (a) one-sided text, (b) two-sided informative text, (c) valid timestamps, (d) a valid historical version, (e) cleared redistribution, and (f) unambiguous player mapping. Extrapolate only within represented strata with uncertainty. The present famous-player examples are unsuitable for that extrapolation.

## 8. Suitability for MulTaBench

| Question | Assessment |
| --- | --- |
| Independently informative structured component? | **Strong hypothesis.** Historical rank, strength and surface-specific form plausibly predict match outcomes. The fields exist. No independent AUROC was measured here. |
| Complementary natural language? | **Plausible and supported by content examples, not performance evidence.** Injury recovery, movement, tactical suitability and fatigue can describe distinctions absent from our initial numerical schema. Many articles also restate ranks/H2H or provide a nearly complete expert forecast, which could instead make structured inputs redundant. |
| Potential TAR gain beyond frozen E5? | **Uncertain.** Tennis-specific negation, conditional injury statements and tactical language may need target adaptation. Frozen E5 may already capture much of this; sparse/repeated text and noisy match labels may leave little room for improvement. |
| Is enough data established? | **No.** The structured match pool is large; complete legally reusable text coverage has not been quantified. |
| Temporal construction possible? | **Yes in principle, with major audit work.** Requires actual match dates, dated historical prose versions, shifted histories and an appropriate evaluation protocol. The raw sources are not already leakage-safe. |
| Feasible before October 26? | **Conditional and currently high risk.** It becomes credible only if reusable text and broad coverage can be confirmed quickly and CUDA resources are available. A multi-site copyright-permission/crawling project would consume the remaining schedule. |

### A particular TAR concern in this architecture

The official encoder adaptation uses each text column separately with the row label; it does not directly condition the encoder loss on both players' structured statistics or both narratives jointly. A repeatedly reused player profile can appear with different win labels depending on opponent and surface. Column-role prefixes help distinguish A from B, and the downstream learner can model interactions, but this does not guarantee that fine-tuning has a clean signal. That makes testing TAR, rather than assuming it follows from interesting prose, especially important.

Pre-match expert predictions are not temporally leaked merely because they correctly name a future winner. However, they can make the text condition independently strong enough to eliminate Joint Signal. Restricting prose to actual performance/context rather than an explicit betting pick would need a fixed, motivated policy established before evaluation—not test-driven paragraph editing.

The historical sports analogs do not resolve this question: FIFA22 Wages passes only two learners and NBA Draft Value passes none in the published grid. They use different tasks/text and do not establish that a tennis scouting-style dataset would succeed or fail.

## 9. Principal risks and controls

| Risk | Practical implication |
| --- | --- |
| Text rights not cleared | URLs or a data license do not grant article redistribution. A URL-only manifest is not the requested free-text CSV deliverable. Obtain a suitable license/permission or use a different adequately documented text source before collection. No permission requests were sent in this phase. |
| Tournament date mistaken for match date | Can invalidate both text cutoffs and historical features. Resolve real dates or restrict to a genuinely verified conservative cutoff. |
| Future training labels in IID splits | Even leakage-safe row features do not make random holdout future forecasting. Keep required IID curation separate from chronological robustness; agree on the course protocol. |
| Famous-player/late-round selection | Articles favor stars and majors; matched rows may not represent the ATP population. Measure coverage by rank band, round, level, year and surface. Do not promise generalization to unseen lower-ranked players. |
| Reused profiles/passages | Apparent large N can hide a small number of texts and overlapping article groups. Audit passage counts, reuse and train/test overlap. |
| Abbreviated names and aliases | The daily source has no stable player IDs; erroneous joins can attach the wrong player/history. Use a reviewed ID crosswalk with event/date/round checks, not fuzzy name matching alone. |
| Retrospective charting/edited articles | Match date or original publication date alone does not show when the actual data/text became available. Require first-seen/version evidence. |
| Strong rankings/odds or rich previews dominate | One modality may explain most of the target. Do not weaken legitimate predictors after reading test results merely to force benchmark passage. |
| Sparse paragraphs / truncation / text detection | A column named “text” is not proof that the official encoder uses it. Inspect contents, unique passages, detection and the 512-token limit. |
| Installation and compute schedule | The unmodified full evaluation entails 100 logical runs, including 25 TAR fits; published medians suggest roughly 20 hours of frozen/TAR compute before other overhead. GPU availability and retries can dominate the deadline. |

## 10. Decision recommendation

**Investigate further; do not start large collection or choose tennis yet.** The positive finding is that the intended qualitative context exists and selected reports can be mapped to later matches. The unresolved evidence is whether enough of that context can be legally redistributed, dated reliably, and joined beyond a handful of famous-player matches.

If we proceed after review, spend at most approximately one day on a permissions-and-coverage gate. Require an explicit reusable prose basis, a representative small join audit, a reproducible date/ID strategy, and a realistic estimate of unique matches and unique passages. In parallel, confirm an available CUDA environment and the official/temporal evaluation distinction. If these remain unresolved, **reject tennis as the main project for this deadline**, while recognizing its longer-term research appeal.

The remaining schedule must cover integration, curation, development diagnostics, full evaluation, reproducibility and the four-page paper. It cannot safely be spent building a large statistics table while assuming the text problem will solve itself.

**Stopping point:** both investigations are for review. No final dataset, model run, large collection, or push has been authorized or performed in this phase, and no claim of passing MulTaBench is made.

## 11. Reproduction notes for the bounded audit

Kaggle samples were version-addressed HTTP ranges, with returned `206`/`Content-Range` verified, not a full dataset download. The sijovm endpoint returned a ZIP prefix; its first entry was partially decompressed to inspect only complete CSV records. The daily endpoint returned plain CSV prefixes/tail. A hard response cap was used if range support failed.

```bash
curl -fLsS --range 0-65535 --max-filesize 131072 'https://www.kaggle.com/api/v1/datasets/download/sijovm/atpdata/atp_matches_till_2022.csv?datasetVersionNumber=3'
curl -fLsS --range 0-65535 --max-filesize 131072 'https://www.kaggle.com/api/v1/datasets/download/dissfya/atp-tennis-2000-2023daily-pull/atp_tennis.csv?datasetVersionNumber=1219'
curl -fLsS --range -65536 --max-filesize 131072 'https://www.kaggle.com/api/v1/datasets/download/dissfya/atp-tennis-2000-2023daily-pull/atp_tennis.csv?datasetVersionNumber=1219'
```

For the limited annual-file inspection, use the pinned [2024 file](https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive/blob/8ac86f7c62889b5d5c4cfec4c6403edcb8c5d993/atp/atp_matches_2024.csv) (650,136 bytes) and [2026 file](https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive/blob/8ac86f7c62889b5d5c4cfec4c6403edcb8c5d993/atp/atp_matches_2026.csv) (300,795 bytes). MCP counts came from the pinned [men's metadata](https://github.com/JeffSackmann/tennis_MatchChartingProject/blob/1813a1309b7ed7ebf1c7e884b32bf675d00e4edf/charting-m-matches.csv) and [women's metadata](https://github.com/JeffSackmann/tennis_MatchChartingProject/blob/1813a1309b7ed7ebf1c7e884b32bf675d00e4edf/charting-w-matches.csv), not from the large point files. Infer dates with explicit formats and count distinct IDs separately from rows.

All counts and access observations refer to this audit date/snapshot. The two Kaggle sources' full row counts remain unknown; no metadata-derived estimate is presented as a measured full-file count.
