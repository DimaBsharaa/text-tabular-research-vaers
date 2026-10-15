# Cornell tennis interview audit

Investigated October 15, 2026. **The corpus has useful, real performance language and recoverable dates. It is not yet a cleared, temporally certified prediction dataset.** Notebook [03](../notebooks/03_tennis_interviews_eda.ipynb) contains executed outputs. Raw text stays outside Git.

## Sources and changes to the proposed approach

The [ConvoKit description](https://convokit.cornell.edu/documentation/tennis.html), [original project](https://www.cs.cornell.edu/~liye/tennis.html), and [original README](https://www.cs.cornell.edu/~liye/tennis_README.txt) describe different representations of the same interview collection. We downloaded the actual ConvoKit v3 archive (71,215,068 bytes) and original release (15,944,110 bytes), inspected their native JSON, and compared them. We did **not** install ConvoKit, SpaCy, Torch, or a language model locally. Direct inspection avoids a large software environment and makes legacy-field handling explicit. No claim is made that the current ConvoKit Python loader was runtime-tested.

The ConvoKit ZIP expands to approximately 663 MB, of which about 580 MB is parsed linguistic annotation. We extracted only the approximately 83 MB core JSON and left the parsed payload compressed. The original full-transcript JSON is approximately 44 MB. Archives and extracted text live in `../tennis-research-cache/`, outside the research Git repository. The source manifest records download URLs, byte counts and SHA-256 hashes; the complete bounded investigation downloaded approximately 104 MB including the structured sources and reader.

The prompt's strongest improvement is to inspect both releases. The original release explicitly describes `date` as the match day. The actual ConvoKit utterances also contain these date strings: they are not null or Unix timestamps. All 163,948 utterance dates agree with their original interview records. This establishes conversion consistency; it does not independently establish historical online publication times.

## What the hierarchy actually contains

| Level | Actual count | Interpretation |
| --- | ---: | --- |
| Users | 359 | 358 players plus `REPORTER` |
| Players | 358 | 191 men, 167 women |
| Utterances | 163,948 | Exactly 81,974 questions and 81,974 answers |
| Conversations | 81,974 | Each contains one question and its answer |
| Interview records | 6,467 | 3,633 men's / 2,834 women's interviews |
| Distinct sporting matches linked after exclusions | 5,031 | From 6,366 accepted interview links; not the count of all original sporting matches |

**The conversation metadata field `match_id` is an original interview-record key.** There are 6,467 distinct values, matching the original JSON keys exactly. Two players' interviews about the same sporting match can have different values. Deduplicating by this field would not deduplicate sporting matches.

The archive uses the legacy utterance keys `id`, `root`, `user`, `reply-to`, `timestamp`, `text`, and `meta`, rather than the current documentation's `speaker`/`conversation_id` spelling. An ID such as `1681_0.a` identifies answer 0 of interview 1681; its `root` is `1681_0.q`. The question/answer flags and `pair_idx` are in `meta`. Join `root` to `conversations.json`, then group by original interview key and sort by numerical pair index. Reconstructing this way gives **zero answer-text mismatches** against the original `QandA`, zero reply-link mismatches, and no conversation with other than two utterances.

Conversation fields are `match_id`, `opponent`, `result`, `stage`, `tournament`, `tournament_type`, and `player_ranking`. The original records add `player`, `gender`, `date`, and the complete `QandA`. Speaker gender is in `users.json`. Original metadata contains **940 literal string values `"nan"` for unknown opponents**; a null-only test misses them. One ranking is missing.

## Coverage, lengths and selection

Dates span **2007-01-15 to 2015-10-18**, with no missing original dates. Annual counts, player counts, stage distributions and tournament counts are saved as CSVs and plotted in notebook 03.

- Winning players account for **68.0% of source interviews**. That is an interview-selection bias, not the class balance of our outcome-independent A/B target.
- The ten most interviewed players account for **34.6% of interviews**.
- Grand Slams account for **43.6% of interviews**. The rest includes multiple ATP/WTA event levels. Interview counts alone do not quantify population coverage for every player or event.
- Median full player-answer length is **769 words**; the 10th/90th percentiles are approximately **281 / 1,484 words**. Around **71.6% exceed 512 words**. Word counts are not tokenizer counts; this nevertheless demonstrates that naive full-interview concatenation risks substantial E5 truncation.
- **7,810 / 81,974 answers (9.5%)** have at most five words. None are empty. A narrow exact boilerplate rule flags **1,749 (2.1%)**, not every uninformative answer.
- There are **zero exact duplicate full answer documents**, 3,239 repeated normalized short/long answer occurrences beyond their first occurrence, and **zero repeated normalized answer sections of at least 30 words** under the implemented rule. These are exact checks, not a claim that all near-duplicates have been eliminated.

Winning-interview bias, star concentration, uneven tournament coverage and transcript availability all affect the eventual target population. Mixing ATP/WTA also changes levels, match formats and population composition; results must remain stratified by gender and year before treating them as a single task.

## Human inspection: 36 complete player-answer sets

We read one short/medium interview from each year × gender × reported-outcome stratum: **36 interviews**, 2007–2015. The fixed sample requires 70–350 player-answer words and is deliberately convenient for inspection. It excludes most long interviews and **cannot estimate corpus-wide substantive-content prevalence**.

The auditable [review table](../data/tennis_prototypes/manual_content_review.csv) records our paraphrases, topics and an explicitly subjective specificity score. Of these 36, **22** contain relatively detailed context, **9** contain some context, and **5** are generic or off-topic. This is a human judgment of this purposive sample, not a validated annotation dataset.

Examples of potentially complementary content include Schiavone's ankle limitation (854), Gasquet's preparation/back/leg difficulties (1380), Baghdatis's shoulder-related stroke limitations (2557), Youzhny's compressed indoor-to-outdoor transition (3406), Nishikori's late-finish recovery problem (5733), Wozniacki's wrist limitation (76), and Bencic's restricted grass preparation (656). These are historical self-reports, not verified medical diagnoses. Ferrer's speculation about an opponent's condition (3303) is explicitly uncertain. Almagro's refusal to discuss a controversy (4799) and Santoro's retirement plans (1987) show why not all interview text supplies useful performance context.

Some short answers are meaningful only with the question: Nishikori's 2008 grass interview (1617) is a clear case. Our first prototype uses player answers only, retaining questions in the local source for review. A later development-stage comparison could prepend the question to genuinely dependent answers, with a fixed policy and the same historical cutoff. This must not turn into test-label-driven paragraph selection.

Corpus-wide keyword flags are also computed for tactics, serve/return, condition, mentality and surface. They are **unvalidated lexical flags**; mention of a term is not proof of useful, truthful, timely information or incremental predictive value.

## Joining and source-quality findings

We used the actual historical [Tennis-Data tables](https://www.tennis-data.co.uk/alldata.php), the same named match-data source used by the original Cornell work. This avoids guessing exact match days from Sackmann tournament-week dates. The [Tennis-Data dictionary](https://www.tennis-data.co.uk/notes.txt) says dates after 2002 are match dates, and rankings are as of tournament start. Its current index links historical Excel files; obsolete guessed `/2014/2014.csv` URLs return 404. Our downloader discovers the current links and hashes cached files.

Matching uses deterministic normalized surname/initial candidates, gender, exact day and round, plus opponent where available. It never uses the reported result to choose between candidate matches. The result is checked only afterwards. We preserve raw names, source file/row, candidates and exclusions in [the join audit](../data/tennis_prototypes/cornell_match_join_audit.csv).

Of 6,467 interviews: **6,366 accepted, 95 unmatched, 5 outcome conflicts excluded, and 1 further transcript conflict excluded after manual reading**. The accepted-link rate is **98.44%**. This is a recoverable-record rate, **not 98.44% independently verified accuracy**: Cornell's original links were themselves based on Tennis-Data and can carry the same erroneous date assignment.

The manual counterexample is interview **1499**: its answers explicitly discuss losing, despite source metadata marking a win. Even the deterministic date/round join initially returned a consistent winning result. This shows why metadata agreement alone does not validate transcript identity. We quarantined the record without guessing another date or outcome.

Two bounded publisher-content checks corroborate original IDs 0 and 497 against [Murray's June 21 final transcript](https://www.asapsports.com/show_interview.php?id=110504) and [August 22 transcript](https://www.asapsports.com/show_interview.php?id=112348). A deliberately wrong [same-day semifinal transcript](https://www.asapsports.com/show_interview.php?id=110501) does not match interview 0's opening answer. Several conferences can share a player and day; source IDs and content matter. These checks corroborate dated content, **not original online publication timestamps**.

## Gate status and what is still blocked

The original dates support strict previous-day calendar construction. They do not provide first-publication timestamps or historical page-version records. Accordingly the **4,052-row A CSV is provisional**, with `publication_verified=False` on every row. It is useful for coverage, provenance and examples, but **zero rows have passed the stricter publication-availability gate**. We did not quietly treat interview event dates as certified web-publication times, and did not train A on that unresolved assumption.

Neither downloaded archive contains an explicit transcript redistribution license; the tennis documentation and original README do not supply one. ConvoKit's software license is not a transcript license. Public availability of Cornell/ASAP copies does not by itself establish permission for our public free-text CSV. No raw interviews were added to Git, no permission request was sent, and no legal clearance is claimed.

**Recommendation:** A merits one focused next gate—confirm an acceptable historical-availability basis and a reusable text license with the corpus/provider or course staff. The data-volume and content findings are positive; rights and reliable transcript identity remain material. Do not choose it as the final dataset or launch the full benchmark until those issues are resolved.
