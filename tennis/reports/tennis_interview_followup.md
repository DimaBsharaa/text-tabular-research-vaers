# Tennis interviews: publication, reuse and evaluation follow-up

**Historical-stage report.** The later [text study](tennis_a_text_preprocessing.md) and [leakage validation](tennis_a_leakage_validation.md) supersede this stage for dataset counts and model status: 3,627 conditional v1 rows and 13 completed CPU condition fits. Publication remains unverified, but an explicit conservative calendar-delay assumption supports local investigation. No outreach is required or planned; draft messages remain unsent reference material. Notebook 05 now displays the current v1 study.

Investigated October 15, 2026. Open the executed [notebook 05](../notebooks/03_tennis_interview_validation.ipynb) for the tables and plots. This follows the A/B/C investigation and leaves notebook 02 and the official benchmark unchanged.

**Decision: keep A as a conditional research candidate; do not start the official sweep or publish its transcript CSV.** We found genuine publisher corroboration, a concrete current restriction on the structured source, and an additional evaluation risk. No new learner was fitted. The 4,052 rows remain provisional, with **zero rows independently cleared for historical transcript availability**.

## 1. Publisher corroboration is stronger, but availability remains unresolved

We made ten bounded publisher checks against nine Cornell interview records. Nine correct-page comparisons match the first substantial answer prefix; a deliberately wrong same-day Andy Murray interview does not. Across all answers with at least 20 words, the correct pages match 63/64 tested prefixes. The unmatched prefix is retained as a discrepancy, not silently repaired. This is a purposive diagnostic sample, not a corpus-wide accuracy estimate.

| Cornell interview | Player / source date | Publisher record | Finding |
| --- | --- | --- | --- |
| 0 | Andy Murray, 2015-06-21 | [110504](https://www.asapsports.com/show_interview.php?id=110504) | Final interview; 15/16 substantial answer prefixes match |
| 497 | Andy Murray, 2015-08-22 | [112348](https://www.asapsports.com/show_interview.php?id=112348) | 9/9 prefixes match |
| 854 | Francesca Schiavone, 2007-08-23 | [45075](https://www.asapsports.com/show_interview.php?id=45075) | 6/6; concrete self-report about ankle pain |
| 2557 | Marcos Baghdatis, 2010-01-23 | [61198](https://www.asapsports.com/show_interview.php?id=61198) | 5/5; shoulder limitation and retirement |
| 3406 | Mikhail Youzhny, 2011-02-22 | [69159](https://www.asapsports.com/show_interview.php?id=69159) | 5/5; preparation and conditions |
| 5733 | Kei Nishikori, 2014-11-01 | [104385](https://www.asapsports.com/show_interview.php?id=104385) | 4/4; fatigue, recovery and serve readiness |
| 76 | Caroline Wozniacki, 2015-01-12 | [105578](https://www.asapsports.com/show_interview.php?id=105578) | 6/6; wrist limitation on the backhand |
| 5019 | Serena Williams, 2013-02-20 | [86673](https://www.asapsports.com/show_interview.php?id=86673) | 9/9; **withdrawal interview**, not an interview after a played match |
| 1499 | Aleksandra Wozniak, 2008-07-30 | [51230](https://www.asapsports.com/show_interview.php?id=51230) | 4/4; publisher scoreline confirms a loss, contradicting the corpus win label; remains excluded |
| 0, negative control | Andy Murray, same date | [110501](https://www.asapsports.com/show_interview.php?id=110501) | Same player/day, different match; 0/16 prefixes match |

An explicit narrow repair of the publisher's observed `Â` plus nonbreaking-space encoding artifact was needed for comparison. The audit does not rewrite the transcript corpus or benchmark. Prefix agreement establishes content corroboration, not character-for-character identity or an independent validation of every sporting-match link. Player/date matching alone fails the same-day negative control.

ASAP's [company description](https://www.asapsports.com/about.php) says transcripts are distributed to media within minutes of interviews. Its [product description](https://www.asapsports.com/about-fast-scripts.php) describes rapid delivery, website display and a historical archive. This supports a **general timely-availability hypothesis**. It does not provide the original public upload time for each 2007–2015 page, establish which version existed then, or grant dataset redistribution rights.

None of the ten checked page responses exposes a recognized first-publication metadata tag. Their headings give event dates; HTTP `Date` is the current response time. The [Cornell README](https://www.cs.cornell.edu/~liye/tennis_README.txt) defines its date as the match day and warns of automated linking errors. Its July 2016 release date cannot certify availability before 2007–2015 target matches.

A bounded archive probe of Murray's exact `www` URL returned an empty CDX result through 2016; the separate availability endpoint returned HTTP 429. No usable pre-target capture was recovered. **This does not establish that no historical capture or timely publication exists.** We did not retry aggressively, infer a publication date from an interview ID, or label every row unavailable forever.

## 2. Current structured-source terms require a change or clarification

The [Tennis-Data homepage](https://www.tennis-data.co.uk/), retrieved successfully with curl on October 15, explicitly limits use to private individuals and excludes “commerical or data training products using automated bots/scrapers/AI.” The misspelling is in the source. Its page also carries an all-rights-reserved notice. The saved raw response and its hash are recorded in [the follow-up source manifest](../data/prototypes/interview_followup_sources.json).

This is a material change from our earlier status of merely finding no general redistribution grant. **Do not assume that free downloads authorize our ML training or public derived CSV.** Whether a specific academic project is permitted needs source clarification; this report does not make a legal determination about existing local research or historical terms. No additional Tennis-Data match files or training runs were requested during this follow-up. The earlier Kaggle fixture has its own source provenance and release terms, which this audit has not newly cleared; it remains unchanged.

The [Cornell project page](https://www.cs.cornell.edu/~liye/tennis.html) and [ConvoKit tennis documentation](https://convokit.cornell.edu/documentation/tennis.html) identify the sources and contacts but supply no explicit transcript redistribution license in the inspected material. A toolkit software license does not establish rights to third-party transcripts. A public download, or permission to quote an excerpt, is not represented here as a grant for a public training corpus.

Prepared [clarification drafts](tennis_source_clarification_drafts.md) cover publication evidence, academic model use and public dataset release separately. **No message was sent.** A metadata/URL-only release might reduce transcript redistribution concerns, but does not itself resolve training terms, historical availability or any course requirement to deliver a usable CSV.

## 3. A licensed structured alternative exists, with substantial coverage costs

Both original Sackmann ATP/WTA repository API endpoints returned **404** during this audit. This establishes current unavailability, not why the repositories are unavailable. We did not clone them or substitute a moving fork without recording provenance.

The previously identified [Hugging Face archive](https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive/tree/8ac86f7c62889b5d5c4cfec4c6403edcb8c5d993) remains accessible. We pinned revision `8ac86f7c62889b5d5c4cfec4c6403edcb8c5d993`, inspected its [license notice](https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive/blob/8ac86f7c62889b5d5c4cfec4c6403edcb8c5d993/LICENSE), preserved upstream READMEs and match dictionary, and fetched **only the 2015 ATP and WTA tour-level CSVs: 1,144,798 bytes combined**. The preserved author notices state **CC BY-NC-SA 4.0**, requiring attribution, noncommercial use and share-alike treatment. This is a substantially clearer reuse basis for those structured files; it does not license Cornell/ASAP text or independently authenticate every mirror record.

The dictionary says `tourney_date` is usually the Monday of tournament week. Therefore we counted source coverage under three **hypothetical cutoffs**: before that anchor, seven days earlier and fourteen days earlier. Interviews must be strictly earlier than the scenario cutoff and at most 90 days before the anchor; ambiguous latest-day documents are excluded. We use exact normalized full-name matches by gender and exclude known-conflict interview 1499. No Tennis-Data feature table is reused for this alternative probe.

| 2015 coverage scenario | Men | Women | Total |
| --- | ---: | ---: | ---: |
| Completed matches, both interviews before week anchor; eventual pairing assumed | 223 | 200 | **423** |
| Completed first-round matches, both interviews before week anchor | 26 | 28 | **54** |
| Completed first-round matches, both interviews seven days before anchor | 26 | 22 | **48** |
| Completed first-round matches, both interviews fourteen days before anchor | 21 | 21 | **42** |

These are measured coverage counts, **not cleared forecasting rows**. The retained source population excludes team events/finals and incomplete/retired/withdrawn matches. First round means the earliest `R<n>` round represented in the source event. Actual event-start and draw-publication dates still need verification; a week margin is not a substitute for that evidence. Later-round pairings are not generally known before a tournament, and a seven-day-early cutoff can also precede the first-round draw. An eventual-pairing conditional task must be described as such. We did not invent individual match days, build a replacement feature CSV, or multiply one year's counts into a claimed nine-year dataset size.

This fallback loses fresh within-tournament interviews and leaves relatively few first-round examples. That weakens its ability to test the original fatigue/condition hypothesis. It is a source option, not a ready fix.

## 4. Random splitting creates an additional outcome-exposure risk

We inspected the 4,052 provisional A rows without fitting models. These two **diagnostic** splits both use roughly 80/20 train/test; the random comparison is not an execution of MulTaBench's official holdouts.

| Measured property | Chronological | Stratified random, seed 42 |
| --- | ---: | ---: |
| Train / test rows | 3,239 / 813 | 3,241 / 811 |
| Latest training / earliest test day | 2014-06-18 / 2014-06-19 | 2015-11-22 / 2007-01-20 |
| Duplicate target match IDs across split | 0 | 0 |
| Shared interview documents, considering both player sides | 24 | 796 |
| Test rows using an interview also present in training | 64 / 813 (**7.9%**) | 666 / 811 (**82.1%**) |
| Held-out target matches referenced as interview-source matches in training | **0** | **242** |
| Training rows containing one of those source references | **0** | **538** |

The mechanism is concrete: an early match is held out as a target, but a later training example uses an interview given after that early match. That text may disclose its outcome. The saved [reference examples](../data/prototypes/interview_split_reference_examples.csv) identify the later training row, interview and earlier held-out match. Counting a source reference does not prove that every truncated answer explicitly states the result, nor measure how much a classifier exploits it. It establishes an outcome-exposure route that must be audited. Rank/form histories can also contain held-out results when future observations enter training.

Reusing a genuinely earlier interview is not automatically leakage for rolling future prediction. The chronological split allows that legitimate reuse while avoiding this reverse-time source-reference route. Later test rows may use earlier test-period outcomes only under an explicitly rolling prediction protocol. An unseen-player evaluation would be a different question; 102 players occur on both sides of the chronological split.

**Evaluation implication:** preserve their official implementation and label its random-holdout results accurately, but do not treat those scores alone as evidence of prospective tennis forecasting. A chronological supplement is necessary. Training-only preprocessing remains required, and random inner validation must not be presented as chronological validation. We have not modified their split code or fitted either comparison.

A fixed, model-free coverage control removes any candidate whose source interview match occurs anywhere in the original 4,052-target pool. It retains **1,265 rows** and eliminates that explicit source/target membership overlap. It does not solve rank/form dependencies, source rights, dates, duplicate documents or the limitations of random-time evaluation. No replacement dataset was selected or trained.

## 5. A publication-delay buffer has a measurable cost, not proof of safety

Requiring both source event dates to be at least 1, 2, 3, 7 or 14 calendar days before the target retains **4,052 / 3,245 / 2,444 / 2,056 / 1,756** original A candidates. These are sensitivity scenarios. None establishes an actual upload timestamp or turns the strict verified count above zero. A long fixed delay would also remove many fresh, potentially informative within-tournament observations.

## Recommendation and next decision

**Current direction after discussion:** continue local CSV/text/split inspection; no outreach. A separate private review workbench now exposes the existing A inputs, audit fields, chronological roles and training-only clipping audit. Its data are still provisional. Earlier clarification drafts remain unsent reference material and are not a prerequisite for continuing these local inspections.

The substantive language hypothesis remains credible. The current construction has three unresolved practical problems: historical publication evidence, source-use terms, and a source/target dependency that makes random scores hard to interpret as forecasting. We should resolve them before spending the A100 budget or selecting this as the final dataset.

The remaining source/evaluation questions could be resolved through published evidence, an alternative usable source, or clarification. No response or permission is assumed or fabricated. For now, the local review is sufficient to inspect the candidate and its preprocessing tradeoffs, without requiring the user to contact anyone. Given the October 26 deadline, avoid accumulating more unverified rows. If public evidence and a defensible construction cannot settle the requirements promptly, investigate a different domain with explicit dataset rights and a cleaner observation unit. No alternative domain has been selected here.

Reproduce the offline audit with `python3 -m tennis.scripts.interview_followup` using the cached inputs. The public follow-up manifest records the exact URLs, statuses, body sizes and hashes; raw pages and two small year files remain outside Git. Recorded new source bodies total about **2.13 MB**, plus headers and two tiny archive responses. New tests check date boundaries, freshness, ambiguity, cross-split source references, encoding and the same-day negative control. No model installation, training, official-source edit, commit or push occurred in this follow-up.
