"""Render the A text/date/leakage investigation from actual saved evidence."""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tennis.scripts.a_study import STUDY
from tennis.scripts.common import OUT, PRIVATE, CACHE, ROOT, digest, write_json


def md_table(frame):
    def cell(value):
        if isinstance(value, float):
            value = f"{value:.4f}" if pd.notna(value) else "—"
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "| " + " | ".join(["---"]*len(frame.columns)) + " |"]
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def collect():
    summary = json.loads((OUT/'a_v1_summary.json').read_text())
    frame = pd.read_csv(STUDY/'metadata.csv')
    original = json.loads((CACHE/'cornell/transcripts_matchinfo.json').read_text())
    scores = pd.concat([pd.read_csv(OUT/'a_v1_development_scores.csv'),pd.read_csv(OUT/'a_v1_context_blocks_development_scores.csv')],ignore_index=True)
    structured = float(scores.loc[scores.condition=='structured','auroc'].iloc[0])
    comparison = scores[scores.condition!='structured'].pivot(index='policy',columns='condition',values='auroc').reset_index()
    comparison['structured_only'] = structured
    comparison['joint_gain'] = comparison.joint_tfidf - comparison[['text_tfidf','structured_only']].max(axis=1)
    comparison['TAR'] = 'not run'
    comparison.to_csv(OUT/'a_v1_development_comparison.csv',index=False)
    text = pd.read_csv(OUT/'a_v1_text_policy_summary.csv')
    context = pd.read_csv(OUT/'a_v1_context_blocks_metrics.csv')
    train_context = context[context.in_training_documents]
    row={'policy':'qa_blocks','training_documents':len(train_context),'median_formatted_tokens':float(train_context.formatted_tokens_a.median()),
         'max_formatted_tokens':int(context[['formatted_tokens_a','formatted_tokens_b']].max().max()),'legacy_over_limit_docs':0}
    for k in ['condition','tactics','preparation','mentality','surface']:
        row[k+'_retention']=float(train_context.loc[train_context[k+'_full'],k+'_selected'].mean())
    text=pd.concat([text,pd.DataFrame([row])],ignore_index=True)
    text.to_csv(OUT/'a_v1_all_text_policies.csv',index=False)
    selections={}
    ids={666,672,673}
    with gzip.open(STUDY/'passages_private.jsonl.gz','rt') as stream:
        for line in stream:
            item=json.loads(line)
            if item['interview_id'] in ids and item['policy'] in ['legacy300','topical_answers']:
                selections[(item['interview_id'],item['policy'])]=item
    notes={
        666: ('She hasn', 'The retained injury statement concerns the previous opponent. Extracted keywords do not establish the interviewed player has an injury.'),
        672: ('During the match I felt pain', 'Later answers describe the player’s own shoulder strain and treatment; the prefix omitted this context.'),
        673: ("I hate that situation", 'Tired of a situation is figurative. The Q&A context separately denies tiring during play; cue counts are not fatigue labels.'),
    }
    examples=[]
    for iid in sorted(ids):
        source=original[str(iid)];old=selections[(iid,'legacy300')]['text'];new=selections[(iid,'topical_answers')]['text']
        phrase,note=notes[iid];start=new.find(phrase)
        after=' '.join(new[max(0,start):].split()[:12])
        examples.append({'interview_id':iid,'player':source['player'],'source_date':source['date'],
            'before_first_10_words':' '.join(old.split()[:10]),'after_selected_12_words':after,
            'what_to_notice':note,'source':'Cornell original transcripts_matchinfo.json; Q&A indices retained in private passage lineage'})
    pd.DataFrame(examples).to_csv(STUDY/'before_after_examples_private.csv',index=False)
    # Public review notes contain paraphrases/IDs only. Real concise excerpts are
    # displayed once in the executed notebook, not replicated in the report.
    pd.DataFrame(examples).drop(columns=['before_first_10_words','after_selected_12_words']).to_csv(OUT/'a_v1_training_review_notes.csv',index=False)
    split=[]
    for name,g in frame.groupby('split',sort=False):
        split.append({'partition':name,'rows':len(g),'first_day':g.match_date.min(),'last_day':g.match_date.max()})
    pd.DataFrame(split).to_csv(OUT/'a_v1_split_summary.csv',index=False)
    next_links=pd.read_csv(OUT/'a_v1_next_match_links.csv').set_index('interview_id')
    is_next=[]
    for side in ['a','b']:
        is_next.append(frame[f'{side}_interview_id'].map(next_links.next_match_id).eq(frame.match_id))
    metadata={'original_rows':4052,'v1_rows':len(frame),'excluded_rows':4052-len(frame),
        'both_sources_immediate_previous_match':int((is_next[0]&is_next[1]).sum()),
        'at_least_one_source_immediate_previous_match':int((is_next[0]|is_next[1]).sum()),
        'median_interview_age_a_days':float(frame.a_calendar_age_days.median()),
        'median_interview_age_b_days':float(frame.b_calendar_age_days.median()),
        'completed_local_diagnostic_fits':len(scores),'superseded_debug_fits_before_scoring':2,
        'reserved_test_model_scores':0,'tokenizer_model_revision':'ffb93f3bd4047442299a41ebb6fa998a38507c52'}
    write_json(OUT/'a_v1_review_summary.json',metadata)
    return summary,frame,comparison,text,examples,metadata


def reports(summary,frame,comparison,text,examples,meta):
    audit=json.loads((OUT/'a_v1_leakage_summary.json').read_text())
    checks=pd.read_csv(OUT/'a_v1_leakage_checks.csv')
    sensitivity=pd.read_csv(OUT/'a_v1_date_sensitivity.csv')
    split=pd.read_csv(OUT/'a_v1_split_summary.csv')
    secondary=json.loads((OUT/'a_v1_secondary_diagnostics.json').read_text())
    test_path=OUT/'a_v1_automated_test_results.json'
    tests=json.loads(test_path.read_text()) if test_path.exists() else {'tests':0,'successful':False}
    text_view=text[['policy','median_formatted_tokens','max_formatted_tokens','condition_retention','tactics_retention']].copy()
    text_report=f'''# Cornell A: text preprocessing comparison

October 15, 2026. See executed [notebook 05](../notebooks/03_tennis_interview_validation.ipynb) for actual short before/after examples, token counts and plots. This study reuses the original 4,052 provisional A targets; all six text comparisons use the **same 3,627 refined rows** and unchanged structured inputs. Text-bearing files stay in the external cache. No outreach, push or official benchmark edit occurred.

## Exact official budget

The pinned paper-version code uses 512 tokens in both `encode_texts_with_e5` and `TextLabelDataset`. Its actual input is `passage: player_a_text: <text>` or the corresponding B column. Prefix and BERT special tokens consume **10 tokens**, leaving **502** for content with these names. This was measured with the actual `intfloat/e5-small-v2` tokenizer JSON, revision `ffb93f3bd4047442299a41ebb6fa998a38507c52`, rather than an estimated word/token conversion. Both column formats were checked on every passage. [Official encoder source](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/e5/e5_finetune.py), [formatting source](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/e5/constants.py), [tokenizer configuration](https://huggingface.co/intfloat/e5-small-v2/blob/ffb93f3bd4047442299a41ebb6fa998a38507c52/tokenizer_config.json).

Only the 0.71 MB tokenizer JSON/config and a checksum-verified 3.10 MB tokenizer wheel were used, extracted into an isolated external-cache vendor folder. No global package installation, Torch/Transformers stack, embedding weights, E5 inference or TAR training was used. The local tokenizer engine is 0.23.3; source assets, hashes and formatting checks are recorded. This measures tokenization, not official model scores.

## Compared inputs

1. **legacy300:** existing first 300 player-answer words, then enforce the true token limit. In this refined cohort none exceeded 512 formatted tokens; the issue is lost later content and unused budget, not an observed overflow.
2. **token_head:** take the original answer stream up to the actual formatted limit. It keeps opening context and more content but still omits later answers.
3. **head_tail:** retain beginning and ending answer spans with an explicit omission marker. It can recover a closing condition/preparation discussion and can miss the middle. Short interviews are not duplicated.
4. **topical_answers:** extract original answer sentences using fixed condition/tactics/preparation/mentality/surface cues and diversity, retaining source order and at most three sentences per answer. No labels, future opponent statistics or validation/test score enters selection. It can detach pronouns, subjects and qualifications from their question.
5. **topical_qa:** the same sentence method, adding the original question for short answers. This helps elliptical answers, but repeated questions consume budget and carry interviewer framing.
6. **qa_blocks:** a separately frozen training-review extension selecting whole answer blocks, with questions for answers under 80 words. Whole blocks better preserve qualifications/negation. Long blocks may be excluded; if none fits, a clearly recorded clipped-block fallback is used. Questions on long answers can still be omitted, so subject ambiguity is not universally solved.

All outputs are extractive. No LLM summary, invented injury, diagnostic medical label or multi-chunk embedding pooling is introduced. Each primary extraction retains original Q&A indexes, normalized answer offsets and source hashes. Whole blocks retain source pair indexes and question/answer hashes. The CSV-to-interview mapping is checked independently, including cases where the date rule selects an older document.

## Training-only coverage diagnostics

The following covers **2,011 unique training-source interviews**. Retention means that an existing lexical cue is still found, not that the passage is medically/tactically correct or useful. For QA variants cues can also occur in the retained question. Different documents and both genders are included; these are not prevalence estimates of injuries.

{md_table(text_view)}

The keyword extracts nearly maximize the dictionary they were designed around. This is not an independent success measure. The first-300-word control retains about 59% of the condition-cue documents; whole blocks about 98%. The token-head control uses more of the official budget without relying on keyword relevance.

## Actual before/after inspection

The executed notebook shows short, literal before/after excerpts from these training interviews. Full candidate passages and original Q&A remain locally inspectable; reports do not publish bulk transcripts.

- **672, Anna Chakvetadze, January 24, 2007:** the prefix discusses the opponent and aggressive play. Later answers contain a first-person account of shoulder strain and treatment. Extraction retrieves that additional context. It is self-report, not a verified medical diagnosis.
- **666, Kim Clijsters, January 25, 2007:** a retained injury statement concerns the previous opponent. Attributing it to Clijsters would be wrong. Question and answer context matter.
- **673, Serena Williams, January 23, 2007:** a figurative tiredness phrase is picked up by the cue detector. Another answer denies tiring during play. A binary keyword flag is not a fatigue annotation.

Other training examples inspected include Federer 665, Sharapova 667 and Hingis 668. They show preserved negation, statements about previous opponents, a blister explanation requiring question context, and historical injuries unrelated to the target. These six examples were purposively selected to expose extraction weaknesses; no population quality rate is inferred. The sentence selection rule was not silently revised to make those cases look better. The separately logged whole-block extension addresses context loss while retaining the original five-policy results.

Repeated documents also create legitimate target ambiguity: **{secondary['training_interview_side_groups_with_both_labels']} training interview/column groups** occur with both future target labels. The same past account can precede a win against one opponent and a loss against another. This is not leakage; per-column TAR supervision may struggle because it does not see the opponent's full structured context while adapting that text. It is another reason not to promise TAR Gain from richer prose alone.

## Small local diagnostic: no joint gain found

Fixed chronological training/development rows: **2,162 / 716**. The **749 reserved-test rows were not scored**. One fixed CPU logistic-regression learner uses train-only median/scaling/one-hot/TF-IDF preprocessing and the same hyperparameters for all conditions. These are **TF-IDF diagnostics, not Joint Frozen E5 or official MulTaBench results**. No TAR score exists.

{md_table(comparison[['policy','structured_only','text_tfidf','joint_tfidf','joint_gain']])}

All joint gains are negative on this one development period. Higher keyword coverage did not produce higher predictive performance; token-head has the strongest text-only diagnostic here, but no policy is declared a passing dataset. The original primary topical-answer policy was frozen before these scores. Whole blocks were added following training-only qualitative review, separately frozen before their two fits; the same development set is reused, so these are exploratory comparisons. Two earlier debug fits aborted before scoring and are recorded separately from the 13 completed diagnostic conditions. No reserved-test-driven policy choice occurred.

**Recommendation:** retain token-head as the transparent baseline and whole blocks as the context-preserving curation candidate. Do not replace the baseline merely because keyword retention is high, and do not select a winner by repeatedly optimizing this development table. Before any E5/TAR study, freeze the intended input and keep the reserved test for a single final assessment. Existing evidence does not demonstrate Joint Signal or TAR Gain.

Reproduce with the existing cache: `python3 -m tennis.scripts.a_study`, then `a_context_blocks`, `a_leakage`, `a_additional_checks`, `a_review_outputs` in the same module namespace. The original five-policy freeze rejects changed selector/temporal source hashes; the saved artifact manifest also guards against input drift on rerun. Use an explicitly new study version for changes. Full row-level metadata and hashes are in `tennis/data/prototypes/a_v1_*`; the private CSVs are in `tennis-research-cache/derived/candidate_A_v1/`.
'''
    (ROOT/'tennis/reports/tennis_a_text_preprocessing.md').write_text(text_report)
    check_view=checks[['check_id','scenario','status','tested','violations','scope']]
    leakage=f'''# Cornell A leakage validation

October 15, 2026. Evidence is in [a_v1_leakage_checks.csv](../data/prototypes/a_v1_leakage_checks.csv), its JSON counterpart, [notebook 05](../notebooks/03_tennis_interview_validation.ipynb), the mutation results and the private row-level provenance files. This covers the **entire original 4,052-row pool**, all 48,780 source match records, and the refined **3,627-row** conditional research candidate.

**Conclusion:** {audit['conditional_conclusion']} In particular, a three-day assumed delay is not a verified publication timestamp. No current-v1 check returned FAIL. Historical random-split and deliberately leaky negative-control failures remain visible. ASSUMPTION and NOT_RUN are not counted as verified PASS.

## Dates: yes, an interview can predict a later match

The original Cornell field is the source match day, not an unknown/missing date. Accepted deterministic links provide **6,366 interviews with the same date as their linked source record**; **6,319** follow Completed/Retired recorded events, while 47 attach to unplayed/other events. There are no ambiguous accepted same-player/day groups. Original A has 8,104 interview uses with source day strictly earlier than target day and different source/target match IDs. Nine publisher-content checks corroborate specific transcripts, while known corpus conflicts remain excluded. These checks support event association, not universal publisher-date accuracy or online first-publication time.

We can therefore associate a valid interview with the event just ended and examine the player's next recorded match. We explicitly distinguish that event-time relation from an upload-time claim. A withdrawal interview can contain useful condition information, but is not silently called a post-played-match interview in the current construction.

**Working conservative rule:** target day minus interview day must be **at least three calendar days and at most 90**. Assume that buffer is sufficient for online availability; select the most recent qualifying interview for each player, falling back to an older qualifying one. Exclude same-day/future sources, self-match sources, known bad/ambiguous links and source events without recorded play. The pairing and context are conditional inputs presumed announced by prediction time. Exact local timestamps and cross-timezone alignment are unresolved. This is an explicitly conditional retrospective experiment, not proof of a historical deployment.

{md_table(sensitivity[['lag_days','retained_rows','excluded_rows','rows_using_older_eligible_interview','rows_both_sources_are_immediate_previous_match']])}

The working three-day rule leaves **3,627** targets, removing **425** from the original pool; **1,219** retained targets use an older eligible document on at least one side. Median source ages are **{meta['median_interview_age_a_days']:.0f}/{meta['median_interview_age_b_days']:.0f} days** for A/B. A delay buffer is a policy sensitivity, never proof that publication occurred.

### Immediate next match versus latest available interview

The next-match audit does not skip an inconvenient retirement/walkover to find a later completed outcome. It resolves the first strictly later recorded day, rejects ambiguous multiple matches that day, and only then applies completion, age and delay eligibility. At three days it yields **2,166 interview-to-next-match links over 2,095 distinct target matches across the source collection**. That is a one-player document count, not a ready two-sided dataset.

Among our 3,627 retained two-sided rows, only **70** have both source interviews attached to each player's immediate previous recorded match; 385 have at least one. Most rows therefore use older interviews. The stricter rule preserves volume by accepting stale text, which may weaken fatigue/condition relevance. Calling every row a fresh previous-match interview would be wrong. A two-day sensitivity retains 3,857 rows and 439 two-sided immediate-next cases; it is a different declared assumption, not an automatically better dataset. We did not choose delay based on model AUROC.

## Structured features and target isolation

The match ID and A/B orientation are recomputed from date/event/round and an **unordered** player pair with a fixed hash seed. Winner status is used only after orientation to construct the label and attach each player's source rank. All 4,052 original orientations, labels and 8,104 rank assignments are checked against raw records. The actual feature whitelist contains ten historical numerical fields, six categorical/numerical context fields and two text fields. Target, match/source IDs, names, source files, scores, odds, completion flags and winner/loser roles are audit-only or excluded.

The dictionary defines entry rank at tournament start. All **97,560 source player-side observations** were also checked for within-player/tournament/year-source consistency: no varying entry-rank groups were found. This corroborates the source contract without independently dating every weekly ranking release.

Historical predictors are replayed in whole-day batches: a target sees only strictly earlier days. Every A/B history has a complete ordered source-ID list, the last-ten IDs and surface-specific IDs. The independent auditor reconstructs those memberships and values from raw outcomes. Original numeric features are also reproduced under their old convention before comparing the refinement.

**Definition correction:** walkovers and the single disqualified event no longer count as played matches or reset rest days. Completed and retired past matches do count, since both involved recorded play. This is a data-definition improvement, not discovery that every previous walkover was temporal leakage. The change affects at least one history field in 3,140 retained rows, largely through accumulated prior-match counts. Same-day past matches have deterministic hash ordering; true within-day order is unavailable, so a last-ten boundary through a tied past day remains an approximation. Current-day results are always excluded.

## Chronological and preprocessing boundaries

Original date cutoffs are frozen, so eligibility filtering does not move the final boundary:

{md_table(split)}

The test protocol is **rolling historical information with a fixed trained model**: an earlier test-period match outcome may inform a later test-period row after it is observed. It is not batch prediction of the entire future block at the start date. Interview documents can be reused when genuinely older than every use. Here **{audit['reuse']['shared_interview_ids_between_partitions']} interview IDs** cross partition boundaries, used in **{audit['reuse']['rows_using_shared_interviews']} rows**; every source date predates use. Same documents receive the same target-independent passage. These dependencies reduce effective sample size and do not represent unseen-player/document generalization.

There are zero exact full-answer duplicates across distinct used interview IDs. A fit-free hashed unigram/bigram nearest-training-document screen checks **{audit['reuse']['distinct_later_documents_screened']} distinct later documents**, excluding already shared IDs, and finds **{audit['reuse']['near_duplicate_pairs_at_0_95']}** pairs at cosine ≥0.95. This is a lexical screen, not proof that no semantic near-duplicate exists.

A secondary name-plus-score screen flags **{secondary['target_pair_plus_score_text_flags']} selected text cells** mentioning both sufficiently long target surnames and a score-like string. A flag can describe a previous encounter and is not automatically target leakage; no flagged cell uses the target itself as its source match. The screen excludes short/ambiguous surnames and does not exhaust semantic outcome detection. Its metadata are in `a_v1_target_reference_text_flags.csv`. The source/target tournament-name crosswalk is also saved; sponsor-name differences are not treated as mismatches without event review. Independent publisher-to-match gold validation remains a source-quality limitation.

All 13 completed local classifier conditions fit median/scaler/category/vocabulary/IDF/model parameters on the 2,162 training rows only. Fit row IDs and fitted-state hashes are saved; validation-only word/outlier injection leaves the state unchanged. A deliberately combined train/validation vocabulary fit exposes its sentinel and is correctly marked FAIL. Passage selectors use source Q&A and fixed rules, with no labels or future target statistics. Manual passage-policy review uses training interviews; whole blocks are a separately recorded exploratory extension. No reserved-test model score was computed. Corpus metadata/EDA had been inspected earlier, so the claim is an unscored final model holdout, not a never-viewed corpus.

## Check results

The saved automated boundary/adversarial test run contains **{tests['tests']} tests**, successful={tests['successful']}. Individual test IDs and PASS/FAIL results are in [a_v1_automated_test_results.json](../data/prototypes/a_v1_automated_test_results.json). These regression tests are separate from the 45 dataset/protocol audit entries below.

{md_table(check_view)}

`SPLIT03` retains the original random 80/20 diagnostic failure: interviews used by later training rows refer to **242 held-out target matches**. Such a reference is an outcome-exposure route; it does not prove every clipped answer explicitly reveals the result or quantify score inflation. The current chronological partitions have zero reverse-time source references. `PREP04` is an intentionally leaky negative control that the audit detects. The complete evidence/why-it-matters explanations are in the CSV/JSON, not only this summary table.

## What the checks do not establish

- Publication times and timezones remain assumed; the three-day delay does not certify them.
- Source match identity and entry-rank timing rely partly on documented source contracts. Automated joins share a source with the original corpus; high join coverage is not independent gold accuracy.
- Completion filtering defines a retrospective completed-match population. It does not prove that completion, later-round pairings or every context field was known for every scheduled match at cutoff.
- Modern pretrained E5 could have encountered historical results or transcripts. This study does not audit its pretraining corpus or establish a 2007-era deployable system.
- The official benchmark's random outer/inner holdouts remain unchanged. Official curation scores must be labeled separately from this chronological diagnostic; no E5/TAR evaluation was run here.
- Source-use/redistribution questions remain recorded separately. No email/contact step or public transcript release is part of this work.

The audit snapshots **149 reference/cache/working-infrastructure files** and verifies unchanged bytes afterward, including the official source cache and working notebook 02. New code resides in our research modules and optional temporal adapter. Input/output hashes and source/span lineage make the result reproducible and inspectable, without claiming cryptographic proof of execution history or universal absence of leakage.

Reproduce: `python3 -m tennis.scripts.a_leakage`; run boundary/adversarial checks with `python3 -m unittest tennis.scripts.test_research tennis.scripts.test_interview_followup tennis.scripts.test_a_validation -v`. The actual CSV/JSON evidence, fit IDs, negative controls and mutation outputs should accompany any later dataset claim.
'''
    (ROOT/'tennis/reports/tennis_a_leakage_validation.md').write_text(leakage)
    (STUDY/'README.md').write_text('''# Cornell A v1 — private conditional research candidate

3,627 rows from the original 4,052 targets. Three-day calendar-delay assumption; publication not verified.
Open the updated repository notebook 05 and the text/leakage reports for results and actual short examples.

- features_legacy300.csv, features_token_head.csv, features_head_tail.csv: controlled baselines.
- features_topical_answers.csv / schema.json: original pre-score-frozen research extraction.
- features_topical_qa.csv: sentence extraction with question context for short answers.
- features_qa_blocks.csv / schema_qa_blocks.json: separately frozen training-review whole-answer-block candidate.
- audit.csv: dates, player/source identities, assumed availability, chronological role. These are not predictors.
- splits.csv: 2,162 train / 716 development / 749 reserved test. Reserved test has not been model-scored.
- metadata.csv: independently rebuilt structured fields and source association, without stale legacy text.
- history_lineage.jsonl.gz: every contributor to each numerical history feature.
- passages_private.jsonl.gz / qa_blocks_private.jsonl.gz: source indexes, spans/hashes, selected text.

All six CSVs have the same targets and structured fields. match_id is audit-only and excluded from X.
No official E5/TAR score exists. None of the local TF-IDF joint conditions beats structured-only on development.
Do not call the dictionary flags verified injuries or the extraction a proven best passage policy.
These transcript-bearing files stay outside Git; nothing was pushed or sent to anyone.
''')


def notebook(summary,frame,comparison,text,examples,meta):
    from tennis.scripts.notebooks import code, markdown, save_execute
    cells=[markdown('''
        # Cornell A: text curation, conservative dates and leakage validation

        Executed October 15, 2026. Current study: original **4,052** targets → **3,627** conditional research rows.
        This reuses the existing interviews, structured sources and notebooks. No new large data, official-code edit,
        E5 model, TAR run, email or push. Actual local TF-IDF diagnostics are clearly distinguished below.
        [Leakage report](../reports/tennis_a_leakage_validation.md) · [Text report](../reports/tennis_a_text_preprocessing.md).
    '''),code('''
        import json
        from pathlib import Path
        import sys
        project=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'tennis/scripts').is_dir())
        if str(project) not in sys.path:sys.path.insert(0,str(project))
        import pandas as pd
        import matplotlib.pyplot as plt
        from IPython.display import display
        from tennis.scripts.common import OUT
        from tennis.scripts.a_study import STUDY
        review=json.loads((OUT/'a_v1_review_summary.json').read_text())
        audit=json.loads((OUT/'a_v1_leakage_summary.json').read_text())
        checks=pd.read_csv(OUT/'a_v1_leakage_checks.csv')
        display(pd.DataFrame([review]))
        print('Private candidate directory:',STUDY)
        print('Conditional date rule: >=3 calendar days; this is NOT verified publication.')
    '''),code('''
        dates=pd.read_csv(OUT/'a_v1_date_sensitivity.csv')
        display(dates)
        display(pd.read_csv(OUT/'a_v1_next_match_counts.csv'))
        fig,axes=plt.subplots(1,2,figsize=(11,3))
        axes[0].bar(dates.lag_days.astype(str),dates.retained_rows,color='#4f86a8')
        axes[0].set(title='Latest eligible interview on both sides',xlabel='Assumed minimum calendar-day gap',ylabel='Original-pool target rows')
        axes[1].bar(dates.lag_days.astype(str),dates.rows_both_sources_are_immediate_previous_match,color='#c47a55')
        axes[1].set(title='Both interviews concern immediate previous match',xlabel='Assumed minimum calendar-day gap',ylabel='Target rows')
        plt.tight_layout();plt.show()
        display(pd.read_csv(OUT/'a_v1_split_summary.csv'))
    '''),markdown('''
        **Insight:** a next-match association is possible. Exact upload times are not supplied, so a declared calendar
        buffer supports conditional research. Selecting an older eligible interview retains 3,627 rows under three days;
        simply dropping a too-recent latest interview would lose more. Only 70 retained rows have both source interviews
        attached to the immediate previous match. Most use older documents, weakening a fresh-fatigue interpretation.
        A one-player next-match link count is not a two-sided dataset size. The delay is an assumption, not proof.
    '''),code('''
        text=pd.read_csv(OUT/'a_v1_all_text_policies.csv')
        display(text)
        ax=text.set_index('policy').median_formatted_tokens.plot.bar(figsize=(8,3),color='#4f86a8',rot=25)
        ax.axhline(512,color='#b34d4d',linestyle='--',label='512 tokens including format/specials')
        ax.set(ylabel='Median formatted E5 input tokens',title='Actual tokenizer budget, unique training interviews')
        ax.legend();plt.tight_layout();plt.show()
    '''),markdown('''
        **Insight:** fixed first-300-word input underuses the available budget here; none of these legacy passages
        exceeded the true limit. The official format and special tokens consume 10 of 512 tokens. All six variants fit.
        Keyword retention is a mechanical dictionary measure, not a medical/tactical quality label or model score.
        Sentence extracts often retain keywords while losing their subject, negation context or time reference.
    '''),code('''
        examples=pd.read_csv(STUDY/'before_after_examples_private.csv')
        display(examples[['interview_id','player','source_date','before_first_10_words','after_selected_12_words','what_to_notice']])
        print('Source: original Cornell transcripts_matchinfo.json; exact source Q&A spans/hashes are retained privately.')
        display(pd.read_csv(OUT/'a_v1_training_review_notes.csv'))
    '''),markdown('''
        **What the examples show:** Chakvetadze's later answers add a first-person account of shoulder trouble absent
        from the prefix. Clijsters' retained injury statement concerns an opponent; Serena's tiredness phrase is figurative.
        These are purposefully selected training examples, not a representative prevalence audit. Whole-answer blocks were
        added as a separately frozen curation probe after inspecting training-context failures. No hypothetical injury
        description is substituted for an actual quote. The excerpts stay short; full candidate passages stay outside Git.
    '''),code('''
        comparison=pd.read_csv(OUT/'a_v1_development_comparison.csv')
        display(comparison)
        ax=comparison.set_index('policy').joint_gain.plot.bar(figsize=(8,3),color='#c47a55',rot=25)
        ax.axhline(0,color='black',linewidth=.7)
        ax.set(ylabel='Joint TF-IDF AUROC minus stronger unimodal AUROC',title='No local joint gain on this development period')
        plt.tight_layout();plt.show()
    '''),markdown('''
        **Insight:** structured-only AUROC is 0.7684; every joint TF-IDF condition is below it. There is no positive
        Joint Signal result here. These are 13 completed CPU logistic conditions on 2,162 train / 716 development rows,
        not official E5/TAR or a five-learner curation result. The 749 reserved-test rows were not scored. Two earlier
        debug fits aborted before scoring and are logged separately. High dictionary retention did not improve prediction.
        The original selected policy was frozen before scores; whole blocks are a documented exploratory extension.
    '''),code('''
        display(checks[['check_id','scenario','status','tested','violations','scope']])
        print('Audit summary:',audit['by_status'])
        print('Actual v1 FAIL checks:',audit['v1_actual_failures'])
        print('Retained legacy/negative-control FAIL checks:',audit['retained_legacy_or_negative_control_failures'])
        tests=json.loads((OUT/'a_v1_automated_test_results.json').read_text())
        print('Automated boundary/adversarial tests:',tests['tests'],'successful:',tests['successful'])
        display(pd.read_csv(OUT/'a_v1_history_examples.csv'))
        display(pd.DataFrame([json.loads((OUT/'a_v1_mutation_results.json').read_text())]))
    '''),markdown('''
        **Evidence:** the entire original target pool is checked for identity/orientation/labels/source dates; refined
        histories are independently reconstructed from all raw played-match records. Future/current-day result mutation
        and a future interview injection leave eligible predictors unchanged. The exported input text is compared with
        its actual selected source ID, which matters when using an older interview. All text variants keep identical
        structured values and targets. Statistical preprocessing/model fitting uses training row IDs only.

        FAIL is intentionally retained for the legacy random forecast interpretation and a deliberately leaky vocabulary
        fit. ASSUMPTION/NOT_RUN are not PASS. The JSON/CSV include why each check matters and its evidence reference.
    '''),code('''
        print('Reuse/similarity evidence:',audit['reuse'])
        display(pd.read_csv(OUT/'a_v1_reused_interview_rows.csv').head(12))
        display(pd.read_csv(OUT/'a_v1_near_duplicate_screen.csv'))
        from tennis.scripts.tennis_temporal_pilot import load_temporal_candidate
        _,schema,train,development,basis=load_temporal_candidate(STUDY/'features_qa_blocks.csv',STUDY/'schema_qa_blocks.json')
        print('Optional worker data checks accept the declared policy:',len(train),'train /',len(development),'development')
        print('Availability basis:',basis)
        print('Verified publication flag remains:',schema['strict_publication_temporal_gate_passed'])
        print('No GPU worker/model was executed by this check.')
    '''),markdown('''
        **Interpretation:** 42 interview IDs cross partitions legitimately because their source events predate every use.
        That dependence can reduce effective sample size without leaking a future event. Distinct documents are also
        screened for exact/lexical similarity; this does not prove absence of all semantic near-duplicates.

        **Remaining assumptions:** source-event linking and ranking contract; publication delay/timezones; pairing/context
        known by cutoff; retrospective completed-match population; possible modern-E5 pretraining contamination. A rolling
        test may use outcomes observed earlier in the test period, while keeping model parameters fixed. Their random
        inner validation remains random if official classes are later used; no benchmark code is changed here.

        **Recommendation:** retain token-head as a transparent baseline and context-preserving whole blocks as a curation
        candidate. Do not call an extractor best because it maximizes its own keywords. Current local results give no
        evidence of complementary predictive gain; leave the final test unscored and make any further development choice
        explicit. Source-use/redistribution status remains recorded; there is no outreach or automatic date-based blocker.
    ''')]
    save_execute('03_tennis_interview_validation',cells)


def manifest():
    from tennis.scripts.a_text import TOKENIZER_DIR,MODEL_REVISION,count_tokens,tokenizer
    active_tokenizer=tokenizer()
    import tokenizers
    import platform
    import sys
    from importlib.metadata import version
    write_json(OUT/'a_v1_environment.json',{'python':sys.version,'platform':platform.platform(),
        'packages':{name:version(name) for name in ['pandas','numpy','scikit-learn','scipy','matplotlib']},
        'tokenizers_isolated_vendor_version':tokenizers.__version__,'torch_models_loaded':False})
    assets=[]
    for filename in ['tokenizer.json','tokenizer_config.json','tokenizers.whl']:
        path=TOKENIZER_DIR/filename
        assets.append({'filename':filename,'bytes':path.stat().st_size,'sha256':digest(path.read_bytes())})
    write_json(OUT/'a_v1_tokenizer_manifest.json',{'model':'intfloat/e5-small-v2','model_revision':MODEL_REVISION,
        'asset_base_url':f'https://huggingface.co/intfloat/e5-small-v2/resolve/{MODEL_REVISION}/',
        'engine_version':tokenizers.__version__,'assets':assets,'formatted_empty_input_tokens':count_tokens(''),
        'model_weights_downloaded':False,'system_environment_installed_packages':False,
        'wheel_expected_sha256_from_pypi':'68649e97d5b43c44c031d8d848874a6eecae8f8fe40ea989aa777a5a83aca716',
        'runtime_probe_input':'passage: player_a_text: My shoulder feels fine. I did not feel tired.',
        'runtime_probe_ids':active_tokenizer.encode('passage: player_a_text: My shoulder feels fine. I did not feel tired.').ids})
    entries=[]
    for path in sorted(OUT.glob('a_v1_*')):
        if path.is_file() and path.name!='a_v1_artifact_manifest.json':
            entries.append({'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':digest(path.read_bytes())})
    private=[]
    for path in sorted(STUDY.iterdir()):
        if path.is_file():private.append({'filename':path.name,'bytes':path.stat().st_size,'sha256':digest(path.read_bytes())})
    inputs=[]
    for path in [PRIVATE/'variant_A.csv',PRIVATE/'structured_source_rows.csv',CACHE/'cornell/transcripts_matchinfo.json']:
        inputs.append({'filename':path.name,'bytes':path.stat().st_size,'sha256':digest(path.read_bytes())})
    implementation=[]
    for path in list((ROOT/'tennis/scripts').glob('a_*.py'))+[ROOT/'tennis/scripts/tennis_temporal_pilot.py',ROOT/'tennis/notebooks/03_tennis_interview_validation.ipynb',ROOT/'tennis/reports/tennis_a_text_preprocessing.md',ROOT/'tennis/reports/tennis_a_leakage_validation.md']:
        implementation.append({'path':str(path.relative_to(ROOT)),'sha256':digest(path.read_bytes())})
    write_json(OUT/'a_v1_artifact_manifest.json',{'inputs':inputs,'repo_outputs':entries,'private_outputs':private,'implementation_and_reports':implementation,
        'notes':'Full transcripts and selected passage CSVs are private. Hashes support reproducibility, not proof of first publication.'})


def main():
    if not (OUT/'a_v1_secondary_diagnostics.json').exists():
        from tennis.scripts.a_additional_checks import main as secondary
        secondary()
    values=collect();reports(*values);notebook(*values);manifest()
    print('Updated notebook 05, two reports, private README and artifact manifest')


if __name__=='__main__':main()
