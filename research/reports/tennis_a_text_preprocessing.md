# Cornell A: text preprocessing comparison

October 15, 2026. See executed [notebook 05](../notebooks/05_tennis_interview_validation.ipynb) for actual short before/after examples, token counts and plots. This study reuses the original 4,052 provisional A targets; all six text comparisons use the **same 3,627 refined rows** and unchanged structured inputs. Text-bearing files stay in the external cache. No outreach, push or official benchmark edit occurred.

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

| policy | median_formatted_tokens | max_formatted_tokens | condition_retention | tactics_retention |
| --- | --- | --- | --- | --- |
| legacy300 | 389.0000 | 461 | 0.5890 | 0.8402 |
| token_head | 512.0000 | 512 | 0.6794 | 0.8942 |
| head_tail | 501.0000 | 512 | 0.6346 | 0.8543 |
| topical_answers | 182.0000 | 512 | 1.0000 | 1.0000 |
| topical_qa | 459.0000 | 512 | 1.0000 | 1.0000 |
| qa_blocks | 446.0000 | 512 | 0.9819 | 0.9853 |

The keyword extracts nearly maximize the dictionary they were designed around. This is not an independent success measure. The first-300-word control retains about 59% of the condition-cue documents; whole blocks about 98%. The token-head control uses more of the official budget without relying on keyword relevance.

## Actual before/after inspection

The executed notebook shows short, literal before/after excerpts from these training interviews. Full candidate passages and original Q&A remain locally inspectable; reports do not publish bulk transcripts.

- **672, Anna Chakvetadze, January 24, 2007:** the prefix discusses the opponent and aggressive play. Later answers contain a first-person account of shoulder strain and treatment. Extraction retrieves that additional context. It is self-report, not a verified medical diagnosis.
- **666, Kim Clijsters, January 25, 2007:** a retained injury statement concerns the previous opponent. Attributing it to Clijsters would be wrong. Question and answer context matter.
- **673, Serena Williams, January 23, 2007:** a figurative tiredness phrase is picked up by the cue detector. Another answer denies tiring during play. A binary keyword flag is not a fatigue annotation.

Other training examples inspected include Federer 665, Sharapova 667 and Hingis 668. They show preserved negation, statements about previous opponents, a blister explanation requiring question context, and historical injuries unrelated to the target. These six examples were purposively selected to expose extraction weaknesses; no population quality rate is inferred. The sentence selection rule was not silently revised to make those cases look better. The separately logged whole-block extension addresses context loss while retaining the original five-policy results.

Repeated documents also create legitimate target ambiguity: **579 training interview/column groups** occur with both future target labels. The same past account can precede a win against one opponent and a loss against another. This is not leakage; per-column TAR supervision may struggle because it does not see the opponent's full structured context while adapting that text. It is another reason not to promise TAR Gain from richer prose alone.

## Small local diagnostic: no joint gain found

Fixed chronological training/development rows: **2,162 / 716**. The **749 reserved-test rows were not scored**. One fixed CPU logistic-regression learner uses train-only median/scaling/one-hot/TF-IDF preprocessing and the same hyperparameters for all conditions. These are **TF-IDF diagnostics, not Joint Frozen E5 or official MulTaBench results**. No TAR score exists.

| policy | structured_only | text_tfidf | joint_tfidf | joint_gain |
| --- | --- | --- | --- | --- |
| head_tail | 0.7684 | 0.5992 | 0.7640 | -0.0043 |
| legacy300 | 0.7684 | 0.5801 | 0.7628 | -0.0056 |
| qa_blocks | 0.7684 | 0.5811 | 0.7586 | -0.0098 |
| token_head | 0.7684 | 0.6045 | 0.7660 | -0.0024 |
| topical_answers | 0.7684 | 0.5572 | 0.7615 | -0.0069 |
| topical_qa | 0.7684 | 0.5792 | 0.7594 | -0.0090 |

All joint gains are negative on this one development period. Higher keyword coverage did not produce higher predictive performance; token-head has the strongest text-only diagnostic here, but no policy is declared a passing dataset. The original primary topical-answer policy was frozen before these scores. Whole blocks were added following training-only qualitative review, separately frozen before their two fits; the same development set is reused, so these are exploratory comparisons. Two earlier debug fits aborted before scoring and are recorded separately from the 13 completed diagnostic conditions. No reserved-test-driven policy choice occurred.

**Recommendation:** retain token-head as the transparent baseline and whole blocks as the context-preserving curation candidate. Do not replace the baseline merely because keyword retention is high, and do not select a winner by repeatedly optimizing this development table. Before any E5/TAR study, freeze the intended input and keep the reserved test for a single final assessment. Existing evidence does not demonstrate Joint Signal or TAR Gain.

Reproduce with the existing cache: `python3 -m research.scripts.tennis.a_study`, then `a_context_blocks`, `a_leakage`, `a_additional_checks`, `a_review_outputs` in the same module namespace. The original five-policy freeze rejects changed selector/temporal source hashes; the saved artifact manifest also guards against input drift on rerun. Use an explicitly new study version for changes. Full row-level metadata and hashes are in `research/data/tennis_prototypes/a_v1_*`; the private CSVs are in `tennis-research-cache/derived/candidate_A_v1/`.
