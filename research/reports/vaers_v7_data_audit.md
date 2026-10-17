# VAERS v7: audit of the actual candidate and source records

October 17, 2026. **The recorded evaluation mechanics and source joins check out. The stronger claim that the retained text contains no care information does not.** Keep v7 as a development candidate; resolve the documented policy gaps before treating it as the final dataset. These findings do not establish that leakage explains the measured Joint Signal or TAR gains.

The supplied files are the exact **build-8 v7** screened previously: 60,000 rows, 4,717 serious reports (7.8617%), 12 structured and five text predictors. Both candidate fingerprints match the archived runs. All nine raw source-member fingerprints match the construction schema. The dataset, builder, official benchmark and completed model scores were left unchanged. No GPU models were trained and nothing was pushed.

## Evidence and verdicts

| Check | Result | Evidence and scope |
|---|---|---|
| Exact CSV/schema identity | PASS | CSV SHA-256 `9ffdfad8…`; schema `3294eed0…`, matching the archived configurations |
| Recorded evaluation boundaries | PASS | Corrected auditor: 168 PASS lines, zero mechanics/source-example FLAG lines across the 24 LightGBM/CatBoost runs; matches the earlier independent archive audit |
| Source alignment and label derivation | PASS | Every one of 60,000 candidate IDs maps uniquely to the supplied source; all labels agree with the six recorded seriousness flags |
| Reproducibility of inputs | PASS | All 17 predictors reproduce for all 60,000 rows: five text transformations and 12 structured fields; zero mismatches |
| Sampled original-text comparison | PASS, limited scope | 12 serious + 12 non-serious reports matched; original and retained text saved locally; source identity and extractor replay checked |
| No care information in retained onset text | **FAIL: written-policy conformance** | Confirmed retained descriptions of blood transfusion, surgery and medical leave; examples below. Existing regex counters nevertheless all equal zero |
| Background necessarily predates the event | **FAIL as a blanket claim** | Examples explicitly describe the reaction, subsequent testing or later care inside background fields |
| No repeated clinical event across train/test | NOT VERIFIED globally | Two bounded detectors found little/no cross-split overlap, with coverage and limitations below |
| Signal entirely explained by year/vaccine group | Not supported by these diagnostics | Text retains conditional association within adequately supported groups; most fine-grained groups are too small to estimate |
| Signal entirely explained by measured style | Not supported by these diagnostics | Text adds +0.0664 to +0.0871 above structure+seven style features in CPU proxies; unmeasured reporting effects remain possible |
| Availability at symptom onset | NOT VERIFIED | Narrative order is not a publication or availability timestamp. The sources support retrospective report classification, not prospective prediction at vaccination/onset |

Replaying the same code is a reproducibility check, not independent validation of the meaning or sufficiency of its rules. Likewise, zero mechanical flags is not an overall leakage-free verdict.

## Confirmed text problems

The schema's policy says that onset text stops at the first sentence describing care, treatment, testing or a clinician. Actual retained examples contradict that claim:

| Report ID | Label | Retained fragment | Why it matters |
|---|---:|---|---|
| `809322` | 1 | “hemolytic anemia requiring blood transfusion x 2” | Describes intervention, rather than only the initial experienced reaction |
| `820772` | 1 | “intussusception requiring surgery” | Care information survives the purported stop rule |
| `2554658` | 1 | “Shortness of breath that resulted in triple bypass surgery” | Explicit later treatment is retained |
| `2760881` | 1 | “I underwent surgery on my left foot” and “I was out on medical leave” | Later care and functional consequences appear in the selected opening sentences |
| `831877`, `current_illness` | 1 | “Child underwent surgery for ears 3x” | A field called current illness includes the later course after vaccination |
| `804620`, `allergies` | 0 | “allergy testing following this adverse reaction” | Even a non-serious example demonstrates that background columns are not uniformly pre-event |

These fragments were checked against their source records. They are **policy counterexamples**, not proof that each record's input logically determines its exact target. Surgery is not itself one of the six flag columns, and historical surgery can be legitimate background. A transfusion/care description can still be an indirect seriousness proxy, which is what the stated policy sought to exclude.

An independent scan finds surgery/surgical wording in **88 onset excerpts** (44 serious) and procedure-related wording in 20 (14 serious). These sets can overlap and include legitimate prior history or other false positives; they are not counts of confirmed leaking rows. The builder's existing outcome/care/chart/vital/capitalization counters are all zero because the retained words fall outside those regex definitions. Thus “zero care words” must be replaced with the narrower statement “zero matches to the implemented patterns.”

The extractor removes dates and keeps up to three opening sentences. It can retain a treatment described in sentence one, or months of evolution summarized in the first three sentences. It cannot establish availability at onset. A severe symptom or a named condition such as a stroke is not automatically leakage in a retrospective classification task; the relevant questions are the task's declared information boundary and whether the input restates an outcome or supplies downstream care information.

There are also **177 excerpts with no ASCII alphabetic word of at least three letters**, including `854775`: `- : 14:00.`. This is a narrow low-information review trigger, not a language detector or a proof that all 177 rows are unusable. Non-empty text alone does not guarantee a useful account of a reaction.

## Duplicate-event diagnostics

The excerpt/metadata detector searches shared age, sex, state, vaccine type, onset interval and report year, using token Jaccard ≥0.6. It finds **48 candidate clusters, 119 rows**, including one cluster with inconsistent labels. It evaluates 16,474 within-block pairs across 50,890 blocks; no large blocks were skipped in this actual dataset. Coarse metadata and generic short text can match unrelated people, while changed metadata or different descriptions can conceal repeated events.

| Fold | Candidate test members with a training counterpart | Joint CPU proxy AUROC, all test | AUROC excluding those members |
|---:|---:|---:|---:|
| 0 | 1 / 2,000 | 0.8605 | 0.8604 |
| 1 | 1 / 2,000 | 0.8445 | 0.8445 |
| 2 | 2 / 2,000 | 0.8294 | 0.8293 |

All removed candidates are non-serious. Reported changes round to 0.0000 or −0.0001; rounded endpoint scores need not show the same last digit as their separately rounded difference. This bounded detector does not suggest that its detected overlap explains the score, but it does not rule out other repeated events. The proxy was not refitted after exclusions; it was rescored on a changed test population. These are not adjusted official E5/TAR scores.

A separate source-backed detector uses complete vaccination date, onset date, age, sex, state, primary vaccine type and lot. **40,833 / 60,000 rows have all key fields**; 13 candidate clusters contain 26 rows. **None straddles outer train/test** in the three inspected folds. This check does not cover missing/changed keys, and shared keys are not a confirmed patient identifier.

## Conditional signal and reporting style

All following scores are small CPU TF-IDF/logistic diagnostics fit on each recorded outer-training set. Imputation, scaling, categorical encoding and TF-IDF fitting use training rows only. They do not retrain the five official learners or demonstrate official Joint Signal/TAR gains. Test folds overlap and are not pooled.

| Fold | Text-only proxy, all five fields | Excerpt-only TF-IDF | Seven style features only | Structure + style | Structure + style + excerpt |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.8468 | 0.8652 | 0.6468 | 0.8016 | 0.8680 |
| 1 | 0.8176 | 0.8338 | 0.6137 | 0.7826 | 0.8522 |
| 2 | 0.8046 | 0.8397 | 0.5912 | 0.7522 | 0.8393 |

The seven measured style features are length, sentence count, first-person references, third-party references, uppercase share, digit share and family references. They can encode population/content as well as style. Their moderate scores show some association; the extra excerpt signal shows that these particular features do not exhaust it. Unmeasured reporting style and residual care/clinical wording remain possible explanations of part of the extra signal.

These are random mixed-year screens. Training can contain later-year reports than a particular test report; the results estimate retrospective classification in that mixed population, not chronological future generalization. The source schema also declares label-free category grouping over the construction cohort. That dataset-definition step uses full-cohort frequencies, unlike the train-only fitted proxy transformations. It should be disclosed rather than described as strictly training-only preprocessing. Repeated inspection/selection among v7/v8/v9 and earlier builds makes these development results, not a fresh confirmatory test.

Within **2023 × COVID19**, the five-field text proxy AUROCs are **0.7629, 0.8231, 0.7844**, based on 49/41/40 serious reports among 204/199/186 test rows. Within 2024 × COVID19, only fold 0 meets the predefined support rule: 0.8693, 23 serious and 154 non-serious. Most other year × specific-vaccine groups have fewer than 20 examples of a class and are explicitly NOT_ESTIMATED. Signal also persists within each year and both broad COVID/non-COVID groups. These observations reject a simple account in which only between-year/vaccine differences produce all discrimination; they do not rule out care or reporting shortcuts within groups.

Background placeholders are common: history 35,559; current illness 50,454; medications 31,119; allergies 38,144. **64.7% of background cells**, or **51.8% of all five-column text passages**, are exactly `none reported`. Four of five fields being background is a different claim from 80% of passages being empty. Merging fields may help representations, but these diagnostics do not establish that an underfed encoder caused weak TAR gains.

## Decision and next step

The evidence is more reassuring about experiment mechanics and data alignment than before. It is also now concrete that the textual information boundary is imperfect. **Do not claim that the dataset is leakage-free, that background is invariably pre-event, or that the opening sentences are an onset-time forecast.**

Before another curation change, inspect the saved care-pattern contexts and specify one consistent retrospective task/policy. If retaining the current no-care onset policy, implement its missing obvious interventions in a separately versioned build, assess the changed cohort and compare with v7 as a development sensitivity study. Review background fields for post-event material under the same declared boundary. Do not strip informative symptoms merely because they correlate with seriousness, and do not keep adding words based on which changes improve benchmark scores.

Keep v7 and its results immutable. v8/v9 merge or resample the same extraction policy; those transformations do not themselves repair these semantic gaps. After a policy decision, freeze the candidate and separate further development comparisons from the final registered official evaluation. No policy/build changes have been made by this audit.

## Files and reproduction

- [Corrected audit tool](../scripts/vaers/run_audit.py) and [tests](../scripts/vaers/test_run_audit.py): mechanics, duplicate sensitivity, conditional/style proxies, matched source examples. Version `update15-corrected-2` also handles source IDs with leading zeros, checking normalization collisions.
- [All-row source auditor](../scripts/vaers/audit_source_rows.py) and [counterexample tests](../scripts/vaers/test_audit_source_rows.py): all 60,000 labels/predictors, source event keys, independent review patterns and text-quality counts.
- [Notebook 07](../notebooks/07_vaers_run_audit.ipynb): both audits, tests and visible summary tables when opened with these local changes available.
- [Compact evidence](vaers_v7_data_audit.json): counts, fingerprints and limitations; full medical-text contexts remain outside Git.

Validation: **65 tests passed**, including modified-label/background and wrong-source-fingerprint counterexamples. Notebook code cells parse, and the builder/runner remain unchanged. The earlier archive evidence remains a separate 894-check verification of saved run records.

Full local outputs are under `/Users/dimabshara/Desktop/NLP/vaers-research-cache/audits/`: `v7_build8_a12d3b1c04dbbdd7.md`, its `.examples.json` sidecar and `v7_source_rows.json`. The source files remain in `Downloads/v7/` and the three yearly ZIPs in `Downloads/`. The safely extracted 24-run archive is `/private/tmp/vaers-real-audit-a12d3b1c04dbbdd7`.

```sh
python3 -m research.scripts.vaers.run_audit /Users/dimabshara/Downloads/v7/vaers_v7.csv /Users/dimabshara/Downloads/v7/schema.json /private/tmp/vaers-real-audit-a12d3b1c04dbbdd7 --raw /Users/dimabshara/Downloads/2019VAERSData.zip /Users/dimabshara/Downloads/2023VAERSData.zip /Users/dimabshara/Downloads/2024VAERSData.zip --out /Users/dimabshara/Desktop/NLP/vaers-research-cache/audits/v7_build8_a12d3b1c04dbbdd7.md

python3 -m research.scripts.vaers.audit_source_rows /Users/dimabshara/Downloads/v7/vaers_v7.csv /Users/dimabshara/Downloads/v7/schema.json --raw /Users/dimabshara/Downloads/2019VAERSData.zip /Users/dimabshara/Downloads/2023VAERSData.zip /Users/dimabshara/Downloads/2024VAERSData.zip --results-dir /private/tmp/vaers-real-audit-a12d3b1c04dbbdd7 --out /Users/dimabshara/Desktop/NLP/vaers-research-cache/audits/v7_source_rows.json
```
