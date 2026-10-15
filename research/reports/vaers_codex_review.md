# Independent review of the VAERS candidate

October 16, 2026. Reviewed after safely fast-forwarding `main` from `45225eb` to `24014901b6080b11f5a04eed5d49086771639faa` (five GitHub upload commits). None of the nine incoming paths overlapped the seven locally changed/untracked files; their pre/post SHA-256 hashes match. The uploaded VAERS scripts, notebook, report and adapter changes were inspected without editing them. This new review is separate from the implementation.

**Assessment:** keep VAERS as a research candidate. Its direct report-ID join, substantial source coverage, useful patient/vaccine context and complementary clinical language are promising. The reported fold-0 numbers justify investigation, but do not establish qualification, clinical utility or absence of leakage. Tennis is dropped by project decision; archive it honestly as a negative preliminary study, not proof that every interview/E5 approach failed.

## Checks completed and limits

`python3 -m unittest research.scripts.vaers.test_build_vaers evaluation.test_smoke_infrastructure -v` passed **33 tests: 14 VAERS and 19 existing infrastructure tests**. This supports the ID-column adapter change and existing tennis smoke compatibility. It does not validate the real clinical records. All additional probes below were small synthetic in-memory/temp-file examples using the actual builder. No model fitting, dependency installation or large source download occurred.

No real VAERS CSV/ZIP or exported VAERS benchmark result was present in the project/cache or the inspected VAERS-named Downloads files. The supplied learner gains therefore remain **user-reported**, not independently reaggregated. Real-world leakage prevalence, retained-row counts, class support and exact code/data hashes for the running jobs remain unverified.

## Findings, in priority order

### 1. High: define report classification versus prospective prediction

The builder describes inputs as known when a report is filed, yet `VAERSSYMPTOMS` contains analyst-coded MedDRA terms derived from reported information. The [official guide](https://vaers.hhs.gov/docs/VAERSDataUseGuide_en_March2025.pdf) describes that coding process. A public processed record does not establish that every diagnosis/finding was known at symptom onset, before hospitalization or at the first submission.

The defensible current task is **retrospective classification of recorded report seriousness from a restricted set of report contents and metadata**. A claim about forecasting future hospitalization/death or triaging at first symptoms would need original-time inputs and outcome timelines not established here. Diagnoses and test findings are not automatically leakage for retrospective document classification; their validity depends on the stated information boundary. Direct restatements of the target can make that task trivial and should be audited separately.

Keep the target explicitly as a report having one of the selected recorded flags. Blank flags mean not reported as positive in the export, not adjudicated clinical absence. VAERS reports do not establish vaccine causation or population risk; the [official FAQ](https://vaers.hhs.gov/faq.html) explains these limits. The actual builder includes prolonged hospitalization and birth defects, which the prompt's shorthand omitted.

### 2. High: background columns are not screened for direct outcome statements

At [build_vaers.py](../scripts/vaers/build_vaers.py), `clean_text` only handles whitespace/null-like strings. The four background fields are passed through it at line 212. `merge_background` concatenates them; it adds no temporal or target-content validation.

**Confirmed synthetic probe:** adding `Hospitalized after this event; patient died.` to the HISTORY field of a synthetic positive report leaves that statement intact in both v1 `history` and v3 `patient_background`. This demonstrates an unguarded path, not that real records necessarily contain that exact sentence. The 14 existing tests do not cover it.

The symptom matcher also leaves the synthetic strings `Life-threatening illness` and `Congenital anomaly` intact. These are examples of filter coverage gaps, not measured frequencies or a claim that both exact strings occur in the supplied data. Examining only the 40 highest positive TF-IDF coefficients is a useful screen, not a comprehensive leak audit; rare markers, negative coefficients, synonyms and background text can be missed.

**Recommended next work:** audit real snippets from every text field, quantify direct-target statements by field/label/year, and distinguish historical hospitalization from hospitalization for the reported event. Do not blindly erase all mentions of hospital care: an old admission can be legitimate background. Preserve removal reasons and before/after examples in an audit sidecar. Do not add keyword rules repeatedly based on evaluation gains.

### 3. High: secondary-report deduplication is a heuristic, not an event identifier

Lines 183–188 identify duplicates from vaccination/onset dates, age, sex, state, first-listed vaccine type and lot. Distinct people can share those values, especially in a common vaccine batch; actual secondary reports may differ on a key or have missing values and escape the rule.

**Confirmed synthetic probe:** two deliberately distinct patients with different background descriptions but identical seven-key tuples were reduced from two reports to one. The synthetic fixture shows that the rule can merge distinct patients; it cannot estimate the real collision rate. Missing key values bypass this deduplication entirely.

This matters because CDC/FDA explicitly state that public files have included secondary reports of already-reported events since May 2025. See [the source announcement](https://vaers.hhs.gov/data.html). Unique report IDs and exact-row deduplication do not prove event independence. Inspect suspected clusters, contradictory labels and near-duplicate text; retain/quarantine uncertain groups with provenance rather than calling every collision a confirmed duplicate. Report residual cross-split similarity separately from legitimate repeated background descriptions.

### 4. Medium: full-cohort frequency grouping precedes the train/test split

`cap_categories` at line 154 chooses its most frequent values from all input reports before sampling/splitting. Held-out covariates can therefore change how training categories are represented.

**Confirmed probe:** training values `A,A,B,C` map to `A,A,B,OTHER` with `k=3`. Adding twelve held-out `C` values changes those same training rows to `A,A,OTHER,C`. No labels are used, so this is transductive preprocessing, not direct outcome leakage. It still differs from the strict train-only transformation policy used in our earlier audit.

Use a documented fixed grouping from an external code vocabulary, or train-fitted handling inside the supported preprocessing path. Do not patch official model implementations to resolve this. Verify actual training-pool modality detection. Rare-category pooling solely to get below a heuristic threshold should not replace a defensible semantic feature definition.

### 5. Medium: ZIP plus extracted CSV input doubles vaccine counts

The notebook builder's line 54 gathers both yearly ZIPs and matching CSVs. The loader concatenates all inputs; data report IDs are deduplicated, but vaccine rows are counted without first removing duplicate source ingestion.

**Confirmed probe:** a synthetic two-vaccine report gets `n_vaccines=2` from its ZIP alone and `n_vaccines=4` from that ZIP plus the same extracted three CSVs. Choose one representation per year or reject duplicate source content. This does not imply the current Colab folder contains both representations.

Also, first-listed vaccine is not independently established as a clinically primary vaccine. Preserve an explicit deterministic convention or encode the administered set without implying causation or vaccine priority. The [official file dictionary](https://vaers.hhs.gov/docs/VAERSDataUseGuide_en_September2021.pdf) documents multiple vaccine rows per report.

## Evaluation and course-rule corrections

Taking the supplied Eilam clarifications as the course requirements:

- Count Joint Signal passes and TAR Gain passes **separately** across learners. The two sets of at least three may differ. This supersedes our earlier same-three-learner interpretation. `evaluation/results.py:50` currently displays a per-learner conjunction; that can remain informative, but is insufficient as the course-level dataset decision. The dataset summary needs separate raw-gain counts.
- Preserve unrounded scores and gains. The CPU proxy rounds condition AUROCs before subtracting; it is unsuitable for tiny-margin course decisions. The official-result adapter keeps raw AUROCs, but its legacy/rounded flags are not the course rule.
- Confirmed official release commit: [`89a18541443852a27e2cea50eec44fd92bacaf83`](https://github.com/alanarazi7/MulTaBench/commit/89a18541443852a27e2cea50eec44fd92bacaf83). Its `benchmark.py` calls the registered-dataset evaluation route.
- Our adapter and notebook 02 still pin `3bb95079a6146ed35d45187e98bcad5b063273a1`. Calling unchanged official classes at that version is a screen, not the newly required final pipeline. `MODE="full"` in notebook 02 currently expands that older screening route; it does not register VAERS or satisfy the final `benchmark.py` requirement.
- Prepare the eventual final run in a **separate checkout at the required commit**, with only the necessary dataset loader/registration changes documented. Do not silently replace the reference under running screening jobs or change official model implementations. Compare dependencies/settings across versions before attributing score differences to the dataset.

The supplied LightGBM gains (+0.020 Joint, −0.007 TAR) and CatBoost gains (+0.012 Joint, +0.002 TAR) give two positive Joint point estimates and one positive TAR point estimate among completed learners. CatBoost's small positive TAR gain is encouraging, not evidence of stability. One fold alone does not establish a ±0.01 uncertainty bound. Inspect paired per-fold gains and preserve raw artifacts; do not pool v1 and v3 or different CSV/code/reference hashes. v3 also changes the number of E5/PCA feature blocks, not just the wording.

## Interpretation of notebook 01

The historical row-count table is numerically reproducible: the >20k group has 10 passing datasets out of 11 compatible profiles, versus 2/8 at ≤5k, 4/6 at 5–10k and 5/9 at 10–20k. This is a selected observational sample, not a 91% probability that a new large dataset passes. Classification task counts are 4/6 binary, 9/12 multiclass and 10/38 regression. The reported mixed-task Spearman correlation of text-minus-structure with TAR is approximately 0.544, but task/selection confounding prevents a causal interpretation.

The report's statement that missingness/structured-column counts hardly matter and that text must be at least as strong as structure goes beyond the evidence. Joint gain can exist whichever modality is stronger. Comparing tennis's local TF-IDF gap with historical official E5 gaps is also not a like-for-like experiment. Keep these as screening hypotheses, not dataset design laws. Do not remove useful symptoms solely to force a fusion gain; use that change as a transparent ablation or a separately justified task definition.

## Suggested next step

Let the current jobs finish on their frozen inputs; preserve their logs, schema/CSV/source/code hashes and raw scores. Before another large sweep, inspect a bounded real-data sample across all text fields, quantify filtering and duplicate/label-conflict rates, and decide the task's information boundary. Reports with no retained coded reaction after filtering are outside the chosen input scope; that does not prove they had no real reaction.

After reviewing those findings, agree on a versioned correction rather than editing the running v1/v3 build in place. Freeze a final candidate, text policy and evaluation plan before the required official-version run. Track selection over development results honestly. Keep the original uploaded files and the currently running experiments intact during this review.
