# VAERS v8/v9: results and exact-data comparison

October 17, 2026. **V9 is the more consistent TAR development result on the two inspected learners. V8 is a representation tradeoff, not a clear overall improvement over v7.** Neither candidate has established the three-of-five requirement, and both retain the documented care-text policy gaps.

## Verified inputs

| Variant | Plan / result archive ID | Rows | Serious reports | Text fields |
|---|---|---:|---:|---|
| v7 reference | `a12d3b1c04dbbdd7` | 60,000 | 4,717 (7.8617%) | Four background fields + onset excerpt |
| v8 | `c7b5b71aeb656d52` | 60,000 | 4,717 (7.8617%) | Merged background + onset excerpt |
| v9 | `20b24baf903ae7ac` | 20,136 | 5,034 (25%) | Merged background + onset excerpt |

The supplied candidate CSV and schema bytes match every corresponding run configuration. Both archives have **24 successful runs**, exactly LightGBM/CatBoost × folds 0–2 × four conditions, and no recorded process errors. The independent comparison tool completed **378 named check instances per archive, 756 PASS and zero FAIL**. These are repeated mechanics/arithmetic/provenance checks, not independent proofs of absence of leakage.

Checks include the recorded train/test and encoder boundaries, condition matching, exact predictor whitelist, schema/CSV identity, exported CSV versus original result JSON, recomputed summary arithmetic, recorded pinned reference inventory/import hashes and log feature/passage counts. In two temporary negative controls, inserting a test ID into encoder training and changing an exported score were both detected. Original archives were unchanged.

The benchmark pin is `3bb95079a6146ed35d45187e98bcad5b063273a1`; the adapter fingerprint matches the prior v7 screen. V8 used an A100 and v9 an L4. These remain screens through the CSV adapter, not the final registered run at the required release commit. Recorded partition files are reconstructions, not independent tensor traces.

## Scores: means across the three folds

All numbers are AUROC. Joint Signal = mean(Joint Frozen) − max(mean(Structured), mean(Text)); TAR Gain = mean(Joint TAR) − mean(Joint Frozen).

| Variant | Learner | Structured | Text | Joint Frozen | Joint TAR | Joint Signal | TAR Gain |
|---|---|---:|---:|---:|---:|---:|---:|
| v7 | LightGBM | 0.773974 | 0.810996 | 0.847245 | 0.850326 | +0.036248 | +0.003081 |
| v8 | LightGBM | 0.773974 | 0.817842 | 0.842684 | 0.849787 | +0.024842 | +0.007103 |
| v9 | LightGBM | 0.796495 | 0.831913 | 0.860509 | 0.871277 | +0.028596 | +0.010768 |
| v7 | CatBoost | 0.800875 | 0.815876 | 0.853733 | 0.863339 | +0.037857 | +0.009606 |
| v8 | CatBoost | 0.800875 | 0.813316 | 0.853704 | 0.864255 | +0.040388 | +0.010551 |
| v9 | CatBoost | 0.805825 | 0.841192 | 0.867786 | 0.877664 | +0.026593 | +0.009878 |

All four v8/v9 learner means have positive Joint Signal and TAR Gain. This establishes positive descriptive screening means for **two learners**, not acceptance on three of five.

### V8 compared with v7: a matched comparison

V7 and v8 have identical ordered report IDs, labels, structured fields and onset excerpts. Every merged v8 background exactly matches the builder's merge of v7's four background fields. Recorded outer train/test IDs match in all three folds. Structured-only scores are identical, as expected.

- **LightGBM:** the TAR gap grows by **+0.004022**, but final Joint TAR falls by **−0.000538**, while Frozen falls by **−0.004561**. Thus the larger gap does not demonstrate a better final predictor. Joint Signal also falls by −0.011406.
- **CatBoost:** final Joint TAR rises by **+0.000916**; Frozen is virtually unchanged (−0.000029), and the TAR gap rises by +0.000945. This is a small descriptive change, with no paired uncertainty estimate establishing a reliable advantage. Joint Signal rises by +0.002531, partly because text-only weakens.

Merging changes passage weighting, context, per-field token budgets and PCA dimensions: five text blocks become two, giving 60 text components and 72 joint features instead of 150/162. It is a legitimate construction comparison, but does not isolate a single encoder-underfeeding hypothesis.

### V9: more consistent TAR, different population

V9's final TAR AUROCs are higher than the other variants on its own test sets. Both learner means show about **+0.010 TAR Gain**. However, CatBoost's Joint Signal is smaller than v7/v8, and its TAR gain is slightly smaller than v8's. There is no variant that dominates all requested gaps.

V9 contains **19,819 rows shared with v8 plus 317 additional rows**; all fields match exactly on the common rows. The additional rows account for the extra 317 serious reports. This confirms v9 is not simply a subset of the particular 60,000-row v8 sample.

Each v8 outer-training set contains **785 serious reports among 9,998**, versus **2,499 among 9,998 for v9**. Test sets contain **158 versus 500 serious reports among 2,000**. The greater positive training support plausibly helps learning. This comparison changes both training and test populations, so it does not isolate a training-balance effect.

For a fixed scoring rule on unchanged class-conditional distributions, prevalence alone does not mechanically raise AUROC. Here the training distribution, selected rows and fitted scoring rules change, and the test rows differ. The observed higher scores cannot be described as a clean improvement on the same test population. The 25% rate is an intentionally case-enriched benchmark cohort, not a population adverse-event risk estimate.

## Fold-level TAR Gain

| Variant | Learner | Fold 0 | Fold 1 | Fold 2 | Positive folds |
|---|---|---:|---:|---:|---:|
| v8 | LightGBM | +0.017761 | +0.004446 | −0.000897 | 2/3 |
| v8 | CatBoost | +0.013449 | −0.003161 | +0.021365 | 2/3 |
| v9 | LightGBM | +0.017489 | +0.008849 | +0.005965 | 3/3 |
| v9 | CatBoost | +0.005645 | +0.013157 | +0.010831 | 3/3 |

Every inspected per-fold Joint Signal is positive. V8 CatBoost fold 2 has Structured slightly ahead of Text; therefore averaging per-fold Joint Signal is **not** the same as the reported difference of condition means. The comparison uses the latter consistently.

V8's test sets share 65/71/84 report IDs across fold pairs 0–1/0–2/1–2. V9's share 187/179/219. Folds are overlapping repeated splits, not independent replications. Three positive TAR folds are encouraging consistency, not a significance test or a guarantee of five-fold results. Saved aggregate AUROCs do not provide the paired predictions needed for a paired score-difference interval.

## Actual source checks and unresolved content risk

All nine raw source-member hashes match each candidate schema. Source replay on **all 60,000 v8 rows and all 20,136 v9 rows** finds:

- Zero target mismatches against the six source seriousness flags.
- Zero onset-excerpt mismatches against the current extractor.
- Zero merged-background mismatches against masking and merging the source background fields.

These establish reproducibility and source alignment. They do not establish that the extraction policy is semantically adequate. V8 retains **88** surgery/surgical-word excerpt matches and **one blood-transfusion match**; v9 retains **58 and one**, respectively. Keyword matches include legitimate history and are not counts of confirmed leaking rows. The already-confirmed care examples and the limits of background timing are documented in [the v7 content audit](vaers_v7_data_audit.md). The same policy is inherited, and neither merging nor case enrichment fixes it.

## Recommendation

Keep v7 as the frozen reference and v8 as the matched merge experiment. Treat v9 as the leading **case-enriched development candidate** for TAR consistency on these two learners, while retaining v8's stronger CatBoost Joint Signal as a real tradeoff. The remaining three learners can determine whether either construction generalizes across learners; their results are not present in these archives.

Do not begin extra score-driven variants from this comparison. Resolve the documented information boundary and care-policy examples before the final official evaluation. Any correction must have a new build identity and a recorded sensitivity comparison. Preserve these files and disclose development selection. Positive gains and clean recorded mechanics do not substitute for content validity.

## Reproduction and saved evidence

[Comparison tool](../scripts/vaers/compare_variant_runs.py) and [machine-readable evidence](vaers_v8_v9_results_comparison.json). The evidence includes full per-fold scores, named checks, actual class counts, fingerprints, overlap and source replay counts. Full candidate CSVs and source text remain outside Git. The existing v7-specific archive auditor and official benchmark were not modified.

```sh
python3 -m research.scripts.vaers.compare_variant_runs --v8-run /Users/dimabshara/Downloads/tennis_diagnostic_c7b5b71aeb656d52_results.zip --v9-run /Users/dimabshara/Downloads/tennis_diagnostic_20b24baf903ae7ac_results.zip --v8-data /Users/dimabshara/Downloads/vaers_v8-20261017T080750Z-1-001.zip --v9-data /Users/dimabshara/Downloads/vaers_v9-20261017T080754Z-1-001.zip --v7-csv /Users/dimabshara/Downloads/v7/vaers_v7.csv --v7-run /Users/dimabshara/Downloads/v7/tennis_diagnostic_a12d3b1c04dbbdd7_results.zip --tree research/data/sources/paper_tree.json --raw /Users/dimabshara/Downloads/2019VAERSData.zip /Users/dimabshara/Downloads/2023VAERSData.zip /Users/dimabshara/Downloads/2024VAERSData.zip --out research/reports/vaers_v8_v9_results_comparison.json
```

The two negative controls were run separately on temporary copies and are recorded in the JSON. No model training, dataset modification or push was performed.
