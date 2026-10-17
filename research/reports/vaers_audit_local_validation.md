# VAERS auditor corrections and local validation

October 17, 2026. Auditor version: `update15-corrected-2`.

**Real-data update:** the user subsequently supplied the exact CSV/schema and three source ZIPs. Both audits have now run locally. See [the actual-data findings](vaers_v7_data_audit.md): source reconstruction passes for all 60,000 rows, but manual examples demonstrate care-policy gaps. The original missing-input status below is superseded by this evidence.

The update-15 audit tool and notebook 07 have been integrated and corrected locally. This validates the **auditor's behavior**, not the absence of leakage in VAERS. No dataset builder, official benchmark implementation or existing model result was changed. Nothing from this task has been pushed.

## Validation evidence

| Evidence | Result | Scope |
|---|---|---|
| Auditor regression suite | PASS: 14 tests | Includes counterexamples that the supplied auditor missed, leading-zero source IDs/collisions and a complete synthetic-data audit |
| All-row source auditor suite | PASS: 3 tests | All-field reconstruction, modified labels/background and mismatched source fingerprints |
| Existing builder suite | PASS: 29 tests | Existing construction behavior remains covered |
| Existing smoke infrastructure suite | PASS: 19 tests | Existing runner/infrastructure behavior remains covered |
| Independent rerun on the two supplied real result archives | PASS: 894 checks across 36 runs, 0 failures | Recorded hashes, partitions, configurations and reference implementation evidence; not report-content validation |
| Notebook 07 code cells | PASS: Python syntax validation | Generated from the corrected notebook builder |
| Actual CSV content audit | RUN on the exact screened bytes | Duplicate/conditional/style diagnostics complete; policy concerns remain |
| Original versus retained report text | RUN: 24 source-matched examples; all-row replay for 60,000 rows | Reproducible extraction does not validate its semantics |

Combined test command, run from the repository root:

```sh
python3 -m unittest research.scripts.vaers.test_run_audit research.scripts.vaers.test_audit_source_rows research.scripts.vaers.test_build_vaers evaluation.test_smoke_infrastructure -q
```

The final combined run passed **65 tests in 20.794 seconds**. Synthetic fixtures demonstrate detection and correct behavior on controlled examples; they do not establish the correctness of real VAERS rows.

## Corrections

- **Recorded boundaries:** require unique partition IDs, disjoint encoder training/validation, exact coverage of learner fitting, no held-out encoder rows, successful runs and complete four-condition comparisons. Input fingerprint or mechanics failures stop diagnostic fitting and save the failed checks.
- **Duplicate sensitivity:** remove only candidate test members whose cluster also intersects that fold's training set. Test-only clusters no longer cause exclusions. Removed class counts are reported. These are candidate text/metadata matches, not confirmed duplicate clinical events.
- **Search coverage:** exact token-set matches are checked in large blocks; approximate comparisons skipped in large blocks are explicitly reported. Changed metadata can still conceal repeated events.
- **Conditional signal:** report each fold separately, including year × specific vaccine type. Require at least 20 examples of each class for a stratum AUROC; show insufficient-support groups rather than pooling overlapping folds.
- **Style diagnostics:** add structure+style versus structure+style+excerpt on matched rows. The features are imperfect proxies, and score differences do not establish a causal fraction attributable to style.
- **Source examples:** require actual raw report-ID matches, verify source member fingerprints when available, compare reconstructed sampled labels and extraction, and save full matched examples to a local JSON sidecar. Missing raw evidence is explicitly unverified.
- **Notebook 07:** run regression tests first, print the auditor version, require the corrected module version and save progressive output. Use a separate CPU runtime if GPU experiments are running elsewhere.

Regression cases include encoder train/validation overlap, missing encoder partitions/conditions, 401 identical excerpts in one block, test-only duplicate clusters, insufficient class support, missing source IDs, schema fingerprint mismatch before fitting and numeric JSON split IDs.

## Inputs for the real local audit

Notebook 06 creates the candidate CSV and schema on Drive. Notebook 02 consumes them and writes result archives. Neither the final candidate CSV nor its schema is tracked on `origin/main`; the two supplied result ZIPs do not contain them.

For the existing build-8 v7 run, the expected byte fingerprints from the archived configurations are:

| Input | SHA-256 |
|---|---|
| `vaers_v7.csv` | `9ffdfad811ef6d0910fdf4e69af9d1e44a69870f31f308e80fd59750259a8922` |
| `schema.json` | `3294eed03d7634fe319323caa9a55938dce2508fe5c25900d34eb60dbe721051` |

Download both unchanged, preferably into `Downloads/vaers_v7/`. Include `audit.txt` if available; raw yearly ZIPs are optional for the content diagnostics but needed for source-text comparisons. A differently rebuilt CSV cannot silently substitute for the screened bytes.

The actual supplied folder is now `Downloads/v7/`; both expected hashes match. All raw source-member hashes also match. Current invocation paths and complete results are recorded in the actual-data report linked above.

After safely extracting the 24-run `a12d3b1c04dbbdd7` archive into a temporary results folder, the invocation is:

```sh
python3 -m research.scripts.vaers.run_audit /path/to/vaers_v7.csv /path/to/schema.json /path/to/a12d3b1c04dbbdd7 --out /path/to/local-audits/v7.md
```

Add `--raw /path/to/2019VAERSData.zip /path/to/2023VAERSData.zip /path/to/2024VAERSData.zip` when the exact source files are available. Keep full medical-text examples in the local audit output rather than automatically committing them.

## Remaining interpretation limits

The existing split records are adapter reconstructions, not independent runtime tensor traces. CPU TF-IDF/logistic diagnostics do not reproduce official E5/TAR scores. Candidate duplicate exclusions change the scored population; a score change is not by itself proof of inflation. Passing implemented checks cannot rule out indirect outcome wording, unidentified secondary reports, reporting bias or development-set selection effects. Review the actual retained text and results before deciding whether further model evaluation is justified.
