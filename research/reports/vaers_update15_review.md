# Review of the proposed notebook 07 / update 15 audit

October 17, 2026. Reviewed `/Users/dimabshara/Downloads/vaers-update-15.zip` in an isolated temporary snapshot, using the existing builder and adapter dependencies. The archive contains six `research/` files, not a Git history. Its three supplied tests all passed. No archive content was installed into the working repository, no models were trained on real reports, and nothing was pushed.

**Subsequent correction:** at the user's request, update 15 was installed locally and the defects below were corrected in the audit tool and notebook. See [local validation](vaers_audit_local_validation.md) for the implementation, tests and remaining missing real-data inputs. The findings below describe the original ZIP, not the corrected tool. No builder or official benchmark changes were made.

**Assessment:** useful diagnostic direction, but correct the mechanics and duplicate-sensitivity checks before interpreting its output as an audit verdict. A zero `FLAG` count currently misses real check failures. These are problems in the proposed auditor; they do not demonstrate leakage in the already-audited v7 runs.

## Confirmed counterexamples

### Encoder train/validation overlap is not checked

Using the actual 24-run archive records, append one encoder-validation ID to `encoder_train` in every TAR configuration. `mechanics()` still produces **zero FLAG lines**. It checks the union is a subset of learner fitting, but does not check encoder train/validation disjointness, within-list uniqueness, or that their union equals the intended fitting pool.

Require unique IDs in every partition; encoder train ∩ validation = empty; their union = learner fitting; and no encoder/test/learner-validation overlap. Require both encoder partitions for every TAR condition. The separate v7 archive auditor already checks these properties on the real archives and found no recorded violations.

The proposed loader also accepts configuration/split directories without checking successful completion or all expected conditions. Missing Frozen and TAR conditions can satisfy `None == None` in the predictor-comparison check. The data-audit entry point should stop before fitting proxies if input identity or mechanics fails, rather than continue on mismatched inputs.

### Duplicate sensitivity removes rows unrelated to training overlap

`duplicate_sensitivity()` constructs `dup_ids` from **every candidate cluster in the entire CSV**, then removes all their test members. Consider a cluster `{test_a, test_b}` with no member in training: both test rows are removed anyway. Thus its score change is not specifically sensitivity to train/test duplication.

For each fold separately, identify clusters intersecting **both that fold's training and test sets**, and remove only their test members. Report the removed test-positive/negative counts and resulting population. The resulting TF-IDF sensitivity remains a proxy experiment; it is not the official E5/TAR AUROC after deduplication, and a changed score may reflect changed case mix.

### Large duplicate blocks silently disappear

A synthetic block of **401 identical excerpts**, sharing every blocking key, returns **zero clusters** because blocks larger than 400 are skipped without coverage reporting. This includes easy exact matches, not just expensive approximate comparisons.

Detect exact matches even in large blocks. Explicitly count blocks/rows and approximate comparisons not searched; report them as unverified coverage. No detected clusters must not be interpreted as no duplicate events.

## Other interpretation limits

- The key uses age, sex, state, vaccine type, onset days and **report year**, not actual vaccination/onset dates or lot. Similar short reports from different people can match; secondary reports with changed metadata can fail to match. Call these **candidate text/metadata clusters**, not confirmed duplicate clinical events. Label agreement does not establish shared identity.
- `strata()` pools predictions from all repeated folds. Some report IDs occur more than once, and independently fitted models may have different score scales. Report per-fold estimates and support, then summarize. Displaying the pooled row count as independent observations overstates support.
- Year, COVID/non-COVID and facility are checked separately. This does not answer whether signal survives **within year × specific vaccine type**, and COVID/non-COVID is much coarser than actual vaccine type. Add joint strata with minimum positive and negative support; report insufficient-support groups rather than silently treating them as evidence of absence.
- The proposed style features are useful but not pure measurements of style. Length, pronouns and parent references can reflect patient population or event content. Compare structure + style with structure + style + text on matched rows as an additional diagnostic. Separate AUROCs do not quantify a percentage of signal due to style.
- Original examples display only the first 400 raw characters and 300 cleaned background characters. A missing source-ID match silently becomes an empty original. The fixture replaces source IDs, so its `ORIGINAL:` assertion checks only formatting, not actual linkage. Require matched raw IDs/hashes and test real linkage; clearly label truncated displays and retain longer examples locally when needed.
- The overall `FLAG` count measures only implemented mechanics flags. Duplicate counts, weak strata, missing raw data and suspicious text do not automatically add flags. It is not an overall leakage pass/fail score.

## Recommended next step

Keep completed and running model experiments unchanged. Correct these audit defects, retain the useful CPU proxy diagnostics, and add counterexample tests for encoder partition overlap, test-only duplicate clusters, large exact-match blocks and raw-ID linkage. Run notebook 07 in a **separate fresh CPU runtime**, without switching or deleting a running GPU notebook's runtime. Read its outputs as diagnostic evidence with explicit PASS/FAIL/NOT_VERIFIED scopes.

Alternatively, the 20 MB CSV can be read directly in our local workspace once supplied; the other assistant's 10 MB connector limit does not apply to local files here. The exact v7 CSV and schema would allow the already-verified archive boundaries to be joined to actual text without relying on that assistant's download access.

No builder, official model implementation, dataset policy or benchmark result was changed by this review.
