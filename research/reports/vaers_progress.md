# VAERS: progress in one page

**Read this first.** It tells the whole story in order. Every other VAERS report is evidence for one row of this file (see the map at the bottom). Last updated October 21, 2026: **final.** v11b was chosen, and the official run passes both checks on 5/5 learners (rows 17–20).

**Task.** One row is one VAERS report. The target `serious` = 1 if the report's flags record death, a life-threatening event, hospitalisation (or a longer stay), disability or a birth defect. These flags build the label only and are never inputs. This is retrospective classification of recorded seriousness. It is not a forecast and not a causal claim about vaccines.

**Pipeline (reproducible from the raw files):**

1. `2019VAERSData.zip`, `2023VAERSData.zip` and `2024VAERSData.zip` from vaers.hhs.gov (exact copies in `research/data/vaers_raw/`).
2. Development notebook 03 runs `build_vaers.py` and produces v1…v9 (build 8).
3. Development notebook 04 runs `refine_v9.py` on v9 and the same raw ZIPs and produces v10 (build 9).
4. Development notebook 05 screens each version with MulTaBench's paper-version code: Structured, Text, Joint Frozen and Joint TAR.
5. `build_v11b.py` rebuilds the final dataset end to end (full cohort → v11a matched controls → v11b care cleanup 3); notebook 02 runs it and shows the evidence.
6. Notebook 07 runs the official `benchmark.py` at commit `c955466` (`master` as of 26 Jul 2026; the same code as `89a1854` apart from leaderboard analysis files). Notebook 08 analyses the results.

## The story: problem → fix → did it work?

| # | Version / build | What we saw (problem) | What we changed | Did it work? |
|---|---|---|---|---|
| 1 | Source choice | Tennis: text weaker than the table (−0.17). Accepted datasets have text at least as strong as the table. | Chose VAERS: large, binary, free text plus a table. | Yes, text and table are both informative. |
| 2 | v1, build 2 | Coded symptom terms contained outcome words ("Death", "Hospitalisation"). | Removed 93 outcome, 745 care and 150 administration terms; excluded all post-event fields. | Yes. Leak counter 0. |
| 3 | v1, official screen (fold 0) | Joint Signal positive on 4/4 learners, but **TAR gain ≈ 0**. | Hypothesis: coded terms are a fixed vocabulary that frozen E5 already handles, so we need free text. | Explains the next step. |
| 4 | v4/v5, build 3 | The full narrative's top words were `remdesivir`, `iv`, `o2`, `tx`: hospital-chart text. | **Rejected** v4/v5 and kept only the opening of the narrative (cut at the first care sentence). | Yes; this became v6/v7. |
| 5 | Build 4 | `onset_days` alone scored AUROC 0.877; 2023 serious rate 42%. Mandated reporting of breakthrough COVID cases made long-onset COVID reports "serious". | Cohort = reactions with onset of 0–42 days; excluded infection and vaccine-failure reports. | Yes. `onset_days` fell to 0.717 and the 2023 rate to 16%. |
| 6 | Build 5 | ICD wording in the excerpt ("unspecified", "init encntr"). | Cut at ICD/EHR wording. | Yes. 0 hits. |
| 7 | Build 6 | Empty excerpt in 29% of serious vs 12% of non-serious reports, so emptiness gave the label away. | Kept only reports with a non-empty excerpt, in every variant. | Removes the signal, but changes the cohort (stated in the schema). |
| 8 | Build 7 | Capitalised hospital problem lists and vital signs survived. | Cut rules for both. | **Partly.** "ANEMIA **OF** CHRONIC…" still got through. |
| 9 | Build 8 (frozen) | Patching rule by rule is endless (reviewer). | A **written excerpt policy**: runs of 4+ capital words, numeric vitals only, disability statements, event dedup, provenance. | Yes on its own checks. Later audit: see row 13. |
| 10 | **v7** (build 8) | — | 12 table columns + 5 text columns (4 background + excerpt); 60k rows; 7.9% serious. | **Screen:** Joint +0.036, TAR +0.010 (median over 5 learners). LightGBM TAR only +0.003 with one negative fold; TabM/TabPFN on 1 fold only. |
| 11 | **v8** | TAR fine-tunes on every text cell, and 4 of v7's 5 columns are mostly "none reported". | Merged the background into 1 column (2 text columns). Same rows and splits as v7. | **No.** Median Joint +0.028, TAR +0.007. LightGBM's TAR gain grew only because Joint Frozen fell. Dropped. |
| 12 | **v9** | Only 7.9% serious: about 785 serious rows per training fold, 158 per test fold. | Case-enriched to 25% serious: all serious reports plus sampled non-serious; 20,136 rows. | **Most consistent so far.** Median Joint +0.027, TAR +0.010; both positive in **15/15** learner-fold runs. The population differs, so it is not "the same data, improved". |
| 13 | Content audit of v7 | Excerpts still held care text that no rule caught (blood transfusion, surgery, medical leave). Some background fields describe care **after** the vaccine. | → v10. | — |
| 14 | **v10** (build 9, care-boundary-2) | (row 13) | Cut excerpts before care, record pointers and functional outcomes; dropped rows whose background is explicitly post-event. 19,832 rows, 24.7% serious (304 v9 rows removed). | **Done** (LightGBM + CatBoost × 3 folds, Drive `benchmark-results/4f9a4eda6271a02b`). Joint gain +0.034 (v9_control +0.030), 6/6 pass. TAR gain median +0.008 (v9_control +0.007; means +0.006 vs +0.007; the gap is noise, since a 10-split CPU check gives v9 0.0216, v9_control 0.0201, v10 0.0204), 5/6 runs positive (LightGBM fold 2: −0.002; LightGBM mean still +0.004). Text-only fell 0.005 while Joint Frozen stayed the same, and the TAR gain did not change. So the removed care text was slightly informative, but it was not what drove TAR. |
| 15 | **v9_control** | v10 changes two things at once (removes 304 rows and cleans text), so v10 vs v9 can't say which change mattered. | Same 19,832 rows as v10 but with v9's old text, so the splits are identical. v10 vs v9_control = the effect of the text cleanup alone. | **Done** (LightGBM + CatBoost × 3 folds, Drive `benchmark-results/0d1c69e9ba434cbe`). Both checks pass in 6/6. Mean TAR gain: LightGBM +0.006, CatBoost +0.008. Lower than v9 on the same folds; only 304 rows changed, so run-to-run noise is about ±0.005. |
| 16 | **Idea tests (CPU proxy)** | TAR gain only at the benchmark median. Which dataset designs could raise it? | Rebuilt the full cohort from the raw ZIPs and tested 12 designs on a CPU proxy (section below). | **Matched controls (v11a) is the only clear win**: proxy TAR gain +0.030 vs +0.021 for v10, with a higher Joint gain too. Dropping background, longer excerpts, 50% serious, separate background columns and symptom matching did not help. A +0.1 TAR gain is out of reach for this text. |
| 17 | **v11a** (matched controls) | Before matching, the vaccine alone told a lot (3% serious for shingles, 17% for COVID-19), so the table did the work and the text had little left to add. | Every serious report plus up to 3 non-serious reports in the same vaccine type × report year × age band × sex stratum, chosen by a fixed hash; no text used. 18,863 rows, 26% serious. | **Yes.** Screen (LightGBM + CatBoost × folds 0–2): Joint +0.038, **TAR +0.021**, about 3× v10. |
| 18 | **v11a_control** | Is the TAR gain driven by the rows or by the text? | v11a's text on v11b's rows (4 rows fewer). | Joint +0.026, TAR +0.035. Only 4 rows differ from v11a, yet TAR moves by 0.013: **differences under about 0.01 are split noise.** |
| 19 | **v11b** (care-boundary-3) | Some excerpts still named medications and procedures ("Gave TYLENOL…", "resolved with prednisone", "IVIG", "NICU"). | 503 excerpts cut to a shorter literal prefix and 4 reports removed, recorded in an edit ledger; label-blind. Also screened: v11b_context (one merged text column) and v11c (manufacturer added to the matching). | v11b: Joint +0.031, TAR +0.029; v11b_context +0.032 / +0.027; v11c +0.027 / +0.019. All within the noise band of each other; **v11b chosen as the cleanest text.** |
| 20 | **Official run** | — | `benchmark.py` at `c955466`, unchanged; 5 learners × 4 conditions × 5 folds on v11b. | **Both checks pass on 5/5 learners.** Median Joint +0.031, TAR +0.020; all 25 learner-fold runs positive on both; folds 3–4, unseen during the version choice, stay positive. On the screened learners and folds it reproduces the v11b screen exactly. |


## Screen results so far

Paper-version code, development only. Values are medians over learners.

| Version | Learners × folds | Structured | Text | Joint Frozen | Joint TAR | Joint gain | TAR gain | Pass (Joint / TAR) |
|---|---|---|---|---|---|---|---|---|
| v7 | 2×3 + 3×1 | 0.807 | 0.833 | 0.871 | 0.884 | +0.036 | +0.010 | 5/5 / 5/5 |
| v8 | 3×3 + 1×2 | 0.790 | 0.824 | 0.857 | 0.866 | +0.028 | +0.007 | 4/4 / 4/4 |
| v9 | 5×3 | 0.806 | 0.844 | 0.868 | 0.878 | +0.027 | +0.010 | 5/5 / 5/5 |
| v9_control | 2×3 | 0.816 | 0.839 | 0.868 | 0.875 | +0.030 | +0.007 | 2/2 / 2/2 |
| v10 | 2×3 | 0.816 | 0.834 | 0.867 | 0.875 | +0.034 | +0.008 | 2/2 / 2/2 |
| *Accepted MulTaBench datasets (median of 23)* | | | | | | *+0.023* | *+0.0085* | |

Per-learner and per-fold numbers: `vaers_v8_v9_results_comparison.md`, `research/data/screens/vaers_v9_v9control_v10_results.csv` and `research/data/screens/`. Rows here are medians over the 6 learner-fold runs (LightGBM + CatBoost × 3).

**Final version comparison** — mean of LightGBM and CatBoost over folds 0–2 (the bars in notebook 08; `research/data/screens/version_screens_light_cat_folds012.csv`):

| Version | Structured | Text | Joint Frozen | Joint TAR | Joint gain | TAR gain |
|---|---|---|---|---|---|---|
| v9 | 0.801 | 0.837 | 0.864 | 0.875 | +0.028 | +0.010 |
| v9_control | 0.814 | 0.838 | 0.867 | 0.874 | +0.030 | +0.007 |
| v10 | 0.814 | 0.833 | 0.867 | 0.873 | +0.034 | +0.006 |
| v11a | 0.739 | 0.775 | 0.813 | 0.834 | +0.038 | +0.021 |
| v11a_control | 0.723 | 0.786 | 0.812 | 0.847 | +0.026 | +0.035 |
| **v11b (chosen)** | 0.723 | 0.786 | 0.816 | 0.846 | +0.031 | +0.029 |
| v11b_context | 0.723 | 0.774 | 0.805 | 0.832 | +0.032 | +0.027 |
| v11c | 0.732 | 0.788 | 0.815 | 0.833 | +0.027 | +0.019 |

**Official v11b result** (all 5 learners × 5 folds, `c955466`): medians Structured 0.735, Text 0.795, Joint Frozen 0.823, Joint TAR 0.843; Joint gain +0.031, TAR gain +0.020; both checks 5/5. See `results/official_c955466/` and notebook 08.

## Status, in one sentence each

- **Mechanics** (same test rows across conditions, TAR trained on training rows only, exact CSV per run): verified on every archive.
- **Leakage:** the known text paths are fixed through care-boundary-3 (v11b); universal absence cannot be proven. Notebook 02 part C shows the remaining checks.
- **Official result:** done. Native registration plus `benchmark.py` at `c955466`, run once on the frozen v11b: both checks pass on 5/5 learners.

## Idea tests on a CPU proxy (October 17)

**How.** The raw ZIPs were rebuilt here. v9, v10, v9_control and v10_matched all match their recorded sha256, so the reproducibility check is done. Each design was then scored with a CPU stand-in for the four conditions:
- Frozen = TF-IDF compressed to 30 numbers per text column.
- TAR = the same, plus a supervised text score per column.
- Same split rules (seed 42 + fold, 10% test, 10k training cap); LightGBM.

The proxy matches the real Structured and Joint Frozen levels. It overstates the TAR gain by 2–2.7× (v7: +0.013 vs real +0.006; v9: +0.022 vs +0.010; v9_control: +0.018 vs +0.007), but it ranks v7 < v9_control < v9 in the same order as the real runs. Scripts: `research/scripts/vaers/proxy_ideas/`. All numbers: `research/data/screens/idea_proxy_results.csv`.

| Design | Proxy Joint gain | Proxy TAR gain | Verdict |
|---|---|---|---|
| v10 (baseline, 5 folds) | +0.035 | +0.021 | — |
| **A2 = v11a: matched controls** (vaccine × year × age band × sex) | **+0.041** | **+0.030** | **Best; run on GPU** |
| A3: also match on onset band | +0.028 | +0.030 | Same TAR gain, weaker Joint gain |
| A / A4: fewer matching fields | +0.044 | +0.026 | Smaller win |
| B: excerpt only (drop background) | +0.039 | +0.015 | Worse: background carries comorbidity signal |
| C: up to 5 sentences | +0.037 | +0.017 | Worse: longer text helps Frozen more than TAR |
| D: 50% serious (vs same-size 25%) | +0.043 | +0.022 (vs +0.023) | No effect |
| G: 4 separate background columns | +0.037 | +0.018 | Worse |
| F: symptom-matched pairs | +0.032 | +0.020 | No gain; text signal collapses (T 0.70) |

**Why +0.1 is impossible here.** Suppose frozen E5 caught nothing at all from the text. Even then, adding the best supervised text model to the table raises AUROC by only +0.063 (v10) or +0.099 (v11a). The real TAR gain must be smaller, because frozen text already catches most of it (and must, for Joint Signal to pass). Realistic target: a real TAR gain of about +0.011 to +0.014 for v11a, versus +0.008 to +0.010 for v10. That would put it in the top third of the accepted datasets (about 5th to 9th of 24), well above the median of +0.0085.

**Cut-position check (leakage).** The excerpt stops at the first care sentence, and serious reports hit that stop more often (38% vs 19%). So serious excerpts are slightly shorter (22.8 vs 24.7 words), even though serious raw narratives are longer (6.0 vs 3.9 sentences).
- Shape features (length, sentence count, ending punctuation) alone reach AUROC 0.59.
- Adding them to the joint proxy adds +0.009.
- This is a small artifact of how the excerpt is cut; report it in the paper.
- Mitigations to consider: always end the excerpt with a period, and report results with the shape features as a control.

## Next steps (in order)

1. ~~Run v10~~ done. 2. ~~Compare v10 vs v9_control~~ done (row 14).
3. ~~Run v11a~~ done (row 17): its TAR gain was clearly higher than v10's and Joint Signal was positive for both learners, so the matched design replaced v10.
4. ~~Checks before freezing~~ done: fresh rebuild from the raw ZIPs reproduces v11a and v11b byte for byte (`build_v11b.py`); excerpt lengths no longer differ by class (notebook 02, C4).
5. ~~Official run~~ done (row 20), at `c955466`.
6. Write the paper.

## Project facts

- Curation rule: Joint Frozen > max(Structured, Text) **and** Joint TAR > Joint Frozen, each on ≥3 of 5 learners (LightGBM, CatBoost, TabM, TabPFN-v2, TabPFN-2.5), counted separately for the two checks, on unrounded scores.
- Development screens: notebook 05 (development), paper-version code at 3bb9507, results on Drive in `MyDrive/text-tabular-research-2/benchmark-results/<plan_id>/results.csv`.
- Data builds: development notebooks 03 (`build_vaers.py`, v1–v9) and 04 (`refine_v9.py`, v10 / v10_matched / v9_control); the final dataset with `build_v11b.py`.
- Plan IDs: v7 `a12d3b1c04dbbdd7` + `721a59225d840c64`; v8 `c7b5b71aeb656d52` + `0343d0f8f1fc61a6`; v9 `20b24baf903ae7ac` + `a84e46ed50352f6a`; v10_matched `211cb027c8cf4486`; v9_control `0d1c69e9ba434cbe`; v11a `2c0763a91fe71895`; v11a_control `46d53676b1560c44`; v11b `46f84d4ab9bf09cf`; v11b_context `114d0c6c64cd8324`; v11c `fcb6180218b90e0e`.
- Official run: Drive `official-final/vaers_v11b_c955466/`, copied to `results/official_c955466/`.
- Rules: never print or commit secrets (they live in Colab Secrets); one notebook run at a time; open notebooks fresh from GitHub.
- Repo history: on 2026-10-18 the history was rewritten (author identity corrected) and moved to `DimaBsharaa/text-tabular-research-vaers` (content unchanged; tree hashes identical). Old commit ids recorded in Drive results map to new ones via [`commit_id_map.csv`](commit_id_map.csv).

## Map of the other VAERS reports

| File | Role |
|---|---|
| `vaers_candidate.md` | Chronological build notes (builds 1–9), detailed |
| `vaers_decision_log.md` | Each decision as saw → inferred → did → checked |
| `vaers_codex_review.md`, `vaers_v8_v9_review.md`, `vaers_update15_review.md` | Independent reviews |
| `vaers_v7_archive_audit.md`, `vaers_v7_data_audit.md`, `vaers_audit_local_validation.md` | Audits of runs, data and auditor |
| `vaers_v8_v9_results_comparison.md` | Exact numbers for v7/v8/v9 |
