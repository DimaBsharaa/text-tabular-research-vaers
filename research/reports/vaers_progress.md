# VAERS: progress in one page

**Read this first.** It tells the whole story in order. Every other VAERS report is evidence for one row of this file (see the map at the bottom). Last updated October 17, 2026 (v9_control finished; v10 3-fold screen next).

**Task.** One row is one VAERS report. The target `serious` = 1 if the report's flags record death, a life-threatening event, hospitalisation (or a longer stay), disability or a birth defect. These flags build the label only and are never inputs. This is retrospective classification of recorded seriousness. It is not a forecast and not a causal claim about vaccines.

**Pipeline (reproducible from the raw files):**

1. Download `2019VAERSData.zip`, `2023VAERSData.zip` and `2024VAERSData.zip` from vaers.hhs.gov.
2. Notebook 06 runs `build_vaers.py` and produces v1…v9 (build 8).
3. Notebook 08 runs `refine_v9.py` on v9 and the same raw ZIPs and produces v10 (build 9).
4. Notebook 02 runs the official MulTaBench code: Structured, Text, Joint Frozen and Joint TAR.

The final run uses `benchmark.py` at commit 89a1854.

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
| 14 | **v10** (build 9, care-boundary-2) | (row 13) | Cut excerpts before care, record pointers and functional outcomes; dropped rows whose background is explicitly post-event. 19,832 rows, 24.7% serious (304 v9 rows removed). | CPU proxy unchanged (joint 0.8625 → 0.8624). v10_matched fold 0: Joint/TAR positive for LightGBM and CatBoost. **3-fold v10 screen next** (see row 15). |
| 15 | **v9_control** | v10 changes two things at once (removes 304 rows and cleans text), so v10 vs v9 can't say which change mattered. | Same 19,832 rows as v10 but with v9's old text, so the splits are identical. v10 vs v9_control = the effect of the text cleanup alone. | **Done** (LightGBM + CatBoost × 3 folds, Drive `benchmark-results/0d1c69e9ba434cbe`). Both checks pass in 6/6. Mean TAR gain: LightGBM +0.006, CatBoost +0.008. Lower than v9 on the same folds; only 304 rows changed, so run-to-run noise is about ±0.005. |

## Screen results so far

Paper-version code, development only. Values are medians over learners.

| Version | Learners × folds | Structured | Text | Joint Frozen | Joint TAR | Joint gain | TAR gain | Pass (Joint / TAR) |
|---|---|---|---|---|---|---|---|---|
| v7 | 2×3 + 3×1 | 0.807 | 0.833 | 0.871 | 0.884 | +0.036 | +0.010 | 5/5 / 5/5 |
| v8 | 3×3 + 1×2 | 0.790 | 0.824 | 0.857 | 0.866 | +0.028 | +0.007 | 4/4 / 4/4 |
| v9 | 5×3 | 0.806 | 0.844 | 0.868 | 0.878 | +0.027 | +0.010 | 5/5 / 5/5 |
| v9_control | 2×3 | 0.814 | 0.838 | 0.867 | 0.874 | +0.030 | +0.007 | 2/2 / 2/2 |
| v10 | next: LightGBM + CatBoost × 3 folds | | | | | | | |
| *Accepted MulTaBench datasets (median of 23)* | | | | | | *+0.023* | *+0.0085* | |

Per-learner and per-fold numbers: `vaers_v8_v9_results_comparison.md` and `research/data/screens/`.

## Status, in one sentence each

- **Mechanics** (same test rows across conditions, TAR trained on training rows only, exact CSV per run): verified on every archive.
- **Leakage:** the known text paths are fixed in v10; universal absence cannot be proven.
- **Official result:** not run yet. It needs native registration and `benchmark.py` at 89a1854, once, on the frozen final version.

## Next steps (in order)

1. Run v10 in notebook 02 (`CSV_PATH`/`SCHEMA_PATH` = `candidates/build9/vaers_v10/`, `MODELS_OVERRIDE=["light","cat"]`, `FOLDS_OVERRIDE=[0,1,2]`).
2. Compare v10 vs v9_control fold by fold. Same rows and splits, so the difference is the text cleanup alone.
3. Try **one** TAR enhancement (best bet: TAR on the reaction text only, without the background columns), or freeze v10.
4. Checks before freezing: truncation artifact (excerpts without ending punctuation, by label); fresh rebuild of v9/v10 from the raw ZIPs matches the recorded sha256.
5. A tiny native test at MulTaBench 89a1854, then the single official run (5 learners × 5 folds) with `benchmark.py`.
6. Write the paper. Deadline Oct 26, 2026.

## Starting a new chat? Read this

- Course: Technion NLP 097215, final project, Track 2 (benchmark). The dataset must pass MulTaBench curation: Joint Frozen > max(Structured, Text) **and** Joint TAR > Joint Frozen, each on ≥3 of 5 learners (LightGBM, CatBoost, TabM, TabPFN-v2, TabPFN-2.5). Bigger gains mean a better grade.
- Development screens: notebook 02, paper-version code at 3bb9507, results on Drive in `MyDrive/text-tabular-research-2/benchmark-results/<plan_id>/results.csv`.
- Data builds: notebook 06 (`build_vaers.py`, v1–v9) and notebook 08 (`refine_v9.py`, v10 / v10_matched / v9_control), both from the raw VAERS ZIPs on Drive `vaers_raw/`.
- Plan IDs: v7 `a12d3b1c04dbbdd7` + `721a59225d840c64`; v8 `c7b5b71aeb656d52` + `0343d0f8f1fc61a6`; v9 `20b24baf903ae7ac` + `a84e46ed50352f6a`; v10_matched `211cb027c8cf4486`; v9_control `0d1c69e9ba434cbe`.
- Rules: never print or commit secrets (they live in Colab Secrets); one notebook run at a time; open notebooks fresh from GitHub.

## Map of the other VAERS reports

| File | Role |
|---|---|
| `vaers_candidate.md` | Chronological build notes (builds 1–9), detailed |
| `vaers_decision_log.md` | Each decision as saw → inferred → did → checked |
| `vaers_codex_review.md`, `vaers_v8_v9_review.md`, `vaers_update15_review.md` | Independent reviews |
| `vaers_v7_archive_audit.md`, `vaers_v7_data_audit.md`, `vaers_audit_local_validation.md` | Audits of runs, data and auditor |
| `vaers_v8_v9_results_comparison.md` | Exact numbers for v7/v8/v9 |
