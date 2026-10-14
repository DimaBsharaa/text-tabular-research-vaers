# MulTaBench dataset-selection research

Start with [the executed notebook](notebooks/01_multabench_dataset_analysis.ipynb) or [the standalone HTML](notebooks/01_multabench_dataset_analysis.html). Outputs are already saved: 18 executed Python cells, 13 charts, and the research tables. The notebook uses ordinary Python and can be run in Jupyter. It also has an existing-dataset selector for inspecting per-column text distributions and targets.

Scope: Standard Track 2; exploratory research to inform a new dataset choice. No new dataset selected or constructed; no new model runs. Both course PDFs remain in the parent NLP folder. The shared conversation was recovered after a direct retry; its recent decisions, the supplied context, and PDFs were read. Sports ideas remain inspiration, not a dataset selection.

## Read next

- [Availability and exact sources](reports/source_availability.md)
- [Confirmed findings, hypotheses, and limitations](reports/research_findings.md)
- [Comparable domains](reports/dataset_comparison.md)
- [56-candidate inventory](data/dataset_inventory.csv)
- [Learner-level historical curation scores](data/learner_curation_results.csv)
- [Dataset profiles](data/dataset_profiles.csv) and [text-column profiles](data/text_column_profiles.csv)

## Reproduce offline

From the repository root, using Python 3.14 and the packages in `research/requirements.txt`:

```bash
python3 research/scripts/analyze.py
python3 research/scripts/build_notebook.py
python3 research/scripts/execute_notebook.py
python3 research/scripts/write_reports.py
```

No network is needed for analysis, execution, or reports. Requirements pin direct analysis packages; they are not a complete transitive environment lock. No packages were installed or upgraded during this work. Normal Jupyter execution needs a Python kernel; the included runner uses the existing Python process, captures `display`, stdout, and matplotlib plots, and writes standard `.ipynb` outputs plus a standalone HTML page. It does not implement arbitrary notebook magics or kernel isolation.

The optional `fetch_sources.py` and `fetch_previews.py` scripts require network access and reuse cached files. They fetch selected small official files and at most 50 rows per released dataset. They never download full parquet datasets, model weights, or an entire official repository. Historical sources are pinned to commit `3bb95079a6146ed35d45187e98bcad5b063273a1`. Current-release cache manifests preserve exact URLs, SHA-256 hashes, revisions, and preview offsets.

## Scientific boundaries

- Published outcomes remain separate from reconstruction. Two learner flags differ; overall accepted candidates agree.
- The code's rounded `>=0.001` threshold differs from the paper's strict `>0.001` equation; sensitivity is visible.
- Core/extended is not pass/fail. Six extended releases change the original task.
- All 56 candidates have scores; 40 have release profiles; only 34 have unchanged task prefixes. Even these may have preprocessing differences.
- Preview properties cover seeded contiguous 50-row blocks, not full populations. No global class-balance, missingness, or duplicate claims.
- Leakage flags require manual review. No new feature-selection decisions or benchmark experiments use these labels.
- Original source URLs and partial benchmark memberships are verified registry references, not proof of working original downloads.
- All changes are local; no push was performed.
