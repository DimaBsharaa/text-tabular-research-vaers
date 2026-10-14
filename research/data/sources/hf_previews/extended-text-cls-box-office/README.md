---
tags:
- multabench
- tabular
---

# MUL_TEXT_BOX_OFFICE

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `revenue`
- **Rows:** 7408
- **Original source:** https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

Load the benchmark splits with `multabench.load_split("MUL_TEXT_BOX_OFFICE", fold=0)`.

| Column | Type |
|---|---|
| adult | bool |
| collection | string |
| budget | int |
| genres | string |
| original_language | categorical |
| original_title | string |
| overview | string |
| popularity | float |
| production_companies | string |
| production_countries | string |
| release_date | datetime |
| runtime | float |
| spoken_languages | string |
| status | categorical |
| tagline | string |
| title | string |
| video | bool |
| vote_average | float |
| vote_count | float |
| revenue | string |
