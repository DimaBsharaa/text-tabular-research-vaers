---
tags:
- multabench
- tabular
---

# REG_TEXT_ANIME_PLANET

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `Rating Score`
- **Rows:** 14391
- **Original source:** https://www.kaggle.com/datasets/hernan4444/animeplanet-recommendation-database-2020

Load the benchmark splits with `multabench.load_split("REG_TEXT_ANIME_PLANET", fold=0)`.

| Column | Type |
|---|---|
| Name | string |
| Alternative Name | string |
| Number Votes | int |
| Tags | string |
| Content Warning | string |
| Type | categorical |
| Episodes | int |
| Finished | bool |
| Duration | float |
| StartYear | float |
| EndYear | float |
| Season | string |
| Studios | string |
| Synopsis | string |
| Rating Score | float |
