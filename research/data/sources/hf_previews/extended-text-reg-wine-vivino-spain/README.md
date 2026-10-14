---
tags:
- multabench
- tabular
---

# REG_TEXT_WINE_VIVINO_SPAIN

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `Rating`
- **Rows:** 8650
- **Original source:** https://www.kaggle.com/datasets/joshuakalobbowles/vivino-wine-data

Load the benchmark splits with `multabench.load_split("REG_TEXT_WINE_VIVINO_SPAIN", fold=0)`.

| Column | Type |
|---|---|
| Winery | string |
| Year | categorical |
| Wine | string |
| num_review | int |
| price | float |
| Country | categorical |
| Region | categorical |
| Rating | float |
