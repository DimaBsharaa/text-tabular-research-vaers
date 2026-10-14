---
tags:
- multabench
- tabular
---

# REG_TEXT_ROTTEN_TOMATOES

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `RatingValue`
- **Rows:** 7158
- **Original source:** http://pages.cs.wisc.edu/~anhai/data/784_data/movies1/csv_files/rotten_tomatoes.csv

Load the benchmark splits with `multabench.load_split("REG_TEXT_ROTTEN_TOMATOES", fold=0)`.

| Column | Type |
|---|---|
| Name | string |
| Year | int |
| Release Date | string |
| Director | string |
| Creator | string |
| Actors | string |
| Cast | string |
| Language | string |
| Country | string |
| Duration | float |
| RatingCount | string |
| ReviewCount | string |
| Genre | string |
| Filming Locations | string |
| Description | string |
| RatingValue | float |
