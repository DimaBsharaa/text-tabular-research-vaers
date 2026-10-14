---
tags:
- multabench
- tabular
---

# REG_TEXT_BABIES_PRICES

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `price`
- **Rows:** 5085
- **Original source:** http://pages.cs.wisc.edu/~anhai/data/784_data/baby_products/csv_files/babies_r_us.csv

Load the benchmark splits with `multabench.load_split("REG_TEXT_BABIES_PRICES", fold=0)`.

| Column | Type |
|---|---|
| title | string |
| is_discounted | int |
| category | categorical |
| company_struct | string |
| company_free | string |
| weight | categorical |
| length | categorical |
| width | categorical |
| height | categorical |
| fabrics | categorical |
| colors | string |
| materials | categorical |
| price | float |
