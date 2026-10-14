---
tags:
- multabench
- tabular
---

# REG_TEXT_MERCARI_MARKETPLACE

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `log_price`
- **Rows:** 100000
- **Original source:** https://www.openml.org/search?type=data&id=46660

Load the benchmark splits with `multabench.load_split("REG_TEXT_MERCARI_MARKETPLACE", fold=0)`.

| Column | Type |
|---|---|
| name | string |
| item_condition_id | int |
| category_name | string |
| brand_name | string |
| shipping | int |
| item_description | string |
| cat1 | categorical |
| cat2 | string |
| cat3 | string |
| log_price | float |
