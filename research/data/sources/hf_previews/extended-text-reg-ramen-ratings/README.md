---
tags:
- multabench
- tabular
---

# REG_TEXT_RAMEN_RATINGS

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `Stars`
- **Rows:** 4105
- **Original source:** https://www.kaggle.com/datasets/ankanhore545/top-ramen-ratings-2022

Load the benchmark splits with `multabench.load_split("REG_TEXT_RAMEN_RATINGS", fold=0)`.

| Column | Type |
|---|---|
| Brand | string |
| Variety | string |
| Style | categorical |
| Country | categorical |
| Stars | float |
