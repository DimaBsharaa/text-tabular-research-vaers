---
tags:
- multabench
- tabular
---

# REG_TEXT_ZOMATO_RESTAURANTS

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `rate`
- **Rows:** 41665
- **Original source:** https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants

Load the benchmark splits with `multabench.load_split("REG_TEXT_ZOMATO_RESTAURANTS", fold=0)`.

| Column | Type |
|---|---|
| address | string |
| name | string |
| online_order | categorical |
| book_table | categorical |
| votes | int |
| phone | string |
| location | categorical |
| rest_type | categorical |
| dish_liked | string |
| cuisines | string |
| approx cost(for two people) | categorical |
| reviews_list | string |
| menu_item | string |
| listed_in(type) | categorical |
| listed_in(city) | categorical |
| rate | float |
