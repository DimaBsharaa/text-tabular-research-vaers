---
tags:
- multabench
- tabular
---

# MUL_TEXT_AMERICAN_EAGLE_PRICES

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `price`
- **Rows:** 22662
- **Original source:** https://www.openml.org/search?type=data&id=46656

Load the benchmark splits with `multabench.load_split("MUL_TEXT_AMERICAN_EAGLE_PRICES", fold=0)`.

| Column | Type |
|---|---|
| product_name | string |
| mrp | categorical |
| pdp_url | string |
| brand_name | categorical |
| product_category | categorical |
| retailer | categorical |
| description | string |
| rating | float |
| review_count | float |
| style_attributes | string |
| total_sizes | categorical |
| available_size | string |
| color | categorical |
| price | string |
