---
tags:
- multabench
- tabular
---

# MUL_TEXT_WINE_REVIEW

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `variety`
- **Rows:** 84123
- **Original source:** https://www.openml.org/search?type=data&id=46653

Load the benchmark splits with `multabench.load_split("MUL_TEXT_WINE_REVIEW", fold=0)`.

| Column | Type |
|---|---|
| country | categorical |
| description | string |
| points | int |
| price | float |
| province | string |
| variety | string |
