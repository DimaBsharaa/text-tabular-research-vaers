---
tags:
- multabench
- tabular
---

# REG_TEXT_CHOCOLATE_BAR_RATINGS

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `Rating`
- **Rows:** 1795
- **Original source:** https://www.kaggle.com/datasets/rtatman/chocolate-bar-ratings

Load the benchmark splits with `multabench.load_split("REG_TEXT_CHOCOLATE_BAR_RATINGS", fold=0)`.

| Column | Type |
|---|---|
| Company  (Maker if known) | string |
| Specific Bean Origin or Bar Name | string |
| REF | int |
| Review Date | int |
| Cocoa Percent | float |
| Company Location | categorical |
| Bean Type | categorical |
| Broad Bean Origin | string |
| Rating | float |
