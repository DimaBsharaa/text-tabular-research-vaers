---
tags:
- multabench
- tabular
---

# REG_TEXT_WIKILIQ_PRICES

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `Price`
- **Rows:** 12569
- **Original source:** https://www.kaggle.com/datasets/limtis/wikiliq-dataset

Load the benchmark splits with `multabench.load_split("REG_TEXT_WIKILIQ_PRICES", fold=0)`.

| Column | Type |
|---|---|
| Name | string |
| Country | string |
| Brand | string |
| Categories | string |
| Tasting Notes | string |
| ABV | float |
| Base Ingredient | categorical |
| Years Aged | float |
| Rating | float |
| Rate Count | int |
| Volume | categorical |
| Description | string |
| Price | float |
