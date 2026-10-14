---
tags:
- multabench
- tabular
---

# REG_TEXT_WINE_POLISH_MARKET

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `price (PLN)`
- **Rows:** 2247
- **Original source:** https://www.kaggle.com/datasets/skamlo/wine-price-on-polish-market

Load the benchmark splits with `multabench.load_split("REG_TEXT_WINE_POLISH_MARKET", fold=0)`.

| Column | Type |
|---|---|
| name | string |
| country | categorical |
| region | string |
| appellation | string |
| vineyard | string |
| vintage | float |
| volume (liters) | float |
| alcohol (%) | float |
| serving temperature (C) | categorical |
| color | categorical |
| kind | categorical |
| taste | categorical |
| style | categorical |
| medals | categorical |
| wegan | bool |
| natural | bool |
| punctation | float |
| grapes | string |
| price (PLN) | float |
