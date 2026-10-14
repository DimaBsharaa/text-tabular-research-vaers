---
tags:
- multabench
- tabular
---

# REG_TEXT_USED_CAR_PAKISTAN

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `Price`
- **Rows:** 72655
- **Original source:** https://www.kaggle.com/datasets/mustafaimam/used-car-prices-in-pakistan-2021

Load the benchmark splits with `multabench.load_split("REG_TEXT_USED_CAR_PAKISTAN", fold=0)`.

| Column | Type |
|---|---|
| Make | categorical |
| Model | string |
| Version | string |
| Make_Year | int |
| CC | int |
| Assembly | categorical |
| Mileage | int |
| Registered City | string |
| Transmission | categorical |
| Price | float |
