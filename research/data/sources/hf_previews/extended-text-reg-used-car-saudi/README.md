---
tags:
- multabench
- tabular
---

# REG_TEXT_USED_CAR_SAUDI

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `Price`
- **Rows:** 8035
- **Original source:** https://www.kaggle.com/datasets/turkibintalib/saudi-arabia-used-cars-dataset

Load the benchmark splits with `multabench.load_split("REG_TEXT_USED_CAR_SAUDI", fold=0)`.

| Column | Type |
|---|---|
| Make | categorical |
| Type | string |
| Year | int |
| Origin | categorical |
| Color | categorical |
| Options | categorical |
| Engine_Size | float |
| Fuel_Type | categorical |
| Gear_Type | categorical |
| Mileage | int |
| Region | categorical |
| Negotiable | bool |
| Price | int |
