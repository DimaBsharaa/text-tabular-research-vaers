---
tags:
- multabench
- tabular
---

# MUL_TEXT_MICHELIN_RESTAURANTS

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `Award`
- **Rows:** 18843
- **Original source:** https://www.kaggle.com/datasets/ngshiheng/michelin-guide-restaurants-2021

Load the benchmark splits with `multabench.load_split("MUL_TEXT_MICHELIN_RESTAURANTS", fold=0)`.

| Column | Type |
|---|---|
| Name | string |
| Address | string |
| Location | string |
| Price | categorical |
| Cuisine | string |
| Longitude | float |
| Latitude | float |
| PhoneNumber | float |
| GreenStar | int |
| FacilitiesAndServices | string |
| Description | string |
| Award | string |
