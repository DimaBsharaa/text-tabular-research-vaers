---
tags:
- multabench
- tabular
---

# MUL_TEXT_KOREAN_DRAMA

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `Score`
- **Rows:** 1647
- **Original source:** https://www.kaggle.com/datasets/noorrizki/top-korean-drama-list-1500

Load the benchmark splits with `multabench.load_split("MUL_TEXT_KOREAN_DRAMA", fold=0)`.

| Column | Type |
|---|---|
| Name | string |
| Year | int |
| Genre | string |
| Main Cast | string |
| Sinopsis | string |
| Content Rating | categorical |
| Tags | string |
| Network | string |
| Number of Episodes | float |
| Score | string |
