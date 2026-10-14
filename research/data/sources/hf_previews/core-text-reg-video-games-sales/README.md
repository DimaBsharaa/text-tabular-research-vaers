---
tags:
- multabench
- tabular
---

# REG_TEXT_VIDEO_GAMES_SALES

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `Global_Sales`
- **Rows:** 16598
- **Original source:** https://www.kaggle.com/datasets/gregorut/videogamesales

Load the benchmark splits with `multabench.load_split("REG_TEXT_VIDEO_GAMES_SALES", fold=0)`.

| Column | Type |
|---|---|
| Name | string |
| Platform | categorical |
| Year | float |
| Genre | categorical |
| Publisher | string |
| Global_Sales | float |
