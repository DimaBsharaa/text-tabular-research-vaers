---
tags:
- multabench
- tabular
---

# MUL_TEXT_WOMEN_CLOTHING_REVIEW

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `Rating`
- **Rows:** 18788
- **Original source:** https://www.openml.org/search?type=data&id=46659

Load the benchmark splits with `multabench.load_split("MUL_TEXT_WOMEN_CLOTHING_REVIEW", fold=0)`.

| Column | Type |
|---|---|
| Unnamed  0 | int |
| Clothing ID | int |
| Age | int |
| Title | string |
| Review Text | string |
| Recommended IND | int |
| Positive Feedback Count | int |
| Division Name | categorical |
| Department Name | categorical |
| Class Name | categorical |
| Rating | int |
