---
tags:
- multabench
- tabular
---

# REG_TEXT_SCIMAGOJR_IMPACT

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `H index`
- **Rows:** 31136
- **Original source:** https://www.scimagojr.com/journalrank.php?out=xls

Load the benchmark splits with `multabench.load_split("REG_TEXT_SCIMAGOJR_IMPACT", fold=0)`.

| Column | Type |
|---|---|
| Rank | int |
| Title | string |
| Type | categorical |
| Issn | string |
| SJR | string |
| SJR Best Quartile | categorical |
| Total Docs  (2024) | int |
| Total Docs  (3years) | int |
| Total Refs  | int |
| Total Citations (3years) | int |
| Citable Docs  (3years) | int |
| Citations / Doc  (2years) | string |
| Ref  / Doc  | string |
| %Female | categorical |
| Overton | int |
| SDG | int |
| Country | string |
| Region | categorical |
| Publisher | string |
| Coverage | string |
| Categories | string |
| Areas | string |
| H index | int |
