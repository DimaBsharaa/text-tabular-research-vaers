---
tags:
- multabench
- tabular
---

# REG_TEXT_VANCOUVER_SALARIES

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `remuneration`
- **Rows:** 44574
- **Original source:** https://opendata.vancouver.ca/api/records/1.0/download/?dataset=employee-remuneration-and-expenses-earning-over-75000&format=csv

Load the benchmark splits with `multabench.load_split("REG_TEXT_VANCOUVER_SALARIES", fold=0)`.

| Column | Type |
|---|---|
| year | int |
| name | string |
| department | categorical |
| title | string |
| expenses | float |
| remuneration | float |
