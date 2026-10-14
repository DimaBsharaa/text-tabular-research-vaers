---
tags:
- multabench
- tabular
---

# BIN_TEXT_FAKE_JOB_POSTING

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `fraudulent`
- **Rows:** 12725
- **Original source:** https://www.openml.org/search?type=data&id=46655

Load the benchmark splits with `multabench.load_split("BIN_TEXT_FAKE_JOB_POSTING", fold=0)`.

| Column | Type |
|---|---|
| title | string |
| salary_range | string |
| description | string |
| required_experience | categorical |
| required_education | categorical |
| fraudulent | int |
