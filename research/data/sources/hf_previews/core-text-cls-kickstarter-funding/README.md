---
tags:
- multabench
- tabular
---

# BIN_TEXT_KICKSTARTER_FUNDING

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `Funding Status`
- **Rows:** 86502
- **Original source:** https://www.openml.org/search?type=data&id=46668

Load the benchmark splits with `multabench.load_split("BIN_TEXT_KICKSTARTER_FUNDING", fold=0)`.

| Column | Type |
|---|---|
| name | string |
| desc | string |
| goal | float |
| keywords | string |
| disable_communication | bool |
| country | categorical |
| currency | categorical |
| deadline | datetime |
| created_at | datetime |
| Funding Status | string |
