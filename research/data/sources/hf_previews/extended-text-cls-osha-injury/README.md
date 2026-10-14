---
tags:
- multabench
- tabular
---

# BIN_TEXT_OSHA_INJURY

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `Task Assigned`
- **Rows:** 4847
- **Original source:** https://www.kaggle.com/ruqaiyaship/osha-accident-and-injury-data-1517

Load the benchmark splits with `multabench.load_split("BIN_TEXT_OSHA_INJURY", fold=0)`.

| Column | Type |
|---|---|
| Event Date | datetime |
| Abstract Text | string |
| Event Description | string |
| Event Keywords | string |
| Construction End Use | categorical |
| Building Stories | categorical |
| Project Cost | categorical |
| Project Type | categorical |
| Degree of Injury | categorical |
| Nature of Injury | categorical |
| Part of Body | categorical |
| Event type | categorical |
| Environmental Factor | categorical |
| Human Factor | categorical |
| hazsub | categorical |
| fat_cause | int |
| Task Assigned | string |
