---
tags:
- multabench
- tabular
---

# MUL_TEXT_CONSUMER_COMPLAINT

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `Company response to consumer`
- **Rows:** 100000
- **Original source:** https://www.kaggle.com/selener/consumer-complaint-database

Load the benchmark splits with `multabench.load_split("MUL_TEXT_CONSUMER_COMPLAINT", fold=0)`.

| Column | Type |
|---|---|
| Date received | datetime |
| Product | categorical |
| Sub-product | categorical |
| Issue | string |
| Sub-issue | string |
| Consumer complaint narrative | string |
| Company public response | categorical |
| Company | string |
| State | categorical |
| ZIP code | string |
| Tags | categorical |
| Consumer consent provided? | categorical |
| Submitted via | categorical |
| Date sent to company | datetime |
| Timely response? | categorical |
| Consumer disputed? | categorical |
| Company response to consumer | string |
