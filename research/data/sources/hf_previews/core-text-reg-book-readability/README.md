---
tags:
- multabench
- tabular
---

# REG_TEXT_BOOK_READABILITY

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `New Dale Chall Readability Formula`
- **Rows:** 4724
- **Original source:** https://www.kaggle.com/datasets/verracodeguacas/clear-corpus

Load the benchmark splits with `multabench.load_split("REG_TEXT_BOOK_READABILITY", fold=0)`.

| Column | Type |
|---|---|
| Author | string |
| Title | string |
| Anthology | string |
| URL | string |
| Source | categorical |
| Pub Year | float |
| Category | categorical |
| Location | categorical |
| License | categorical |
| MPAA Max | categorical |
| MPAA  #Max | float |
| MPAA #Avg | float |
| Excerpt | string |
| Google WC | float |
| Joon WC v1 | float |
| British WC | float |
| British Words | string |
| Sentence Count v1 | float |
| Sentence Count v2 | float |
| Paragraphs | float |
| BT Easiness | float |
| BT s e  | float |
| Flesch-Reading-Ease | float |
| Flesch-Kincaid-Grade-Level | float |
| Automated Readability Index | float |
| SMOG Readability | float |
| CAREC | float |
| CAREC_M | float |
| CARES | float |
| CML2RI | float |
| New Dale Chall Readability Formula | float |
