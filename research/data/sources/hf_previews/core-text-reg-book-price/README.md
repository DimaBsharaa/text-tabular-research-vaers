---
tags:
- multabench
- tabular
---

# REG_TEXT_BOOK_PRICE

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `Price in USD`
- **Rows:** 4989
- **Original source:** https://www.openml.org/search?type=data&id=46663

Load the benchmark splits with `multabench.load_split("REG_TEXT_BOOK_PRICE", fold=0)`.

| Column | Type |
|---|---|
| Title | string |
| Author | string |
| Edition | string |
| Reviews | float |
| Ratings | float |
| Synopsis | string |
| Genre | string |
| BookCategory | categorical |
| Price in USD | float |
