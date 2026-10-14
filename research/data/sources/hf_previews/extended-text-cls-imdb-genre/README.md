---
tags:
- multabench
- tabular
---

# BIN_TEXT_IMDB_GENRE

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `Genre`
- **Rows:** 800
- **Original source:** https://www.openml.org/search?type=data&id=46667

Load the benchmark splits with `multabench.load_split("BIN_TEXT_IMDB_GENRE", fold=0)`.

| Column | Type |
|---|---|
| Rank | int |
| Title | string |
| Description | string |
| Director | string |
| Actors | string |
| Year | int |
| Runtime (Minutes) | int |
| Rating | float |
| Votes | int |
| Revenue (Millions) | float |
| Metascore | float |
| Genre | string |
