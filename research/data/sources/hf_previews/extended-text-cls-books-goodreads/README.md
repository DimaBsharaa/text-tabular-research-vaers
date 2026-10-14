---
tags:
- multabench
- tabular
---

# MUL_TEXT_BOOKS_GOODREADS

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `Rating`
- **Rows:** 3967
- **Original source:** http://pages.cs.wisc.edu/~anhai/data/784_data/books2/csv_files/goodreads.csv

Load the benchmark splits with `multabench.load_split("MUL_TEXT_BOOKS_GOODREADS", fold=0)`.

| Column | Type |
|---|---|
| Title | string |
| Description | string |
| ISBN | string |
| ISBN13 | string |
| PageCount | int |
| FirstAuthor | string |
| SecondAuthor | string |
| ThirdAuthor | string |
| NumberofRatings | int |
| NumberofReviews | string |
| Publisher | string |
| PublishDate | datetime |
| Format | categorical |
| Language | categorical |
| Rating | string |
