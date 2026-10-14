---
tags:
- multabench
- tabular
---

# MUL_TEXT_NEWS_CHANNEL

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `News Category`
- **Rows:** 20284
- **Original source:** https://www.openml.org/search?type=data&id=46652

Load the benchmark splits with `multabench.load_split("MUL_TEXT_NEWS_CHANNEL", fold=0)`.

| Column | Type |
|---|---|
| n_tokens_content | float |
| n_unique_tokens | float |
| n_non_stop_words | float |
| n_non_stop_unique_tokens | float |
| num_hrefs | float |
| num_self_hrefs | int |
| num_imgs | int |
| num_videos | int |
| average_token_length | float |
| num_keywords | int |
| global_subjectivity | float |
| global_sentiment_polarity | float |
| global_rate_positive_words | float |
| global_rate_negative_words | float |
| rate_positive_words | float |
| rate_negative_words | float |
| article_title | string |
| News Category | string |
