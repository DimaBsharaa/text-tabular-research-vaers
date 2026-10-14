---
tags:
- multabench
- tabular
---

# MUL_TEXT_SPOTIFY_GENRES

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `track_genre`
- **Rows:** 114000
- **Original source:** https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

Load the benchmark splits with `multabench.load_split("MUL_TEXT_SPOTIFY_GENRES", fold=0)`.

| Column | Type |
|---|---|
| artists | string |
| album_name | string |
| track_name | string |
| popularity | int |
| duration_ms | int |
| explicit | bool |
| danceability | float |
| energy | float |
| key | int |
| loudness | float |
| mode | int |
| speechiness | float |
| acousticness | float |
| instrumentalness | float |
| liveness | float |
| valence | float |
| tempo | float |
| time_signature | int |
| track_genre | string |
