# Source availability · October 14, 2026

The pasted context and both course PDFs were read. The shared ChatGPT conversation was retrieved after a direct retry and its recent project decisions were read. NFL drafting and tennis were exploratory examples, not a chosen dataset. Scope: Standard Track 2, research and EDA before choosing a new dataset. No new learner runs, encoder inference, training, or fine-tuning were performed. No push, fork, submodule, or full MulTaBench clone was made. The two original PDFs remain in the parent NLP folder.

## Evidence found

- All **56** original names and published five-learner outcome flags: paper v1, Appendix C.4.
- **23 passed, 33 failed** historically; **20** became core. The other successes are Consumer Complaints, News Channel, and Hearthstone Cards. The first two have extended releases; Hearthstone Cards is unreleased.
- Original registered source URLs for **56/56** candidates. They are provenance references; their download endpoints were not tested.
- Current public, ungated Hugging Face repositories with `data.parquet` listed: **40/56** (20 core, 20 extended). These artifacts were not downloaded in full.
- Four-condition historical scores for **276/280** learner–dataset pairs: **5,519** fold rows. Four missing pairs are both TabPFN variants on Wine Review and Spotify Genres. One CarDekho/TabPFN 2.0 TAR fold is missing.
- Historical released summaries: N, total structured columns, text columns, task, classes for **40** datasets. They are release properties, not verified untouched original-candidate properties.
- Current typed cards and metadata: target, numeric/categorical/boolean/date/text columns, row counts, artifact size, source and revision. All 40 viewer previews returned 50 rows, yielding **2,000** rows and **214** text-column profiles. No reported cell truncation.
- Individual original benchmark membership is verified for **21/56** candidates from the paper's overlap table and TextTabBench exports. Membership lists remain partial; blanks are unknown, never fabricated.
- Original-source target column names were not verified. `target_column` remains blank; `released_target_column` records the verified current-release name. A matching task prefix alone is insufficient to equate them.

## Exact official artifacts

Historical reference: [paper_version commit 3bb95079](https://github.com/alanarazi7/MulTaBench/tree/3bb95079a6146ed35d45187e98bcad5b063273a1). `official-paper-version` was not separately fetched or assumed identical. `master` is a newer leaderboard protocol and was used only to find current release identities.

- `multabench/leaderboard/results/analysis_curation_sensitivity/pool_scores_long.csv`: canonical historical folds. Official `committee_pool.py` traces the original five-learner scores to `tabstar_corpus/text_50_datasets.csv` and `texttabench_datasets.csv`.
- `analysis_curation_sensitivity/pass_matrix.csv` and `leaderboard/analysis/pass_matrix.py`: reproduction flags and exact arithmetic.
- `datasets_summary.csv` / `datasets_summary_extra.csv`: structure of core and extended releases.
- `leaderboard/analysis/pool_names.py`: explicit mappings between candidate IDs and release IDs.
- `datasets/all_datasets.py`: original sources and release IDs; targeted `benchmark/datasets/*_TEXT_*.py` recipes describe preparation.
- [Paper v1](https://arxiv.org/html/2605.10616v1): published acceptance flags and definitions.
- [Hugging Face organization](https://huggingface.co/multabench): current dataset cards, `metadata.json`, API sizes, revision IDs, and public viewer rows.

The bounded cache is approximately 6.68 MB, plus the small repository-tree listing. Full currently released datasets total about 352.0 MB and were not downloaded. SHA-256 hashes, exact source URLs, per-release revisions, and offsets are recorded in `data/sources/manifest.json` and `data/sources/hf_previews/manifest.json`.

## Accessible released artifacts

Public accessibility here means an ungated repository, listed parquet file, fetched metadata, and a successful bounded preview—not full original source availability. `data/dataset_inventory.csv` contains clickable source and release URLs.

| dataset_name | released_collection | task_type | released_target_column | number_of_passing_learners |
| --- | --- | --- | --- | --- |
| Consumer Complaints | extended | binary | Company response to consumer | 4 |
| Fake Job Postings | core | binary | fraudulent | 3 |
| Kickstarter | core | binary | Funding Status | 5 |
| IMDB Genre | extended | binary | Genre | 1 |
| Jigsaw Toxicity | core | binary | Is Toxic | 5 |
| OSHA Injury | extended | binary | Task Assigned | 2 |
| Product Sentiment | core | multiclass | Sentiment | 5 |
| Women's Clothing | core | multiclass | Rating | 5 |
| Michelin Guide | core | multiclass | Award | 5 |
| Wine Review | core | multiclass | variety | 3 |
| Airbnb Melbourne | extended | multiclass | price_label | 1 |
| Data Scientist Salary | core | multiclass | salary | 3 |
| News Channel | extended | multiclass | News Category | 5 |
| Spotify Genres | core | multiclass | track_genre | 3 |
| US Accidents | core | multiclass | Severity | 4 |
| American Eagle | extended | regression | price | 2 |
| Baby Products | core | regression | price | 5 |
| Book Price | core | regression | Price in USD | 4 |
| Mercari Marketplace | core | regression | log_price | 4 |
| WikiLiq Spirits | extended | regression | Price | 2 |
| Chocolate Bars | extended | regression | Rating | 2 |
| Ramen Ratings | extended | regression | Stars | 0 |
| Wine Poland | extended | regression | price (PLN) | 1 |
| Wine Vivino Spain | extended | regression | Rating | 2 |
| Zomato Restaurants | core | regression | rate | 4 |
| Airbnb Seattle | extended | regression | price | 0 |
| California Prices | extended | regression | Sold_Price | 2 |
| Vancouver Salaries | core | regression | remuneration | 5 |
| Montgomery Salaries | core | regression | current_annual_salary | 3 |
| SciMagojr Impact | core | regression | H index | 5 |
| Anime Planet | extended | regression | Rating Score | 0 |
| Goodreads Books | extended | regression | Rating | 1 |
| Book Readability | core | regression | New Dale Chall Readability Formula | 5 |
| Korean Drama | extended | regression | Score | 1 |
| Box Office | extended | regression | revenue | 0 |
| Rotten Tomatoes | core | regression | RatingValue | 4 |
| Video Games Sales | core | regression | Global_Sales | 5 |
| FIFA22 Wages | extended | regression | Wage in Euros | 2 |
| Used Cars Pakistan | extended | regression | Price | 1 |
| Used Cars Saudi | extended | regression | Price | 1 |

## Candidates requiring original-source retrieval for row-level EDA

| dataset_name | task_type | passed_curation | original_dataset_url |
| --- | --- | --- | --- |
| Yelp Reviews | multiclass | False | https://www.kaggle.com/datasets/omkarsabnis/yelp-reviews-dataset |
| Google QA | multiclass | False | https://www.openml.org/search?type=data&id=46658 |
| Hearthstone Cards | multiclass | True | https://www.kaggle.com/datasets/jeradrose/hearthstone-cards |
| BikeWale Bikes | regression | False | http://pages.cs.wisc.edu/~anhai/data/784_data/bikes/csv_files/bikewale.csv |
| CarDekho Cars | regression | False | https://www.kaggle.com/datasets/sukritchatterjee/used-cars-dataset-cardekho |
| JC Penney | regression | False | https://www.openml.org/search?type=data&id=46661 |
| Indian Laptops | regression | False | https://www.kaggle.com/datasets/dhanushbommavaram/laptop-dataset |
| Beer Ratings | regression | False | https://www.kaggle.com/datasets/ruthgn/beer-profile-and-ratings-data-set |
| Coffee Reviews | regression | False | https://www.kaggle.com/datasets/hanifalirsyad/coffee-scrap-coffeereview |
| SF Permits | regression | False | https://www.kaggle.com/datasets/aparnashastry/building-permit-applications-data |
| Company Size | regression | False | https://www.kaggle.com/datasets/peopledatalabssf/free-7-million-company-dataset |
| AI Job Salaries | regression | False | https://ai-jobs.net/salaries/download/salaries.csv |
| FilmTV Ratings | regression | False | https://www.kaggle.com/datasets/stefanoleone992/filmtv-movies-dataset |
| US Museums | regression | False | https://www.kaggle.com/datasets/markusschmitz/museums |
| NBA Draft Value | regression | False | https://www.kaggle.com/datasets/mattop/nba-draft-basketball-player-data-19892021 |
| Used Cars Mercedes | regression | False | https://www.kaggle.com/datasets/bogdansorin/second-hand-mercedes-benz-registered-2000-2023-ita |

## Version and task differences

These six releases change the original task prefix. Their current rows must not be treated as profiles of the original evaluation task:

| dataset_name | task_type | released_task_type | released_target_column |
| --- | --- | --- | --- |
| Consumer Complaints | binary | cls | Company response to consumer |
| American Eagle | regression | cls | price |
| California Prices | regression | cls | Sold_Price |
| Goodreads Books | regression | cls | Rating |
| Korean Drama | regression | cls | Score |
| Box Office | regression | cls | revenue |

The reproduction code rounds each condition mean to three decimals and uses gain **>=0.001**. Paper Appendix A.3 uses **>0.001**. Both are shown in the notebook. Two published model cells disagree with reconstruction (CarDekho/LightGBM and AI Job Salaries/CatBoost); the candidate-level accepted set remains unchanged. These are retained separately, not overwritten.

## Analysis plan carried out

1. Preserve published inventory and reconstruct learner-level gains from pinned scores.
2. Compare failures, learners, thresholds, quorum, and fold signs on the complete candidate pool.
3. Compare historical release structure and current preview text only with explicit coverage and task compatibility.
4. Keep core/extended analysis separate, inspect analog domains, and present a decision worksheet.
5. Save executed notebook outputs, standalone HTML, tables, figures, and these reports for review. Stop before new dataset construction or benchmark training.
