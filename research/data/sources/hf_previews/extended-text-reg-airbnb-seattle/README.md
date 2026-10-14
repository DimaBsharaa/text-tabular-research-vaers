---
tags:
- multabench
- tabular
---

# REG_TEXT_AIRBNB_SEATTLE

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `price`
- **Rows:** 3818
- **Original source:** https://www.kaggle.com/datasets/airbnb/seattle

Load the benchmark splits with `multabench.load_split("REG_TEXT_AIRBNB_SEATTLE", fold=0)`.

| Column | Type |
|---|---|
| name | string |
| summary | string |
| space | string |
| description | string |
| neighborhood_overview | string |
| notes | string |
| transit | string |
| host_name | string |
| host_since | datetime |
| host_location | string |
| host_about | string |
| host_response_time | categorical |
| host_response_rate | float |
| host_acceptance_rate | float |
| host_is_superhost | categorical |
| host_neighbourhood | string |
| host_listings_count | float |
| host_total_listings_count | float |
| host_verifications | string |
| host_has_profile_pic | categorical |
| host_identity_verified | categorical |
| street | string |
| neighbourhood | categorical |
| neighbourhood_cleansed | categorical |
| neighbourhood_group_cleansed | categorical |
| city | categorical |
| state | categorical |
| zipcode | categorical |
| smart_location | categorical |
| latitude | float |
| longitude | float |
| is_location_exact | categorical |
| property_type | categorical |
| room_type | categorical |
| accommodates | int |
| bathrooms | float |
| bedrooms | float |
| beds | float |
| bed_type | categorical |
| amenities | string |
| square_feet | float |
| guests_included | int |
| minimum_nights | int |
| maximum_nights | int |
| calendar_updated | categorical |
| availability_30 | int |
| availability_60 | int |
| availability_90 | int |
| availability_365 | int |
| number_of_reviews | int |
| first_review | datetime |
| last_review | datetime |
| review_scores_rating | float |
| review_scores_accuracy | float |
| review_scores_cleanliness | float |
| review_scores_checkin | float |
| review_scores_communication | float |
| review_scores_location | float |
| review_scores_value | float |
| instant_bookable | categorical |
| cancellation_policy | categorical |
| require_guest_profile_picture | categorical |
| require_guest_phone_verification | categorical |
| calculated_host_listings_count | int |
| reviews_per_month | float |
| price | float |
