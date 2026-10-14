---
tags:
- multabench
- tabular
---

# MUL_TEXT_MELBOURNE_AIRBNB

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `price_label`
- **Rows:** 18316
- **Original source:** https://www.openml.org/search?type=data&id=46665

Load the benchmark splits with `multabench.load_split("MUL_TEXT_MELBOURNE_AIRBNB", fold=0)`.

| Column | Type |
|---|---|
| name | string |
| summary | string |
| space | string |
| description | string |
| neighborhood_overview | string |
| notes | string |
| transit | string |
| access | string |
| interaction | string |
| house_rules | string |
| host_id | int |
| host_name | string |
| host_since | datetime |
| host_location | string |
| host_about | string |
| host_response_time | categorical |
| host_response_rate | categorical |
| host_is_superhost | categorical |
| host_neighborhood | string |
| host_verifications | string |
| host_has_profile_pic | categorical |
| host_identity_verified | categorical |
| street | string |
| neighborhood | categorical |
| city | categorical |
| suburb | string |
| state | categorical |
| zipcode | string |
| smart_location | string |
| country_code | categorical |
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
| guests_included | int |
| extra_people | int |
| minimum_nights | int |
| maximum_nights | int |
| calendar_updated | categorical |
| has_availability | categorical |
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
| requires_license | categorical |
| license | categorical |
| instant_bookable | categorical |
| cancellation_policy | categorical |
| require_guest_profile_picture | categorical |
| require_guest_phone_verification | categorical |
| calculated_host_listings_count | int |
| reviews_per_month | float |
| host_verifications_jumio | bool |
| host_verifications_government_id | bool |
| host_verifications_kba | bool |
| host_verifications_zhima_selfie | bool |
| host_verifications_facebook | bool |
| host_verifications_work_email | bool |
| host_verifications_google | bool |
| host_verifications_sesame | bool |
| host_verifications_manual_online | bool |
| host_verifications_manual_offline | bool |
| host_verifications_offline_government_id | bool |
| host_verifications_selfie | bool |
| host_verifications_reviews | bool |
| host_verifications_identity_manual | bool |
| host_verifications_sesame_offline | bool |
| host_verifications_weibo | bool |
| host_verifications_email | bool |
| host_verifications_sent_id | bool |
| host_verifications_phone | bool |
| price_label | int |
