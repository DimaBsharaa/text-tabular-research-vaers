---
tags:
- multabench
- tabular
---

# MUL_TEXT_US_ACCIDENTS

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `Severity`
- **Rows:** 100001
- **Original source:** https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

Load the benchmark splits with `multabench.load_split("MUL_TEXT_US_ACCIDENTS", fold=0)`.

| Column | Type |
|---|---|
| Source | categorical |
| Start_Time | datetime |
| End_Time | datetime |
| Start_Lat | float |
| Start_Lng | float |
| End_Lat | float |
| End_Lng | float |
| Distance(mi) | float |
| Description | string |
| Street | string |
| City | string |
| County | string |
| State | categorical |
| Zipcode | string |
| Country | categorical |
| Timezone | categorical |
| Airport_Code | string |
| Weather_Timestamp | datetime |
| Temperature(F) | float |
| Wind_Chill(F) | float |
| Humidity(%) | float |
| Pressure(in) | float |
| Visibility(mi) | float |
| Wind_Direction | categorical |
| Wind_Speed(mph) | float |
| Precipitation(in) | float |
| Weather_Condition | categorical |
| Amenity | bool |
| Bump | bool |
| Crossing | bool |
| Give_Way | bool |
| Junction | bool |
| No_Exit | bool |
| Railway | bool |
| Roundabout | bool |
| Station | bool |
| Stop | bool |
| Traffic_Calming | bool |
| Traffic_Signal | bool |
| Turning_Loop | bool |
| Sunrise_Sunset | categorical |
| Civil_Twilight | categorical |
| Nautical_Twilight | categorical |
| Astronomical_Twilight | categorical |
| Severity | int |
