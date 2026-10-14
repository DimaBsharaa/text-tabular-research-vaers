# Comparable domains and different outcomes

These are analog groups, not matched causal comparisons. Scores are median original learner gains. Profile values are absent when the released task differs or the candidate was not released. Numeric targets and units differ, so compare signs and criteria rather than ranking raw AUC and R² together.

## Salary (4 candidates)

| dataset_name | task_type | number_of_passing_learners | historical_released_n_rows | historical_released_structured_features | sample_longest_column_mean_chars | joint_signal_gain | tar_gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Data Scientist Salary | multiclass | 3 | 15841.0 | 1.0 | 98.9024 | 0.003 | 0.002 |
| Vancouver Salaries | regression | 5 | 44574.0 | 3.0 | 20.54 | 0.168 | 0.012 |
| Montgomery Salaries | regression | 3 | 9228.0 | 8.0 | 24.02 | 0.021 | 0.002 |
| AI Job Salaries | regression | 1 | unknown | unknown | unknown | 0.073 | -0.006 |

## Product price (7 candidates)

| dataset_name | task_type | number_of_passing_learners | historical_released_n_rows | historical_released_structured_features | sample_longest_column_mean_chars | joint_signal_gain | tar_gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baby Products | regression | 5 | 5085.0 | 8.0 | 51.16 | 0.024 | 0.016 |
| BikeWale Bikes | regression | 1 | unknown | unknown | unknown | -0.046 | 0.057 |
| Book Price | regression | 4 | 4989.0 | 3.0 | 1125.28 | 0.017 | 0.007 |
| CarDekho Cars | regression | 1 | unknown | unknown | unknown | -0.005 | -0.009 |
| JC Penney | regression | 2 | unknown | unknown | unknown | -0.001 | 0.031 |
| Indian Laptops | regression | 0 | unknown | unknown | unknown | 0.015 | -0.015 |
| Mercari Marketplace | regression | 4 | 100000.0 | 3.0 | 136.86 | 0.038 | 0.012 |

## Review / rating (6 candidates)

| dataset_name | task_type | number_of_passing_learners | historical_released_n_rows | historical_released_structured_features | sample_longest_column_mean_chars | joint_signal_gain | tar_gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Women's Clothing | multiclass | 5 | 18788.0 | 8.0 | 346.6 | 0.016 | 0.005 |
| Wine Review | multiclass | 3 | 84123.0 | 3.0 | 243.08 | 0.002 | 0.007 |
| Beer Ratings | regression | 0 | unknown | unknown | unknown | -0.003 | 0.001 |
| Coffee Reviews | regression | 0 | unknown | unknown | unknown | -0.001 | 0.0 |
| Ramen Ratings | regression | 0 | 4105.0 | 2.0 | 35.94 | 0.024 | -0.011 |
| Anime Planet | regression | 0 | 14391.0 | 7.0 | 266.24 | 0.085 | -0.013 |

## Real estate (4 candidates)

| dataset_name | task_type | number_of_passing_learners | historical_released_n_rows | historical_released_structured_features | sample_longest_column_mean_chars | joint_signal_gain | tar_gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Airbnb Melbourne | multiclass | 1 | 18316.0 | 68.0 | 789.38 | 0.002 | 0.001 |
| Airbnb Seattle | regression | 0 | 3818.0 | 51.0 | 841.16 | 0.009 | -0.004 |
| California Prices | regression | 2 | unknown | unknown | unknown | 0.005 | 0.0 |
| SF Permits | regression | 2 | unknown | unknown | unknown | 0.003 | 0.005 |

## Sports (historical analogs only) (2 candidates)

| dataset_name | task_type | number_of_passing_learners | historical_released_n_rows | historical_released_structured_features | sample_longest_column_mean_chars | joint_signal_gain | tar_gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FIFA22 Wages | regression | 2 | 19178.0 | 27.0 | 7.94 | 0.014 | 0.0 |
| NBA Draft Value | regression | 0 | unknown | unknown | unknown | -0.008 | 0.004 |

## What is supported

Salary examples include both successful datasets and AI Job Salaries, which fails the quorum. Product-price examples similarly span successful Baby Products/Book Price/Mercari and rejected products. TAR and joint gains, rather than the domain name, reveal different bottlenecks. The Notebook contains all five learners' scores and preview column profiles so these analogies can be inspected.

## What remains unknown

We did not run feature-level predictive analyses or new models, and several failed candidates lack row previews. Therefore we cannot explain outcomes by specific strong structured columns or establish that a particular text feature causes improvement. Housing releases with changed targets are excluded from original-task profile comparisons. Review source target processing before asserting that targets within an analog group are comparable.
