# MVP Summary Report

Generated at: 2026-03-23T23:14:37.777744

## Overview

- Logged data-quality batches: 4
- Logged training records: 4
- Logged updates: 1
- Current best model: CatBoostRegressor
- Current best model key: catboost
- Current best model version: 20260323_224900

## Data Quality

- Processed batches: 4
- Total rows across batches: 0
- Total missing values: 0
- Total duplicate rows: 0
- Total rows with negative target: 0
- Total rows with invalid duration: 0
- Total rows with invalid coordinates: 0

## Training

- Total training records: 4

### Latest selected best model

- created_at: 2026-03-23T16:13:24.103358
- model_name: CatBoostRegressor
- model_key: catboost
- model_version: 20260323_210055
- selection_strategy: mini_grid_search
- train_rows: 10239739
- test_rows: 2116911
- train_batches: 4
- test_batch: batch_0005_2015-01-26_2015-01-31.csv
- val_rmse: 2.024962188696533
- val_mae: 1.3416905176711935
- val_r2: 0.9615336430065582
- test_rmse: 1.704772868409165
- test_mae: 1.167740634649533
- test_r2: 0.95546794927041

### Training history

| created_at                 | model_name            | model_key     | selection_strategy   |   val_rmse |   val_mae |   val_r2 |   test_rmse |   test_mae |   test_r2 |   is_best |
|:---------------------------|:----------------------|:--------------|:---------------------|-----------:|----------:|---------:|------------:|-----------:|----------:|----------:|
| 2026-03-23T16:13:24.103358 | CatBoostRegressor     | catboost      | mini_grid_search     |    2.02496 |   1.34169 | 0.961534 |     1.70477 |    1.16774 |  0.955468 |       nan |
| 2026-03-23T16:31:50.508864 | DecisionTreeRegressor | decision_tree | mini_grid_search     |    2.03121 |   1.34638 | 0.961296 |     1.74902 |    1.18009 |  0.953126 |       nan |
| 2026-03-23T21:00:55.570471 | MLPRegressor          | mlp           | mini_grid_search     |    2.11679 |   1.36629 | 0.957966 |     1.71514 |    1.16963 |  0.954924 |       nan |
| 2026-03-23T16:13:24.103358 | CatBoostRegressor     | catboost      | mini_grid_search     |    2.02496 |   1.34169 | 0.961534 |     1.70477 |    1.16774 |  0.955468 |         1 |

## Updates

- Total update records: 1

### Latest update

- created_at: 2026-03-23T22:49:00.029705
- processed_batch_name: batch_0005_2015-01-26_2015-01-31.csv
- quality_row_count: 2181379
- incremental_update_attempted: True
- updated_candidate_models: 3
- post_update_model_name: CatBoostRegressor
- post_update_model_key: catboost
- post_update_model_type: catboost
- post_update_strategy: catboost_continue_fit
- post_update_model_version: 20260323_224900
- post_update_batch_mae: 1.139807378363458
- post_update_batch_rmse: 1.622376255306706
- post_update_batch_r2: 0.9596686454542276

### Update history

| created_at                 | processed_batch_name                 | post_update_model_name   | post_update_model_key   | post_update_strategy   |   post_update_batch_rmse |
|:---------------------------|:-------------------------------------|:-------------------------|:------------------------|:-----------------------|-------------------------:|
| 2026-03-23T22:49:00.029705 | batch_0005_2015-01-26_2015-01-31.csv | CatBoostRegressor        | catboost                | catboost_continue_fit  |                  1.62238 |

## Best Model

- model_name: CatBoostRegressor
- model_key: catboost
- model_type: catboost
- version: 20260323_224900
- preprocessing: native_catboost
- saved_at: 2026-03-23T22:49:00.022637
- updated_at: 2026-03-23T22:45:48.652038
- update_strategy: catboost_continue_fit
- selection_strategy: mini_grid_search

### Metrics

- val_mae: 1.3416905176711935
- val_rmse: 2.024962188696533
- val_r2: 0.9615336430065582
- test_mae: 1.167740634649533
- test_rmse: 1.704772868409165
- test_r2: 0.95546794927041

### Hyperparameters

- iterations: 300
- depth: 6
- learning_rate: 0.1
- loss_function: RMSE
- eval_metric: RMSE
- random_seed: 42
- verbose: False

- train_rows: 10239739
- test_rows: 2116911
- train_batches: 4
- test_batch: batch_0005_2015-01-26_2015-01-31.csv
- last_incremental_batch_rows: 2116911

### Last incremental batch metrics

- mae: 1.1398073783634581
- rmse: 1.622376255306706
- r2: 0.9596686454542277

### Feature columns

- VendorID
- passenger_count
- trip_distance
- pickup_longitude
- pickup_latitude
- RatecodeID
- store_and_fwd_flag
- dropoff_longitude
- dropoff_latitude
- pickup_hour
- pickup_weekday
- pickup_month
- trip_duration_min

### Categorical features

- VendorID
- RatecodeID
- store_and_fwd_flag

### Numerical features

- passenger_count
- trip_distance
- pickup_longitude
- pickup_latitude
- dropoff_longitude
- dropoff_latitude
- pickup_hour
- pickup_weekday
- pickup_month
- trip_duration_min
