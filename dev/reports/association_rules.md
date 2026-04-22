# Association Rules Report

## Configuration

- min_support: 0.05
- min_confidence: 0.6
- batch_count: 6
- row_count_after_preparation: 13282499

## Purpose

Правила используются как вспомогательные паттерны для проверки правдоподобия и корректности данных.

## Top-5 rules

### Rule 1

- rule: `pickup_night & trip_duration_high -> total_amount_high & trip_distance_high`
- support: 0.0522
- confidence: 0.8871
- lift: 4.2453

### Rule 2

- rule: `pickup_weekend & total_amount_high -> trip_distance_high & trip_duration_high`
- support: 0.0576
- confidence: 0.7364
- lift: 4.0526

### Rule 3

- rule: `pickup_night & trip_distance_high & trip_duration_high -> total_amount_high`
- support: 0.0522
- confidence: 0.9642
- lift: 3.9788

### Rule 4

- rule: `pickup_night & total_amount_high -> trip_distance_high & trip_duration_high`
- support: 0.0522
- confidence: 0.7098
- lift: 3.9061

### Rule 5

- rule: `total_amount_high -> trip_distance_high & trip_duration_high`
- support: 0.1715
- confidence: 0.7079
- lift: 3.8953

## Full rules table

| rule                                                                          |   support |   confidence |    lift | A_text                                                   | B_text                                  |
|:------------------------------------------------------------------------------|----------:|-------------:|--------:|:---------------------------------------------------------|:----------------------------------------|
| pickup_night & trip_duration_high -> total_amount_high & trip_distance_high   | 0.0521587 |     0.887119 | 4.24534 | pickup_night & trip_duration_high                        | total_amount_high & trip_distance_high  |
| pickup_weekend & total_amount_high -> trip_distance_high & trip_duration_high | 0.0575865 |     0.736448 | 4.05261 | pickup_weekend & total_amount_high                       | trip_distance_high & trip_duration_high |
| pickup_night & trip_distance_high & trip_duration_high -> total_amount_high   | 0.0521587 |     0.964161 | 3.97876 | pickup_night & trip_distance_high & trip_duration_high   | total_amount_high                       |
| pickup_night & total_amount_high -> trip_distance_high & trip_duration_high   | 0.0521587 |     0.709824 | 3.9061  | pickup_night & total_amount_high                         | trip_distance_high & trip_duration_high |
| total_amount_high -> trip_distance_high & trip_duration_high                  | 0.171532  |     0.707853 | 3.89526 | total_amount_high                                        | trip_distance_high & trip_duration_high |
| trip_distance_high & trip_duration_high -> total_amount_high                  | 0.171532  |     0.943926 | 3.89526 | trip_distance_high & trip_duration_high                  | total_amount_high                       |
| pickup_night & total_amount_high & trip_duration_high -> trip_distance_high   | 0.0521587 |     0.961779 | 3.8483  | pickup_night & total_amount_high & trip_duration_high    | trip_distance_high                      |
| pickup_weekend & trip_distance_high & trip_duration_high -> total_amount_high | 0.0575865 |     0.929862 | 3.83722 | pickup_weekend & trip_distance_high & trip_duration_high | total_amount_high                       |
| pickup_night & trip_duration_high -> total_amount_high                        | 0.0542315 |     0.922373 | 3.80632 | pickup_night & trip_duration_high                        | total_amount_high                       |
| pickup_night & total_amount_high -> trip_distance_high                        | 0.0698107 |     0.950048 | 3.80136 | pickup_night & total_amount_high                         | trip_distance_high                      |
| pickup_weekend & total_amount_high & trip_duration_high -> trip_distance_high | 0.0575865 |     0.926723 | 3.70803 | pickup_weekend & total_amount_high & trip_duration_high  | trip_distance_high                      |
| pickup_weekend & total_amount_high -> trip_distance_high                      | 0.0721221 |     0.922338 | 3.69048 | pickup_weekend & total_amount_high                       | trip_distance_high                      |
| pickup_night & trip_duration_high -> trip_distance_high                       | 0.0540975 |     0.920094 | 3.68151 | pickup_night & trip_duration_high                        | trip_distance_high                      |
| pickup_weekend & trip_duration_high -> total_amount_high & trip_distance_high | 0.0575865 |     0.769196 | 3.68101 | pickup_weekend & trip_duration_high                      | total_amount_high & trip_distance_high  |
| trip_distance_high -> total_amount_high                                       | 0.208963  |     0.836109 | 3.45033 | trip_distance_high                                       | total_amount_high                       |
| total_amount_high -> trip_distance_high                                       | 0.208963  |     0.862319 | 3.45033 | total_amount_high                                        | trip_distance_high                      |
| total_amount_high & trip_duration_high -> trip_distance_high                  | 0.171532  |     0.859333 | 3.43839 | total_amount_high & trip_duration_high                   | trip_distance_high                      |
| trip_distance_high -> total_amount_high & trip_duration_high                  | 0.171532  |     0.686338 | 3.43839 | trip_distance_high                                       | total_amount_high & trip_duration_high  |
| pickup_weekend & trip_duration_high -> total_amount_high                      | 0.06214   |     0.830017 | 3.4252  | pickup_weekend & trip_duration_high                      | total_amount_high                       |
| pickup_night & trip_distance_high -> total_amount_high                        | 0.0698107 |     0.822959 | 3.39607 | pickup_night & trip_distance_high                        | total_amount_high                       |
