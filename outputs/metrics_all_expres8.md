# Results (all_expres8)

**EXPLORATORY.** The test conferences had already been inspected before this run; reason for re-running: experiment: exclude Church President talks under 8 min. Treat test numbers as a second look, not a fresh hold-out. The next untouched conference is the first one after the data ends.

**Exclusion experiment:** 46 Church President talks shorter than 8 min excluded from history, fitting and scoring. Test rows differ from the default run, so numbers are not comparable with it row for row.

Split: train 2010-04 .. 2022-04 (859 timed talks), val 2022-10 .. 2024-04 (129), test 2024-10 .. 2026-04 (132).

| model | selected params | val MAE | test MAE | test median AE | seen MAE (n) | unseen MAE (n) | 80% half-width | 80% coverage |
|---|---|---|---|---|---|---|---|---|
| naive | `{}` | 1.78 | 1.98 | 1.66 | 1.88 (103) | 2.35 (29) | 2.31 min | 0.73 |
| baseline | `{'min_n': 1, 'shrink': 3.0}` | 0.98 | 1.25 | 0.83 | 1.33 (103) | 0.96 (29) | 1.44 min | 0.67 |
| ridge | `{'alpha': 30.0, 'speaker_onehot': False}` | 0.82 | 1.13 | 0.90 | 1.15 (103) | 1.04 (29) | 1.24 min | 0.67 |
| catboost | `{'depth': 6, 'loss': 'RMSE', 'iterations': 98, 'early_stopping': False}` | 0.71 | 1.12 | 0.68 | 1.17 (103) | 0.95 (29) | 0.97 min | 0.60 |

Recommended by validation MAE: **catboost**.

## Test MAE by segment (minutes, same rows for every model)

| segment | n | naive | baseline | ridge | catboost |
|---|---|---|---|---|---|
| all | 132 | 1.98 | 1.25 | 1.13 | 1.12 |
| church_president | 3 | 1.31 | 2.19 | 1.70 | 3.15 |
| seventy | 52 | 2.30 | 0.87 | 0.95 | 0.89 |
| seventy_presidency | 2 | 1.58 | 0.47 | 1.09 | 0.09 |
| unseen_speaker | 29 | 2.35 | 0.96 | 1.04 | 0.95 |
| seen_speaker | 103 | 1.88 | 1.33 | 1.15 | 1.17 |

## CatBoost vs the baselines on identical test rows

Paired difference in absolute error (CatBoost minus other, minutes; negative favours CatBoost).

| comparison | n | mean diff | 95% bootstrap CI | share CatBoost better |
|---|---|---|---|---|
| catboost vs naive | 132 | -0.86 | [-1.06, -0.63] | 77% |
| catboost vs baseline | 132 | -0.13 | [-0.25, +0.00] | 54% |
| catboost vs ridge | 132 | -0.00 | [-0.10, +0.10] | 53% |

All MAE values are in minutes.
