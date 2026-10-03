# Results (legacy)

**EXPLORATORY.** The test conferences had already been inspected before this run; reason for re-running: legacy durations verified/filled from official pages; CatBoost speaker target-encoding removed; test errors were inspected in the earlier run. Treat test numbers as a second look, not a fresh hold-out. The next untouched conference is the first one after the data ends.

Split: train 2010-04 .. 2016-10 (521 timed talks), val 2017-04 .. 2018-10 (139), test 2019-04 .. 2020-10 (131).

| model | selected params | val MAE | test MAE | test median AE | seen MAE (n) | unseen MAE (n) | 80% half-width | 80% coverage |
|---|---|---|---|---|---|---|---|---|
| naive | `{}` | 2.92 | 2.82 | 2.33 | 2.73 (105) | 3.19 (26) | 3.38 min | 0.76 |
| baseline | `{'min_n': 1, 'shrink': 3.0}` | 1.59 | 1.92 | 1.32 | 2.06 (105) | 1.35 (26) | 2.14 min | 0.75 |
| ridge | `{'alpha': 30.0, 'speaker_onehot': False}` | 1.51 | 1.88 | 1.23 | 1.95 (105) | 1.59 (26) | 2.03 min | 0.72 |
| catboost | `{'depth': 6, 'loss': 'RMSE', 'iterations': 668, 'early_stopping': False}` | 1.38 | 1.52 | 1.00 | 1.56 (105) | 1.35 (26) | 1.79 min | 0.70 |

Recommended by validation MAE: **catboost**.

## Test MAE by segment (minutes, same rows for every model)

| segment | n | naive | baseline | ridge | catboost |
|---|---|---|---|---|---|
| all | 131 | 2.82 | 1.92 | 1.88 | 1.52 |
| church_president | 16 | 5.73 | 5.77 | 5.14 | 3.80 |
| seventy | 22 | 2.86 | 0.73 | 1.03 | 1.01 |
| seventy_presidency | 4 | 2.19 | 0.27 | 0.66 | 0.33 |
| unseen_speaker | 26 | 3.19 | 1.35 | 1.59 | 1.35 |
| seen_speaker | 105 | 2.73 | 2.06 | 1.95 | 1.56 |

## CatBoost vs the baselines on identical test rows

Paired difference in absolute error (CatBoost minus other, minutes; negative favours CatBoost).

| comparison | n | mean diff | 95% bootstrap CI | share CatBoost better |
|---|---|---|---|---|
| catboost vs naive | 131 | -1.30 | [-1.56, -1.04] | 84% |
| catboost vs baseline | 131 | -0.40 | [-0.61, -0.19] | 56% |
| catboost vs ridge | 131 | -0.36 | [-0.60, -0.15] | 60% |

All MAE values are in minutes.
