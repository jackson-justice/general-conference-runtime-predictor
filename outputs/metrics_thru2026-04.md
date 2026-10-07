# Results (thru2026-04)

**EXPLORATORY.** The test conferences had already been inspected before this run; reason for re-running: re-run of the October 2026 frozen configuration with the data restricted to 2026-04, to score it on October 2026 as a hold-out. Treat test numbers as a second look, not a fresh hold-out. The next untouched conference is the first one after the data ends.

Predictors as before October 2026 (without the expected-session-size columns).

Split: train 2010-04 .. 2022-04 (900 timed talks), val 2022-10 .. 2024-04 (132), test 2024-10 .. 2026-04 (134).

| model | selected params | val MAE | test MAE | test median AE | seen MAE (n) | unseen MAE (n) | range half-width (80th pct val error) | test coverage of range |
|---|---|---|---|---|---|---|---|---|
| naive | `{}` | 1.84 | 2.10 | 1.68 | 2.04 (105) | 2.32 (29) | 2.46 min | 0.73 |
| baseline | `{'min_n': 1, 'shrink': 3.0}` | 1.20 | 1.39 | 0.83 | 1.50 (105) | 0.96 (29) | 1.60 min | 0.69 |
| ridge | `{'alpha': 30.0, 'speaker_onehot': False, 'session_size': False}` | 1.26 | 1.30 | 1.00 | 1.37 (105) | 1.08 (29) | 1.65 min | 0.74 |
| catboost | `{'depth': 6, 'loss': 'RMSE', 'session_size': False, 'iterations': 55, 'early_stopping': False}` | 0.95 | 1.19 | 0.68 | 1.24 (105) | 1.01 (29) | 1.06 min | 0.62 |

Recommended by validation MAE: **catboost**.

## Test MAE by segment (minutes, same rows for every model)

| segment | n | naive | baseline | ridge | catboost |
|---|---|---|---|---|---|
| all | 134 | 2.10 | 1.39 | 1.30 | 1.19 |
| church_president | 5 | 4.73 | 5.48 | 4.77 | 3.49 |
| seventy | 52 | 2.26 | 0.87 | 1.02 | 0.96 |
| seventy_presidency | 2 | 1.49 | 0.47 | 0.89 | 0.44 |
| unseen_speaker | 29 | 2.32 | 0.96 | 1.08 | 1.01 |
| seen_speaker | 105 | 2.04 | 1.50 | 1.37 | 1.24 |

## CatBoost vs the baselines on identical test rows

Paired difference in absolute error (CatBoost minus other, minutes; negative favours CatBoost).

| comparison | n | mean diff | 95% bootstrap CI | share CatBoost better |
|---|---|---|---|---|
| catboost vs naive | 134 | -0.91 | [-1.07, -0.74] | 82% |
| catboost vs baseline | 134 | -0.20 | [-0.38, -0.04] | 57% |
| catboost vs ridge | 134 | -0.12 | [-0.24, -0.00] | 54% |

## Hold-out: 2026-10 (37 timed talks)

Predicted by the final models (fit on everything through the `--through` conference).

| model | hold-out MAE |
|---|---|
| naive | 2.43 |
| baseline | 2.66 |
| ridge | 2.40 |
| catboost | 1.94 |

All MAE values are in minutes.
