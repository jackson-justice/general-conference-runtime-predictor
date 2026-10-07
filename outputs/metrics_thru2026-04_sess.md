# Results (thru2026-04_sess)

**EXPLORATORY.** The test conferences had already been inspected before this run; reason for re-running: expected-session-size history features added after the October 2026 live run; test conferences were inspected earlier. Treat test numbers as a second look, not a fresh hold-out. The next untouched conference is the first one after the data ends.

Ridge and CatBoost use the expected session size learned from earlier conferences (`session_n_prev`, `session_n_recent`).

Split: train 2010-04 .. 2022-04 (900 timed talks), val 2022-10 .. 2024-04 (132), test 2024-10 .. 2026-04 (134).

| model | selected params | val MAE | test MAE | test median AE | seen MAE (n) | unseen MAE (n) | range half-width (80th pct val error) | test coverage of range |
|---|---|---|---|---|---|---|---|---|
| naive | `{}` | 1.84 | 2.10 | 1.68 | 2.04 (105) | 2.32 (29) | 2.46 min | 0.73 |
| baseline | `{'min_n': 1, 'shrink': 3.0}` | 1.20 | 1.39 | 0.83 | 1.50 (105) | 0.96 (29) | 1.60 min | 0.69 |
| ridge | `{'alpha': 0.3, 'speaker_onehot': False, 'session_size': True}` | 1.23 | 1.37 | 1.06 | 1.43 (105) | 1.14 (29) | 1.76 min | 0.75 |
| catboost | `{'depth': 6, 'loss': 'RMSE', 'session_size': True, 'iterations': 68, 'early_stopping': False}` | 0.89 | 1.14 | 0.66 | 1.18 (105) | 0.98 (29) | 1.02 min | 0.61 |

Recommended by validation MAE: **catboost**.

## Test MAE by segment (minutes, same rows for every model)

| segment | n | naive | baseline | ridge | catboost |
|---|---|---|---|---|---|
| all | 134 | 2.10 | 1.39 | 1.37 | 1.14 |
| church_president | 5 | 4.73 | 5.48 | 4.74 | 3.55 |
| seventy | 52 | 2.26 | 0.87 | 1.08 | 0.91 |
| seventy_presidency | 2 | 1.49 | 0.47 | 1.41 | 0.09 |
| unseen_speaker | 29 | 2.32 | 0.96 | 1.14 | 0.98 |
| seen_speaker | 105 | 2.04 | 1.50 | 1.43 | 1.18 |

## CatBoost vs the baselines on identical test rows

Paired difference in absolute error (CatBoost minus other, minutes; negative favours CatBoost).

| comparison | n | mean diff | 95% bootstrap CI | share CatBoost better |
|---|---|---|---|---|
| catboost vs naive | 134 | -0.96 | [-1.14, -0.78] | 80% |
| catboost vs baseline | 134 | -0.25 | [-0.42, -0.09] | 57% |
| catboost vs ridge | 134 | -0.23 | [-0.38, -0.09] | 62% |

## Hold-out: 2026-10 (37 timed talks)

Predicted by the final models (fit on everything through the `--through` conference).

| model | hold-out MAE |
|---|---|
| naive | 2.43 |
| baseline | 2.66 |
| ridge | 1.96 |
| catboost | 1.58 |

All MAE values are in minutes.
