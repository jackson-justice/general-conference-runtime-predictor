# Results (all)

**EXPLORATORY.** The test conferences had already been inspected before this run; reason for re-running: October 2026 added to the data and expected-session-size history features added after its live run; the test conferences were inspected earlier. The clean hold-out for this configuration is April 2027. Treat test numbers as a second look, not a fresh hold-out. The next untouched conference is the first one after the data ends.

Ridge and CatBoost use the expected session size learned from earlier conferences (`session_n_prev`, `session_n_recent`).

Split: train 2010-04 .. 2022-10 (935 timed talks), val 2023-04 .. 2024-10 (131), test 2025-04 .. 2026-10 (137).

| model | selected params | val MAE | test MAE | test median AE | seen MAE (n) | unseen MAE (n) | range half-width (80th pct val error) | test coverage of range |
|---|---|---|---|---|---|---|---|---|
| naive | `{}` | 1.82 | 2.28 | 1.76 | 2.14 (103) | 2.69 (34) | 2.58 min | 0.65 |
| baseline | `{'min_n': 1, 'shrink': 3.0}` | 1.05 | 1.86 | 1.54 | 1.99 (103) | 1.47 (34) | 1.44 min | 0.47 |
| ridge | `{'alpha': 0.3, 'speaker_onehot': False, 'session_size': True}` | 1.08 | 1.70 | 1.37 | 1.77 (103) | 1.48 (34) | 1.52 min | 0.56 |
| catboost | `{'depth': 6, 'loss': 'MAE', 'session_size': True, 'iterations': 440, 'early_stopping': False}` | 0.96 | 1.50 | 1.18 | 1.57 (103) | 1.26 (34) | 1.24 min | 0.51 |

Recommended by validation MAE: **catboost**.

## Test MAE by segment (minutes, same rows for every model)

| segment | n | naive | baseline | ridge | catboost |
|---|---|---|---|---|---|
| all | 137 | 2.28 | 1.86 | 1.70 | 1.50 |
| church_president | 6 | 5.20 | 6.08 | 5.44 | 3.59 |
| seventy | 53 | 2.68 | 1.30 | 1.39 | 1.18 |
| seventy_presidency | 2 | 2.13 | 0.94 | 1.44 | 1.23 |
| unseen_speaker | 34 | 2.69 | 1.47 | 1.48 | 1.26 |
| seen_speaker | 103 | 2.14 | 1.99 | 1.77 | 1.57 |

## CatBoost vs the baselines on identical test rows

Paired difference in absolute error (CatBoost minus other, minutes; negative favours CatBoost).

| comparison | n | mean diff | 95% bootstrap CI | share CatBoost better |
|---|---|---|---|---|
| catboost vs naive | 137 | -0.78 | [-1.02, -0.56] | 75% |
| catboost vs baseline | 137 | -0.37 | [-0.56, -0.19] | 63% |
| catboost vs ridge | 137 | -0.20 | [-0.36, -0.05] | 57% |

All MAE values are in minutes.
