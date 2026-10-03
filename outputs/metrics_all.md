# Results (all)

Split: train 2010-04 .. 2022-04 (764 timed talks), val 2022-10 .. 2024-04 (132), test 2024-10 .. 2026-04 (134).

| model | selected params | val MAE | test MAE | test median AE | seen MAE (n) | unseen MAE (n) | 80% half-width | 80% coverage |
|---|---|---|---|---|---|---|---|---|
| naive | `{}` | 1.84 | 2.10 | 1.68 | 2.04 (105) | 2.32 (29) | 2.46 min | 0.73 |
| baseline | `{'min_n': 1, 'shrink': 3.0}` | 1.20 | 1.37 | 0.78 | 1.48 (105) | 0.95 (29) | 1.67 min | 0.72 |
| ridge | `{'alpha': 30.0, 'speaker_onehot': False}` | 1.25 | 1.29 | 0.97 | 1.34 (105) | 1.09 (29) | 1.56 min | 0.71 |
| catboost | `{'depth': 6, 'loss': 'RMSE', 'iterations': 143, 'early_stopping': False}` | 0.87 | 1.15 | 0.65 | 1.21 (105) | 0.94 (29) | 1.02 min | 0.61 |

Recommended by validation MAE: **catboost**.

All MAE values are in minutes.
