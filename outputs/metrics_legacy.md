# Results (legacy)

Split: train 2010-04 .. 2016-10 (433 timed talks), val 2017-04 .. 2018-10 (108), test 2019-04 .. 2020-10 (114).

| model | selected params | val MAE | test MAE | test median AE | seen MAE (n) | unseen MAE (n) | 80% half-width | 80% coverage |
|---|---|---|---|---|---|---|---|---|
| naive | `{}` | 2.91 | 2.71 | 2.32 | 2.61 (93) | 3.13 (21) | 3.32 min | 0.74 |
| baseline | `{'min_n': 3, 'shrink': 3.0}` | 1.56 | 1.82 | 1.23 | 1.95 (93) | 1.22 (21) | 1.98 min | 0.75 |
| ridge | `{'alpha': 10.0, 'speaker_onehot': True}` | 1.53 | 1.87 | 1.26 | 1.98 (93) | 1.37 (21) | 1.99 min | 0.72 |

Recommended by validation MAE: **ridge**.

All MAE values are in minutes.
