# General Conference talk duration predictor

Predicts how long a General Conference talk will run using only what is known
when the speaker is announced: who they are, their calling, the session, their
speaking order, the month, and how long their earlier talks ran.

## Setup

```
uv sync
```

Python 3.12, dependencies in `pyproject.toml` (pandas, scikit-learn, CatBoost,
requests, beautifulsoup4).

## Commands

| Step | Command |
|---|---|
| Baseline on the original CSV only | `uv run python scripts/train.py --legacy-only` |
| Collect 2021-04 .. 2026-04 talks (+ Oct 2020 for the compatibility check) | `uv run python scripts/collect.py --extra 2020-10 --compare-legacy` |
| Train and compare all models on legacy + collected data | `uv run python scripts/train.py` |
| Predict a talk | `uv run python scripts/predict.py predict --speaker "Dale G. Renlund" --calling "Of the Quorum of the Twelve Apostles" --session sunday-morning --order 3 --log` |
| Record the actual duration afterwards | `uv run python scripts/predict.py log-actual --speaker "Dale G. Renlund" --session sunday-morning --actual 14:12` |
| Score logged predictions | `uv run python scripts/predict.py score` |

`--conference` on `predict`/`log-actual` defaults to `2026-10`. `--calling` takes
the role as printed on the talk page (e.g. `Of the Seventy`, `President of the
Church`, `Relief Society General President`).

## Data

- `data/raw/talks.csv` (unchanged): 794 rows, April 2010 to October 2020, with
  `runtime` as `m:ss`. 130 runtimes are missing, 6 are unparseable (`17:60`,
  `11:60`, ...), and 2 are impossible given the word count (a 1,960-word talk
  at `1:23`, a 1,345-word talk at `1:55`). All of these are excluded from the
  target and never imputed. One solemn assembly row is dropped as a non-talk.
- `data/processed/talks_collected.csv`: scraped from
  `churchofjesuschrist.org/study/general-conference/<year>/<month>`. Each talk
  page embeds a base64 JSON state; the talk body carries a `<video>` element
  with `data-duration` (milliseconds) and `data-duration-string` (`m:ss`).
  That is the recording duration used here. The mp3 on the same page is
  longer (it includes intro/outro), so it is recorded as a URL only and never
  used as a duration.
- Compatibility check (`outputs/duration_compatibility.md`): for the 26
  October 2020 talks timed in both sources, the scraped video duration matches
  the CSV runtime to within 0.5 s, so the two sources are the same measurement.
  The legacy CSV remains the source of record for 2010-2020; scraped rows for
  overlapping conferences are only used for this check.
- Sustainings, audit/statistical reports, solemn assemblies and video
  interludes are not talks. Speaking order is renumbered over the remaining
  talks in each session, matching how the legacy CSV counted.

## Modeling

Predictors (all known before the talk begins): speaker, calling (printed role
and a 10-way calling group), session, speaking order, month, and history
features computed from strictly earlier conferences: speaker mean / median /
std / last / min / max duration and counts, speaker mean in the current
calling, calling mean (all-time and last two years), role mean, session mean,
and global mean. Title, text, kicker, word count and the talk's own runtime are
never predictors.

Models compared:

1. **Baseline**: speaker historical mean, shrunk toward the calling mean with
   `shrink` pseudo-talks, falling back to the calling mean (then global mean)
   when the speaker has fewer than `min_n` timed talks.
2. **Ridge regression** on the history features plus one-hot calling group,
   session, month and (optionally) speaker.
3. **CatBoost regression** with speaker, calling, role, session and month as
   categorical features and the history features as numeric.

Splits are chronological by whole conference: the most recent 4 conferences
are the test set, the 4 before are validation. Hyperparameters (small fixed
grids) are chosen on validation; the chosen setting is refit on train+val and
scored once on test, then refit on everything for prediction.

Intervals are empirical: the half-width is the 80th percentile of validation
absolute errors, and the README reports the coverage that interval actually
achieved on the test conferences. They are not calibrated beyond that check.

## Results

### Stage 1: legacy CSV only (`outputs/metrics_legacy.md`)

655 timed talks, 2010-04 .. 2020-10. Train 2010-04 .. 2016-10, validation
2017-04 .. 2018-10, test 2019-04 .. 2020-10 (114 timed talks, 21 by speakers
never seen before).

| model | selected | val MAE | test MAE | seen | unseen |
|---|---|---|---|---|---|
| naive (recent global mean) | - | 2.91 | 2.71 | 2.62 | 3.13 |
| baseline (speaker mean, shrink 3, min 3 talks) | `min_n=3, shrink=3` | 1.56 | 1.82 | 1.95 | 1.22 |
| ridge | `alpha=10, speaker one-hot` | 1.53 | 1.87 | 1.98 | 1.37 |

Unseen speakers are *easier* than seen ones: almost all are Seventies or
auxiliary leaders whose talks cluster tightly around 10 minutes, so the
calling fallback is accurate. The large errors are concentrated in the
President of the Church (MAE ~5 min), who gives both 2-6 minute remarks and
18-24 minute sermons in the same conference; nothing in the allowed predictors
separates the two.

### Stage 2: legacy + scraped, all models (`outputs/metrics_all.md`)

1,030 timed talks, 33 conferences (2010-04 .. 2026-04). Train 2010-04 ..
2022-04 (764 timed), validation 2022-10 .. 2024-04 (132), test 2024-10 ..
2026-04 (134 timed talks, 29 by unseen speakers). All MAE in minutes.

| model | selected | val MAE | test MAE | seen (105) | unseen (29) | 80% interval half-width | test coverage |
|---|---|---|---|---|---|---|---|
| naive | - | 1.84 | 2.10 | 2.04 | 2.32 | 2.46 | 73% |
| baseline | `min_n=1, shrink=3` | 1.20 | 1.37 | 1.48 | 0.96 | 1.67 | 72% |
| ridge | `alpha=30, no speaker one-hot` | 1.25 | 1.29 | 1.34 | 1.09 | 1.56 | 71% |
| **catboost** | `depth=6, RMSE, 143 iters` | **0.87** | **1.15** | 1.21 | 0.94 | 1.02 | **61%** |

Test MAE by conference (catboost): 2024-10 0.76, 2025-04 0.54, 2025-10 1.01,
2026-04 2.26. April 2026 was the first conference under a new First Presidency,
ran four sessions with more talks each, and averaged 9.8 minutes per talk versus
roughly 12.4 for the preceding years; every model over-predicted it by 2.2-2.8
minutes on average. That regime shift is also why the empirical intervals
under-cover on test: the catboost 80% interval (built from validation errors)
covered 61% of test talks, the others 71-73%. Treat the intervals as rough
guides, not calibrated bounds; `predict.py` prints the measured coverage next
to every interval.

Remaining error is dominated by the President of the Church and the President
of the Twelve (MAE 3.6 min each), for the same remarks-vs-sermon reason as in
stage 1. Seventies are predicted within about 0.8 minutes.

### What would help next

- Interpret `mm:60` runtimes in the legacy CSV as `(mm+1):00` (6 talks) and
  fill the 130 missing legacy runtimes from the official pages; the October
  2020 check showed the two sources agree exactly, so this is recovery of
  recorded data rather than imputation, but it needs scraping 2010-2020.
- A "times this speaker has already spoken in this conference" feature would
  separate opening/closing remarks from full talks for the First Presidency.
  It is known before the talk begins but is outside the predictor list used
  here, so it was left out.
- Conformal-style intervals re-fit after each conference, so coverage tracks
  format changes like April 2026.

### October 2026

Predictions can be logged as speakers are announced and scored once the talk
pages publish durations:

```
uv run python scripts/predict.py predict --speaker "..." --calling "..." --session saturday-morning --order 1 --log
uv run python scripts/predict.py log-actual --speaker "..." --session saturday-morning --actual 12:34
uv run python scripts/predict.py score --conference 2026-10
```

## Layout

```
scripts/collect.py   scrape index + talk pages, cache HTML, log failures
scripts/train.py     build dataset, history features, chronological eval, save models/bundle.joblib
scripts/predict.py   predict / log-actual / score
src/general_conference_runtime_predictor/{data,features,models,paths}.py
data/raw/            original CSV (do not modify)
data/processed/      talks_collected.csv, talks_dataset.csv
data/cache/          HTML cache (gitignored)
outputs/             metrics_*.md/json, test_predictions_*.csv, data_report.json,
                     duration_compatibility.md, collect.log, predictions_log.csv
models/              bundle.joblib (gitignored)
```
