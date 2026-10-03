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
| Check unusable legacy runtimes against the official talk pages | `uv run python scripts/verify_durations.py` |
| Train and compare all models on legacy + collected data | `uv run python scripts/train.py` |
| Rule tests (history leakage, speaking order, parsing, aliases) | `uv run python -m pytest -q` |
| Predict a talk | `uv run python scripts/predict.py predict --speaker "Dale G. Renlund" --calling "Of the Quorum of the Twelve Apostles" --session sunday-morning --order 3 --log` |
| Record the actual duration afterwards | `uv run python scripts/predict.py log-actual --speaker "Dale G. Renlund" --session sunday-morning --actual 14:12` |
| Score logged predictions | `uv run python scripts/predict.py score` |

`--conference` on `predict`/`log-actual` defaults to `2026-10`. `--calling` takes
the role as printed on the talk page (e.g. `Of the Seventy`, `President of the
Church`, `Relief Society General President`).

### How to count `--order` during the live broadcast

`--order` is the talk's position among the **talks** of its session, starting
at 1. Count only spoken talks. Do not count, and do not leave a gap for:

- the sustaining of General Authorities and officers,
- the Church Auditing Department report or statistical report,
- a solemn assembly,
- videos, musical numbers, hymns and prayers.

Example: Saturday afternoon opens with the sustaining, then the audit report,
then the first speaker. That speaker is `--order 1`. If a talk precedes the
sustaining (as in October 2025), it is `--order 1` and the talk after the
sustaining is `--order 2`. This is how both data sources are numbered: the
legacy CSV already counted talks only, and the scraper drops the same items
and renumbers; the two agree on every October 2020 session.

## Data

- `data/raw/talks.csv` (unchanged): 794 rows, April 2010 to October 2020, with
  `runtime` as `m:ss`. 130 runtimes are missing, 6 are unparseable (`17:60`,
  `11:60`, ...), and 2 are impossible given the word count (a 1,960-word talk
  at `1:23`, a 1,345-word talk at `1:55`). One solemn assembly row is dropped
  as a non-talk.
- `data/processed/legacy_duration_fixes.csv` (`scripts/verify_durations.py`):
  each of those 138 rows was looked up on its official talk page and the
  `<video>` duration recorded. Result: all 130 missing runtimes were present
  on the page and are filled from it; the 6 `mm:60` strings all correspond to
  a page duration of `mm:59` (not `(mm+1):00`, so the source rounded up) and
  are corrected to the page value; the `1:23` talk is `18:09` on the page and
  is corrected; the `1:55` talk is `1:54` on the official page too, which is
  impossible for 1,345 words, so it stays excluded. Corrected rows carry
  `duration_source = video_data_duration` and a `duration_note`; the original
  string is kept in `duration_raw`. Nothing is imputed: a value is changed only
  when the official recording metadata supplies it.
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
  talks in each session, matching how the legacy CSV counted (see "How to
  count `--order`" above).
- Speaker aliases are an explicit table (`data.SPEAKER_ALIASES`): `Becky
  Craven` -> `Rebecca L. Craven` (site byline vs CSV), `Larry Echo Hawk` ->
  `Larry J. Echo Hawk` and `L. Harkness` -> `Lisa L. Harkness` (two spellings
  inside the CSV). `speaker` is the canonical name used for histories;
  `speaker_source` keeps the spelling from the source. A surname/initial scan
  of all 1,168 bylines found no other variants.
- Duplicates: none by URL and none by (conference, session, speaker, title).
  134 rows are a speaker's second or later talk in the same conference (mostly
  the President of the Church); these are real separate talks.

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
3. **CatBoost regression** with calling group, role, session and month as
   one-hot categorical features and the history features as numeric. It gets
   no `speaker` categorical: CatBoost's target statistics would average a
   speaker's other talks, including same-conference ones, which breaks the
   strictly-earlier rule. Speaker information enters only through the history
   features.

Splits are chronological by whole conference: the most recent 4 conferences
are the test set, the 4 before are validation. Hyperparameters (small fixed
grids) are chosen on validation; the chosen setting is refit on train+val and
scored once on test, then refit on everything for prediction.

History features never see the same conference: `features.add_history_features`
builds each conference's features from rows with a strictly smaller conference
index, and `history_features` raises if asked otherwise.
`tests/test_pipeline_rules.py` checks that two talks by one speaker in the same
conference do not enter each other's history.

**Test-set status.** The first run's test errors were inspected (by conference
and by calling) before the data fixes and the CatBoost change above. The
re-run is therefore labelled *exploratory* in `outputs/metrics_*.md`
(`train.py --exploratory`). The first untouched conference is October 2026:
predictions logged with `predict.py --log` before each talk and scored with
`predict.py score` are the clean evaluation.

Intervals are empirical: the half-width is the 80th percentile of validation
absolute errors, and the README reports the coverage that interval actually
achieved on the test conferences. They are not calibrated beyond that check.

## Results

Both stages below are the exploratory re-run after the duration fixes (791 of
793 legacy talks now timed, up from 655).

### Stage 1: legacy CSV only (`outputs/metrics_legacy.md`)

791 timed talks, 2010-04 .. 2020-10. Train 2010-04 .. 2016-10 (521),
validation 2017-04 .. 2018-10 (139), test 2019-04 .. 2020-10 (131 timed talks,
26 by speakers never seen before). All MAE in minutes.

| model | selected | val MAE | test MAE | seen (105) | unseen (26) | Church President (16) | Seventy (22) |
|---|---|---|---|---|---|---|---|
| naive (recent global mean) | - | 2.92 | 2.82 | 2.73 | 3.19 | 5.73 | 2.86 |
| baseline (speaker mean, shrunk to calling) | `min_n=1, shrink=3` | 1.59 | 1.92 | 2.06 | 1.35 | 5.77 | 0.73 |
| ridge | `alpha=30, no speaker one-hot` | 1.51 | 1.88 | 1.95 | 1.59 | 5.14 | 1.03 |
| catboost | `depth=6, RMSE, 668 iters` | **1.38** | **1.52** | 1.56 | 1.35 | 3.80 | 1.01 |

Unseen speakers are *easier* than seen ones: almost all are Seventies or
auxiliary leaders whose talks cluster tightly around 10 minutes, so the
calling fallback is accurate. The large errors are concentrated in the
President of the Church (MAE ~5 min), who gives both 2-6 minute remarks and
18-24 minute sermons in the same conference; nothing in the allowed predictors
separates the two.

### Stage 2: legacy + scraped, all models (`outputs/metrics_all.md`)

1,166 timed talks, 33 conferences (2010-04 .. 2026-04). Train 2010-04 ..
2022-04 (900 timed), validation 2022-10 .. 2024-04 (132), test 2024-10 ..
2026-04 (134 timed talks, 29 by unseen speakers). Every model is scored on
exactly the same 134 test rows. All MAE in minutes.

| model | selected | val MAE | test MAE | test median AE | 80% interval half-width | test coverage |
|---|---|---|---|---|---|---|
| naive | - | 1.84 | 2.10 | 1.68 | 2.46 | 73% |
| baseline | `min_n=1, shrink=3` | 1.20 | 1.39 | 0.83 | 1.60 | 69% |
| ridge | `alpha=30, no speaker one-hot` | 1.26 | 1.30 | 1.00 | 1.65 | 74% |
| **catboost** | `depth=6, RMSE, 55 iters` | **0.95** | **1.19** | 0.68 | 1.06 | **62%** |

Test MAE by segment (same rows for every model):

| segment | n | naive | baseline | ridge | catboost |
|---|---|---|---|---|---|
| all | 134 | 2.10 | 1.39 | 1.30 | 1.19 |
| Church President | 5 | 4.73 | 5.48 | 4.77 | 3.49 |
| Seventy (`Of the Seventy`) | 52 | 2.26 | 0.87 | 1.02 | 0.96 |
| Presidency of the Seventy | 2 | 1.49 | 0.47 | 0.89 | 0.44 |
| unseen speaker | 29 | 2.32 | 0.96 | 1.08 | 1.01 |
| seen speaker | 105 | 2.04 | 1.50 | 1.37 | 1.24 |

CatBoost against the baselines, paired per talk on the same 134 rows
(difference in absolute error, CatBoost minus other, minutes; bootstrap 95% CI):

| comparison | mean diff | 95% CI | share of talks CatBoost closer |
|---|---|---|---|
| vs naive | -0.91 | [-1.07, -0.74] | 82% |
| vs baseline (speaker mean) | -0.20 | [-0.38, -0.04] | 57% |
| vs ridge | -0.12 | [-0.24, -0.00] | 54% |

So CatBoost's edge over the speaker-mean baseline is real but modest (about 12
seconds per talk on average) and comes almost entirely from the First
Presidency and the Twelve; for Seventies the plain speaker mean is as good or
slightly better (0.87 vs 0.96).

Test MAE by conference (catboost / baseline): 2024-10 0.75 / 0.86, 2025-04
0.48 / 0.68, 2025-10 1.04 / 1.17, 2026-04 2.44 / 2.80. April 2026 was the
first conference under a new First Presidency, ran four sessions with more
talks each, and averaged 9.8 minutes per talk versus roughly 12.4 for the
preceding years; every model over-predicted it. That regime shift is also why
the empirical intervals under-cover on test: the catboost 80% interval (built
from validation errors) covered 62% of test talks, the others 69-74%. Treat
the intervals as rough guides, not calibrated bounds; `predict.py` prints the
measured coverage next to every interval.

Remaining error is dominated by the President of the Church (3.5 min) and the
President of the Twelve (3.6 min), for the same remarks-vs-sermon reason as in
stage 1.

### What would help next

- A "times this speaker has already spoken in this conference" feature would
  separate opening/closing remarks from full talks for the First Presidency.
  It is known before the talk begins but is outside the predictor list used
  here, so it was left out.
- Conformal-style intervals re-fit after each conference, so coverage tracks
  format changes like April 2026.

### October 2026 (the clean hold-out)

Log a prediction when each speaker is announced, before the talk starts, then
record the actual once it is known (from a stopwatch, or the talk page's
`m:ss` once published). Sessions: `saturday-morning`, `saturday-afternoon`,
`saturday-evening`, `sunday-morning`, `sunday-afternoon`.

```
# before the talk (count --order over talks only, see above)
uv run python scripts/predict.py predict --speaker "Dale G. Renlund" --calling "Of the Quorum of the Twelve Apostles" --session saturday-morning --order 2 --log

# after the talk
uv run python scripts/predict.py log-actual --speaker "Dale G. Renlund" --session saturday-morning --actual 14:12

# any time
uv run python scripts/predict.py score --conference 2026-10
```

Re-running `predict` for the same (speaker, session, order) replaces the
earlier logged prediction and keeps an actual already recorded. Once the
official pages are up, `collect.py --start 2026-10 --end 2026-10` adds the
conference to the dataset for the next retrain.

## Layout

```
scripts/collect.py   scrape index + talk pages, cache HTML, log failures
scripts/verify_durations.py  look up unusable legacy runtimes on the official pages -> legacy_duration_fixes.csv
scripts/train.py     build dataset, history features, chronological eval, save models/bundle.joblib
scripts/predict.py   predict / log-actual / score
src/general_conference_runtime_predictor/{data,features,models,paths}.py
data/raw/            original CSV (do not modify)
data/processed/      talks_collected.csv, legacy_duration_fixes.csv, talks_dataset.csv
tests/               pytest rule checks
data/cache/          HTML cache (gitignored)
outputs/             metrics_*.md/json, test_predictions_*.csv, data_report.json,
                     duration_compatibility.md, collect.log, verify_durations.log, predictions_log.csv
models/              bundle.joblib (gitignored)
```
