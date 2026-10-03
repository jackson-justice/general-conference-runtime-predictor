# General Conference talk duration predictor

Small Python ML project: predict how long a General Conference talk will run
from information known when the speaker is announced.

## Commands

```
uv sync                                   # install deps (Python 3.12, see pyproject.toml)
uv run python scripts/train.py --legacy-only          # baseline on data/raw/talks.csv only
uv run python scripts/collect.py --extra 2020-10 --compare-legacy   # scrape 2021-04..2026-04 (+ 2020-10 check)
uv run python scripts/verify_durations.py             # check unusable legacy runtimes against the official pages
uv run python scripts/train.py                        # legacy + scraped, all models
uv run python scripts/train.py --exploratory "why"    # same, but label test numbers as a re-look
uv run python -m pytest -q                            # rule tests (history leakage, order, parsing, aliases)
uv run python scripts/predict.py predict --speaker "..." --calling "Of the Seventy" --session sunday-morning --order 3 --log
uv run python scripts/predict.py log-actual --speaker "..." --session sunday-morning --actual 12:34
uv run python scripts/predict.py fill-actuals --conference 2026-10   # after collect.py scraped that conference
uv run python scripts/predict.py score
```

Always run Python through `uv run`. Shared code lives in
`src/general_conference_runtime_predictor/` (data, features, models, paths);
`scripts/` are thin CLIs.

## Layout

- `data/raw/talks.csv` - original dataset (Apr 2010 - Oct 2020). Never modify.
- `data/processed/talks_collected.csv` - scraper output (metadata, full text, durations).
- `data/processed/legacy_duration_fixes.csv` - verify_durations.py output: for each
  legacy row with a missing/unparseable/implausible runtime, the official video
  duration and the action taken (`corrected`, `filled`, `unresolved_*`).
- `data/processed/talks_dataset.csv` - merged modeling table written by train.py.
- `data/cache/` - raw HTML cache (gitignored).
- `outputs/` - metrics, test predictions, data report, scrape logs, prediction log.
- `models/bundle.joblib` - fitted models + empirical intervals (gitignored).

## Conventions

- Target is `duration_sec`. Rows with missing, unparseable or implausible
  durations keep `duration_status != "ok"` and are excluded from fitting and
  scoring. Never impute the target. Bad legacy runtimes (e.g. `17:60`) are
  not guessed at: `verify_durations.py` looks the talk up on the official page
  and `data.apply_duration_fixes` replaces the value only when that page
  carries a video duration (`duration_source = video_data_duration`,
  `duration_note` explains, `duration_raw` keeps the original string).
- Durations come from two sources that were verified to agree: the legacy CSV
  `runtime` column and the `data-duration` attribute on the talk page's
  `<video>` element. Do not mix in other measurements (e.g. mp3 length) without
  re-running `collect.py --compare-legacy` and documenting the result.
- Predictors are only: speaker, calling (`role_norm`, `calling_group`),
  session, `speaker_order`, month, and history features from strictly earlier
  conferences (`features.history_features`). Title, text, kicker, word count
  and the talk's own runtime are never predictors (word count is used only to
  flag impossible durations). No talk may contribute to another talk's history
  in the same conference; this also rules out target-encoding `speaker` inside
  a model (CatBoost gets no speaker categorical and one-hot encodes the rest).
  `tests/test_pipeline_rules.py` checks this.
- Speaker names: `speaker` is canonical via the explicit `data.SPEAKER_ALIASES`
  table; `speaker_source` keeps the spelling from the source. Add an alias only
  with a comment saying where each spelling was seen.
- Splits are chronological by whole conference: last `--n-test` conferences
  are test, the `--n-val` before them are validation. Settings are selected on
  validation; test is reported once. If test errors have been inspected and
  the pipeline changed afterwards, re-run with `--exploratory "<reason>"` so
  the outputs say so, and treat the next conference (logged via predict.py)
  as the real hold-out.
- Sustainings, audit reports, statistical reports, solemn assemblies and
  video interludes are not talks; `speaker_order` is renumbered over the
  remaining talks. At prediction time `--order` therefore counts talks only
  (the legacy CSV counted the same way, verified on 2020-10).
- Keep tuning grids small (see `models.GRIDS`). Keep the scraper polite:
  cache every page, `--delay` >= 1.5 s, log failures instead of raising.
- Intervals are empirical (validation residual quantiles) and are reported
  with their measured test coverage; do not describe them as calibrated
  without that check.
