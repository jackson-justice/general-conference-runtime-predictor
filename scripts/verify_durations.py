"""Check legacy rows without a usable duration against the official talk page.

For every legacy CSV row whose runtime is missing, unparseable (e.g. "17:60")
or implausible for its word count, fetch the talk page (cached, rate-limited)
and read the recording duration from the <video> element. Writes
data/processed/legacy_duration_fixes.csv; data.py applies rows whose action is
"corrected" or "filled". The original runtime string is kept in duration_raw.

Usage:
    uv run python scripts/verify_durations.py [--delay 1.5] [--refresh]
"""
from __future__ import annotations

import argparse
import logging
import re
import sys

import pandas as pd

from collect import Fetcher, parse_talk
from general_conference_runtime_predictor.data import WPM_MAX, WPM_MIN, load_legacy, parse_runtime
from general_conference_runtime_predictor.paths import LEGACY_FIXES_CSV, OUTPUTS

log = logging.getLogger("verify")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delay", type=float, default=1.5)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(OUTPUTS / "verify_durations.log", encoding="utf-8")])

    legacy = load_legacy(apply_fixes=False)
    todo = legacy[legacy.duration_status != "ok"]
    log.info("checking %d legacy rows without a usable duration", len(todo))
    fetcher = Fetcher(delay=args.delay, refresh=args.refresh)
    rows = []
    for r in todo.itertuples():
        uri = re.sub(r"^https?://www\.churchofjesuschrist\.org", "", r.url)
        html = fetcher.get(f"{r.url}?lang=eng", stage="verify")
        video_ms = video_str = None
        if html is not None:
            try:
                talk = parse_talk(html, uri)
                video_ms, video_str = talk["duration_ms"], talk["duration_string"]
            except Exception as e:  # noqa: BLE001
                fetcher.failures.append({"url": r.url, "stage": "verify-parse", "error": repr(e)})
                log.error("could not parse %s: %r", r.url, e)
        video_sec = None if video_ms is None else video_ms / 1000.0
        raw = None if pd.isna(r.duration_raw) else str(r.duration_raw)
        # For "mm:60" strings the natural reading is (mm+1):00; record whether the video agrees.
        rounded_sec = None
        m = re.match(r"^(\d+):60$", raw or "")
        if m:
            rounded_sec = (int(m.group(1)) + 1) * 60.0
        wpm = None if video_sec is None or pd.isna(r.num_words) else r.num_words / (video_sec / 60.0)
        if video_sec is None:
            action = "unresolved_no_video_duration"
        elif wpm is not None and not (WPM_MIN <= wpm <= WPM_MAX):
            action = "unresolved_video_implausible"
        elif r.duration_status == "missing":
            action = "filled"
        else:
            action = "corrected"
        rows.append({
            "url": r.url, "conference": r.conference, "speaker": r.speaker, "title": r.title,
            "legacy_raw": raw, "legacy_status": r.duration_status, "num_words": r.num_words,
            "video_duration_ms": video_ms, "video_duration_string": video_str, "video_sec": video_sec,
            "mm60_rounded_sec": rounded_sec,
            "mm60_matches_video": None if rounded_sec is None or video_sec is None else abs(rounded_sec - video_sec) <= 1.5,
            "words_per_min": None if wpm is None else round(wpm, 1),
            "action": action, "source": "video_data_duration" if video_sec is not None else None,
        })
    out = pd.DataFrame(rows)
    LEGACY_FIXES_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(LEGACY_FIXES_CSV, index=False, encoding="utf-8")
    log.info("wrote %s (network requests: %d)", LEGACY_FIXES_CSV, fetcher.n_network)
    log.info("actions: %s", out.action.value_counts().to_dict())
    if fetcher.failures:
        pd.DataFrame(fetcher.failures).to_csv(OUTPUTS / "verify_failures.csv", index=False, encoding="utf-8")
        log.warning("%d failures logged to outputs/verify_failures.csv", len(fetcher.failures))
    return 0


if __name__ == "__main__":
    sys.exit(main())
