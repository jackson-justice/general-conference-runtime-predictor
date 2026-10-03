"""Collect durations for non-talk program items (sustaining, audit report, solemn assembly).

These are not talks and never enter the model or the talk count. They are kept
in data/processed/program_items.csv so predict.py live can give a simple
historical-average estimate when asked, and score it separately.

Usage:
    uv run python scripts/collect_program_items.py            # items listed in outputs/collect_skipped_items.csv
"""
from __future__ import annotations

import argparse
import logging
import re
import sys

import pandas as pd

from collect import BASE, Fetcher, parse_talk
from general_conference_runtime_predictor.paths import OUTPUTS, PROGRAM_ITEMS_CSV

log = logging.getLogger("program-items")

KIND_RE = [
    ("sustaining", re.compile(r"^(The )?Sustaining of", re.I)),
    ("audit_report", re.compile(r"^Church Auditing Department Report", re.I)),
    ("statistical_report", re.compile(r"^Statistical Report", re.I)),
    ("solemn_assembly", re.compile(r"^Solemn Assembly", re.I)),
]


def kind_of(title: str) -> str | None:
    for k, rx in KIND_RE:
        if rx.match(title or ""):
            return k
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delay", type=float, default=1.5)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    skipped = pd.read_csv(OUTPUTS / "collect_skipped_items.csv", encoding="utf-8")
    skipped["kind"] = skipped.title.map(kind_of)
    items = skipped[skipped.kind.notna() & skipped.uri.notna()]
    fetcher = Fetcher(delay=args.delay)
    rows = []
    for r in items.itertuples():
        html = fetcher.get(f"{BASE}{r.uri}?lang=eng", stage="program-item")
        if html is None:
            continue
        try:
            t = parse_talk(html, r.uri)
        except Exception as e:  # noqa: BLE001
            log.error("could not parse %s: %r", r.uri, e)
            continue
        rows.append({"conference": r.conference, "session": r.session, "site_slot": r.site_slot, "kind": r.kind,
                     "title": r.title, "presenter": t["speaker"] or r.subtitle,
                     "duration_sec": None if t["duration_ms"] is None else t["duration_ms"] / 1000.0,
                     "duration_string": t["duration_string"], "url": f"{BASE}{r.uri}"})
    out = pd.DataFrame(rows)
    if PROGRAM_ITEMS_CSV.exists():  # keep earlier conferences; collect.py rewrites the skipped list per run
        old = pd.read_csv(PROGRAM_ITEMS_CSV, encoding="utf-8")
        out = pd.concat([old[~old.url.isin(out.url)], out], ignore_index=True)
    out = out.sort_values(["conference", "session", "site_slot"])
    out.to_csv(PROGRAM_ITEMS_CSV, index=False, encoding="utf-8")
    log.info("wrote %d program items to %s (network requests: %d)", len(out), PROGRAM_ITEMS_CSV, fetcher.n_network)
    print(out[["conference", "session", "kind", "presenter", "duration_string"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
