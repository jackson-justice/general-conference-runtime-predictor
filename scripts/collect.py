"""Collect General Conference talks from the official Church study pages.

For each conference index page, read the session list and talk entries from
the embedded __INITIAL_STATE__ JSON, then fetch every talk page for metadata,
full text, word count, and the recording duration exposed on the page's
<video> element (data-duration in milliseconds, data-duration-string m:ss).

Usage:
    uv run python scripts/collect.py                            # 2021-04 .. 2026-04
    uv run python scripts/collect.py --extra 2020-10 --compare-legacy
    uv run python scripts/collect.py --start 2026-04 --end 2026-04 --refresh
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import logging
import re
import sys
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from general_conference_runtime_predictor.data import (
    NON_TALK_RE, clean_text, conference_key, load_legacy, parse_conference_key, parse_runtime,
)
from general_conference_runtime_predictor.paths import COLLECTED_CSV, DATA_CACHE, OUTPUTS

BASE = "https://www.churchofjesuschrist.org"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; general-conference-runtime-predictor/0.1; personal research)",
    "Accept-Language": "en",
}
log = logging.getLogger("collect")


class Fetcher:
    """Cached, rate-limited GET with a few retries. Failures are recorded, not raised."""

    def __init__(self, delay: float, refresh: bool = False):
        self.delay, self.refresh = delay, refresh
        self.session = requests.Session()
        self.last_request = 0.0
        self.failures: list[dict] = []
        self.n_network = 0
        DATA_CACHE.mkdir(parents=True, exist_ok=True)

    def get(self, url: str, stage: str) -> str | None:
        path = DATA_CACHE / (hashlib.sha1(url.encode()).hexdigest() + ".html")
        if path.exists() and not self.refresh:
            return path.read_text(encoding="utf-8")
        for attempt in range(3):
            wait = self.delay - (time.monotonic() - self.last_request)
            if wait > 0:
                time.sleep(wait)
            try:
                self.last_request = time.monotonic()
                self.n_network += 1
                r = self.session.get(url, headers=HEADERS, timeout=30)
                if r.status_code == 200:
                    path.write_text(r.text, encoding="utf-8")
                    return r.text
                if r.status_code == 404:
                    log.warning("404 %s", url)
                    self.failures.append({"url": url, "stage": stage, "error": "HTTP 404"})
                    return None
                log.warning("HTTP %s on %s (attempt %d)", r.status_code, url, attempt + 1)
                err = f"HTTP {r.status_code}"
            except requests.RequestException as e:
                log.warning("request error on %s (attempt %d): %s", url, attempt + 1, e)
                err = str(e)
            time.sleep(self.delay * (2 ** attempt))
        self.failures.append({"url": url, "stage": stage, "error": err})
        return None


def parse_state(html: str) -> dict:
    m = re.search(r"__INITIAL_STATE__\s*=\s*(.*?);?\s*</script>", html, re.S)
    if not m:
        raise ValueError("no __INITIAL_STATE__ in page")
    raw = json.loads(m.group(1).strip())
    if isinstance(raw, str):
        raw = json.loads(base64.b64decode(raw).decode("utf-8"))
    return raw


def conference_sessions(fetcher: Fetcher, year: int, month_num: int) -> list[dict] | None:
    url = f"{BASE}/study/general-conference/{year}/{month_num:02d}?lang=eng"
    html = fetcher.get(url, stage="index")
    if html is None:
        return None
    try:
        state = parse_state(html)
        books = state["reader"]["bookStore"]
        book = next(v for k, v in books.items() if k.endswith(f"/general-conference/{year}/{month_num:02d}"))
    except Exception as e:  # noqa: BLE001
        fetcher.failures.append({"url": url, "stage": "index-parse", "error": repr(e)})
        log.error("could not parse index %s: %r", url, e)
        return None
    sessions = []
    for entry in book.get("entries", []):
        sec = entry.get("section")
        if not sec:
            continue
        slug = sec["uri"].rsplit("/", 1)[-1].removesuffix("-session")
        items = []
        for pos, e in enumerate(sec.get("entries", []), start=1):
            c = e.get("content", e)
            items.append({"uri": c.get("uri"), "title": clean_text(c.get("title")),
                          "subtitle": clean_text(c.get("subtitle")), "site_slot": pos})
        sessions.append({"session": slug, "session_title": sec.get("title"), "items": items})
    return sessions


SPEAKER_PREFIX_RE = re.compile(r"^(?:by|presented by)\s+", re.I)
HONORIFIC_RE = re.compile(r"^(?:President|Elder|Sister|Brother|Bishop)\s+", re.I)


def clean_speaker(raw: str | None) -> str | None:
    if not raw:
        return None
    s = SPEAKER_PREFIX_RE.sub("", clean_text(raw))
    return HONORIFIC_RE.sub("", s).strip()


def parse_talk(html: str, uri: str) -> dict:
    state = parse_state(html)
    store = state["reader"]["contentStore"]
    tail = uri.split("/study", 1)[-1]
    cs = next(v for k, v in store.items() if k.endswith(tail))
    meta = cs.get("meta", {})
    soup = BeautifulSoup(cs["content"]["body"], "html.parser")

    def sel(css):
        node = soup.select_one(css)
        return clean_text(node.get_text(" ", strip=True)) if node else None

    out = {
        "title": clean_text(meta.get("title")) or sel("h1"),
        "speaker_raw": sel(".author-name"),
        "role": sel(".author-role"),
        "kicker": sel(".kicker"),
        "audio_url": (meta.get("audio") or [{}])[0].get("mediaUrl"),
        "duration_ms": None, "duration_string": None, "video_id": None, "duration_flag": "no_video_element",
    }
    out["speaker"] = clean_speaker(out["speaker_raw"])
    video = soup.find("video")
    if video is not None:
        ms = video.get("data-duration")
        out["duration_ms"] = int(ms) if ms and ms.isdigit() else None
        out["duration_string"] = video.get("data-duration-string")
        out["video_id"] = video.get("data-video-id")
        if out["duration_ms"] is None:
            out["duration_flag"] = "video_without_duration"
        else:
            secs, status = parse_runtime(out["duration_string"])
            if status == "ok" and abs(secs - out["duration_ms"] / 1000.0) > 1.5:
                out["duration_flag"] = "ms_string_mismatch"
            else:
                out["duration_flag"] = "ok"
    body = soup.select_one(".body-block")
    if body is not None:
        for sup in body.select("sup, .note-ref, a.note-ref"):
            sup.decompose()
        paragraphs = [clean_text(p.get_text(" ", strip=True)) for p in body.find_all(["p", "h2", "h3", "li"])]
        text = "\n".join(p for p in paragraphs if p)
    else:
        text = ""
    out["text"] = text
    out["num_words"] = len(re.findall(r"\S+", text)) if text else None
    return out


def collect(conferences: list[tuple[int, int]], fetcher: Fetcher, limit: int | None = None) -> pd.DataFrame:
    rows, skipped = [], []
    for year, month_num in conferences:
        sessions = conference_sessions(fetcher, year, month_num)
        if sessions is None:
            log.error("skipping %s: no index", conference_key(year, month_num))
            continue
        n_conf = 0
        for s in sessions:
            order = 0
            for item in s["items"]:
                if limit is not None and len(rows) >= limit:
                    break
                if not item["uri"] or not item["subtitle"] or NON_TALK_RE.match(item["title"] or ""):
                    skipped.append({"conference": conference_key(year, month_num), "session": s["session"], **item})
                    continue
                url = f"{BASE}{item['uri']}?lang=eng"
                html = fetcher.get(url, stage="talk")
                if html is None:
                    continue
                try:
                    talk = parse_talk(html, item["uri"])
                except Exception as e:  # noqa: BLE001
                    fetcher.failures.append({"url": url, "stage": "talk-parse", "error": repr(e)})
                    log.error("could not parse talk %s: %r", url, e)
                    continue
                order += 1
                n_conf += 1
                rows.append({
                    "conference": conference_key(year, month_num), "year": year, "month_num": month_num,
                    "session": s["session"], "session_title": s["session_title"],
                    "site_slot": item["site_slot"], "speaker_order": order,
                    "speaker": talk["speaker"] or item["subtitle"], "speaker_raw": talk["speaker_raw"],
                    "role": talk["role"], "title": talk["title"] or item["title"], "kicker": talk["kicker"],
                    "url": url.split("?")[0], "uri": item["uri"], "num_words": talk["num_words"],
                    "duration_ms": talk["duration_ms"], "duration_string": talk["duration_string"],
                    "duration_sec": None if talk["duration_ms"] is None else talk["duration_ms"] / 1000.0,
                    "duration_source": "video_data_duration" if talk["duration_ms"] is not None else None,
                    "duration_flag": talk["duration_flag"], "video_id": talk["video_id"],
                    "audio_url": talk["audio_url"], "text": talk["text"],
                    "collected_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                })
        log.info("%s: %d talks collected (%d sessions)", conference_key(year, month_num), n_conf, len(sessions))
    if skipped:
        pd.DataFrame(skipped).to_csv(OUTPUTS / "collect_skipped_items.csv", index=False, encoding="utf-8")
        log.info("skipped %d non-talk program items (see outputs/collect_skipped_items.csv)", len(skipped))
    return pd.DataFrame(rows)


def compare_with_legacy(collected: pd.DataFrame) -> str:
    legacy = load_legacy()
    merged = legacy.merge(collected[["url", "duration_sec", "duration_string", "speaker", "role", "num_words"]],
                          on="url", how="inner", suffixes=("_legacy", "_scraped"))
    lines = ["# Duration compatibility: legacy CSV vs scraped video duration", ""]
    if merged.empty:
        lines.append("No overlapping talks between the legacy CSV and the scraped set.")
        return "\n".join(lines)
    both = merged[merged.duration_status.eq("ok") & merged.duration_sec_scraped.notna()]
    diff = (both.duration_sec_legacy - both.duration_sec_scraped).abs()
    lines += [
        f"Overlapping talks (by URL): {len(merged)}",
        f"Talks timed in both: {len(both)}",
        f"Exact matches (|diff| <= 1 s): {int((diff <= 1).sum())}",
        f"Within 5 s: {int((diff <= 5).sum())}",
        f"Max |diff|: {diff.max():.1f} s" if len(both) else "",
        f"Legacy missing/invalid but scraped has a duration: "
        f"{int((merged.duration_status.ne('ok') & merged.duration_sec_scraped.notna()).sum())}",
        f"Scraped missing but legacy timed: {int((merged.duration_status.eq('ok') & merged.duration_sec_scraped.isna()).sum())}",
        f"Speaker name mismatches: {int((merged.speaker_legacy != merged.speaker_scraped).sum())}",
        f"Role mismatches: {int((merged.role_legacy != merged.role_scraped).sum())}",
        "",
    ]
    bad = both[diff > 1]
    if len(bad):
        lines.append("| talk | legacy | scraped |")
        lines.append("|---|---|---|")
        for r in bad.itertuples():
            lines.append(f"| {r.conference} {r.speaker_legacy} | {r.duration_raw} | {r.duration_string} |")
    names = merged[merged.speaker_legacy != merged.speaker_scraped]
    if len(names):
        lines += ["", "Name differences:"] + [f"- {r.speaker_legacy!r} vs {r.speaker_scraped!r}" for r in names.itertuples()]
    roles = merged[merged.role_legacy != merged.role_scraped]
    if len(roles):
        lines += ["", "Role differences:"] + [f"- {r.role_legacy!r} vs {r.role_scraped!r}" for r in roles.itertuples()]
    wd = (merged.num_words_legacy - merged.num_words_scraped)
    lines += ["", f"Word count difference (legacy - scraped): mean {wd.mean():.1f}, max |diff| {wd.abs().max():.0f}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2021-04")
    ap.add_argument("--end", default="2026-04")
    ap.add_argument("--extra", nargs="*", default=[], help="additional conferences, e.g. 2020-10 for the legacy check")
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between network requests")
    ap.add_argument("--refresh", action="store_true", help="ignore the HTML cache")
    ap.add_argument("--limit", type=int, default=None, help="stop after N talks (debugging)")
    ap.add_argument("--compare-legacy", action="store_true", help="write outputs/duration_compatibility.md")
    args = ap.parse_args()

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(OUTPUTS / "collect.log", encoding="utf-8")])

    sy, sm = parse_conference_key(args.start)
    ey, em = parse_conference_key(args.end)
    confs = [(y, m) for y in range(sy, ey + 1) for m in (4, 10) if (y, m) >= (sy, sm) and (y, m) <= (ey, em)]
    confs = sorted(set(confs) | {parse_conference_key(x) for x in args.extra})
    log.info("collecting %s", [conference_key(*c) for c in confs])

    fetcher = Fetcher(delay=args.delay, refresh=args.refresh)
    df = collect(confs, fetcher, limit=args.limit)
    if df.empty:
        log.error("nothing collected")
        return 1
    COLLECTED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(COLLECTED_CSV, index=False, encoding="utf-8")
    log.info("wrote %d talks to %s (network requests: %d)", len(df), COLLECTED_CSV, fetcher.n_network)
    log.info("duration flags: %s", df.duration_flag.value_counts().to_dict())
    log.info("talks without duration: %d", int(df.duration_ms.isna().sum()))

    if fetcher.failures:
        pd.DataFrame(fetcher.failures).to_csv(OUTPUTS / "collect_failures.csv", index=False, encoding="utf-8")
        log.warning("%d failures logged to outputs/collect_failures.csv", len(fetcher.failures))
    elif (OUTPUTS / "collect_failures.csv").exists():
        (OUTPUTS / "collect_failures.csv").unlink()

    if args.compare_legacy:
        report = compare_with_legacy(df)
        (OUTPUTS / "duration_compatibility.md").write_text(report, encoding="utf-8")
        print("\n" + report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
