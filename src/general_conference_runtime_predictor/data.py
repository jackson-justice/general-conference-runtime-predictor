"""Loading, cleaning and validating talk data from the legacy CSV and the scraper."""
from __future__ import annotations

import json
import re
import unicodedata

import numpy as np
import pandas as pd

from .paths import COLLECTED_CSV, DATASET_CSV, LEGACY_CSV, LEGACY_FIXES_CSV, OUTPUTS

MONTH_NUM = {"April": 4, "October": 10}
MONTH_NAME = {4: "April", 10: "October"}

# Program items that are not talks. The legacy CSV already drops sustainings and
# audit reports but keeps one untimed solemn assembly; we drop all of them.
NON_TALK_RE = re.compile(
    r"^(Sustaining of|Solemn Assembly|Church Auditing Department Report|"
    r"Statistical Report|The Sustaining of)",
    re.I,
)
RUNTIME_RE = re.compile(r"^\s*(?:(\d{1,2}):)?(\d{1,3}):(\d{2})\s*$")

# Spoken English in conference runs roughly 90-220 words per minute in the
# legacy data. Anything outside this window is a data error, not a fast talker.
WPM_MIN, WPM_MAX = 50.0, 300.0

STATUS_ORDER = ["ok", "missing", "unparseable", "implausible"]

# Explicit alias table: source spelling -> canonical name. The source spelling is
# preserved in `speaker_source`; `speaker` (canonical) is what histories key on.
# Found by collect.py --compare-legacy and a surname/initial scan of all bylines.
SPEAKER_ALIASES = {
    "Becky Craven": "Rebecca L. Craven",      # site byline (2020-10) vs legacy CSV / 2022-04 byline
    "Larry Echo Hawk": "Larry J. Echo Hawk",  # two spellings within the legacy CSV
    "L. Harkness": "Lisa L. Harkness",        # two spellings within the legacy CSV
    "James O. Fantone": "James G. O. Fantone",  # 2026-10 live log (name as announced) vs site byline (2026-10)
}


def canonical_speaker(value) -> str | None:
    name = clean_text(value)
    return SPEAKER_ALIASES.get(name, name)


def clean_text(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = unicodedata.normalize("NFC", str(value)).replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def parse_runtime(value) -> tuple[float, str]:
    """Parse 'm:ss' / 'mm:ss' / 'h:mm:ss' into seconds. Returns (seconds, status)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan, "missing"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return np.nan, "missing"
    m = RUNTIME_RE.match(text)
    if not m:
        return np.nan, "unparseable"
    h, mm, ss = m.groups()
    if int(ss) >= 60:
        return np.nan, "unparseable"
    seconds = (int(h) if h else 0) * 3600 + int(mm) * 60 + int(ss)
    if seconds <= 0:
        return np.nan, "unparseable"
    return float(seconds), "ok"


def format_seconds(seconds) -> str:
    if seconds is None or pd.isna(seconds):
        return "n/a"
    seconds = int(round(float(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


def normalize_role(role: str | None) -> str | None:
    if role is None or (isinstance(role, float) and np.isnan(role)):
        return None
    r = re.sub(r"^Recently Released\s+", "", str(role).strip(), flags=re.I)
    if re.match(r"^President of The Church of Jesus Christ", r, re.I):
        r = "President of the Church"
    return r


def calling_group(role: str | None) -> str:
    """Collapse the printed role into a small set of callings with similar time budgets."""
    r = normalize_role(role) or ""
    if r.startswith("President of the Church"):
        return "church_president"
    if "Counselor in the First Presidency" in r:
        return "first_presidency_counselor"
    if re.match(r"^(Acting )?President of the Quorum of the Twelve", r):
        return "twelve_president"
    if "Quorum of the Twelve" in r:
        return "twelve"
    if "Presidency of the Seventy" in r:
        return "seventy_presidency"
    if "Seventy" in r:
        return "seventy"
    if "Presiding Bishop" in r:
        return "presiding_bishopric"
    if re.search(r"Counselor in the .*General Presidency", r):
        return "auxiliary_counselor"
    if re.search(r"General President$", r):
        return "auxiliary_president"
    return "other"


def conf_index(year: int, month_num: int) -> int:
    return int(year) * 2 + (1 if int(month_num) == 10 else 0)


def conference_key(year: int, month_num: int) -> str:
    return f"{int(year)}-{int(month_num):02d}"


def parse_conference_key(key: str) -> tuple[int, int]:
    m = re.match(r"^(\d{4})-(0?4|10)$", key.strip())
    if not m:
        raise ValueError(f"Conference key must look like 2026-10, got {key!r}")
    return int(m.group(1)), int(m.group(2))


def _finish(df: pd.DataFrame, source: str, duration_source: str) -> pd.DataFrame:
    df = df.copy()
    df["conference"] = [conference_key(y, m) for y, m in zip(df.year, df.month_num)]
    df["conf_index"] = [conf_index(y, m) for y, m in zip(df.year, df.month_num)]
    df["month"] = df.month_num.map(MONTH_NAME)
    df["role_norm"] = df.role.map(normalize_role)
    df["calling_group"] = df.role.map(calling_group)
    df["source"] = source
    df["duration_source"] = duration_source

    non_talk = df.title.fillna("").str.match(NON_TALK_RE)
    dropped = int(non_talk.sum())
    df = df[~non_talk].copy()

    df = df.sort_values(["conf_index", "session", "orig_order"]).reset_index(drop=True)
    df["speaker_order"] = df.groupby(["conference", "session"]).cumcount() + 1
    df.attrs["dropped_non_talks"] = dropped
    return df


def flag_implausible(df: pd.DataFrame) -> pd.DataFrame:
    """Mark durations that are impossible given the talk's word count."""
    attrs = dict(df.attrs)
    df = df.copy()
    ok = df.duration_status.eq("ok") & df.num_words.notna() & (df.num_words > 0)
    wpm = df.num_words / (df.duration_sec / 60.0)
    bad = ok & ((wpm < WPM_MIN) | (wpm > WPM_MAX))
    df.loc[bad, "duration_status"] = "implausible"
    df.loc[bad, "duration_sec"] = np.nan
    df["words_per_min"] = wpm.where(ok & ~bad)
    df.attrs.update(attrs)
    return df


def apply_duration_fixes(df: pd.DataFrame, path=LEGACY_FIXES_CSV) -> pd.DataFrame:
    """Replace unusable legacy runtimes with durations verified on the official talk page.

    Only rows that scripts/verify_durations.py marked "corrected" or "filled"
    are touched. The original runtime string stays in duration_raw.
    """
    attrs = dict(df.attrs)
    df = df.copy()
    fixes = pd.read_csv(path, encoding="utf-8")
    fixes = fixes[fixes.action.isin(["corrected", "filled"]) & fixes.video_sec.notna()].set_index("url")
    sec = df.url.map(fixes.video_sec)
    hit = sec.notna() & df.duration_status.ne("ok")
    before = df.loc[hit, "duration_status"].value_counts().to_dict()
    df.loc[hit, "duration_note"] = [
        f"legacy runtime {raw!r} ({st}) replaced by official video duration"
        for raw, st in zip(df.loc[hit, "duration_raw"], df.loc[hit, "duration_status"])
    ]
    df.loc[hit, "duration_sec"] = sec[hit].astype(float)
    df.loc[hit, "duration_status"] = "ok"
    df.loc[hit, "duration_source"] = "video_data_duration"
    df.attrs.update(attrs)
    df.attrs["duration_fixes_applied"] = {k: int(v) for k, v in before.items()}
    return df


def load_legacy(path=LEGACY_CSV, apply_fixes: bool = True) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8")
    df = pd.DataFrame(
        {
            "year": raw.year.astype(int),
            "month_num": raw.month.map(MONTH_NUM).astype(int),
            "session": raw.session.map(clean_text),
            "orig_order": raw.speaker_order.astype(int),
            "speaker": raw.speaker.map(canonical_speaker),
            "speaker_source": raw.speaker.map(clean_text),
            "role": raw.role.map(clean_text),
            "title": raw.talk_name.map(clean_text),
            "kicker": raw.kicker.map(clean_text),
            "url": raw.link.str.replace(r"\?.*$", "", regex=True),
            "num_words": raw.num_words.astype(float),
            "duration_raw": raw.runtime,
        }
    )
    parsed = df.duration_raw.map(parse_runtime)
    df["duration_sec"] = [p[0] for p in parsed]
    df["duration_status"] = [p[1] for p in parsed]
    df["duration_note"] = None
    df = _finish(df, source="legacy_csv", duration_source="legacy_csv_runtime")
    if apply_fixes and LEGACY_FIXES_CSV.exists():
        df = apply_duration_fixes(df)
    return flag_implausible(df)


def load_collected(path=COLLECTED_CSV) -> pd.DataFrame | None:
    if not path.exists():
        return None
    raw = pd.read_csv(path, encoding="utf-8")
    df = pd.DataFrame(
        {
            "year": raw.year.astype(int),
            "month_num": raw.month_num.astype(int),
            "session": raw.session.map(clean_text),
            "orig_order": raw.site_slot.astype(int),
            "speaker": raw.speaker.map(canonical_speaker),
            "speaker_source": raw.speaker.map(clean_text),
            "role": raw.role.map(clean_text),
            "title": raw.title.map(clean_text),
            "kicker": raw.kicker.map(clean_text),
            "url": raw.url.str.replace(r"\?.*$", "", regex=True),
            "num_words": raw.num_words.astype(float),
            "duration_raw": raw.duration_string,
        }
    )
    ms = pd.to_numeric(raw.duration_ms, errors="coerce")
    df["duration_sec"] = ms / 1000.0
    df["duration_status"] = np.where(ms.notna() & (ms > 0), "ok", "missing")
    df.loc[df.duration_status != "ok", "duration_sec"] = np.nan
    df["duration_note"] = None
    df = _finish(df, source="scraped", duration_source="video_data_duration")
    return flag_implausible(df)


def duration_report(df: pd.DataFrame) -> dict:
    rep: dict = {"n_talks": int(len(df))}
    rep["by_status"] = {s: int((df.duration_status == s).sum()) for s in STATUS_ORDER}
    rep["by_source_status"] = {
        src: {s: int(v) for s, v in g.duration_status.value_counts().items()}
        for src, g in df.groupby("source")
    }
    rep["not_ok_by_conference"] = {
        c: int(v) for c, v in df[df.duration_status != "ok"].groupby("conference").size().items()
    }
    bad = df[df.duration_status.isin(["unparseable", "implausible"])]
    rep["invalid_rows"] = [
        {
            "conference": r.conference,
            "speaker": r.speaker,
            "title": r.title,
            "duration_raw": None if pd.isna(r.duration_raw) else str(r.duration_raw),
            "num_words": None if pd.isna(r.num_words) else int(r.num_words),
            "status": r.duration_status,
        }
        for r in bad.itertuples()
    ]
    cov = (
        df.groupby(["conference", "source"], sort=True)
        .agg(sessions=("session", "nunique"), talks=("url", "size"),
             timed=("duration_status", lambda s: int((s == "ok").sum())),
             from_legacy_runtime=("duration_source", lambda s: int((s == "legacy_csv_runtime").sum())),
             from_video=("duration_source", lambda s: int((s == "video_data_duration").sum())))
        .reset_index()
    )
    cov["not_timed"] = cov.talks - cov.timed
    rep["coverage_by_conference"] = cov.to_dict(orient="records")
    rep["duplicate_check"] = {
        "duplicate_urls": int(df.url.duplicated().sum()),
        "duplicate_conference_session_speaker_title": int(df.duplicated(["conference", "session", "speaker", "title"]).sum()),
        "speakers_with_multiple_talks_in_one_conference": int(df.duplicated(["conference", "speaker"]).sum()),
    }
    aliased = df[df.speaker != df.speaker_source]
    rep["speaker_aliases_applied"] = {
        f"{src} -> {canon}": int(n)
        for (src, canon), n in aliased.groupby(["speaker_source", "speaker"]).size().items()
    }
    rep["duration_fixes_applied"] = df.attrs.get("duration_fixes_applied", {})
    ok = df[df.duration_status == "ok"]
    rep["duration_minutes"] = {
        "n": int(len(ok)),
        "mean": round(float(ok.duration_sec.mean() / 60), 2),
        "median": round(float(ok.duration_sec.median() / 60), 2),
        "std": round(float(ok.duration_sec.std() / 60), 2),
        "min": round(float(ok.duration_sec.min() / 60), 2),
        "max": round(float(ok.duration_sec.max() / 60), 2),
    }
    return rep


def build_dataset(include_collected: bool = True, save: bool = True) -> tuple[pd.DataFrame, dict]:
    """Combine legacy and scraped talks into one chronologically ordered table."""
    legacy = load_legacy()
    parts = [legacy]
    notes: dict = {"dropped_non_talks": {"legacy_csv": legacy.attrs.get("dropped_non_talks")}}
    if include_collected:
        collected = load_collected()
        if collected is not None:
            notes["dropped_non_talks"]["scraped"] = collected.attrs.get("dropped_non_talks")
            overlap = sorted(set(collected.conference) & set(legacy.conference))
            if overlap:
                # Overlapping conferences exist only for the compatibility check.
                # The legacy CSV stays the source of record for its own range.
                notes["scraped_conferences_skipped_in_favor_of_legacy"] = overlap
                collected = collected[~collected.conference.isin(overlap)]
            parts.append(collected)
        else:
            notes["collected"] = "no scraped data found; using legacy CSV only"
    fixes_applied = legacy.attrs.get("duration_fixes_applied", {})
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["conf_index", "session", "speaker_order"]).reset_index(drop=True)
    df.attrs["duration_fixes_applied"] = fixes_applied
    df["talk_id"] = df.url.str.replace(r"^https?://www\.churchofjesuschrist\.org/study/", "", regex=True)
    dup = df.talk_id.duplicated()
    if dup.any():
        notes["duplicate_urls_dropped"] = int(dup.sum())
        df = df[~dup].reset_index(drop=True)
    report = duration_report(df)
    notes["legacy_duration_fixes_file"] = str(LEGACY_FIXES_CSV.relative_to(LEGACY_FIXES_CSV.parents[2])) if LEGACY_FIXES_CSV.exists() else None
    report["notes"] = notes
    report["conferences"] = sorted(df.conference.unique().tolist())
    if save:
        DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATASET_CSV, index=False, encoding="utf-8")
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        (OUTPUTS / "data_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return df, report


def load_dataset(path=DATASET_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/train.py first")
    return pd.read_csv(path, encoding="utf-8")
