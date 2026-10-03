"""Predict a talk's duration from pre-talk information, and log predictions/actuals.

Usage:
    uv run python scripts/predict.py live                  # interactive: one name per talk
    uv run python scripts/predict.py predict --speaker "Dale G. Renlund" \
        --calling "Of the Quorum of the Twelve Apostles" --session sunday-morning --order 3 --log
    uv run python scripts/predict.py log-actual --speaker "Dale G. Renlund" --session sunday-morning --actual 14:12
    uv run python scripts/predict.py remove --speaker "Dale G. Renlnud" --session sunday-morning   # drop a mistyped row
    uv run python scripts/predict.py fill-actuals --conference 2026-10   # after collect.py has scraped it
    uv run python scripts/predict.py score
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd

from general_conference_runtime_predictor.data import (
    MONTH_NAME, calling_group, canonical_speaker, clean_text, conf_index, format_seconds, load_dataset, normalize_role,
    parse_conference_key, parse_runtime,
)
from general_conference_runtime_predictor.features import history_features
from general_conference_runtime_predictor.paths import (
    BUNDLE_PATH, COLLECTED_CSV, OUTPUTS, PREDICTION_LOG, PROGRAM_ITEMS_CSV,
)

LOG_COLS = ["logged_at", "conference", "speaker", "calling", "calling_group", "session", "speaker_order",
            "model", "model_version", "pred_sec", "pred_mmss", "actual_sec", "actual_mmss", "actual_source"]
MANIFEST_PATH = OUTPUTS / "model_manifest.json"


def load_log() -> pd.DataFrame:
    if PREDICTION_LOG.exists():
        log = pd.read_csv(PREDICTION_LOG, encoding="utf-8",
                          dtype={"actual_mmss": str, "pred_mmss": str, "model_version": str, "actual_source": str})
        return log.reindex(columns=LOG_COLS)  # older logs gain the new columns as empty
    return pd.DataFrame(columns=LOG_COLS)


def model_version(bundle: dict) -> str:
    """Identifies the frozen bundle: training timestamp plus the code commit it was trained from (if recorded)."""
    return bundle["trained_at"] + ("@" + bundle["code_commit"] if bundle.get("code_commit") else "")


def write_manifest(bundle: dict) -> dict:
    manifest = {
        "model_version": model_version(bundle),
        "trained_at": bundle["trained_at"],
        "code_commit": bundle.get("code_commit"),
        "tag": bundle["tag"],
        "data_through_conf_index": bundle["max_conf_index"],
        "split_used_for_selection": bundle["split"],
        "models": {name: {"params": e["params"], "val_mae_min": round(e["val_mae_min"], 3),
                          "test_mae_min": round(e["test"]["mae_min"], 3),
                          "range_half_width_min": e["intervals"]["0.8"]["half_width_min"],
                          "range_test_coverage": e["intervals"]["0.8"]["test_coverage"]}
                   for name, e in bundle["models"].items()},
        "recommended": bundle["recommended"],
        "note": "Final models are refit on every timed talk through data_through_conf_index after settings were "
                "selected on validation. Ranges are uncalibrated: half-width is the 80th percentile of validation "
                "error, range_test_coverage is what that range covered on the test conferences.",
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def save_log(log: pd.DataFrame) -> None:
    PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    log[LOG_COLS].to_csv(PREDICTION_LOG, index=False, encoding="utf-8")


SESSIONS = ["saturday-morning", "saturday-afternoon", "sunday-morning", "sunday-afternoon", "saturday-evening"]
CALLING_SHORTCUTS = [
    "Of the Seventy",
    "Of the Quorum of the Twelve Apostles",
    "Of the Presidency of the Seventy",
    "President of the Church",
    "First Counselor in the First Presidency",
    "Second Counselor in the First Presidency",
    "Acting President of the Quorum of the Twelve Apostles",
    "President of the Quorum of the Twelve Apostles",
    "Presiding Bishop",
    "First Counselor in the Presiding Bishopric",
    "Second Counselor in the Presiding Bishopric",
    "Relief Society General President",
    "Young Women General President",
    "Primary General President",
    "Young Men General President",
    "Sunday School General President",
]


def norm_session(value: str) -> str:
    return clean_text(value).lower().replace("_", "-").removesuffix("-session")


def load_context(conference: str):
    """Bundle, dataset, and the history rows usable for `conference`."""
    bundle = joblib.load(BUNDLE_PATH)
    df = load_dataset()
    year, month_num = parse_conference_key(conference)
    ci = conf_index(year, month_num)
    if ci <= df.conf_index.max():
        print(f"note: {conference} is inside the training data; history is truncated to earlier conferences")
    return bundle, df, df[df.conf_index < ci], year, month_num, ci


def predict_talk(bundle, hist, year, month_num, ci, conference, speaker, role, session, order,
                 models="all", verbose=True, compact=False):
    """Predict one talk. Returns (rows, version, group); prints the explanation when verbose.

    compact=True (live mode) prints a short block: who, history in one line, and the
    predictions with the recommended model first.
    """
    group = calling_group(role)
    if session not in bundle["sessions"]:
        print(f"warning: session {session!r} not among known sessions {bundle['sessions']}")
    if group == "other":
        print(f"warning: calling {role!r} did not map to a known calling group; using 'other'")
    row = pd.DataFrame([{
        "speaker": speaker, "role": role, "role_norm": normalize_role(role), "calling_group": group,
        "session": session, "speaker_order": int(order), "month": MONTH_NAME[month_num],
        "conf_index": ci, "year": year,
    }])
    X = row.join(history_features(hist, row))
    f = X.iloc[0]
    version = model_version(bundle)
    if compact:
        hist_line = ("no history (new speaker; using calling and session)" if f.spk_n_talks == 0 else
                     f"{int(f.spk_n_timed)} earlier talks, mean {format_seconds(f.spk_mean)}, last {format_seconds(f.spk_last)}")
        print(f"\n  {speaker}  |  {normalize_role(role)}  |  {session} #{int(order)}")
        print(f"  {hist_line}")
        rows = []
        names = list(bundle["models"]) if models == "all" else [models]
        names.sort(key=lambda n: n != bundle["recommended"])  # recommended first
        for name in names:
            entry = bundle["models"][name]
            pred = float(entry["model"].predict(X)[0])
            hw = entry["intervals"]["0.8"]["half_width_min"] * 60
            if name == bundle["recommended"]:
                print(f"  >> {name:9s} {format_seconds(pred):>6s}   likely {format_seconds(pred - hw)} - {format_seconds(pred + hw)}")
            else:
                print(f"     {name:9s} {format_seconds(pred):>6s}")
            rows.append({"model": name, "pred_sec": round(pred, 1)})
        return rows, version, group
    if verbose:
        print(f"\n{speaker} | {role} ({group}) | {session} #{int(order)} | {conference}")
        if f.spk_n_talks == 0:
            print("speaker history: none in the dataset (unseen speaker; prediction relies on calling/session)")
        else:
            print(f"speaker history: {int(f.spk_n_talks)} prior talks, {int(f.spk_n_timed)} timed, "
                  f"mean {format_seconds(f.spk_mean)}, last {format_seconds(f.spk_last)}, "
                  f"same-calling mean {format_seconds(f.spk_calling_mean)} (n={int(f.spk_calling_n)})")
        print(f"calling mean {format_seconds(f.calling_mean)} | session mean {format_seconds(f.session_mean)} | "
              f"recent global mean {format_seconds(f.global_recent_mean)}")
        print(f"model version {version}")
        print("\nmodel      prediction   estimated range (uncalibrated; share of test talks it covered)   test MAE")
    names = list(bundle["models"]) if models == "all" else [models]
    rows = []
    for name in names:
        entry = bundle["models"][name]
        pred = float(entry["model"].predict(X)[0])
        iv = entry["intervals"]["0.8"]
        hw = iv["half_width_min"] * 60
        star = " <- recommended" if name == bundle["recommended"] else ""
        if verbose:
            print(f"{name:10s} {format_seconds(pred):>8s}     {format_seconds(pred - hw)} - {format_seconds(pred + hw)} "
                  f"(covered {iv['test_coverage']:.0%})                                   {entry['test']['mae_min']:.2f} min{star}")
        rows.append({"model": name, "pred_sec": round(pred, 1)})
    if verbose:
        print("\nThe range is not a calibrated interval. Its half-width is the 80th percentile of validation errors; "
              "'covered' is the share of held-out test talks that fell inside it (62% for catboost).")
    return rows, version, group


def log_predictions(conference, speaker, role, group, session, order, rows, version, force=False, quiet=False):
    """Append prediction rows. Returns 'logged', or 'duplicate' when the talk is already logged and not force."""
    log = load_log()
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    new = pd.DataFrame([{
        "logged_at": now, "conference": conference, "speaker": speaker, "calling": role,
        "calling_group": group, "session": session, "speaker_order": int(order),
        "model": r["model"], "model_version": version,
        "pred_sec": r["pred_sec"], "pred_mmss": format_seconds(r["pred_sec"]),
        "actual_sec": np.nan, "actual_mmss": None, "actual_source": None,
    } for r in rows])
    # One prediction per (conference, speaker, session, order, model). An existing row is
    # never overwritten unless force; use `remove` for typos.
    keys = ["conference", "speaker", "session", "speaker_order", "model"]
    if len(log):
        dup = log.set_index(keys).index.isin(new.set_index(keys).index)
        if dup.any() and not force:
            first = log[dup].iloc[0]
            if quiet:
                print(f"  already logged at {first.logged_at}.")
            else:
                print(f"\nNOT logged: {speaker!r} {session} #{int(order)} already has a prediction "
                      f"(logged {first.logged_at}, version {first.model_version}). "
                      f"Re-run with --force to replace it, or `remove` the row if it was a typo.")
            return "duplicate"
        carried_actual = log[dup].dropna(subset=["actual_sec"])
        if len(carried_actual):
            new = new.merge(carried_actual[keys + ["actual_sec", "actual_source"]]
                            .rename(columns={"actual_sec": "_a", "actual_source": "_s"}), on=keys, how="left")
            new["actual_sec"] = new["_a"]
            new["actual_source"] = new["_s"]
            new["actual_mmss"] = new.actual_sec.map(lambda s: None if pd.isna(s) else format_seconds(s))
            new = new.drop(columns=["_a", "_s"])
        log = log[~dup]
    log = pd.concat([log, new], ignore_index=True)
    save_log(log)
    print("  logged." if quiet else f"logged {len(new)} prediction rows to {PREDICTION_LOG}")
    return "logged"


def cmd_predict(args) -> int:
    bundle, df, hist, year, month_num, ci = load_context(args.conference)
    speaker = canonical_speaker(args.speaker)
    role = clean_text(args.calling)
    session = norm_session(args.session)
    rows, version, group = predict_talk(bundle, hist, year, month_num, ci, args.conference,
                                        speaker, role, session, args.order, models=args.model)
    if args.log:
        log_predictions(args.conference, speaker, role, group, session, args.order, rows, version, force=args.force)
    return 0


# ---------------------------------------------------------------------------
# Interactive live mode
# ---------------------------------------------------------------------------

def speaker_directory(hist: pd.DataFrame) -> pd.DataFrame:
    """Known speakers with their most recent printed role and last conference."""
    h = hist.sort_values(["conf_index", "session", "speaker_order"])
    last = h.groupby("speaker").tail(1)
    return last[["speaker", "role", "conference"]].set_index("speaker")


def match_speaker(text: str, directory: pd.DataFrame) -> list[str]:
    """Candidates for a typed name: exact, then all typed words contained in the name, then fuzzy."""
    import difflib

    import unicodedata

    def fold(x: str) -> str:  # accent-insensitive, case-insensitive ("causse" matches "Caussé")
        return "".join(c for c in unicodedata.normalize("NFKD", x) if not unicodedata.combining(c)).lower()

    t = clean_text(text) or ""
    canon = canonical_speaker(t)
    names = directory.index.tolist()
    folded = {fold(n): n for n in names}
    if canon and fold(canon) in folded:
        return [folded[fold(canon)]]
    words = [w for w in re.split(r"\s+", fold(t)) if w]
    contained = [n for n in names if all(w in fold(n) for w in words)]
    if contained:
        return sorted(contained)
    fuzzy = [folded[f] for f in difflib.get_close_matches(fold(t), list(folded), n=5, cutoff=0.6)]
    if not fuzzy and words:
        # match on surname only
        fuzzy = [n for n in names if difflib.get_close_matches(words[-1], fold(n).split(), n=1, cutoff=0.8)]
    return fuzzy


def ask(prompt: str) -> str | None:
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def choose_session(default: str | None = None) -> str | None:
    opts = SESSIONS[:4]
    menu = "  ".join(f"{i + 1}={s}" for i, s in enumerate(opts))
    while True:
        ans = ask(f"Session? {menu}" + (f"  [Enter={default}]" if default else "") + ": ")
        if ans is None:
            return None
        ans = ans.strip()
        if not ans and default:
            return default
        if ans.isdigit() and 1 <= int(ans) <= len(opts):
            return opts[int(ans) - 1]
        if ans:
            s = norm_session(ans)
            if s in SESSIONS:
                return s
        print("  pick 1-4 or type a session name")


def choose_calling(default: str | None) -> str | None:
    menu = "\n".join(f"  {i + 1:2d}  {c}" for i, c in enumerate(CALLING_SHORTCUTS))
    hint = f" [Enter={default}]" if default else ""
    while True:
        ans = ask(f"  calling{hint}  (Enter to accept, or type it, or ? for list): ")
        if ans is None:
            return None
        ans = ans.strip()
        if not ans and default:
            return default
        if ans == "?":
            print(menu)
            continue
        if ans.isdigit() and 1 <= int(ans) <= len(CALLING_SHORTCUTS):
            return CALLING_SHORTCUTS[int(ans) - 1]
        if ans:
            role = clean_text(ans)
            if calling_group(role) == "other":
                ok = ask(f"  {role!r} does not match a known calling group. Use it anyway? [y/N] ")
                if ok is None or ok.strip().lower() != "y":
                    continue
            return role


# Non-talk program items. They never enter the model or the talk count; live mode
# gives a plain historical-average estimate from data/processed/program_items.csv.
PROGRAM_ITEMS = {
    "sustaining": {"label": "Sustaining of Officers", "aliases": ("sustaining", "sustain", "sus")},
    "audit_report": {"label": "Church Auditing Department Report", "aliases": ("audit", "audit report")},
    "statistical_report": {"label": "Statistical Report", "aliases": ("statistical", "stats", "statistical report")},
    "solemn_assembly": {"label": "Solemn Assembly", "aliases": ("solemn", "solemn assembly")},
}
PROGRAM_ALIAS = {a: k for k, v in PROGRAM_ITEMS.items() for a in v["aliases"]}


def program_item_estimate(kind: str, month_num: int, conference: str):
    """Average of the last three same-month occurrences strictly before `conference`
    (falls back to any month). Returns (sec, used_df) or None."""
    if not PROGRAM_ITEMS_CSV.exists():
        return None
    items = pd.read_csv(PROGRAM_ITEMS_CSV, encoding="utf-8")
    same = items[items.kind.eq(kind) & items.duration_sec.notna() & (items.conference < conference)]
    same = same.sort_values("conference")
    if same.empty:
        return None
    same_month = same[same.conference.str[-2:] == f"{month_num:02d}"]
    use = (same_month if len(same_month) >= 2 else same).tail(3)
    return float(use.duration_sec.mean()), use


def log_program_item(conference, kind, session, pred_sec, n_used, force=False):
    label = PROGRAM_ITEMS[kind]["label"]
    rows = [{"model": "program_mean", "pred_sec": round(pred_sec, 1)}]
    return log_predictions(conference, label, kind, "program_item", session, 0, rows,
                           f"program_items:{n_used}", force=force, quiet=True)


LIVE_HELP = """Commands at the speaker prompt:
  <name>    type the speaker's name (surname is enough) and press Enter -> predicts and logs that talk
  sustaining / audit / solemn   estimate a non-talk program item from past averages (not counted as a talk)
  u         undo: delete the last talk you logged (typo, wrong person) and step the talk number back
  t 12:34   time: record your stopwatch time for the LAST logged talk (optional; the official time replaces it later)
  o 5       order: make the NEXT talk number 5 (if you lost count; the conference study page shows the order)
  s         session: switch to another session (talk number restarts at 1)
  h or ?    show this help
  q         quit (everything is already saved; run `live` again to continue where you left off)
Skip the sustaining, audit report, music, prayers and videos: they are not talks and get no number."""


def cmd_live(args) -> int:
    """Prompt-driven logging for a live session: type a name per talk, everything else is filled in."""
    bundle, df, hist, year, month_num, ci = load_context(args.conference)
    directory = speaker_directory(hist)
    print(f"\nLive mode for {args.conference} (model {bundle['trained_at'][:10]}). "
          f"Type a speaker's name per talk; talks only, no sustaining/audit/music. h = help.")
    session = choose_session()
    if session is None:
        return 0
    order = 1
    last = None  # (speaker, session, order)
    while True:
        ans = ask(f"\n[{session}] talk #{order}   (u=undo  t 12:34=time  o N=set #  s=session  h=help  q=quit)\n"
                  f"  speaker: ")
        if ans is None or ans.strip().lower() == "q":
            print("bye")
            return 0
        text = ans.strip()
        if not text:
            continue
        low = text.lower()
        if low in ("h", "?", "help"):
            print(LIVE_HELP)
            continue
        if low == "s":
            new_session = choose_session(default=session)
            if new_session is None:
                return 0
            if new_session != session:
                session, order, last = new_session, 1, None
            continue
        if low == "u":
            if last is None:
                print("  nothing to undo")
                continue
            sp, se, od = last
            log = load_log()
            mask = log.conference.eq(args.conference) & log.speaker.eq(sp) & log.session.eq(se) \
                & log.speaker_order.astype(float).eq(float(od))
            save_log(log[~mask])
            print(f"  removed {int(mask.sum())} rows for {sp} ({se}" + (f" #{od})" if od else ", program item)"))
            if od:  # program items (order 0) do not move the talk count
                order = od
            last = None
            continue
        m = re.match(r"^[ta]\s+(\S+)$", low)
        if m:
            if last is None:
                print("  no talk logged yet in this run; use `predict.py log-actual` for earlier talks")
                continue
            seconds, status = parse_runtime(m.group(1))
            if status != "ok":
                print(f"  could not parse {m.group(1)!r}; use m:ss")
                continue
            sp, se, od = last
            log = load_log()
            mask = log.conference.eq(args.conference) & log.speaker.eq(sp) & log.session.eq(se) \
                & log.speaker_order.astype(float).eq(float(od))
            log.loc[mask, "actual_sec"] = seconds
            log.loc[mask, "actual_mmss"] = format_seconds(seconds)
            log.loc[mask, "actual_source"] = "hand"
            save_log(log)
            print(f"  time {format_seconds(seconds)} recorded for {sp}.")
            continue
        m = re.match(r"^o\s+(\d+)$", low)
        if m:
            order = int(m.group(1))
            print(f"  next talk is #{order}")
            continue

        # --- a non-talk program item (sustaining, audit report, ...) ---
        if low in PROGRAM_ALIAS:
            kind = PROGRAM_ALIAS[low]
            label = PROGRAM_ITEMS[kind]["label"]
            est = program_item_estimate(kind, month_num, args.conference)
            if est is None:
                print(f"  no past durations for {label}; run scripts/collect_program_items.py first")
                continue
            pred, used = est
            past = ", ".join(f"{c[:4]} {format_seconds(d)}" for c, d in zip(used.conference, used.duration_sec))
            print(f"\n  {label}  |  {session}  (not a talk; no number)")
            print(f"  >> estimate  {format_seconds(pred):>6s}   plain average of the last {len(used)}: {past}")
            status = log_program_item(args.conference, kind, session, pred, len(used))
            if status == "duplicate":
                rep = ask("  replace the existing estimate? [y/N] ")
                if rep is not None and rep.strip().lower() == "y":
                    log_program_item(args.conference, kind, session, pred, len(used), force=True)
                else:
                    continue
            last = (label, session, 0)
            continue

        # --- a speaker name ---
        candidates = match_speaker(text, directory)
        speaker, default_role = None, None
        if len(candidates) == 1:
            speaker = candidates[0]
            default_role = directory.loc[speaker, "role"]
            print(f"  -> {speaker}")
        elif len(candidates) > 1:
            for i, c in enumerate(candidates, 1):
                print(f"  {i}  {c}  ({directory.loc[c, 'role']}, last {directory.loc[c, 'conference']})")
            pick = ask(f"  which one? (1-{len(candidates)}, or n = none of these, new speaker): ")
            if pick is None:
                return 0
            pick = pick.strip().lower()
            if pick.isdigit() and 1 <= int(pick) <= len(candidates):
                speaker = candidates[int(pick) - 1]
                default_role = directory.loc[speaker, "role"]
            elif pick != "n":
                continue
        if speaker is None:
            full = ask(f"  new speaker: full name as shown on screen [Enter={clean_text(text)}]: ")
            if full is None:
                return 0
            speaker = canonical_speaker(full.strip() or text)
            print(f"  -> {speaker} (new speaker)")
        role = choose_calling(default_role)
        if role is None:
            return 0
        rows, version, group = predict_talk(bundle, hist, year, month_num, ci, args.conference,
                                            speaker, role, session, order, compact=True)
        status = log_predictions(args.conference, speaker, role, group, session, order, rows, version, quiet=True)
        if status == "duplicate":
            rep = ask("  replace the existing prediction? [y/N] ")
            if rep is not None and rep.strip().lower() == "y":
                log_predictions(args.conference, speaker, role, group, session, order, rows, version,
                                force=True, quiet=True)
            else:
                continue
        last = (speaker, session, order)
        order += 1
        print()


def cmd_log_actual(args) -> int:
    seconds, status = parse_runtime(args.actual)
    if status != "ok":
        sys.exit(f"could not parse actual duration {args.actual!r} (expected m:ss); nothing written")
    log = load_log()
    speaker = canonical_speaker(args.speaker)
    session = clean_text(args.session).lower().replace("_", "-").removesuffix("-session")
    mask = (log.conference == args.conference) & (log.speaker == speaker) & (log.session == session)
    if args.order is not None:
        mask &= log.speaker_order.astype(int) == int(args.order)
    if mask.any():
        log.loc[mask, "actual_sec"] = seconds
        log.loc[mask, "actual_mmss"] = format_seconds(seconds)
        log.loc[mask, "actual_source"] = "hand"
        print(f"recorded hand-timed actual {format_seconds(seconds)} on {int(mask.sum())} logged prediction rows "
              f"(provisional; fill-actuals replaces it with the official duration)")
    else:
        log = pd.concat([log, pd.DataFrame([{
            "logged_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "conference": args.conference, "speaker": speaker, "calling": clean_text(args.calling) if args.calling else None,
            "calling_group": calling_group(args.calling) if args.calling else None, "session": session,
            "speaker_order": args.order, "model": None, "model_version": None, "pred_sec": np.nan, "pred_mmss": None,
            "actual_sec": seconds, "actual_mmss": format_seconds(seconds), "actual_source": "hand",
        }])], ignore_index=True)
        print(f"no logged prediction matched; recorded actual {format_seconds(seconds)} as a new row")
    save_log(log)
    return 0


def cmd_remove(args) -> int:
    """Delete logged rows for one talk, e.g. after a typo in the speaker name or session."""
    log = load_log()
    speaker = canonical_speaker(args.speaker)
    mask = log.conference.eq(args.conference) & log.speaker.eq(speaker)
    if args.session:
        session = clean_text(args.session).lower().replace("_", "-").removesuffix("-session")
        mask &= log.session.eq(session)
    if args.order is not None:
        mask &= log.speaker_order.astype(float).eq(float(args.order))
    if not mask.any():
        print(f"no logged rows for speaker={speaker!r} in {args.conference}; nothing removed")
        print("logged speakers:", sorted(log[log.conference.eq(args.conference)].speaker.dropna().unique().tolist()))
        return 0
    save_log(log[~mask])
    print(f"removed {int(mask.sum())} rows for {speaker!r}" + (f" ({args.session})" if args.session else ""))
    return 0


def cmd_fill_actuals(args) -> int:
    """Fill actual durations for logged predictions from the scraped talk pages.

    Matches on conference, canonical speaker and session (and speaker order when
    the log has it). The official video duration is the reference measurement, so
    it replaces hand-timed actuals (the difference is printed). Use --keep-hand to
    leave hand-timed rows alone.
    """
    if not COLLECTED_CSV.exists():
        sys.exit(f"{COLLECTED_CSV} not found; run scripts/collect.py --start {args.conference} --end {args.conference} first")
    col = pd.read_csv(COLLECTED_CSV, encoding="utf-8")
    col = col[col.conference == args.conference].copy()
    if col.empty:
        sys.exit(f"no scraped talks for {args.conference}; run scripts/collect.py --start {args.conference} --end {args.conference}")
    col["speaker"] = col.speaker.map(canonical_speaker)
    col = col[col.duration_ms.notna()]
    log = load_log()
    mask = log.conference.eq(args.conference) & log.model.notna() & log.actual_source.ne("video_data_duration")
    if args.keep_hand:
        mask &= log.actual_sec.isna()
    items = pd.DataFrame()
    if PROGRAM_ITEMS_CSV.exists():  # sustaining / audit rows logged by live mode (calling_group == program_item)
        items = pd.read_csv(PROGRAM_ITEMS_CSV, encoding="utf-8")
        items = items[items.conference.eq(args.conference) & items.duration_sec.notna()]
    filled, unmatched, replaced = 0, set(), []
    for i in log[mask].index:
        r = log.loc[i]
        if r.calling_group == "program_item":
            hit = items[(items.kind == r.calling) & (items.session == r.session)]
            if len(hit) != 1:
                unmatched.add((r.speaker, r.session, "program item; run scripts/collect_program_items.py"))
                continue
            sec = float(hit.duration_sec.iloc[0])
        else:
            hit = col[(col.speaker == r.speaker) & (col.session == r.session)]
            if len(hit) > 1 and not pd.isna(r.speaker_order):
                hit = hit[hit.speaker_order == int(r.speaker_order)]
            if len(hit) != 1:
                unmatched.add((r.speaker, r.session, r.speaker_order))
                continue
            sec = float(hit.duration_ms.iloc[0]) / 1000.0
        if not pd.isna(r.actual_sec) and abs(float(r.actual_sec) - sec) > 0.5:
            replaced.append((r.speaker, r.session, format_seconds(r.actual_sec), format_seconds(sec)))
        log.loc[i, "actual_sec"] = sec
        log.loc[i, "actual_mmss"] = format_seconds(sec)
        log.loc[i, "actual_source"] = "video_data_duration"
        filled += 1
    save_log(log)
    print(f"filled {filled} logged prediction rows from official durations for {args.conference}")
    for sp, se, hand, off in sorted(set(replaced)):
        print(f"  {sp} ({se}): hand-timed {hand} replaced by official {off}")
    for u in sorted(unmatched, key=str):
        print(f"  no unique match for speaker={u[0]!r} session={u[1]!r} order={u[2]}")
    return 0


def cmd_info(args) -> int:
    """Print the frozen model's provenance and write outputs/model_manifest.json."""
    bundle = joblib.load(BUNDLE_PATH)
    m = write_manifest(bundle)
    print(f"model version : {m['model_version']}")
    print(f"trained at    : {m['trained_at']}  (code commit {m['code_commit'] or 'not recorded'})")
    print(f"data through  : conf_index {m['data_through_conf_index']} = "
          f"{m['data_through_conf_index'] // 2}-{'10' if m['data_through_conf_index'] % 2 else '04'}")
    print(f"selection     : {m['split_used_for_selection']}")
    for name, e in m["models"].items():
        print(f"  {name:9s} {e['params']}  val {e['val_mae_min']:.2f}  test {e['test_mae_min']:.2f}  "
              f"range +/-{e['range_half_width_min']:.2f} min covered {e['range_test_coverage']:.0%}")
    print(f"wrote {MANIFEST_PATH}")
    return 0


def cmd_score(args) -> int:
    log = load_log()
    scored = log.dropna(subset=["pred_sec", "actual_sec"])
    scored = scored[scored.model.notna()]
    if args.conference:
        scored = scored[scored.conference == args.conference]
    if scored.empty:
        print("no rows with both a prediction and an actual duration yet")
        return 0
    scored = scored.assign(abs_err_min=(scored.pred_sec - scored.actual_sec).abs() / 60)
    print(scored.groupby("model").abs_err_min.agg(n="size", mae_min="mean", median_ae_min="median").round(2))
    src = scored.drop_duplicates(["conference", "speaker", "session", "speaker_order"]).actual_source.fillna("unknown")
    print("actual source:", src.value_counts().to_dict())
    versions = scored.model_version.dropna().unique().tolist()
    print("model versions:", versions or ["not recorded"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="predict duration for an announced speaker")
    p.add_argument("--speaker", required=True)
    p.add_argument("--calling", required=True, help='role as printed, e.g. "Of the Seventy"')
    p.add_argument("--session", required=True, help="e.g. saturday-morning, sunday-afternoon, saturday-evening")
    p.add_argument("--order", required=True, type=int,
                   help="speaking order within the session counting TALKS ONLY (1 = first talk). Do not count the "
                        "sustaining of officers, the auditing/statistical report, a solemn assembly, videos, music "
                        "or prayers; the next talk after one of those keeps counting from where the talks left off")
    p.add_argument("--conference", default="2026-10", help="target conference, YYYY-MM (default 2026-10)")
    p.add_argument("--model", default="all")
    p.add_argument("--log", action="store_true", help="append predictions to outputs/predictions_log.csv")
    p.add_argument("--force", action="store_true", help="replace an already-logged prediction for this talk")
    p.set_defaults(func=cmd_predict)

    lv = sub.add_parser("live", help="interactive mode: type a speaker name per talk; calling, order and logging are handled")
    lv.add_argument("--conference", default="2026-10")
    lv.set_defaults(func=cmd_live)

    a = sub.add_parser("log-actual", help="record the actual duration for a logged talk")
    a.add_argument("--speaker", required=True)
    a.add_argument("--session", required=True)
    a.add_argument("--actual", required=True, help="m:ss as shown on the talk page")
    a.add_argument("--order", type=int, default=None)
    a.add_argument("--calling", default=None)
    a.add_argument("--conference", default="2026-10")
    a.set_defaults(func=cmd_log_actual)

    rm = sub.add_parser("remove", help="delete logged rows for a talk (fix a typo by removing and re-logging)")
    rm.add_argument("--speaker", required=True, help="the name exactly as it was (mis)typed")
    rm.add_argument("--session", default=None)
    rm.add_argument("--order", type=int, default=None)
    rm.add_argument("--conference", default="2026-10")
    rm.set_defaults(func=cmd_remove)

    f = sub.add_parser("fill-actuals", help="fill actuals for a conference from data/processed/talks_collected.csv")
    f.add_argument("--conference", default="2026-10")
    f.add_argument("--keep-hand", action="store_true", help="do not replace hand-timed actuals with official ones")
    f.set_defaults(func=cmd_fill_actuals)

    i = sub.add_parser("info", help="show the frozen model's version and write outputs/model_manifest.json")
    i.set_defaults(func=cmd_info)

    s = sub.add_parser("score", help="MAE of logged predictions that have actuals")
    s.add_argument("--conference", default=None)
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
