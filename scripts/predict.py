"""Predict a talk's duration from pre-talk information, and log predictions/actuals.

Usage:
    uv run python scripts/predict.py predict --speaker "Dale G. Renlund" \
        --calling "Of the Quorum of the Twelve Apostles" --session sunday-morning --order 3 --log
    uv run python scripts/predict.py log-actual --speaker "Dale G. Renlund" --session sunday-morning --actual 14:12
    uv run python scripts/predict.py fill-actuals --conference 2026-10   # after collect.py has scraped it
    uv run python scripts/predict.py score
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

import joblib
import numpy as np
import pandas as pd

from general_conference_runtime_predictor.data import (
    MONTH_NAME, calling_group, canonical_speaker, clean_text, conf_index, format_seconds, load_dataset, normalize_role,
    parse_conference_key, parse_runtime,
)
from general_conference_runtime_predictor.features import history_features
from general_conference_runtime_predictor.paths import BUNDLE_PATH, COLLECTED_CSV, PREDICTION_LOG

LOG_COLS = ["logged_at", "conference", "speaker", "calling", "calling_group", "session", "speaker_order",
            "model", "pred_sec", "pred_mmss", "actual_sec", "actual_mmss"]


def load_log() -> pd.DataFrame:
    if PREDICTION_LOG.exists():
        return pd.read_csv(PREDICTION_LOG, encoding="utf-8", dtype={"actual_mmss": str, "pred_mmss": str})
    return pd.DataFrame(columns=LOG_COLS)


def save_log(log: pd.DataFrame) -> None:
    PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    log[LOG_COLS].to_csv(PREDICTION_LOG, index=False, encoding="utf-8")


def cmd_predict(args) -> int:
    bundle = joblib.load(BUNDLE_PATH)
    df = load_dataset()
    year, month_num = parse_conference_key(args.conference)
    ci = conf_index(year, month_num)
    hist = df[df.conf_index < ci]
    if ci <= df.conf_index.max():
        print(f"note: {args.conference} is inside the training data; history is truncated to earlier conferences")

    speaker = canonical_speaker(args.speaker)
    role = clean_text(args.calling)
    session = clean_text(args.session).lower().replace("_", "-").removesuffix("-session")
    if session not in bundle["sessions"]:
        print(f"warning: session {session!r} not among known sessions {bundle['sessions']}")
    group = calling_group(role)
    if group == "other":
        print(f"warning: calling {role!r} did not map to a known calling group; using 'other'")

    row = pd.DataFrame([{
        "speaker": speaker, "role": role, "role_norm": normalize_role(role), "calling_group": group,
        "session": session, "speaker_order": int(args.order), "month": MONTH_NAME[month_num],
        "conf_index": ci, "year": year,
    }])
    X = row.join(history_features(hist, row))
    f = X.iloc[0]
    print(f"\n{speaker} | {role} ({group}) | {session} #{int(args.order)} | {args.conference}")
    if f.spk_n_talks == 0:
        print("speaker history: none in the dataset (unseen speaker; prediction relies on calling/session)")
    else:
        print(f"speaker history: {int(f.spk_n_talks)} prior talks, {int(f.spk_n_timed)} timed, "
              f"mean {format_seconds(f.spk_mean)}, last {format_seconds(f.spk_last)}, "
              f"same-calling mean {format_seconds(f.spk_calling_mean)} (n={int(f.spk_calling_n)})")
    print(f"calling mean {format_seconds(f.calling_mean)} | session mean {format_seconds(f.session_mean)} | "
          f"recent global mean {format_seconds(f.global_recent_mean)}")

    names = list(bundle["models"]) if args.model == "all" else [args.model]
    rows = []
    print("\nmodel      prediction   80% interval (held-out coverage)   test MAE")
    for name in names:
        entry = bundle["models"][name]
        pred = float(entry["model"].predict(X)[0])
        iv = entry["intervals"]["0.8"]
        hw = iv["half_width_min"] * 60
        star = " <- recommended" if name == bundle["recommended"] else ""
        print(f"{name:10s} {format_seconds(pred):>8s}     {format_seconds(pred - hw)} - {format_seconds(pred + hw)} "
              f"(coverage {iv['test_coverage']:.0%})          {entry['test']['mae_min']:.2f} min{star}")
        rows.append({"model": name, "pred_sec": round(pred, 1)})
    print("\nIntervals are empirical: half-width = 80th percentile of validation errors; "
          "coverage is what that interval achieved on held-out test conferences.")

    if args.log:
        log = load_log()
        now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        new = pd.DataFrame([{
            "logged_at": now, "conference": args.conference, "speaker": speaker, "calling": role,
            "calling_group": group, "session": session, "speaker_order": int(args.order),
            "model": r["model"], "pred_sec": r["pred_sec"], "pred_mmss": format_seconds(r["pred_sec"]),
            "actual_sec": np.nan, "actual_mmss": None,
        } for r in rows])
        # Keep one prediction per (conference, speaker, session, order, model): latest wins.
        keys = ["conference", "speaker", "session", "speaker_order", "model"]
        if len(log):
            dup = log.set_index(keys).index.isin(new.set_index(keys).index)
            carried_actual = log[dup].dropna(subset=["actual_sec"])
            if len(carried_actual):
                new = new.merge(carried_actual[keys + ["actual_sec"]].rename(columns={"actual_sec": "_a"}),
                                on=keys, how="left")
                new["actual_sec"] = new["_a"]
                new["actual_mmss"] = new.actual_sec.map(lambda s: None if pd.isna(s) else format_seconds(s))
                new = new.drop(columns="_a")
            log = log[~dup]
        log = pd.concat([log, new], ignore_index=True)
        save_log(log)
        print(f"logged {len(new)} prediction rows to {PREDICTION_LOG}")
    return 0


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
        print(f"recorded actual {format_seconds(seconds)} on {int(mask.sum())} logged prediction rows")
    else:
        log = pd.concat([log, pd.DataFrame([{
            "logged_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "conference": args.conference, "speaker": speaker, "calling": clean_text(args.calling) if args.calling else None,
            "calling_group": calling_group(args.calling) if args.calling else None, "session": session,
            "speaker_order": args.order, "model": None, "pred_sec": np.nan, "pred_mmss": None,
            "actual_sec": seconds, "actual_mmss": format_seconds(seconds),
        }])], ignore_index=True)
        print(f"no logged prediction matched; recorded actual {format_seconds(seconds)} as a new row")
    save_log(log)
    return 0


def cmd_fill_actuals(args) -> int:
    """Fill actual durations for logged predictions from the scraped talk pages.

    Matches on conference, canonical speaker and session (and speaker order when
    the log has it). A logged row whose actual was typed by hand is left alone
    unless --overwrite is given.
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
    mask = log.conference.eq(args.conference) & log.model.notna()
    if not args.overwrite:
        mask &= log.actual_sec.isna()
    filled, unmatched = 0, set()
    for i in log[mask].index:
        r = log.loc[i]
        hit = col[(col.speaker == r.speaker) & (col.session == r.session)]
        if len(hit) > 1 and not pd.isna(r.speaker_order):
            hit = hit[hit.speaker_order == int(r.speaker_order)]
        if len(hit) != 1:
            unmatched.add((r.speaker, r.session, r.speaker_order))
            continue
        sec = float(hit.duration_ms.iloc[0]) / 1000.0
        log.loc[i, "actual_sec"] = sec
        log.loc[i, "actual_mmss"] = format_seconds(sec)
        filled += 1
    save_log(log)
    print(f"filled {filled} logged prediction rows from scraped durations for {args.conference}")
    for u in sorted(unmatched, key=str):
        print(f"  no unique match for speaker={u[0]!r} session={u[1]!r} order={u[2]}")
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
    p.set_defaults(func=cmd_predict)

    a = sub.add_parser("log-actual", help="record the actual duration for a logged talk")
    a.add_argument("--speaker", required=True)
    a.add_argument("--session", required=True)
    a.add_argument("--actual", required=True, help="m:ss as shown on the talk page")
    a.add_argument("--order", type=int, default=None)
    a.add_argument("--calling", default=None)
    a.add_argument("--conference", default="2026-10")
    a.set_defaults(func=cmd_log_actual)

    f = sub.add_parser("fill-actuals", help="fill actuals for a conference from data/processed/talks_collected.csv")
    f.add_argument("--conference", default="2026-10")
    f.add_argument("--overwrite", action="store_true", help="also replace actuals that were entered by hand")
    f.set_defaults(func=cmd_fill_actuals)

    s = sub.add_parser("score", help="MAE of logged predictions that have actuals")
    s.add_argument("--conference", default=None)
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
