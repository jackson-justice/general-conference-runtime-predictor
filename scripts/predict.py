"""Predict a talk's duration from pre-talk information, and log predictions/actuals.

Usage:
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
from general_conference_runtime_predictor.paths import BUNDLE_PATH, COLLECTED_CSV, OUTPUTS, PREDICTION_LOG

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
    version = model_version(bundle)
    print(f"model version {version}")
    print("\nmodel      prediction   estimated range (uncalibrated; share of test talks it covered)   test MAE")
    for name in names:
        entry = bundle["models"][name]
        pred = float(entry["model"].predict(X)[0])
        iv = entry["intervals"]["0.8"]
        hw = iv["half_width_min"] * 60
        star = " <- recommended" if name == bundle["recommended"] else ""
        print(f"{name:10s} {format_seconds(pred):>8s}     {format_seconds(pred - hw)} - {format_seconds(pred + hw)} "
              f"(covered {iv['test_coverage']:.0%})                                   {entry['test']['mae_min']:.2f} min{star}")
        rows.append({"model": name, "pred_sec": round(pred, 1)})
    print("\nThe range is not a calibrated interval. Its half-width is the 80th percentile of validation errors; "
          "'covered' is the share of held-out test talks that fell inside it (62% for catboost).")

    if args.log:
        log = load_log()
        now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        new = pd.DataFrame([{
            "logged_at": now, "conference": args.conference, "speaker": speaker, "calling": role,
            "calling_group": group, "session": session, "speaker_order": int(args.order),
            "model": r["model"], "model_version": version,
            "pred_sec": r["pred_sec"], "pred_mmss": format_seconds(r["pred_sec"]),
            "actual_sec": np.nan, "actual_mmss": None, "actual_source": None,
        } for r in rows])
        # One prediction per (conference, speaker, session, order, model). An existing row is
        # never overwritten unless --force is given; use `remove` for typos.
        keys = ["conference", "speaker", "session", "speaker_order", "model"]
        if len(log):
            dup = log.set_index(keys).index.isin(new.set_index(keys).index)
            if dup.any() and not args.force:
                first = log[dup].iloc[0]
                print(f"\nNOT logged: {speaker!r} {session} #{int(args.order)} already has a prediction "
                      f"(logged {first.logged_at}, version {first.model_version}). "
                      f"Re-run with --force to replace it, or `remove` the row if it was a typo.")
                return 0
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
    filled, unmatched, replaced = 0, set(), []
    for i in log[mask].index:
        r = log.loc[i]
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
