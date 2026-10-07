"""Train and evaluate duration models with chronological, conference-level splits.

Usage:
    uv run python scripts/train.py                       # legacy CSV + scraped data
    uv run python scripts/train.py --legacy-only         # existing CSV only
    uv run python scripts/train.py --models baseline,ridge
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

import joblib
import numpy as np
import pandas as pd
import subprocess

from general_conference_runtime_predictor.data import build_dataset, conf_index, format_seconds, parse_conference_key
from general_conference_runtime_predictor.features import SESSION_COLS, add_history_features
from general_conference_runtime_predictor.models import GRIDS, SESSION_SIZE_MODELS, make_model
from general_conference_runtime_predictor.paths import BUNDLE_PATH, MODELS, OUTPUTS

INTERVAL_LEVELS = (0.5, 0.8, 0.9)


def mae_min(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, dtype=float) - np.asarray(p, dtype=float))) / 60.0)


# Test rows are reported overall and for these fixed segments. Every model is
# scored on exactly the same rows (the timed talks of the test conferences).
SEGMENTS = {
    "all": lambda d: np.ones(len(d), dtype=bool),
    "church_president": lambda d: d.calling_group.eq("church_president").to_numpy(),
    "seventy": lambda d: d.calling_group.eq("seventy").to_numpy(),
    "seventy_presidency": lambda d: d.calling_group.eq("seventy_presidency").to_numpy(),
    "unseen_speaker": lambda d: d.spk_n_talks.to_numpy() == 0,
    "seen_speaker": lambda d: d.spk_n_talks.to_numpy() > 0,
    "few_history": lambda d: (d.spk_n_talks.to_numpy() > 0) & (d.spk_n_timed.to_numpy() <= 1),
}


def evaluate(df: pd.DataFrame, pred) -> dict:
    err = np.abs(df.duration_sec.to_numpy(dtype=float) - np.asarray(pred, dtype=float)) / 60.0
    seg = {}
    for name, fn in SEGMENTS.items():
        mask = fn(df)
        seg[name] = {"n": int(mask.sum()),
                     "mae_min": float(err[mask].mean()) if mask.any() else None,
                     "median_ae_min": float(np.median(err[mask])) if mask.any() else None}
    return {
        "n": seg["all"]["n"], "mae_min": seg["all"]["mae_min"], "median_ae_min": seg["all"]["median_ae_min"],
        "n_unseen": seg["unseen_speaker"]["n"], "mae_unseen_min": seg["unseen_speaker"]["mae_min"],
        "n_seen": seg["seen_speaker"]["n"], "mae_seen_min": seg["seen_speaker"]["mae_min"],
        "n_few_history": seg["few_history"]["n"], "mae_few_history_min": seg["few_history"]["mae_min"],
        "segments": seg,
    }


def paired_comparison(y, pred_a, pred_b, seed: int = 0, n_boot: int = 2000) -> dict:
    """Per-talk comparison of model A against model B on identical rows (A - B, minutes)."""
    y = np.asarray(y, dtype=float)
    d = (np.abs(y - np.asarray(pred_a, dtype=float)) - np.abs(y - np.asarray(pred_b, dtype=float))) / 60.0
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return {"n": int(len(d)), "mean_diff_min": float(d.mean()),
            "ci95_min": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "share_a_better": float(np.mean(d < 0)), "share_tied": float(np.mean(d == 0))}


def print_data_summary(report: dict) -> None:
    print("\n=== Data ===")
    print(f"talks: {report['n_talks']}  conferences: {report['conferences'][0]} .. {report['conferences'][-1]} "
          f"({len(report['conferences'])})")
    print("duration status:", report["by_status"])
    for src, st in report["by_source_status"].items():
        print(f"  {src}: {st}")
    if report["invalid_rows"]:
        print("invalid durations (excluded from targets, never imputed):")
        for r in report["invalid_rows"]:
            print(f"  {r['conference']} {r['speaker']}: runtime={r['duration_raw']} words={r['num_words']} -> {r['status']}")
    print("notes:", json.dumps(report["notes"]))
    print("coverage by conference (talks / timed / not timed):")
    for c in report["coverage_by_conference"]:
        print(f"  {c['conference']} {c['source']:10s} sessions={c['sessions']} talks={c['talks']:3d} "
              f"timed={c['timed']:3d} not_timed={c['not_timed']}")
    d = report["duration_minutes"]
    print(f"timed talks: {d['n']}  mean {d['mean']} min  median {d['median']}  std {d['std']}  range {d['min']}-{d['max']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--legacy-only", action="store_true", help="ignore scraped data in data/processed")
    ap.add_argument("--models", default="naive,baseline,ridge,catboost")
    ap.add_argument("--n-val", type=int, default=4, help="validation conferences (whole conferences)")
    ap.add_argument("--n-test", type=int, default=4, help="held-out test conferences (most recent)")
    ap.add_argument("--tag", default=None, help="suffix for output files (default: legacy|all)")
    ap.add_argument("--no-bundle", action="store_true", help="do not save models/bundle.joblib")
    ap.add_argument("--exclude-president-under", type=float, default=None, metavar="MIN",
                    help="drop Church President talks shorter than MIN minutes from history, fitting and scoring "
                         "(opt-in experiment; the default keeps every timed talk)")
    ap.add_argument("--exploratory", default=None, metavar="REASON",
                    help="label this run's test numbers as exploratory (the test conferences were already "
                         "inspected before this change), with the reason")
    ap.add_argument("--session-size", dest="session_size", action="store_true", default=True,
                    help="give ridge and catboost the expected session size learned from earlier conferences "
                         "(session_n_prev, session_n_recent; default)")
    ap.add_argument("--no-session-size", dest="session_size", action="store_false",
                    help="predictors as before October 2026: without the expected-session-size columns")
    ap.add_argument("--through", default=None, metavar="YYYY-MM",
                    help="use only conferences up to and including this one for splitting, tuning and the "
                         "final fit (default: everything in the data)")
    ap.add_argument("--predict-next", action="store_true",
                    help="with --through: predict the conferences after it with the final models, as a true "
                         "hold-out (their talks never entered selection or fitting); writes "
                         "outputs/holdout_predictions_<tag>.csv")
    args = ap.parse_args()
    if args.predict_next and not args.through:
        sys.exit("--predict-next needs --through")

    tag = args.tag or ("legacy" if args.legacy_only else "all")
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in model_names:
        if m not in GRIDS:
            sys.exit(f"unknown model {m!r}; choose from {list(GRIDS)}")

    df, report = build_dataset(include_collected=not args.legacy_only)
    print_data_summary(report)
    exclusion = None
    if args.exclude_president_under is not None:
        short = df.calling_group.eq("church_president") & df.duration_status.eq("ok")             & (df.duration_sec < args.exclude_president_under * 60)
        df.loc[short, "duration_status"] = "excluded_short_president"
        exclusion = (f"{int(short.sum())} Church President talks shorter than {args.exclude_president_under:g} min "
                     f"excluded from history, fitting and scoring")
        print(); print(exclusion)
    df = add_history_features(df)
    holdout = None
    if args.through:
        through_ci = conf_index(*parse_conference_key(args.through))
        if through_ci not in set(df.conf_index):
            sys.exit(f"--through {args.through} is not a conference in the data")
        holdout = df[df.conf_index > through_ci].copy()
        df = df[df.conf_index <= through_ci].copy()
        print(f"\nusing conferences through {args.through} only"
              + (f"; {len(holdout)} later talks kept aside as hold-out" if len(holdout) else ""))
    extra = {name: ({"session_size": args.session_size} if name in SESSION_SIZE_MODELS else {})
             for name in model_names}
    print(f"expected-session-size columns {SESSION_COLS} "
          + (f"given to {[n for n in model_names if n in SESSION_SIZE_MODELS]}" if args.session_size else "withheld"))

    confs = sorted(df.conf_index.unique())
    if len(confs) < args.n_val + args.n_test + 2:
        sys.exit("not enough conferences for the requested split")
    test_c = confs[-args.n_test:]
    val_c = confs[-(args.n_test + args.n_val):-args.n_test]
    train_c = confs[: -(args.n_test + args.n_val)]
    timed = df[df.duration_status == "ok"].copy()
    tr = timed[timed.conf_index.isin(train_c)]
    va = timed[timed.conf_index.isin(val_c)]
    te = timed[timed.conf_index.isin(test_c)]
    key = lambda cs: f"{df[df.conf_index == cs[0]].conference.iloc[0]} .. {df[df.conf_index == cs[-1]].conference.iloc[0]}"
    split_info = {
        "train": {"conferences": key(train_c), "n_conf": len(train_c), "n_timed": int(len(tr))},
        "val": {"conferences": key(val_c), "n_conf": len(val_c), "n_timed": int(len(va))},
        "test": {"conferences": key(test_c), "n_conf": len(test_c), "n_timed": int(len(te))},
    }
    print("\n=== Split (chronological, whole conferences) ===")
    for k, v in split_info.items():
        print(f"{k:5s} {v['conferences']}  conferences={v['n_conf']}  timed talks={v['n_timed']}")
    print(f"test talks by speakers never seen before: {(te.spk_n_talks == 0).sum()}")

    results, tuning, bundle_models, test_pred = {}, {}, {}, pd.DataFrame(index=te.index)
    trva = pd.concat([tr, va])
    for name in model_names:
        print(f"\n=== {name} ===")
        tuning[name] = []
        best = None
        for params in GRIDS[name]:
            params = {**params, **extra[name]}
            m = make_model(name, **params).fit(tr, tr.duration_sec, va, va.duration_sec)
            pv = m.predict(va)
            score = mae_min(va.duration_sec, pv)
            rec = {"params": params, "val_mae_min": round(score, 4)}
            if hasattr(m, "best_iterations_"):
                rec["best_iterations"] = m.best_iterations_
            tuning[name].append(rec)
            print(f"  {params} -> val MAE {score:.3f} min" + (f" (iters {m.best_iterations_})" if "best_iterations" in rec else ""))
            if best is None or score < best[0]:
                best = (score, dict(params), m, pv)
        val_score, params, model_val, pv = best
        if name == "catboost":
            params = {**params, "iterations": model_val.best_iterations_, "early_stopping": False}

        # Interval half-widths from out-of-sample validation residuals, then checked on test.
        res_val = np.abs(va.duration_sec.to_numpy(dtype=float) - pv)
        half_widths = {lvl: float(np.quantile(res_val, lvl)) for lvl in INTERVAL_LEVELS}

        model_test = make_model(name, **params).fit(trva, trva.duration_sec)
        pt = model_test.predict(te)
        test_metrics = evaluate(te, pt)
        res_test = np.abs(te.duration_sec.to_numpy(dtype=float) - pt)
        intervals = {
            str(lvl): {
                "half_width_min": round(hw / 60.0, 3),
                "test_coverage": round(float(np.mean(res_test <= hw)), 3),
            }
            for lvl, hw in half_widths.items()
        }
        test_pred[name] = pt

        final = make_model(name, **params).fit(timed, timed.duration_sec)
        bundle_models[name] = {"model": final, "params": params, "intervals": intervals,
                               "val_mae_min": val_score, "test": test_metrics}
        results[name] = {"params": params, "val_mae_min": round(val_score, 4), "test": test_metrics,
                         "intervals": intervals}
        print(f"  selected {params}")
        print(f"  val MAE {val_score:.3f} | test MAE {test_metrics['mae_min']:.3f} "
              f"(seen {test_metrics['mae_seen_min']:.3f} n={test_metrics['n_seen']}; "
              f"unseen {test_metrics['mae_unseen_min'] if test_metrics['mae_unseen_min'] is None else round(test_metrics['mae_unseen_min'], 3)} "
              f"n={test_metrics['n_unseen']})")
        print("  test MAE by segment: " + ", ".join(
            f"{k}={v['mae_min']:.2f} (n={v['n']})" for k, v in test_metrics["segments"].items() if v["mae_min"] is not None))
        print("  intervals (half-width min / test coverage): " +
              ", ".join(f"{k}: {v['half_width_min']:.2f}/{v['test_coverage']:.2f}" for k, v in intervals.items()))

    recommended = min(results, key=lambda n: results[n]["val_mae_min"])

    # True hold-out: conferences after --through, predicted by the final models (fit on everything
    # through --through). Their talks never entered selection or fitting; their history features,
    # including the expected session sizes, come from conferences through --through only.
    holdout_summary = None
    if args.predict_next and holdout is not None and len(holdout):
        ho = holdout[holdout.duration_status == "ok"].copy()
        hcols = ["conference", "session", "speaker_order", "speaker", "role_norm", "calling_group",
                 *SESSION_COLS, "duration_sec"]
        hout = ho[hcols].copy()
        hout["session_n_actual"] = ho.groupby(["conference", "session"]).speaker_order.transform("max")
        hout["actual"] = hout.duration_sec.map(format_seconds)
        holdout_summary = {"conferences": sorted(ho.conference.unique().tolist()), "n_timed": int(len(ho)),
                           "mae_min": {}}
        print(f"\n=== Hold-out {holdout_summary['conferences']} ({len(ho)} timed talks), final models ===")
        for name, entry in bundle_models.items():
            p = entry["model"].predict(ho)
            hout[f"pred_{name}"] = np.round(p, 1)
            hout[f"err_{name}_min"] = ((p - ho.duration_sec) / 60).round(2)
            holdout_summary["mae_min"][name] = round(mae_min(ho.duration_sec, p), 3)
            print(f"  {name:9s} MAE {holdout_summary['mae_min'][name]:.3f} min")
        hout.to_csv(OUTPUTS / f"holdout_predictions_{tag}.csv", index=False, encoding="utf-8")
        print(f"  wrote {OUTPUTS / f'holdout_predictions_{tag}.csv'}")
        sizes = hout.drop_duplicates(["conference", "session"])
        print("  session sizes (expected from history -> actual): " + ", ".join(
            f"{r.session} {r.session_n_prev:.0f}->{int(r.session_n_actual)}" for r in sizes.itertuples()))

    paired = {}
    if "catboost" in test_pred:
        for other in ("naive", "baseline", "ridge"):
            if other in test_pred:
                paired[f"catboost_vs_{other}"] = paired_comparison(te.duration_sec, test_pred["catboost"], test_pred[other])
    summary = {
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "tag": tag,
        "test_status": ("exploratory: " + args.exploratory) if args.exploratory else "single report",
        "exclusion": exclusion,
        "session_size_feature": bool(args.session_size),
        "data_through": args.through,
        "holdout": holdout_summary,
        "split": split_info,
        "recommended_by_val_mae": recommended,
        "results": results,
        "paired_test_comparisons": paired,
        "tuning": tuning,
        "data_report": report,
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / f"metrics_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [f"# Results ({tag})", ""]
    if args.exploratory:
        lines += [f"**EXPLORATORY.** The test conferences had already been inspected before this run; "
                  f"reason for re-running: {args.exploratory}. Treat test numbers as a second look, not a "
                  f"fresh hold-out. The next untouched conference is the first one after the data ends.", ""]
    if exclusion:
        lines += [f"**Exclusion experiment:** {exclusion}. Test rows differ from the default run, so numbers "
                  f"are not comparable with it row for row.", ""]
    lines += [("Ridge and CatBoost use the expected session size learned from earlier conferences "
               "(`session_n_prev`, `session_n_recent`)." if args.session_size else
               "Predictors as before October 2026 (without the expected-session-size columns)."), ""]
    lines += [f"Split: train {split_info['train']['conferences']} ({split_info['train']['n_timed']} timed talks), "
             f"val {split_info['val']['conferences']} ({split_info['val']['n_timed']}), "
             f"test {split_info['test']['conferences']} ({split_info['test']['n_timed']}).", "",
             "| model | selected params | val MAE | test MAE | test median AE | seen MAE (n) | unseen MAE (n) | range half-width (80th pct val error) | test coverage of range |",
             "|---|---|---|---|---|---|---|---|---|"]
    for n, r in results.items():
        t = r["test"]
        fmt = lambda v: "n/a" if v is None else f"{v:.2f}"
        lines.append(f"| {n} | `{r['params']}` | {r['val_mae_min']:.2f} | {t['mae_min']:.2f} | {t['median_ae_min']:.2f} | "
                     f"{fmt(t['mae_seen_min'])} ({t['n_seen']}) | {fmt(t['mae_unseen_min'])} ({t['n_unseen']}) | "
                     f"{r['intervals']['0.8']['half_width_min']:.2f} min | {r['intervals']['0.8']['test_coverage']:.2f} |")
    lines += ["", f"Recommended by validation MAE: **{recommended}**.", ""]
    seg_names = ["all", "church_president", "seventy", "seventy_presidency", "unseen_speaker", "seen_speaker"]
    lines += ["## Test MAE by segment (minutes, same rows for every model)", "",
              "| segment | n | " + " | ".join(results) + " |",
              "|---|---|" + "---|" * len(results)]
    for sname in seg_names:
        n = results[next(iter(results))]["test"]["segments"][sname]["n"]
        cells = []
        for r in results.values():
            v = r["test"]["segments"][sname]["mae_min"]
            cells.append("n/a" if v is None else f"{v:.2f}")
        lines.append(f"| {sname} | {n} | " + " | ".join(cells) + " |")
    if paired:
        lines += ["", "## CatBoost vs the baselines on identical test rows", "",
                  "Paired difference in absolute error (CatBoost minus other, minutes; negative favours CatBoost).", "",
                  "| comparison | n | mean diff | 95% bootstrap CI | share CatBoost better |", "|---|---|---|---|---|"]
        for k, v in paired.items():
            lines.append(f"| {k.replace('_', ' ')} | {v['n']} | {v['mean_diff_min']:+.2f} | "
                         f"[{v['ci95_min'][0]:+.2f}, {v['ci95_min'][1]:+.2f}] | {v['share_a_better']:.0%} |")
    if holdout_summary:
        lines += ["", f"## Hold-out: {', '.join(holdout_summary['conferences'])} ({holdout_summary['n_timed']} timed talks)", "",
                  "Predicted by the final models (fit on everything through the `--through` conference).", "",
                  "| model | hold-out MAE |", "|---|---|"]
        for k, v in holdout_summary["mae_min"].items():
            lines.append(f"| {k} | {v:.2f} |")
    lines += ["", "All MAE values are in minutes."]
    (OUTPUTS / f"metrics_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    cols = ["conference", "session", "speaker_order", "speaker", "speaker_source", "role_norm", "calling_group",
            *SESSION_COLS, "spk_n_talks", "spk_n_timed", "duration_source", "duration_sec"]
    out = te[cols].copy()
    out["actual"] = out.duration_sec.map(format_seconds)
    for n in test_pred.columns:
        out[f"pred_{n}"] = test_pred[n].round(1)
        out[f"err_{n}_min"] = ((test_pred[n] - te.duration_sec) / 60).round(2)
    out.to_csv(OUTPUTS / f"test_predictions_{tag}.csv", index=False, encoding="utf-8")

    if not args.no_bundle:
        MODELS.mkdir(parents=True, exist_ok=True)
        try:
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                    check=True, cwd=MODELS.parent).stdout.strip()
            if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                              text=True, cwd=MODELS.parent).stdout.strip():
                commit += "-dirty"
        except Exception:  # noqa: BLE001
            commit = None
        joblib.dump(
            {"models": bundle_models, "recommended": recommended, "trained_at": summary["trained_at"], "tag": tag,
             "code_commit": commit, "session_size": bool(args.session_size),
             "split": split_info, "max_conf_index": int(df.conf_index.max()),
             "sessions": sorted(df.session.unique().tolist())},
            BUNDLE_PATH,
        )
        print(f"\nsaved {BUNDLE_PATH}")
    print(f"wrote {OUTPUTS / f'metrics_{tag}.md'}, metrics_{tag}.json, test_predictions_{tag}.csv")
    print(f"recommended model (by validation MAE): {recommended}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
