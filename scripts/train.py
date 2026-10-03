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

from general_conference_runtime_predictor.data import build_dataset, format_seconds
from general_conference_runtime_predictor.features import add_history_features
from general_conference_runtime_predictor.models import GRIDS, make_model
from general_conference_runtime_predictor.paths import BUNDLE_PATH, MODELS, OUTPUTS

INTERVAL_LEVELS = (0.5, 0.8, 0.9)


def mae_min(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, dtype=float) - np.asarray(p, dtype=float))) / 60.0)


def evaluate(df: pd.DataFrame, pred) -> dict:
    err = np.abs(df.duration_sec.to_numpy(dtype=float) - np.asarray(pred, dtype=float)) / 60.0
    unseen = df.spk_n_talks.to_numpy() == 0
    few = (df.spk_n_timed.to_numpy() <= 1) & ~unseen
    out = {
        "n": int(len(df)),
        "mae_min": float(err.mean()),
        "median_ae_min": float(np.median(err)),
        "n_unseen": int(unseen.sum()),
        "mae_unseen_min": float(err[unseen].mean()) if unseen.any() else None,
        "n_few_history": int(few.sum()),
        "mae_few_history_min": float(err[few].mean()) if few.any() else None,
        "n_seen": int((~unseen).sum()),
        "mae_seen_min": float(err[~unseen].mean()) if (~unseen).any() else None,
    }
    return out


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
    args = ap.parse_args()

    tag = args.tag or ("legacy" if args.legacy_only else "all")
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in model_names:
        if m not in GRIDS:
            sys.exit(f"unknown model {m!r}; choose from {list(GRIDS)}")

    df, report = build_dataset(include_collected=not args.legacy_only)
    print_data_summary(report)
    df = add_history_features(df)

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
        print("  intervals (half-width min / test coverage): " +
              ", ".join(f"{k}: {v['half_width_min']:.2f}/{v['test_coverage']:.2f}" for k, v in intervals.items()))

    recommended = min(results, key=lambda n: results[n]["val_mae_min"])
    summary = {
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "tag": tag,
        "split": split_info,
        "recommended_by_val_mae": recommended,
        "results": results,
        "tuning": tuning,
        "data_report": report,
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / f"metrics_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [f"# Results ({tag})", "",
             f"Split: train {split_info['train']['conferences']} ({split_info['train']['n_timed']} timed talks), "
             f"val {split_info['val']['conferences']} ({split_info['val']['n_timed']}), "
             f"test {split_info['test']['conferences']} ({split_info['test']['n_timed']}).", "",
             "| model | selected params | val MAE | test MAE | test median AE | seen MAE (n) | unseen MAE (n) | 80% half-width | 80% coverage |",
             "|---|---|---|---|---|---|---|---|---|"]
    for n, r in results.items():
        t = r["test"]
        fmt = lambda v: "n/a" if v is None else f"{v:.2f}"
        lines.append(f"| {n} | `{r['params']}` | {r['val_mae_min']:.2f} | {t['mae_min']:.2f} | {t['median_ae_min']:.2f} | "
                     f"{fmt(t['mae_seen_min'])} ({t['n_seen']}) | {fmt(t['mae_unseen_min'])} ({t['n_unseen']}) | "
                     f"{r['intervals']['0.8']['half_width_min']:.2f} min | {r['intervals']['0.8']['test_coverage']:.2f} |")
    lines += ["", f"Recommended by validation MAE: **{recommended}**.", "", "All MAE values are in minutes."]
    (OUTPUTS / f"metrics_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    cols = ["conference", "session", "speaker_order", "speaker", "calling_group", "spk_n_talks", "spk_n_timed", "duration_sec"]
    out = te[cols].copy()
    out["actual"] = out.duration_sec.map(format_seconds)
    for n in test_pred.columns:
        out[f"pred_{n}"] = test_pred[n].round(1)
        out[f"err_{n}_min"] = ((test_pred[n] - te.duration_sec) / 60).round(2)
    out.to_csv(OUTPUTS / f"test_predictions_{tag}.csv", index=False, encoding="utf-8")

    if not args.no_bundle:
        MODELS.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"models": bundle_models, "recommended": recommended, "trained_at": summary["trained_at"], "tag": tag,
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
