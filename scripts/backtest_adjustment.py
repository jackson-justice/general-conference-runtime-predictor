"""Backtest the in-conference bias adjustment used by `predict.py live`.

For each test conference, walk the talks in broadcast order and shift the frozen
CatBoost prediction by n/(n+c) times the mean miss of the talks already finished
(Church President remarks excluded from the residuals). Reports MAE before/after.

Usage: uv run python scripts/backtest_adjustment.py [--c 4] [--tag all]
       uv run python scripts/backtest_adjustment.py --file outputs/holdout_predictions_thru2026-04_sess.csv --col pred_catboost_guess
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from general_conference_runtime_predictor.paths import OUTPUTS

SESSION_ORDER = {"saturday-morning": 0, "saturday-afternoon": 1, "saturday-evening": 2, "sunday-morning": 3,
                 "sunday-afternoon": 4}


def backtest(t: pd.DataFrame, c: float) -> pd.DataFrame:
    out = []
    for conf, g in t.groupby("conference", sort=False):
        res: list[float] = []
        for r in g.itertuples():
            n = len(res)
            shift = (n / (n + c)) * float(np.mean(res)) if n else 0.0
            out.append((conf, r.calling_group, r.duration_sec, r.pred, r.pred + shift))
            if r.calling_group != "church_president":
                res.append(r.duration_sec - r.pred)
    o = pd.DataFrame(out, columns=["conference", "calling_group", "y", "frozen", "adjusted"])
    o["ae_frozen"] = (o.y - o.frozen).abs() / 60
    o["ae_adjusted"] = (o.y - o.adjusted).abs() / 60
    return o


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c", type=float, default=4.0, help="shrinkage pseudo-talks")
    ap.add_argument("--tag", default="all")
    ap.add_argument("--file", default=None, help="predictions CSV (default outputs/test_predictions_<tag>.csv)")
    ap.add_argument("--col", default="pred_catboost", help="prediction column to adjust")
    args = ap.parse_args()
    t = pd.read_csv(args.file or OUTPUTS / f"test_predictions_{args.tag}.csv", encoding="utf-8")
    t["pred"] = t[args.col]
    t["_s"] = t.session.map(SESSION_ORDER)
    t = t.sort_values(["conference", "_s", "speaker_order"])
    o = backtest(t, args.c)
    print(f"{args.col} c={args.c:g}  MAE (min): frozen {o.ae_frozen.mean():.3f} -> adjusted {o.ae_adjusted.mean():.3f}")
    print(o.groupby("conference")[["ae_frozen", "ae_adjusted"]].mean().round(2).to_string())
    print(o.groupby("calling_group")[["ae_frozen", "ae_adjusted"]].mean().round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
