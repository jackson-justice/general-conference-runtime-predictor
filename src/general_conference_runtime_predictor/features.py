"""Historical duration features computed only from strictly earlier conferences."""
from __future__ import annotations

import numpy as np
import pandas as pd

RECENT_WINDOW = 4  # conferences, i.e. the previous two years

HISTORY_COLS = [
    "spk_n_timed", "spk_mean", "spk_median", "spk_std", "spk_min", "spk_max", "spk_last",
    "spk_n_talks", "spk_confs_since_first", "spk_confs_since_last",
    "spk_calling_n", "spk_calling_mean",
    "calling_n", "calling_mean", "calling_recent_mean",
    "role_n", "role_mean",
    "session_mean",
    "global_n", "global_mean", "global_recent_mean",
]
COUNT_COLS = ["spk_n_timed", "spk_n_talks", "spk_calling_n", "calling_n", "role_n", "global_n"]

# Columns a talk row must carry before history features can be attached.
PRE_TALK_COLS = ["speaker", "role_norm", "calling_group", "session", "speaker_order", "month", "conf_index"]


def _lookup(targets: pd.DataFrame, keys: list[str], table: pd.DataFrame) -> pd.DataFrame:
    """Left-join `table` (indexed by `keys`) onto targets, preserving the target index."""
    if table.empty:
        return pd.DataFrame(index=targets.index, columns=table.columns, dtype=float)
    merged = targets[keys].merge(table, left_on=keys, right_index=True, how="left")
    merged.index = targets.index
    return merged.drop(columns=keys)


def history_features(history: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Features for `targets` using only rows in `history` (all strictly earlier).

    Untimed talks in `history` count toward talk counts but never toward a
    duration statistic. Returns a float frame aligned with `targets.index`.
    """
    if not history.empty and history.conf_index.max() >= targets.conf_index.min():
        raise ValueError("history must come from conferences strictly before the targets")
    out = pd.DataFrame(index=targets.index)
    timed = history[history.duration_status == "ok"]

    g = timed.groupby("speaker").duration_sec
    spk = pd.DataFrame(
        {
            "spk_n_timed": g.size(),
            "spk_mean": g.mean(),
            "spk_median": g.median(),
            "spk_std": g.std(),
            "spk_min": g.min(),
            "spk_max": g.max(),
        }
    )
    if not timed.empty:
        ordered = timed.sort_values(["conf_index", "session", "speaker_order"])
        spk["spk_last"] = ordered.groupby("speaker").duration_sec.last()
    else:
        spk["spk_last"] = np.nan
    out = out.join(_lookup(targets, ["speaker"], spk))

    ga = history.groupby("speaker")
    spk_all = pd.DataFrame(
        {
            "spk_n_talks": ga.size(),
            "spk_first_conf": ga.conf_index.min(),
            "spk_last_conf": ga.conf_index.max(),
        }
    )
    tmp = _lookup(targets, ["speaker"], spk_all)
    out["spk_n_talks"] = tmp.spk_n_talks
    out["spk_confs_since_first"] = targets.conf_index - tmp.spk_first_conf
    out["spk_confs_since_last"] = targets.conf_index - tmp.spk_last_conf

    gc = timed.groupby(["speaker", "calling_group"]).duration_sec
    out = out.join(
        _lookup(targets, ["speaker", "calling_group"],
                pd.DataFrame({"spk_calling_n": gc.size(), "spk_calling_mean": gc.mean()}))
    )

    gcal = timed.groupby("calling_group").duration_sec
    out = out.join(
        _lookup(targets, ["calling_group"], pd.DataFrame({"calling_n": gcal.size(), "calling_mean": gcal.mean()}))
    )

    recent = timed[timed.conf_index > timed.conf_index.max() - RECENT_WINDOW] if not timed.empty else timed
    grc = recent.groupby("calling_group").duration_sec
    out = out.join(_lookup(targets, ["calling_group"], pd.DataFrame({"calling_recent_mean": grc.mean()})))

    gr = timed.groupby("role_norm").duration_sec
    out = out.join(_lookup(targets, ["role_norm"], pd.DataFrame({"role_n": gr.size(), "role_mean": gr.mean()})))

    gs = timed.groupby("session").duration_sec
    out = out.join(_lookup(targets, ["session"], pd.DataFrame({"session_mean": gs.mean()})))

    out["global_n"] = float(len(timed))
    out["global_mean"] = timed.duration_sec.mean() if len(timed) else np.nan
    out["global_recent_mean"] = recent.duration_sec.mean() if len(recent) else np.nan

    for c in COUNT_COLS:
        out[c] = out[c].fillna(0).astype(float)
    return out[HISTORY_COLS].astype(float)


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach history features to every row, one conference at a time."""
    df = df.sort_values(["conf_index", "session", "speaker_order"]).copy()
    pieces = []
    for c in sorted(df.conf_index.unique()):
        hist = df[df.conf_index < c]
        tgt = df[df.conf_index == c]
        pieces.append(history_features(hist, tgt))
    return df.join(pd.concat(pieces))
