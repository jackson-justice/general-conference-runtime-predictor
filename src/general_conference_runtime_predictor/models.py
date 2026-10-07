"""Duration models sharing one fit/predict interface over the feature frame.

Every model sees only pre-talk information: speaker, calling, session,
speaking order, month, and history features from earlier conferences.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import HISTORY_COLS, SESSION_COLS

CAT_COLS = ["calling_group", "session", "month"]
# CatBoost gets no `speaker` categorical: its target statistics would average a
# speaker's other talks, including same-conference ones, which the history-feature
# rule forbids. Speaker history enters only through HISTORY_COLS (strictly earlier
# conferences). The remaining categoricals are one-hot encoded (one_hot_max_size),
# so CatBoost never computes a target statistic from the training rows.
CATBOOST_CAT_COLS = ["calling_group", "role_norm", "session", "month"]
CATBOOST_ONE_HOT_MAX = 64
NUMERIC_COLS = ["speaker_order"] + HISTORY_COLS
SESSION_SIZE_MODELS = ("ridge", "catboost")


def numeric_cols(session_size: bool) -> list[str]:
    """Numeric predictors; `session_size=False` drops the expected-session-size history columns."""
    return [c for c in NUMERIC_COLS if session_size or c not in SESSION_COLS]

# Small, fixed tuning grids. Settings are chosen on the validation conferences.
GRIDS = {
    "naive": [{}],
    "baseline": [{"min_n": n, "shrink": s} for n in (1, 2, 3) for s in (0.0, 1.0, 3.0)],
    "ridge": [{"alpha": a, "speaker_onehot": so} for a in (0.3, 1.0, 3.0, 10.0, 30.0) for so in (False, True)],
    "catboost": [{"depth": d, "loss": loss} for d in (4, 6) for loss in ("RMSE", "MAE")],
}


def fill_fallbacks(X: pd.DataFrame, defaults: dict) -> pd.DataFrame:
    """Fill missing history stats hierarchically: speaker -> calling -> global -> training mean."""
    X = X.copy()
    gm = X.global_mean.fillna(defaults["global_mean"])
    X["global_mean"] = gm
    X["global_recent_mean"] = X.global_recent_mean.fillna(gm)
    cm = X.calling_mean.fillna(gm)
    X["calling_mean"] = cm
    X["calling_recent_mean"] = X.calling_recent_mean.fillna(cm)
    X["role_mean"] = X.role_mean.fillna(cm)
    X["session_mean"] = X.session_mean.fillna(gm)
    X["spk_has_history"] = (X.spk_n_timed.fillna(0) > 0).astype(float)
    for c in ["spk_mean", "spk_median", "spk_min", "spk_max", "spk_last", "spk_calling_mean"]:
        X[c] = X[c].fillna(cm)
    X["spk_std"] = X.spk_std.fillna(defaults["global_std"])
    for c in ["spk_confs_since_first", "spk_confs_since_last"]:
        X[c] = X[c].fillna(0)
    return X


def _defaults(y) -> dict:
    y = np.asarray(y, dtype=float)
    return {"global_mean": float(y.mean()), "global_std": float(y.std())}


class NaiveModel:
    """Reference: the mean duration over the previous two years of conferences."""

    name = "naive"

    def __init__(self):
        self.params = {}

    def fit(self, X, y, X_val=None, y_val=None):
        self.defaults_ = _defaults(y)
        return self

    def predict(self, X):
        return X.global_recent_mean.fillna(X.global_mean).fillna(self.defaults_["global_mean"]).to_numpy()


class BaselineModel:
    """Speaker historical mean, falling back to calling mean, then global mean.

    `min_n` is how many timed prior talks a speaker needs before their own mean
    is used. `shrink` pulls the speaker mean toward the calling mean with the
    weight of `shrink` pseudo-talks (0 means the plain speaker average).
    """

    name = "baseline"

    def __init__(self, min_n: int = 1, shrink: float = 0.0):
        self.min_n = int(min_n)
        self.shrink = float(shrink)
        self.params = {"min_n": self.min_n, "shrink": self.shrink}

    def fit(self, X, y, X_val=None, y_val=None):
        self.defaults_ = _defaults(y)
        return self

    def predict(self, X):
        F = fill_fallbacks(X, self.defaults_)
        n = X.spk_n_timed.fillna(0).to_numpy(dtype=float)
        spk = np.nan_to_num(X.spk_mean.to_numpy(dtype=float))
        cal = F.calling_mean.to_numpy(dtype=float)
        use_speaker = n >= self.min_n
        blended = (n * spk + self.shrink * cal) / np.maximum(n + self.shrink, 1e-9)
        return np.where(use_speaker, blended, cal)


class RidgeModel:
    name = "ridge"

    def __init__(self, alpha: float = 1.0, speaker_onehot: bool = False, session_size: bool = True):
        self.alpha = float(alpha)
        self.speaker_onehot = bool(speaker_onehot)
        self.session_size = bool(session_size)
        self.params = {"alpha": self.alpha, "speaker_onehot": self.speaker_onehot, "session_size": self.session_size}

    def fit(self, X, y, X_val=None, y_val=None):
        self.defaults_ = _defaults(y)
        cats = CAT_COLS + (["speaker"] if self.speaker_onehot else [])
        nums = numeric_cols(self.session_size) + ["spk_has_history"]
        pre = ColumnTransformer(
            [
                ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), nums),
                ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), cats),
            ]
        )
        self.pipe_ = Pipeline([("pre", pre), ("ridge", Ridge(alpha=self.alpha))])
        self.pipe_.fit(fill_fallbacks(X, self.defaults_), np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        return self.pipe_.predict(fill_fallbacks(X, self.defaults_))


class CatBoostModel:
    name = "catboost"

    def __init__(self, depth: int = 6, loss: str = "RMSE", learning_rate: float = 0.05,
                 iterations: int = 2000, l2_leaf_reg: float = 3.0, early_stopping: bool = True, seed: int = 0,
                 session_size: bool = True):
        self.depth, self.loss, self.learning_rate = int(depth), str(loss), float(learning_rate)
        self.iterations, self.l2_leaf_reg = int(iterations), float(l2_leaf_reg)
        self.early_stopping, self.seed = bool(early_stopping), int(seed)
        self.session_size = bool(session_size)
        self.params = {
            "depth": self.depth, "loss": self.loss, "learning_rate": self.learning_rate,
            "iterations": self.iterations, "l2_leaf_reg": self.l2_leaf_reg,
            "early_stopping": self.early_stopping, "seed": self.seed, "session_size": self.session_size,
        }

    def _pool(self, X, y=None):
        from catboost import Pool

        cols = CATBOOST_CAT_COLS + numeric_cols(self.session_size)
        Xc = X[cols].copy()
        for c in CATBOOST_CAT_COLS:
            Xc[c] = Xc[c].fillna("unknown").astype(str)
        return Pool(Xc, None if y is None else np.asarray(y, dtype=float), cat_features=CATBOOST_CAT_COLS)

    def fit(self, X, y, X_val=None, y_val=None):
        from catboost import CatBoostRegressor

        self.model_ = CatBoostRegressor(
            depth=self.depth, loss_function=self.loss, eval_metric="MAE", learning_rate=self.learning_rate,
            iterations=self.iterations, l2_leaf_reg=self.l2_leaf_reg, random_seed=self.seed,
            one_hot_max_size=CATBOOST_ONE_HOT_MAX, verbose=0, allow_writing_files=False, thread_count=-1,
        )
        if X_val is not None and self.early_stopping:
            self.model_.fit(self._pool(X, y), eval_set=self._pool(X_val, y_val),
                            early_stopping_rounds=100, use_best_model=True)
            self.best_iterations_ = int(self.model_.get_best_iteration()) + 1
        else:
            self.model_.fit(self._pool(X, y))
            self.best_iterations_ = self.iterations
        return self

    def predict(self, X):
        return self.model_.predict(self._pool(X))


MODEL_CLASSES = {m.name: m for m in (NaiveModel, BaselineModel, RidgeModel, CatBoostModel)}


def make_model(name: str, **params):
    return MODEL_CLASSES[name](**params)
