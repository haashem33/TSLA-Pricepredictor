"""Re-scoring of the Phase 2 walk-forward with the D17 anchored IC, without refits.

The fold checkpoints (data/processed/phase2_folds/<M>.pkl) keep each fold's fitted parameters (GARCH-t,
HMM, Kalman variances, frac-diff d), its feature versions (raw / rolling-z) and its condprob cells, but not the
feature values. `fold_features` rebuilds a fold's feature frame by applying the stored parameters with the
same causal filters as the walk-forward run (no parameter is re-estimated).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tslaq.backtest.metrics import Z_CLIP
from tslaq.data.pipeline import choose_lambda, load_bars
from tslaq.features.build import CATEGORICAL, fitted_features, stat_features
from tslaq.features.context import add_market, daily_context, event_flags
from tslaq.features.indicators import bar_features
from tslaq.features.normalize import rolling_z
from tslaq.features.regime import GaussianHMM
from tslaq.features.returns import add_returns
from tslaq.features.volatility import add_sigma, seasonal_profile
from tslaq.labels.forward_returns import add_labels
from tslaq.selection.ic import (
    anchored_group_ic,
    delayed,
    feature_scores,
    feature_scores_v2,
    score_anchors,
    score_anchors_v2,
    v2_group_ic,
)
from tslaq.utils.io import interim_path, load_config

H = (1, 3, 12)
DECAY_H = tuple(range(1, 25))
RZ_WINDOW = 1560


def base_frame(raw: pd.DataFrame, lam: float, horizons=DECAY_H, labels: bool = True) -> pd.DataFrame:
    v = load_config("data")["volatility"]
    L = add_sigma(add_returns(raw), lam, v["seasonal_window_sessions"], v["burn_in_sessions"], horizons)
    if not labels:
        return L
    L = add_labels(L, horizons)
    sess = L["session_date"].to_numpy()
    for h in H:  # one-bar-delayed outcomes: from t+1 to t+1+h, aligned to t
        if f"z_{h}" in L:
            L[f"fwdd_{h}"] = delayed(L[f"fwd_{h}"].to_numpy(), sess)
            L[f"zd_{h}"] = delayed(L[f"z_{h}"].to_numpy(), sess)
    return L


def lambdas() -> dict:
    cfg, wf = load_config("data"), load_config("backtest")["walk_forward"]
    return {s: choose_lambda(load_bars(s), wf["first_val_month"], cfg)[0] for s in ("TSLA", "SPY", "QQQ")}


def market_frames(lam: dict) -> dict:
    return {"spy": base_frame(load_bars("SPY"), lam["SPY"], (), labels=False),
            "qqq": base_frame(load_bars("QQQ"), lam["QQQ"], (), labels=False)}


def reaction_sessions():
    return pd.read_parquet(interim_path("TSLA").with_name("TSLA_earnings_sessions.parquet"))["reaction_session"]


def free_features(L: pd.DataFrame, mkt: dict, react) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parameter-free features (raw) and their rolling-z versions: build_features without the fitted part."""
    F = pd.concat([bar_features(L), stat_features(L), add_market(L, mkt), daily_context(L), event_flags(L, react)],
                  axis=1).replace([np.inf, -np.inf], np.nan)
    Frz = pd.DataFrame({f: rolling_z(F[f], RZ_WINDOW) for f in F.columns if f not in CATEGORICAL})
    return F, Frz


def fold_params(r: dict) -> dict:
    """Fitted parameters of a fold, rebuilt from its checkpoint (no refit)."""
    h = r["hmm"]
    hmm = GaussianHMM(np.array(h["startprob"]), np.array(h["transmat"]), np.array(h["means"]), np.array(h["covs"]),
                      h["loglik"], h["n_iter"])
    k = r["kalman"]
    return {"cutoff": f"{r['month']}-01", "garch": r["garch"], "hmm": hmm,
            "kalman": {"params": [k["s2_irregular"], k["s2_level"], k["s2_trend"]]}, "frac_d": r["frac_d"]}


def profile(L: pd.DataFrame):
    prof_df = seasonal_profile(L, load_config("data")["volatility"]["seasonal_window_sessions"])
    return prof_df.to_numpy(), prof_df.index.get_indexer(L["session_date"])


def fold_features(L, F_free, Frz_free, P: dict, prof, pos_s, versions: dict) -> pd.DataFrame:
    """The fold's feature frame X (one column per kept feature, in the fold's version)."""
    FF = fitted_features(L, P, prof, pos_s)
    cols = {}
    for f, v in versions.items():
        if f in FF:
            cols[f] = rolling_z(FF[f], RZ_WINDOW) if v == "rz" else FF[f]
        else:
            cols[f] = Frz_free[f] if v == "rz" else F_free[f]
    return pd.DataFrame(cols).replace([np.inf, -np.inf], np.nan)


def anchored_fold(X: pd.DataFrame, Z: dict, month: np.ndarray, M: str, min_obs: int = 200):
    """D17 anchored IC of every feature of fold M, for every month <= M (training months and the test month),
    with fold M's anchors (median / IQR of each feature and m of each target over the training rows, i.e.
    months < M). Returns (long IC table with columns feature, h, month, ic; centre; scale; m by key)."""
    upto = month <= M
    tr = month < M
    Xa = X.to_numpy(float)[upto]
    centre, scale = score_anchors(X.to_numpy(float)[tr])
    S = feature_scores(Xa, centre, scale)
    mon = month[upto]
    rows, ms = [], {}
    for key, z in Z.items():
        zc = np.clip(z, -Z_CLIP, Z_CLIP)
        m = float(np.nanmean(zc[tr]))
        ms[key] = m
        G = anchored_group_ic(S, zc[upto] - m, mon, min_obs)
        G.columns = X.columns
        g = G.stack(future_stack=True).rename("ic").reset_index()
        g.columns = ["month", "feature", "ic"]
        g["h"] = str(key)
        rows.append(g)
    return pd.concat(rows, ignore_index=True), pd.Series(centre, X.columns), pd.Series(scale, X.columns), ms


def anchored_fold_v2(X: pd.DataFrame, Z: dict, month: np.ndarray, M: str, min_obs: int = 200):
    """D19 IC of every feature of fold M, for every month <= M, with fold M's anchors (median / IQR, training
    mean of the clipped score, V_s, and V_y per target, all from months < M). Returns (long IC table with
    columns feature, h, month, ic; anchors dict; V_y by key)."""
    upto = month <= M
    tr = month < M
    Xv = X.to_numpy(float)
    A = score_anchors_v2(Xv[tr])
    S = feature_scores_v2(Xv[upto], A)
    mon = month[upto]
    rows, vys = [], {}
    for key, z in Z.items():
        y = np.clip(z, -Z_CLIP, Z_CLIP)
        vy = float(np.nanmean(y[tr] ** 2))
        vys[key] = vy
        G = v2_group_ic(S, y[upto], mon, A["vs"], vy, min_obs)
        G.columns = X.columns
        g = G.stack(future_stack=True).rename("ic").reset_index()
        g.columns = ["month", "feature", "ic"]
        g["h"] = str(key)
        rows.append(g)
    return pd.concat(rows, ignore_index=True), A, vys
