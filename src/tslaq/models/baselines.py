"""Phase 1 baselines (CLAUDE.md 10.3): zero / historical-mean / LightGBM on simple returns for the
return forecast; seasonal, EWMA x seasonal and GARCH(1,1)-t x seasonal for volatility."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from tslaq.features.stats_models import fit_garch_t, garch_filter, garch_multi_step

LGB_PARAMS = {
    "objective": "regression", "learning_rate": 0.03, "n_estimators": 300, "num_leaves": 15,
    "min_child_samples": 1000, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
    "reg_lambda": 1.0, "random_state": 0, "deterministic": True, "force_row_wise": True,
    "num_threads": 4, "verbose": -1,
}
Z_CLIP = 6.0


def simple_return_features(df: pd.DataFrame, n_lags: int = 12) -> pd.DataFrame:
    """Past vol-scaled returns only (causal: positive shifts and trailing sums)."""
    zr = (df["r"] / df["sigma_hat"]).clip(-10, 10)
    out = {f"zr_lag{j}": zr.shift(j) for j in range(n_lags)}
    for k in (24, 78):
        out[f"zcum_{k}"] = df["r"].rolling(k).sum() / (df["sigma_hat"] * np.sqrt(k))
    out["bar_index"] = df["bar_index"].astype(float)
    return pd.DataFrame(out, index=df.index)


def fit_predict_lgb(X_tr, y_tr, X_va, params=None) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (val predictions, train predictions, train residual std). Target clipped to +-Z_CLIP."""
    m = lgb.LGBMRegressor(**(params or LGB_PARAMS))
    m.fit(X_tr, np.clip(y_tr, -Z_CLIP, Z_CLIP))
    p_tr = m.predict(X_tr)
    return m.predict(X_va), p_tr, float(np.std(np.clip(y_tr, -Z_CLIP, Z_CLIP) - p_tr))


def lgb_n_params(params=None) -> int:
    """Upper bound on the number of fitted leaf values (trees x leaves)."""
    p = params or LGB_PARAMS
    return int(p["n_estimators"] * p["num_leaves"])


def seasonal_sum_sq(df: pd.DataFrame, h: int) -> np.ndarray:
    """sum_{i=1..h} seasonal^2_{k(t)+i} recovered from S_h / sqrt(level2_next)."""
    return (df[f"S_{h}"] ** 2 / df["level2_next"]).to_numpy()


def garch_var_forecast(df: pd.DataFrame, train_pos: np.ndarray, horizons) -> tuple[dict, dict]:
    """Fit GARCH-t on training u, filter the full series with fixed params, and return the h-bar
    variance forecast sum_i seasonal^2_{k+i} * E[sigma_u^2_{t+i}]. Requires seasonal at future slots,
    which is today's profile (computed from previous sessions)."""
    u = df["u"].to_numpy()
    params = fit_garch_t(u[train_pos])
    _, nxt = garch_filter(u, params)
    prof = df.pivot_table(index="session_date", columns="bar_index", values="seasonal", aggfunc="first")
    P2 = prof.to_numpy() ** 2
    ps = prof.index.get_indexer(df["session_date"])
    k = df["bar_index"].to_numpy()
    out = {}
    for h in horizons:
        ms = garch_multi_step(nxt, params, h)  # (n, h): E[sigma_u^2_{t+i}], i = 1..h
        f = np.zeros(len(df))
        for i in range(1, h + 1):
            kk = np.minimum(k + i, P2.shape[1] - 1)
            f += P2[ps, kk] * ms[:, i - 1]
        f[~np.isfinite(df[f"S_{h}"].to_numpy())] = np.nan
        out[h] = f
    return out, params
