"""Phase 2 feature frame.

Fitted feature objects (GARCH-t, HMM, Kalman noise variances, fractional-differencing d) are fit at
yearly cutoffs on data strictly before the cutoff, then applied with FIXED parameters (causal filters)
to the rows of that year. Rows before the first cutoff use the first cutoff's fit, which is training data
for every walk-forward fold. Everything else has no fitted parameters.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from tslaq.features.context import add_market, daily_context, event_flags
from tslaq.features.indicators import bar_features
from tslaq.features.regime import filtered_probs, fit_hmm, hmm_obs
from tslaq.features.stats_models import (
    choose_frac_d,
    fit_garch_t,
    fit_kalman_llt,
    fracdiff,
    garch_features,
    hurst_rs,
    kalman_llt_filter,
    rolling_ac1,
    rolling_entropy,
    variance_ratio,
)
from tslaq.features.volatility import seasonal_profile

# features that are categorical / discrete (bucketed by value, not by quantile; no rank-IC transforms)
CATEGORICAL = ["sign_first30", "sign_12th30", "hmm_state", "earn_react", "earn_after", "opex", "half_day", "dow"]
FITTED = ["garch_sigma", "garch_resid", "garch_logvar12", "hmm_p0", "hmm_p1", "hmm_p2", "hmm_state",
          "kalman_slope", "kalman_innov", "fracdiff_lnc"]


def stat_features(df: pd.DataFrame) -> pd.DataFrame:
    u = df["u"].where(~df["missing"]).to_numpy()
    return pd.DataFrame({
        "hurst_256": hurst_rs(u, 256),
        "vr12_256": variance_ratio(u, 12, 256),
        "ac1_78": rolling_ac1(df["r"].to_numpy(), 78),
        "entropy_78": rolling_entropy((df["r"] / df["sigma_hat"]).to_numpy(), 78),
    }, index=df.index)


def fit_params(df: pd.DataFrame, cutoff: str, seed: int = 0) -> dict:
    """Fit every fitted feature object on rows with session_date < cutoff."""
    t0 = time.time()
    tr = (df["session_date"] < pd.Timestamp(cutoff)).to_numpy()
    u = df["u"].to_numpy()
    garch = fit_garch_t(u[tr])
    obs = hmm_obs(u, df["sigma_hat"].to_numpy())
    obs[df["missing"].to_numpy()] = np.nan
    hmm, runs = fit_hmm(obs[tr], K=3, n_starts=5, seed=seed)
    lc = np.log(df["close"].to_numpy())
    kal = fit_kalman_llt(lc[tr])
    d, tried = choose_frac_d(lc[tr])
    return {"cutoff": cutoff, "garch": garch, "hmm": hmm, "hmm_runs": runs, "kalman": kal, "frac_d": d,
            "frac_tried": tried, "fit_seconds": round(time.time() - t0, 1)}


def fitted_features(df: pd.DataFrame, P: dict, prof: np.ndarray, pos_s: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for k, v in garch_features(df, P["garch"], prof, pos_s, 12).items():
        out[k] = v
    obs = hmm_obs(df["u"].to_numpy(), df["sigma_hat"].to_numpy())
    obs[df["missing"].to_numpy()] = np.nan
    p = filtered_probs(obs, P["hmm"])
    for j in range(p.shape[1]):
        out[f"hmm_p{j}"] = p[:, j]
    out["hmm_state"] = np.where(np.isfinite(p).all(axis=1), np.nanargmax(np.nan_to_num(p, nan=-1), axis=1), np.nan)
    lc = np.log(df["close"].to_numpy())
    slope, innov = kalman_llt_filter(lc, P["kalman"])
    out["kalman_slope"] = slope / df["sigma_hat"].to_numpy()
    out["kalman_innov"] = innov
    out["fracdiff_lnc"] = fracdiff(lc, P["frac_d"])
    return out


def build_features(df: pd.DataFrame, mkt: dict[str, pd.DataFrame], params: list[dict],
                   reaction_sessions, window: int = 60) -> pd.DataFrame:
    """All Phase 2 features for the rows of `df` (TSLA with returns and sigma). `params` is a list of
    fit_params outputs sorted by cutoff."""
    parts = [bar_features(df), stat_features(df), add_market(df, mkt), daily_context(df),
             event_flags(df, reaction_sessions)]
    prof_df = seasonal_profile(df, window)
    prof = prof_df.to_numpy()
    pos_s = prof_df.index.get_indexer(df["session_date"])
    sd = df["session_date"].to_numpy()
    fit = None
    for i, P in enumerate(params):
        lo = np.datetime64(pd.Timestamp(P["cutoff"])) if i > 0 else np.datetime64("1900-01-01")
        hi = np.datetime64(pd.Timestamp(params[i + 1]["cutoff"])) if i + 1 < len(params) else np.datetime64("2262-01-01")
        rows = (sd >= lo) & (sd < hi)
        if not rows.any():
            continue
        ff = fitted_features(df, P, prof, pos_s)
        if fit is None:
            fit = pd.DataFrame(np.nan, index=df.index, columns=ff.columns)
        fit.loc[rows, ff.columns] = ff.loc[rows].to_numpy()
    parts.append(fit)
    out = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)
    return out
