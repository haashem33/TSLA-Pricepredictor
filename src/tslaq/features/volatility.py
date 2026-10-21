"""Intraday volatility scale (CLAUDE.md 5.2) and horizon scale S_{t,h} (CLAUDE.md 9.1).

    seasonal_k  = median over the previous `window` sessions of |r| at bar_index k (current session excluded)
    u_t         = r_t / seasonal_{k(t)}
    level2_t    = lam * level2_{t-1} + (1 - lam) * u_{t-1}^2         (uses information up to t-1)
    sigma_t     = sqrt(level2_t) * seasonal_{k(t)}                   (sigma_{t|t-1})
    S_{t,h}     = sqrt(level2_{t+1}) * sqrt(sum_{i=1..h} seasonal^2_{k(t)+i})

level2_{t+1} = lam * level2_t + (1 - lam) * u_t^2 is known at the close of bar t, so S_{t,h} uses
information up to and including bar t only. Bar 0 (overnight gap) has its own profile slot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEASONAL_FLOOR = 1e-6
FULL_DAY_BARS = 78   # 09:30-16:00 in 5-minute bars
HALF_DAY_BARS = 42   # NYSE early close 13:00


def seasonal_profile(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Sessions x bar_index matrix of trailing medians of |r|, using past sessions only.

    Missing (forward-filled) bars are excluded from the medians.
    """
    a = df["r"].abs().where(~df["missing"]) if "missing" in df else df["r"].abs()
    mat = pd.DataFrame({"s": df["session_date"].to_numpy(), "k": df["bar_index"].to_numpy(),
                        "a": a.to_numpy()}).pivot(index="s", columns="k", values="a")
    # Per slot: median of the last `window` OBSERVED values strictly before each session. Slots that do
    # not exist on half-days are skipped rather than breaking the window.
    cols = {}
    for k in mat.columns:
        obs = mat[k].dropna()
        med = obs.rolling(window, min_periods=window).median()
        cols[k] = med.reindex(mat.index).ffill().shift(1)
    prof = pd.DataFrame(cols, index=mat.index)
    return prof.clip(lower=SEASONAL_FLOOR)


def ewma_level(u: np.ndarray, lam: float, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (level2_t, level2_{t+1}) arrays. level2_t uses u up to t-1 only.

    Bars where `valid` is False (warm-up, missing bars) do not update the level. Initialised at 1.0 on
    the first valid bar (E[u^2] is O(1) by construction); callers discard a burn-in period.
    """
    n = len(u)
    lvl = np.full(n, np.nan)
    nxt = np.full(n, np.nan)
    cur = np.nan
    for t in range(n):
        if np.isnan(cur) and valid[t]:
            cur = 1.0
        lvl[t] = cur
        if valid[t] and not np.isnan(cur):
            cur = lam * cur + (1.0 - lam) * u[t] * u[t]
        nxt[t] = cur
    return lvl, nxt


def add_sigma(df: pd.DataFrame, lam: float = 0.97, window: int = 60, burn_in_sessions: int = 5,
              horizons: tuple[int, ...] = (1, 3, 12)) -> pd.DataFrame:
    """Add seasonal, u, level2, sigma_hat and S_h columns. Requires column `r` (see add_returns)."""
    df = df.copy()
    prof = seasonal_profile(df, window)
    pos_s = prof.index.get_indexer(df["session_date"])
    pos_k = prof.columns.get_indexer(df["bar_index"])
    P = prof.to_numpy()
    df["seasonal"] = P[pos_s, pos_k]
    u = (df["r"] / df["seasonal"]).to_numpy()
    df["u"] = u
    valid = np.isfinite(u)
    if "missing" in df:
        valid &= ~df["missing"].to_numpy()
    lvl, nxt = ewma_level(u, lam, valid)
    first_valid = np.argmax(valid) if valid.any() else len(df)
    sess = df["session_date"].to_numpy()
    uniq = pd.unique(sess[first_valid:])
    # bars in the first `burn_in_sessions` sessions after the first valid u are warm-up
    warm = np.ones(len(df), bool) if len(uniq) <= burn_in_sessions else sess < uniq[burn_in_sessions]
    lvl[warm] = np.nan
    nxt[warm] = np.nan
    df["level2"] = lvl
    df["level2_next"] = nxt
    df["sigma_hat"] = np.sqrt(lvl) * df["seasonal"].to_numpy()
    # S_{t,h}: cumulative seasonal^2 over the next h slots of the same session
    n_k = P.shape[1]
    zero = np.zeros((P.shape[0], 1))
    cum = np.concatenate([zero, np.cumsum(np.nan_to_num(P**2, nan=0.0), axis=1)], axis=1)
    cum_nan = np.concatenate([zero, np.cumsum(np.isnan(P), axis=1)], axis=1)
    k = df["bar_index"].to_numpy()
    # session length from the calendar (not from the data, which may be truncated at t)
    n_bars = np.where(df["is_half_day"].to_numpy(), HALF_DAY_BARS, FULL_DAY_BARS)
    for h in horizons:
        end = k + 1 + h  # slots k+1 .. k+h -> cum[end] - cum[k+1]
        ok = (k + h < n_bars) & (end <= n_k)
        endc = np.minimum(end, n_k)
        s2 = cum[pos_s, endc] - cum[pos_s, np.minimum(k + 1, n_k)]
        n_nan = cum_nan[pos_s, endc] - cum_nan[pos_s, np.minimum(k + 1, n_k)]
        S = np.sqrt(nxt * s2)
        S[~ok | (n_nan > 0)] = np.nan
        df[f"S_{h}"] = S
    return df


def qlike_loss(r: np.ndarray, var_forecast: np.ndarray) -> float:
    """Mean of r^2 / v + ln v (QLIKE up to a constant; robust to r = 0)."""
    m = np.isfinite(r) & np.isfinite(var_forecast) & (var_forecast > 0)
    return float(np.mean(r[m] ** 2 / var_forecast[m] + np.log(var_forecast[m])))


def tune_lambda(df: pd.DataFrame, grid, window: int = 60, burn_in_sessions: int = 5) -> pd.DataFrame:
    """QLIKE of sigma_hat^2 vs r^2 for each lambda; call on TRAINING data only."""
    rows = []
    for lam in grid:
        d = add_sigma(df, lam, window, burn_in_sessions, horizons=())
        rows.append({"lambda": lam, "qlike": qlike_loss(d["r"].to_numpy(), d["sigma_hat"].to_numpy() ** 2)})
    return pd.DataFrame(rows)
