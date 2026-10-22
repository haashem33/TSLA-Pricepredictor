"""Feature IC tables (CLAUDE.md 6.8, 10.6 / D1): Spearman of feature vs realized z pooled within each
calendar month; ICIR = mean / std across months; t = ICIR * sqrt(n); binomial p of IC > 0 months."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from tslaq.backtest.metrics import fold_ic_stats


def _corr_cols(R: np.ndarray, ry: np.ndarray) -> np.ndarray:
    Rc = R - R.mean(axis=0)
    yc = ry - ry.mean()
    den = np.sqrt((Rc**2).sum(axis=0) * (yc**2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, Rc.T @ yc / den, np.nan)


def spearman_cols(X: np.ndarray, y: np.ndarray, min_obs: int = 200) -> np.ndarray:
    """Spearman correlation of each column of X with y, on pairwise-complete rows."""
    ok = np.isfinite(y)
    X, y = X[ok], y[ok]
    out = np.full(X.shape[1], np.nan)
    if len(y) < min_obs:
        return out
    fin = np.isfinite(X)
    full = fin.all(axis=0)
    if full.any():
        out[full] = _corr_cols(rankdata(X[:, full], axis=0), rankdata(y))
    for j in np.flatnonzero(~full):
        m = fin[:, j]
        if m.sum() >= min_obs:
            out[j] = _corr_cols(rankdata(X[m, j])[:, None], rankdata(y[m]))[0]
    return out


def delayed(z: np.ndarray, session: np.ndarray) -> np.ndarray:
    """Realized z from t+1 to t+1+h, aligned to t (one bar of execution delay; NaN across sessions)."""
    out = np.full(len(z), np.nan)
    same = session[1:] == session[:-1]
    out[:-1] = np.where(same, z[1:], np.nan)
    return out


def monthly_ic(F: pd.DataFrame, Z: dict, month: np.ndarray, min_obs: int = 200) -> pd.DataFrame:
    """Long table: feature, h, month, ic. `Z` maps a label key (e.g. 12 or 'd12') to realized z."""
    rows = []
    X = F.to_numpy(float)
    for m in pd.unique(month):
        idx = month == m
        for key, z in Z.items():
            ic = spearman_cols(X[idx], z[idx], min_obs)
            rows.append(pd.DataFrame({"feature": F.columns, "h": key, "month": m, "ic": ic}))
    return pd.concat(rows, ignore_index=True)


def ic_stats(ic: pd.Series) -> dict:
    """fold_ic_stats on a month-indexed IC series, plus year-sign agreement."""
    s = fold_ic_stats(ic)
    v = ic.dropna()
    yrs = v.groupby(pd.to_datetime(v.index).year).mean()
    sgn = np.sign(v.mean()) if len(v) else 0
    s["year_sign_share"] = float((np.sign(yrs) == sgn).mean()) if len(yrs) else np.nan
    s["n_years"] = len(yrs)
    return s


def signed_walk_forward(ic: pd.Series, eval_months, min_past: int = 12) -> pd.Series:
    """Out-of-sample IC of a single feature used as a forecast: in month M the sign is the sign of its
    mean IC over all months before M (needs `min_past` months)."""
    ic = ic.sort_index()
    out = {}
    for m in eval_months:
        past = ic[ic.index < m].dropna()
        if len(past) >= min_past and m in ic.index and np.isfinite(ic[m]):
            out[m] = np.sign(past.mean()) * ic[m]
    return pd.Series(out, dtype=float)


def slot_ic(F: pd.DataFrame, z: np.ndarray, slot: np.ndarray, min_obs: int = 100) -> pd.DataFrame:
    """Spearman across days at each bar_index (cross-check, 10.6). Rows: slot; columns: features."""
    X = F.to_numpy(float)
    rows = {k: spearman_cols(X[slot == k], z[slot == k], min_obs) for k in np.unique(slot)}
    return pd.DataFrame(rows, index=F.columns).T
