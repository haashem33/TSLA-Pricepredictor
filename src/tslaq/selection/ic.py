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


SCORE_CLIP = 5.0


def score_anchors(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """D17 single-feature anchors from a fold's training rows, per column: binary flags (<= 2 distinct
    values) -> centre = training mean, scale = 1; otherwise centre = training median, scale = IQR (training
    sd if the IQR is 0). A constant column gets scale NaN (no score)."""
    X_train = np.asarray(X_train, float)
    k = X_train.shape[1]
    centre, scale = np.full(k, np.nan), np.full(k, np.nan)
    for j in range(k):
        x = X_train[:, j]
        x = x[np.isfinite(x)]
        if len(x) < 2:
            continue
        mn, mx = x.min(), x.max()
        if ((x == mn) | (x == mx)).all():  # binary flag (or constant)
            centre[j], scale[j] = x.mean(), (1.0 if mx > mn else np.nan)
            continue
        lo, med, hi = np.quantile(x, [0.25, 0.5, 0.75])
        centre[j] = med
        sd = x.std()
        scale[j] = hi - lo if hi > lo else (sd if sd > 0 else np.nan)
    return centre, scale


def feature_scores(X: np.ndarray, centre: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Unsigned D17 score: clip((x - centre) / scale, -5, 5). Multiply by sign_train for a forecast."""
    with np.errstate(invalid="ignore"):
        return np.clip((np.asarray(X, float) - centre) / scale, -SCORE_CLIP, SCORE_CLIP)


def anchored_group_ic(S: np.ndarray, y: np.ndarray, group: np.ndarray, min_obs: int = 200) -> pd.DataFrame:
    """D17 anchored IC of each column of S against y within each group (e.g. calendar month):
    sum(s*y) / sqrt(sum(s^2) * sum(y^2)) on pairwise-finite rows. Rows must be ordered so that each group is
    contiguous. Returns a (group x column) frame."""
    g = np.asarray(group)
    starts = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]
    labels = g[starts]
    if len(set(labels.tolist())) != len(labels):
        raise ValueError("groups must be contiguous")
    S = np.asarray(S, float)
    M = np.isfinite(S) & np.isfinite(y)[:, None]
    S0 = np.where(M, S, 0.0)
    Y0 = np.where(M, np.asarray(y, float)[:, None], 0.0)
    sy = np.add.reduceat(S0 * Y0, starts, axis=0)
    ss = np.add.reduceat(S0 * S0, starts, axis=0)
    yy = np.add.reduceat(Y0 * Y0, starts, axis=0)
    n = np.add.reduceat(M.astype(np.int64), starts, axis=0)
    den = np.sqrt(ss * yy)
    with np.errstate(divide="ignore", invalid="ignore"):
        ic = np.where((den > 0) & (n >= min_obs), sy / den, np.nan)
    return pd.DataFrame(ic, index=labels)


def slot_ic(F: pd.DataFrame, z: np.ndarray, slot: np.ndarray, min_obs: int = 100) -> pd.DataFrame:
    """Spearman across days at each bar_index (cross-check, 10.6). Rows: slot; columns: features."""
    X = F.to_numpy(float)
    rows = {k: spearman_cols(X[slot == k], z[slot == k], min_obs) for k in np.unique(slot)}
    return pd.DataFrame(rows, index=F.columns).T


def score_anchors_v2(X_train: np.ndarray) -> dict:
    """D19 single-feature anchors from a fold's training rows, per column. c(x) = clip((x - median) / IQR,
    +-5) (training sd if the IQR is 0); binary flags (<= 2 distinct values) as 0/1 (1 = the larger value).
    mu = training mean of c(x), vs = training mean of (c(x) - mu)^2. A constant column gets scale NaN."""
    X_train = np.asarray(X_train, float)
    k = X_train.shape[1]
    A = {n: np.full(k, np.nan) for n in ("centre", "scale", "hi", "mu", "vs")}
    A["binary"] = np.zeros(k, bool)
    for j in range(k):
        x = X_train[:, j]
        x = x[np.isfinite(x)]
        if len(x) < 2:
            continue
        mn, mx = x.min(), x.max()
        if ((x == mn) | (x == mx)).all():
            A["binary"][j], A["hi"][j], A["scale"][j] = True, mx, (1.0 if mx > mn else np.nan)
            continue
        lo, med, hi = np.quantile(x, [0.25, 0.5, 0.75])
        sd = x.std()
        A["centre"][j], A["scale"][j] = med, (hi - lo if hi > lo else (sd if sd > 0 else np.nan))
    C = _clipped(X_train, A)
    with np.errstate(invalid="ignore"):
        A["mu"] = np.nanmean(C, axis=0) if len(C) else A["mu"]
        A["vs"] = np.nanmean((C - A["mu"]) ** 2, axis=0) if len(C) else A["vs"]
    return A


def _clipped(X: np.ndarray, A: dict) -> np.ndarray:
    X = np.asarray(X, float)
    with np.errstate(invalid="ignore"):
        cont = np.clip((X - A["centre"]) / A["scale"], -SCORE_CLIP, SCORE_CLIP)
        flag = np.where(np.isfinite(X), (X == A["hi"]).astype(float), np.nan)
        out = np.where(A["binary"], flag, cont)
    return np.where(np.isfinite(A["scale"]), out, np.nan)


def feature_scores_v2(X: np.ndarray, A: dict) -> np.ndarray:
    """Unsigned D19 score c(x) - mean_train(c(x)). Multiply by sign_train for a forecast."""
    return _clipped(X, A) - A["mu"]


def v2_group_ic(S: np.ndarray, y: np.ndarray, group: np.ndarray, vs: np.ndarray, vy: float,
                min_obs: int = 200) -> pd.DataFrame:
    """D19 IC of each column of S against y within each (contiguous) group: sum(s*y) / (n * sqrt(vs * vy)),
    n = rows with finite s and y; vs (per column) and vy are training moments, never the group's own."""
    g = np.asarray(group)
    starts = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]
    labels = g[starts]
    if len(set(labels.tolist())) != len(labels):
        raise ValueError("groups must be contiguous")
    S = np.asarray(S, float)
    M = np.isfinite(S) & np.isfinite(y)[:, None]
    sy = np.add.reduceat(np.where(M, S * np.asarray(y, float)[:, None], 0.0), starts, axis=0)
    n = np.add.reduceat(M.astype(np.int64), starts, axis=0)
    den = n * np.sqrt(np.asarray(vs, float) * vy)
    with np.errstate(divide="ignore", invalid="ignore"):
        ic = np.where((den > 0) & (n >= min_obs), sy / den, np.nan)
    return pd.DataFrame(ic, index=labels)
