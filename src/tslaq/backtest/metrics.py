"""Forecast metrics (CLAUDE.md 10.6)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr


def daily_rank_ic(pred: np.ndarray, realized: np.ndarray, day: np.ndarray, min_obs: int = 10) -> pd.Series:
    """Spearman IC per session. Days where the prediction is constant give NaN."""
    out = {}
    df = pd.DataFrame({"p": pred, "y": realized, "d": day}).dropna()
    for d, g in df.groupby("d"):
        if len(g) >= min_obs and g["p"].nunique() > 1 and g["y"].nunique() > 1:
            out[d] = spearmanr(g["p"], g["y"]).statistic
    return pd.Series(out, dtype=float)


def ic_summary(pred, realized, day) -> dict:
    ic = daily_rank_ic(pred, realized, day)
    m = np.isfinite(pred) & np.isfinite(realized)
    pooled = spearmanr(pred[m], realized[m]).statistic if np.unique(pred[m]).size > 1 else np.nan
    return {"rank_ic": ic.mean() if len(ic) else np.nan,
            "icir": ic.mean() / ic.std() if len(ic) > 1 and ic.std() > 0 else np.nan,
            "pooled_ic": pooled, "n_days": len(ic)}


def per_slot_ic(pred, realized, slot, min_obs: int = 20) -> float:
    """Mean over time-of-day slots of the Spearman IC computed ACROSS days within each slot.

    For a single stock the per-day IC ranks bars over time within a day; with overlapping forward
    windows that is mechanically biased (a random walk gives strongly negative IC for any predictor
    correlated with the day's past path). Ranking across days at a fixed slot avoids this.
    """
    df = pd.DataFrame({"p": pred, "y": realized, "k": slot}).dropna()
    ics = [spearmanr(g["p"], g["y"]).statistic for _, g in df.groupby("k")
           if len(g) >= min_obs and g["p"].nunique() > 1]
    return float(np.mean(ics)) if ics else np.nan


def fold_ic(pred, realized, fold) -> pd.Series:
    """Pooled Spearman IC within each fold (one validation month). Biased for predictors with multi-day
    persistence (D17); reported for information only. The primary metric is `fold_anchored_ic`."""
    df = pd.DataFrame({"p": pred, "y": realized, "f": fold}).dropna()
    return pd.Series({f: spearmanr(g["p"], g["y"]).statistic for f, g in df.groupby("f")
                      if g["p"].nunique() > 1}, dtype=float)


def fold_ic_stats(fic: pd.Series) -> dict:
    """Primary IC summary (CLAUDE.md 10.6): mean fold IC, ICIR = mean/std across folds,
    t = ICIR * sqrt(n_folds), share of folds with IC > 0 and its one-sided binomial p-value."""
    from scipy.stats import binomtest

    v = fic.dropna()
    n = len(v)
    if n < 2:
        return {"ic": v.mean() if n else np.nan, "icir": np.nan, "t": np.nan, "pos_share": np.nan,
                "binom_p": np.nan, "n_folds": n}
    k = int((v > 0).sum())
    icir = v.mean() / v.std()
    return {"ic": v.mean(), "icir": icir, "t": icir * np.sqrt(n), "pos_share": k / n,
            "binom_p": binomtest(k, n, 0.5, alternative="greater").pvalue, "n_folds": n}


Z_CLIP = 5.0


def anchored_target(z: np.ndarray, z_train: np.ndarray) -> np.ndarray:
    """D17 (superseded by D19) target: y = clip(z, -5, 5) - m, with m = mean of clipped z over the fold's training rows."""
    m = np.nanmean(np.clip(z_train, -Z_CLIP, Z_CLIP))
    return np.clip(z, -Z_CLIP, Z_CLIP) - m


def anchored_ic(s: np.ndarray, y: np.ndarray, min_obs: int = 1) -> float:
    """D17 (superseded by D19) anchored IC = sum(s*y) / sqrt(sum(s^2) * sum(y^2)) over rows where both are finite. Nothing is
    centred or ranked with the scored rows' own statistics (s and y carry training-period anchors)."""
    ok = np.isfinite(s) & np.isfinite(y)
    if ok.sum() < min_obs:
        return np.nan
    s, y = s[ok], y[ok]
    den = np.sqrt((s * s).sum() * (y * y).sum())
    return float((s * y).sum() / den) if den > 0 else np.nan


def fold_anchored_ic(pred, z, fold, m_by_fold: dict, min_obs: int = 1) -> pd.Series:
    """Anchored IC per fold. `pred` = the model's predicted z as is; `m_by_fold[f]` = mean clipped z over
    fold f's training rows."""
    pred, z, fold = np.asarray(pred, float), np.asarray(z, float), np.asarray(fold)
    out = {}
    for f in pd.unique(fold):
        i = fold == f
        out[f] = anchored_ic(pred[i], np.clip(z[i], -Z_CLIP, Z_CLIP) - m_by_fold[f], min_obs)
    return pd.Series(out, dtype=float).sort_index()


def ic_v2(s: np.ndarray, y: np.ndarray, v_s: float, v_y: float, min_obs: int = 1) -> float:
    """D19 IC = sum(s*y) / (n * sqrt(v_s * v_y)) over rows where s and y are finite; v_s and v_y are
    moments from data before the scored rows (training rows, or earlier out-of-sample months)."""
    ok = np.isfinite(s) & np.isfinite(y)
    n = int(ok.sum())
    den = n * np.sqrt(v_s * v_y)
    return float((s[ok] * y[ok]).sum() / den) if n >= min_obs and den > 0 else np.nan


def fold_ic_v2_model(pred, z, fold, zbar_by_fold: dict, vy_by_fold: dict, vs_first: float,
                     min_obs: int = 1) -> pd.Series:
    """D19 IC of a model per fold (folds in time order). s = pred - zbar_f (fold f's training mean of clipped
    z), y = clip(z, +-5); v_y = fold f's training mean of y^2; v_s = mean of s^2 over all earlier
    out-of-sample folds (the first fold uses `vs_first`, from the D6 inner-validation block)."""
    pred, z, fold = np.asarray(pred, float), np.asarray(z, float), np.asarray(fold)
    y = np.clip(z, -Z_CLIP, Z_CLIP)
    out, ss, nn = {}, 0.0, 0
    for f in sorted(pd.unique(fold)):
        i = fold == f
        s = pred[i] - zbar_by_fold[f]
        v_s = vs_first if nn == 0 else ss / nn
        out[f] = ic_v2(s, y[i], v_s, vy_by_fold[f], min_obs)
        ok = np.isfinite(s) & np.isfinite(y[i])
        ss, nn = ss + float((s[ok] ** 2).sum()), nn + int(ok.sum())
    return pd.Series(out, dtype=float)


def hit_rate(pred: np.ndarray, realized: np.ndarray) -> float:
    """Share of non-zero predictions whose sign matches a non-zero realized return."""
    m = np.isfinite(pred) & np.isfinite(realized) & (pred != 0) & (realized != 0)
    return float(np.mean(np.sign(pred[m]) == np.sign(realized[m]))) if m.any() else np.nan


def crps_gaussian(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def brier_up(p_up: np.ndarray, realized: np.ndarray) -> float:
    m = np.isfinite(p_up) & np.isfinite(realized) & (realized != 0)
    return float(np.mean((p_up[m] - (realized[m] > 0)) ** 2))


def qlike(rv: np.ndarray, f: np.ndarray) -> float:
    """QLIKE(f, RV) = RV/f - ln(RV/f) - 1; rows with RV <= 0 are excluded."""
    m = np.isfinite(rv) & np.isfinite(f) & (rv > 0) & (f > 0)
    x = rv[m] / f[m]
    return float(np.mean(x - np.log(x) - 1))
