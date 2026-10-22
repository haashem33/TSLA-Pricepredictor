"""Conditional up-probability analysis (CLAUDE.md 6.9).

For a feature x and horizon h, rows are bucketed with edges fit on TRAINING rows (deciles plus 2% / 5%
tails; categorical features by value). Per bucket: P(up | bucket) with exact zeros excluded, lift vs the
base rate, Wilson interval and binomial test with n_eff = n / h (overlapping labels), P(beyond cost),
forward-return statistics. Multiple testing: Benjamini-Hochberg across all cells.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm, spearmanr

DECILES = [f"D{i}" for i in range(1, 11)]
TAILS = ["B2", "B5", "T5", "T2"]
EXTREME = ["D1", "D10", *TAILS]


def wilson(p, n, z: float = 1.96):
    p, n = np.asarray(p, float), np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        den = 1 + z**2 / n
        c = (p + z**2 / (2 * n)) / den
        hw = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - hw, c + hw


def benjamini_hochberg(p: np.ndarray, q: float = 0.10) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.zeros(len(p), bool)
    ok = np.isfinite(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return out
    o = np.argsort(pv)
    thr = q * np.arange(1, m + 1) / m
    below = pv[o] <= thr
    if below.any():
        kmax = np.max(np.flatnonzero(below))
        rej = np.zeros(m, bool)
        rej[o[: kmax + 1]] = True
        out[np.flatnonzero(ok)] = rej
    return out


def fit_edges(x_train: np.ndarray, categorical: bool) -> dict:
    x = x_train[np.isfinite(x_train)]
    if categorical:
        return {"categorical": True, "values": np.unique(x)}
    return {"categorical": False, "deciles": np.quantile(x, np.arange(1, 10) / 10),
            "tails": np.quantile(x, [0.02, 0.05, 0.95, 0.98])}


def bucket_masks(x: np.ndarray, edges: dict) -> dict[str, np.ndarray]:
    fin = np.isfinite(x)
    if edges["categorical"]:
        return {f"v={v:g}": fin & (x == v) for v in edges["values"]}
    d = np.searchsorted(edges["deciles"], x, side="right")
    out = {DECILES[i]: fin & (d == i) for i in range(10)}
    q02, q05, q95, q98 = edges["tails"]
    out.update({"B2": fin & (x <= q02), "B5": fin & (x <= q05), "T5": fin & (x >= q95), "T2": fin & (x >= q98)})
    return out


def _binom_p(k_eff: float, n_eff: float, p0: float) -> float:
    n = int(round(n_eff))
    if n < 1:
        return np.nan
    return float(binomtest(int(round(k_eff)), n, p0).pvalue)


def bucket_table(x: np.ndarray, fwd: np.ndarray, z: np.ndarray, edges: dict, h: int, cost: float) -> pd.DataFrame:
    """One row per bucket. Rows with non-finite x or fwd are ignored."""
    ok = np.isfinite(x) & np.isfinite(fwd)
    x, fwd, z = x[ok], fwd[ok], z[ok]
    nz = fwd != 0
    up = fwd > 0
    p0 = up[nz].mean()
    rows = []
    for name, m in bucket_masks(x, edges).items():
        mz = m & nz
        n = int(mz.sum())
        if n == 0:
            continue
        p = up[mz].mean()
        ne = n / h
        lo, hi = wilson(p, ne)
        rows.append({"bucket": name, "n": n, "n_zero": int((m & ~nz).sum()), "n_eff": ne, "p_up": p, "base": p0,
                     "lift": p - p0, "wilson_lo": lo, "wilson_hi": hi, "p_up_cost": float(np.mean(fwd[m] > cost)),
                     "p_dn_cost": float(np.mean(fwd[m] < -cost)), "mean_fwd": float(fwd[m].mean()),
                     "median_fwd": float(np.median(fwd[m])), "mean_z": float(np.nanmean(z[m])),
                     "pval": _binom_p(p * ne, ne, p0)})
    return pd.DataFrame(rows)


def by_year_table(x, fwd, year, edges, h) -> pd.DataFrame:
    """P(up) and lift per bucket per calendar year (year base rates), same frozen edges."""
    rows = []
    ok = np.isfinite(x) & np.isfinite(fwd) & (fwd != 0)
    for y in np.unique(year[ok]):
        sel = ok & (year == y)
        up = fwd[sel] > 0
        p0 = up.mean()
        for name, m in bucket_masks(x[sel], edges).items():
            if m.sum() == 0:
                continue
            rows.append({"year": int(y), "bucket": name, "n": int(m.sum()), "p_up": up[m].mean(), "lift": up[m].mean() - p0})
    return pd.DataFrame(rows)


def year_sign_share(tab: pd.DataFrame, by_year: pd.DataFrame, min_n_eff: float, h: int) -> pd.Series:
    """Per bucket: share of years whose lift has the full-period sign (years with n/h >= min_n_eff)."""
    full = tab.set_index("bucket")["lift"]
    b = by_year[by_year["n"] / h >= min_n_eff].copy()
    b["agree"] = np.sign(b["lift"]) == np.sign(b["bucket"].map(full))
    return b.groupby("bucket")["agree"].mean()


def shape_label(tab: pd.DataFrame, sig: pd.Series, categorical: bool) -> tuple[str, float, float]:
    """Returns (label, Spearman(bucket index, p_up) over deciles, spread D10 - D1)."""
    if categorical:
        return ("categorical-sig" if sig.any() else "none"), np.nan, np.nan
    t = tab.set_index("bucket")
    dec = [d for d in DECILES if d in t.index]
    if len(dec) < 5:
        return "none", np.nan, np.nan
    p = t.loc[dec, "p_up"].to_numpy()
    rho = float(spearmanr(np.arange(len(p)), p).statistic)
    spread = float(t.loc["D10", "p_up"] - t.loc["D1", "p_up"]) if {"D1", "D10"} <= set(t.index) else np.nan
    s = sig.reindex(t.index).fillna(False)
    if not s.any():
        return "none", rho, spread
    if abs(rho) >= 0.7:
        return "monotonic", rho, spread
    lift = t["lift"]
    mid = [d for d in ["D4", "D5", "D6", "D7"] if d in t.index]
    if {"D1", "D10"} <= set(t.index) and np.sign(lift["D1"]) == np.sign(lift["D10"]) \
            and min(abs(lift["D1"]), abs(lift["D10"])) > lift[mid].abs().max() and (s.get("D1") or s.get("D10")):
        return "U-shaped", rho, spread
    middle = [d for d in DECILES[2:8] if d in t.index]
    if not s[middle].any():
        return "extremes-only", rho, spread
    return "other", rho, spread


def auc_mi(x: np.ndarray, fwd: np.ndarray, edges: dict) -> tuple[float, float]:
    """Single-feature ROC AUC for up (zeros excluded) and mutual information (bits) between decile
    bucket (or category) and up."""
    from sklearn.metrics import roc_auc_score

    ok = np.isfinite(x) & np.isfinite(fwd) & (fwd != 0)
    x, up = x[ok], fwd[ok] > 0
    if len(np.unique(x)) < 2:
        return np.nan, np.nan
    auc = float(roc_auc_score(up, x))
    if edges["categorical"]:
        b = np.searchsorted(edges["values"], x)
    else:
        b = np.searchsorted(edges["deciles"], x, side="right")
    ct = pd.crosstab(b, up).to_numpy().astype(float)
    pj = ct / ct.sum()
    pa, pb = pj.sum(axis=1, keepdims=True), pj.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(np.where(pj > 0, pj * np.log2(pj / (pa @ pb)), 0.0))
    return auc, float(mi)


def session_weights(n_sessions: int, n_draws: int = 1000, seed: int = 0) -> np.ndarray:
    """Day-block bootstrap: (n_draws, n_sessions) resampling counts of whole sessions."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_sessions, np.full(n_sessions, 1.0 / n_sessions), size=n_draws).astype(np.float32)


def bootstrap_lift(x, fwd, session_codes, edges, W: np.ndarray, buckets=EXTREME) -> pd.DataFrame:
    """Day-block bootstrap CI (2.5%, 97.5%) of the lift of the given buckets. `session_codes` are
    0..S-1 integer session ids aligned with W's columns."""
    ok = np.isfinite(x) & np.isfinite(fwd) & (fwd != 0)
    x, up, s = x[ok], (fwd[ok] > 0).astype(float), session_codes[ok]
    S = W.shape[1]
    n_all = np.bincount(s, minlength=S)
    u_all = np.bincount(s, weights=up, minlength=S)
    base = (W @ u_all) / (W @ n_all)
    rows = []
    masks = bucket_masks(x, edges)
    for b in buckets:
        if b not in masks or masks[b].sum() == 0:
            continue
        m = masks[b]
        nb = np.bincount(s[m], minlength=S)
        ub = np.bincount(s[m], weights=up[m], minlength=S)
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = (W @ ub) / (W @ nb) - base
        lo, hi = np.nanpercentile(lift, [2.5, 97.5])
        rows.append({"bucket": b, "boot_lo": lo, "boot_hi": hi, "boot_sd": float(np.nanstd(lift))})
    return pd.DataFrame(rows)


def quintile_codes(x_train, x, categorical=False):
    if categorical:
        vals = np.unique(x_train[np.isfinite(x_train)])
        c = np.full(len(x), -1)
        for i, v in enumerate(vals):
            c[x == v] = i
        return c, [f"{v:g}" for v in vals]
    e = np.quantile(x_train[np.isfinite(x_train)], [0.2, 0.4, 0.6, 0.8])
    c = np.where(np.isfinite(x), np.searchsorted(e, x, side="right"), -1)
    return c, [f"Q{i}" for i in range(1, 6)]


def pair_grid(ca, la, cb, lb, fwd, h, min_n: int = 500) -> pd.DataFrame:
    """Grid of P(up) and lift for two bucketed features; interaction = lift_cell - (lift_a + lift_b)."""
    ok = (ca >= 0) & (cb >= 0) & np.isfinite(fwd) & (fwd != 0)
    up = fwd > 0
    p0 = up[ok].mean()
    la_ = {i: up[ok & (ca == i)].mean() - p0 for i in range(len(la))}
    lb_ = {j: up[ok & (cb == j)].mean() - p0 for j in range(len(lb))}
    rows = []
    for i, a in enumerate(la):
        for j, b in enumerate(lb):
            m = ok & (ca == i) & (cb == j)
            n = int(m.sum())
            p = up[m].mean() if n else np.nan
            suff = n >= min_n
            pv = _binom_p(p * n / h, n / h, p0) if suff else np.nan
            rows.append({"a": a, "b": b, "n": n, "sufficient": suff, "p_up": p if suff else np.nan,
                         "lift": p - p0 if suff else np.nan,
                         "interaction": (p - p0 - la_[i] - lb_[j]) if suff else np.nan, "pval": pv})
    return pd.DataFrame(rows)


def binom_z(p, p0, n_eff):
    return (p - p0) / np.sqrt(p0 * (1 - p0) / n_eff)


__all__ = ["benjamini_hochberg", "bucket_table", "by_year_table", "fit_edges", "norm", "wilson"]
