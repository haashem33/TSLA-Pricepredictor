"""Statistical-model features. Phase 1: GARCH(1,1)-t on de-seasonalized returns (CLAUDE.md 6.4, 5.6).

Fit on training data with `arch`, then run the variance recursion forward manually with the trained
parameters held fixed, so sigma_t uses only u_{<t} and the trained parameters.
"""
from __future__ import annotations

import warnings

import numpy as np
from arch import arch_model
from numba import njit


def fit_garch_t(u_train: np.ndarray) -> dict:
    u = u_train[np.isfinite(u_train)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(u, mean="Zero", vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off")
    p = res.params
    return {"omega": float(p["omega"]), "alpha": float(p["alpha[1]"]), "beta": float(p["beta[1]"]),
            "nu": float(p["nu"]), "train_var": float(np.mean(u**2))}


def garch_filter(u: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Returns (var_t, var_next): var_t = sigma^2_{t|t-1} from u_{<t}; var_next = sigma^2_{t+1|t}.

    Non-finite u (warm-up, missing bars) do not update the state. The state is initialised at the
    training-sample variance, which does not depend on data after t.
    """
    w, a, b = params["omega"], params["alpha"], params["beta"]
    n = len(u)
    var = np.full(n, np.nan)
    nxt = np.full(n, np.nan)
    cur = np.nan
    for t in range(n):
        ok = np.isfinite(u[t])
        if np.isnan(cur) and ok:
            cur = params["train_var"]
        var[t] = cur
        if ok and not np.isnan(cur):
            cur = w + a * u[t] * u[t] + b * cur
        nxt[t] = cur
    return var, nxt


def garch_multi_step(var_next: np.ndarray, params: dict, h: int) -> np.ndarray:
    """E[sigma^2_{t+i} | F_t] for i = 1..h, shape (n, h)."""
    a, b, w = params["alpha"], params["beta"], params["omega"]
    pers = a + b
    lr = w / (1 - pers) if pers < 1 else params["train_var"]
    i = np.arange(h)
    return lr + (pers ** i)[None, :] * (var_next[:, None] - lr)


def garch_features(df, params: dict, prof: np.ndarray, pos_s: np.ndarray, h: int = 12) -> dict:
    """GARCH-t features with fixed trained parameters (6.4): sigma_{t+1|t} of u, standardized residual
    u_t / sigma_{t|t-1}, and log of the h-bar variance forecast sum_i seasonal^2_{k+i} E[sigma_u^2_{t+i}].
    `prof` is the sessions x slot seasonal profile from past sessions (known at the open)."""
    u = df["u"].to_numpy()
    var, nxt = garch_filter(u, params)
    k = df["bar_index"].to_numpy()
    ms = garch_multi_step(nxt, params, h)
    f = np.zeros(len(u))
    n_k = prof.shape[1]
    for i in range(1, h + 1):
        kk = k + i
        f += np.where(kk < n_k, prof[pos_s, np.minimum(kk, n_k - 1)], np.nan) ** 2 * ms[:, i - 1]
    f[~np.isfinite(df[f"S_{h}"].to_numpy())] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"garch_sigma": np.sqrt(nxt), "garch_resid": u / np.sqrt(var), f"garch_logvar{h}": np.log(f)}


# ---------------------------------------------------------------- Kalman local linear trend (6.6)
# y_t = mu_t + eps_t,  mu_{t+1} = mu_t + beta_t + eta_t,  beta_{t+1} = beta_t + zeta_t
# variances (irregular, level, trend) - the statsmodels UnobservedComponents('lltrend') model. Numba filter
# with the exact diffuse start: after the first two observations the filtered state is (y_1, y_1 - y_0) with
# P = [[s_irr, s_irr], [s_irr, 2 s_irr + s_lvl + s_trd]]; those two innovations are excluded from the
# likelihood and set to NaN. Only filtered (causal) quantities are returned.

N_DIFFUSE = 2


@njit(cache=True)
def _llt_filter_nb(y, s_irr, s_lvl, s_trd):
    n = len(y)
    level = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    innov = np.full(n, np.nan)
    a0, a1 = 0.0, 0.0
    p00, p01, p11 = 0.0, 0.0, 0.0
    y_first = np.nan
    k_obs = 0
    ll = 0.0
    for t in range(n):
        if not np.isfinite(y[t]):
            continue
        if k_obs == 0:
            y_first = y[t]
            level[t] = y[t]
            k_obs = 1
            continue
        if k_obs == 1:
            a0, a1 = y[t], y[t] - y_first
            q00, q01, q11 = s_irr, s_irr, 2 * s_irr + s_lvl + s_trd
            k_obs = 2
        else:
            v = y[t] - a0
            F = p00 + s_irr
            k0, k1 = p00 / F, p01 / F
            a0 += k0 * v
            a1 += k1 * v
            q00 = p00 - k0 * p00
            q01 = p01 - k0 * p01
            q11 = p11 - k1 * p01
            innov[t] = v / np.sqrt(F)
            ll += -0.5 * (np.log(2 * np.pi * F) + v * v / F)
            k_obs += 1
        level[t], slope[t] = a0, a1
        # predict t+1: a = T a, P = T P T' + Q,  T = [[1, 1], [0, 1]]
        a0 = a0 + a1
        p00 = q00 + 2 * q01 + q11 + s_lvl
        p01 = q01 + q11
        p11 = q11 + s_trd
    return level, slope, innov, ll


def _llt(y, params):
    s_irr, s_lvl, s_trd = (float(x) for x in params)
    return _llt_filter_nb(np.asarray(y, np.float64), s_irr, s_lvl, s_trd)


def fit_kalman_llt(y_train: np.ndarray, max_obs: int = 60000) -> dict:
    """MLE of (irregular, level, trend) variances on (the last `max_obs` bars of) training ln C, by
    L-BFGS on log-variances of the numba filter's Gaussian log-likelihood."""
    from scipy.optimize import minimize

    y = y_train[np.isfinite(y_train)][-max_obs:]
    dv = float(np.var(np.diff(y)))
    x0 = np.log([0.5 * dv, 0.5 * dv, 1e-4 * dv])

    def nll(x):
        return -_llt(y, np.exp(x))[3] / len(y)

    res = minimize(nll, x0, method="L-BFGS-B", bounds=[(np.log(dv) - 30, np.log(dv) + 5)] * 3)
    return {"params": [float(v) for v in np.exp(res.x)], "names": ["sigma2.irregular", "sigma2.level", "sigma2.trend"],
            "loglik": float(-res.fun * len(y)), "converged": bool(res.success)}


def kalman_llt_filter(y: np.ndarray, kp: dict) -> tuple[np.ndarray, np.ndarray]:
    """Filtered slope and standardized one-step innovation with FIXED parameters (never smoothed)."""
    _, slope, innov, _ = _llt(y, kp["params"])
    return slope, innov


# ---------------------------------------------------------------- fractional differencing (5.5)

def fracdiff_weights(d: float, thresh: float = 1e-4, max_len: int = 5000) -> np.ndarray:
    w = [1.0]
    k = 1
    while k < max_len:
        wk = -w[-1] * (d - k + 1) / k
        if abs(wk) < thresh:
            break
        w.append(wk)
        k += 1
    return np.array(w)


def fracdiff(x: np.ndarray, d: float, thresh: float = 1e-4) -> np.ndarray:
    """Fixed-width-window fractional difference: y_t = sum_k w_k x_{t-k}, NaN until the window is full."""
    w = fracdiff_weights(d, thresh)
    out = np.full(len(x), np.nan)
    if len(x) >= len(w):
        out[len(w) - 1:] = np.convolve(x, w, mode="valid")
    return out


def choose_frac_d(x_train: np.ndarray, grid=None, step: int = 6, pval: float = 0.05) -> tuple[float, list]:
    """Smallest d in [0, 1] (step 0.05) whose fractional difference passes ADF on TRAINING data
    (sampled every `step` bars to keep the test tractable)."""
    from statsmodels.tsa.stattools import adfuller

    grid = np.round(np.arange(0, 1.0001, 0.05), 2) if grid is None else grid
    tried = []
    for d in grid:
        y = fracdiff(x_train, d) if d > 0 else x_train.copy()
        y = y[np.isfinite(y)][::step]
        p = float(adfuller(y, maxlag=12, autolag=None)[1])
        tried.append((float(d), p))
        if p < pval:
            return float(d), tried
    return 1.0, tried


# ---------------------------------------------------------------- rolling statistics (6.6)

def _windows(x: np.ndarray, n: int, start: int, stop: int) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view

    return sliding_window_view(x[max(start - n + 1, 0): stop], n)


def hurst_rs_reference(x: np.ndarray, n: int = 256, scales=(16, 32, 64, 128, 256), chunk: int = 4096) -> np.ndarray:
    """Vectorized-numpy reference of `hurst_rs` (tests only)."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    lx = np.log(np.asarray(scales, float))
    lxc = lx - lx.mean()
    for s0 in range(n - 1, len(x), chunk):
        s1 = min(s0 + chunk, len(x))
        W = _windows(x, n, s0, s1)  # rows = windows ending at s0..s1-1
        logrs = []
        for m in scales:
            c = W.reshape(len(W), n // m, m)
            d = c - c.mean(axis=2, keepdims=True)
            y = np.cumsum(d, axis=2)
            R = y.max(axis=2) - y.min(axis=2)
            S = c.std(axis=2)
            with np.errstate(divide="ignore", invalid="ignore"):
                rs = np.where(S > 0, R / S, np.nan)
            logrs.append(np.log(np.nanmean(rs, axis=1)))
        L = np.column_stack(logrs)
        out[s0:s1] = (L - L.mean(axis=1, keepdims=True)) @ lxc / np.sum(lxc**2)
    return out


def variance_ratio(x, q: int = 12, n: int = 256):
    """VR(q) over the last n values: var of overlapping q-sums / (q * var of single values)."""
    import pandas as pd

    s = pd.Series(np.asarray(x, float))
    v1 = s.rolling(n).var()
    vq = s.rolling(q).sum().rolling(n - q + 1).var()
    return (vq / (q * v1)).to_numpy()


def rolling_ac1(x, n: int = 78):
    import pandas as pd

    s = pd.Series(np.asarray(x, float))
    return s.rolling(n).corr(s.shift(1)).to_numpy()


NORM_DECILES = np.array([-1.2816, -0.8416, -0.5244, -0.2533, 0.0, 0.2533, 0.5244, 0.8416, 1.2816])


def rolling_entropy_reference(z, n: int = 78, edges: np.ndarray = NORM_DECILES) -> np.ndarray:
    """Pandas reference of `rolling_entropy` (tests only). Shannon entropy (normalised to [0, 1]) of 10-bin quantized vol-scaled returns over the last n bars.
    Bins are fixed N(0,1) deciles (no fitting). NaN if any value in the window is missing."""
    import pandas as pd

    z = np.asarray(z, float)
    b = np.searchsorted(edges, z, side="right")
    oh = np.zeros((len(z), len(edges) + 1))
    ok = np.isfinite(z)
    oh[np.flatnonzero(ok), b[ok]] = 1.0
    oh[~ok] = np.nan
    cnt = pd.DataFrame(oh).rolling(n).sum().to_numpy()
    p = cnt / n
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1) / np.log(len(edges) + 1)
    h[~np.isfinite(cnt).all(axis=1)] = np.nan
    return h


@njit(cache=True)
def _hurst_nb(x, n, scales):
    N = len(x)
    out = np.full(N, np.nan)
    ns = len(scales)
    lx = np.log(scales.astype(np.float64))
    lxm = lx.mean()
    sxx = 0.0
    for i in range(ns):
        sxx += (lx[i] - lxm) ** 2
    logrs = np.empty(ns)
    for t in range(n - 1, N):
        w = x[t - n + 1: t + 1]
        bad = False
        for v in w:
            if not np.isfinite(v):
                bad = True
                break
        if bad:
            continue
        ok = True
        for si in range(ns):
            m = scales[si]
            acc, cnt = 0.0, 0
            for c in range(n // m):
                seg = w[c * m: (c + 1) * m]
                mu = seg.mean()
                cum, mx, mn, ss = 0.0, -np.inf, np.inf, 0.0
                for v in seg:
                    d = v - mu
                    cum += d
                    mx = max(mx, cum)
                    mn = min(mn, cum)
                    ss += d * d
                sd = np.sqrt(ss / m)
                if sd > 0:
                    acc += (mx - mn) / sd
                    cnt += 1
            if cnt == 0:
                ok = False
                break
            logrs[si] = np.log(acc / cnt)
        if not ok:
            continue
        lm = logrs.mean()
        num = 0.0
        for i in range(ns):
            num += (logrs[i] - lm) * (lx[i] - lxm)
        out[t] = num / sxx
    return out


def hurst_rs(x: np.ndarray, n: int = 256, scales=(16, 32, 64, 128, 256)) -> np.ndarray:
    """Rolling R/S Hurst exponent over the last n values (window ends at t). NaN if any value is missing."""
    return _hurst_nb(np.asarray(x, np.float64), n, np.asarray(scales, np.int64))


@njit(cache=True)
def _entropy_nb(b, ok, n, nbins):
    N = len(b)
    out = np.full(N, np.nan)
    cnt = np.zeros(nbins)
    nbad = 0
    for t in range(N):
        if ok[t]:
            cnt[b[t]] += 1
        else:
            nbad += 1
        if t >= n:
            if ok[t - n]:
                cnt[b[t - n]] -= 1
            else:
                nbad -= 1
        if t >= n - 1 and nbad == 0:
            h = 0.0
            for j in range(nbins):
                if cnt[j] > 0:
                    pj = cnt[j] / n
                    h -= pj * np.log(pj)
            out[t] = h / np.log(nbins)
    return out


def rolling_entropy(z, n: int = 78, edges: np.ndarray = NORM_DECILES) -> np.ndarray:
    """Shannon entropy (normalised to [0, 1]) of 10-bin quantized vol-scaled returns over the last n bars.
    Bins are fixed N(0,1) deciles (no fitting). NaN if any value in the window is missing."""
    z = np.asarray(z, float)
    ok = np.isfinite(z)
    b = np.searchsorted(edges, np.where(ok, z, 0.0), side="right").astype(np.int64)
    return _entropy_nb(b, ok, n, len(edges) + 1)
