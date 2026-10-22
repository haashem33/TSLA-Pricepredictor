"""Bar-level features with no fitted parameters (CLAUDE.md 6.1-6.5). All causal: windows end at bar t,
EMAs/Wilder averages are recursive (adjust=False), and nothing uses bars after t.

Input: canonical bars with r, sigma_hat, seasonal, level2 (features/volatility.py). Missing (forward-filled)
bars keep their prices; volume-based features are NaN on them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tslaq.features.normalize import rolling_pct_rank, rolling_z, slot_trailing_median

LN2 = np.log(2.0)
N_SESSION_BARS = {False: 78, True: 42}


def wilder(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def _safe_div(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)


def rolling_slope(y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS slope (per bar) of y on 0..n-1 over the window ending at t, and R^2. Centered weights avoid
    cancellation for large-valued y (e.g. OBV)."""
    x = np.arange(n) - (n - 1) / 2.0
    sxx = float(np.sum(x**2))
    y = np.asarray(y, float)
    out = np.full(len(y), np.nan)
    r2 = np.full(len(y), np.nan)
    if len(y) >= n:
        sxy = np.convolve(y, x[::-1], mode="valid")  # sum_j x_j y_{t-n+1+j}
        slope = sxy / sxx
        out[n - 1:] = slope
        var_y = pd.Series(y).rolling(n).var(ddof=0).to_numpy()[n - 1:] * n
        with np.errstate(divide="ignore", invalid="ignore"):
            r2[n - 1:] = np.where(var_y > 0, (sxy**2 / sxx) / var_y, np.nan)
    return out, r2


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)


def candle_features(df: pd.DataFrame, atr: pd.Series) -> dict:
    O, H, L, C = (df[c] for c in ["open", "high", "low", "close"])
    rng = H - L
    f = {
        "body": (C - O) / O,
        "body_ratio": pd.Series(np.where(rng > 0, (C - O) / rng.where(rng > 0), 0.0), df.index),
        "range": rng / O,
        "upper_shadow": (H - np.maximum(O, C)) / O,
        "lower_shadow": (np.minimum(O, C) - L) / O,
        "close_location": pd.Series(np.where(rng > 0, (2 * C - H - L) / rng.where(rng > 0), 0.0), df.index),
    }
    # relation to the previous candle (6.2)
    H1, L1, C1, O1 = H.shift(1), L.shift(1), C.shift(1), O.shift(1)
    f["dH"] = (H - H1) / atr
    f["dL"] = (L - L1) / atr
    f["dC"] = (C - C1) / atr
    prng = H1 - L1
    f["pos_in_prev"] = pd.Series(_safe_div(C - L1, prng), df.index)
    top, bot = np.maximum(O, C), np.minimum(O, C)
    top1, bot1 = np.maximum(O1, C1), np.minimum(O1, C1)
    inter = (np.minimum(top, top1) - np.maximum(bot, bot1)).clip(lower=0)
    union = np.maximum(top, top1) - np.minimum(bot, bot1)
    f["body_overlap"] = pd.Series(np.where(union > 0, inter / union.where(union > 0), 0.0), df.index).where(O1.notna())
    return f


def trend_features(df: pd.DataFrame, atr: pd.Series) -> dict:
    C, H, L = df["close"], df["high"], df["low"]
    lc = np.log(C)
    sig = df["sigma_hat"]
    f = {}
    for k in (3, 6, 12, 24, 78):
        f[f"mom_{k}"] = (lc - lc.shift(k)) / (sig * np.sqrt(k))
    e = {n: ema(C, n) for n in (12, 26, 78)}
    for n in (12, 26, 78):
        f[f"ema_dev_{n}"] = C / e[n] - 1
    f["ema12_26"] = e[12] / e[26] - 1
    macd = e[12] - e[26]
    f["macd_hist"] = (macd - ema(macd, 9)) / (C * sig)
    d = C.diff()
    up, dn = wilder(d.clip(lower=0), 14), wilder((-d).clip(lower=0), 14)
    rsi = 100 - 100 / (1 + up / dn.where(dn > 0))
    rsi = rsi.where(dn > 0, 100.0).where(up.notna())
    f["rsi14"] = (rsi - 50) / 50
    ll, hh = L.rolling(14).min(), H.rolling(14).max()
    f["stoch_k14"] = pd.Series(_safe_div(C - ll, hh - ll), df.index)
    for n in (24, 78):
        s, r2 = rolling_slope(lc.to_numpy(), n)
        f[f"linreg_slope_{n}"] = pd.Series(s * n, df.index)
        f[f"linreg_r2_{n}"] = pd.Series(r2, df.index)
    # ADX(14), Wilder
    upm, dnm = H.diff(), -L.diff()
    pdm = pd.Series(np.where((upm > dnm) & (upm > 0), upm, 0.0), df.index).where(upm.notna())
    ndm = pd.Series(np.where((dnm > upm) & (dnm > 0), dnm, 0.0), df.index).where(dnm.notna())
    pdi, ndi = 100 * wilder(pdm, 14) / atr, 100 * wilder(ndm, 14) / atr
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).where((pdi + ndi) > 0)
    f["adx14"] = wilder(dx, 14) / 100
    return f


def intraday_momentum(df: pd.DataFrame) -> dict:
    """Gao, Han, Li & Zhou (2018) intraday momentum features (6.3)."""
    k = df["bar_index"].to_numpy()
    sess = df["session_date"]
    lc = np.log(df["close"])
    prev_close = lc.shift(1).where(df["bar_index"] == 0)  # previous session's last close, on bar 0
    pc = prev_close.groupby(sess).transform("first")
    c5 = lc.where(df["bar_index"] == 5).groupby(sess).transform("first")
    c65 = lc.where(df["bar_index"] == 65).groupby(sess).transform("first")
    c71 = lc.where(df["bar_index"] == 71).groupby(sess).transform("first")
    r30 = pd.Series(np.where(k >= 5, c5 - pc, np.nan), df.index)
    r12 = pd.Series(np.where(k >= 71, c71 - c65, np.nan), df.index)
    # first-half-hour relative volume and realized vol vs the same quantities over the last 20 sessions
    v = df["volume"].where(~df["missing"])
    first = df["bar_index"] <= 5
    day_v = v.where(first).groupby(sess).sum(min_count=6)
    day_rv = np.sqrt((np.log(df["close"] / df["open"]) ** 2).where(first).groupby(sess).sum(min_count=6))
    rel_v = np.log(day_v / day_v.rolling(20, min_periods=10).median().shift(1))
    rel_rv = np.log(day_rv / day_rv.rolling(20, min_periods=10).median().shift(1))
    rv_b = pd.Series(rel_v.reindex(sess).to_numpy(), df.index).where(df["bar_index"] >= 5)
    rr_b = pd.Series(rel_rv.reindex(sess).to_numpy(), df.index).where(df["bar_index"] >= 5)
    return {"r_first30": r30, "sign_first30": np.sign(r30), "r_12th30": r12, "sign_12th30": np.sign(r12),
            "first30_x_relvol": r30 * rv_b, "first30_x_rv": r30 * rr_b}


def vol_features(df: pd.DataFrame, atr: pd.Series) -> dict:
    O, H, L, C, r = df["open"], df["high"], df["low"], df["close"], df["r"]
    hl2 = np.log(H / L) ** 2
    co2 = np.log(C / O) ** 2
    f = {"atr_c": atr / C}
    for n in (12, 78):
        f[f"parkinson_{n}"] = np.sqrt(hl2.rolling(n).mean() / (4 * LN2))
    f["gk_78"] = np.sqrt((0.5 * hl2 - (2 * LN2 - 1) * co2).rolling(78).mean().clip(lower=0))
    sma, sd = C.rolling(20).mean(), C.rolling(20).std(ddof=0)
    f["bb_pctb"] = pd.Series(_safe_div(C - (sma - 2 * sd), 4 * sd), df.index)
    f["bb_width"] = 4 * sd / sma
    f["vol_ratio"] = r.rolling(12).std() / r.rolling(78).std()
    # volatility regime: percentile within the trailing 20 sessions (1560 bars)
    f["vreg_sigma_pct"] = rolling_pct_rank(df["sigma_hat"], 1560)
    f["vreg_level_pct"] = rolling_pct_rank(df["level2"], 1560)
    return f


def volume_features(df: pd.DataFrame) -> dict:
    sess, k = df["session_date"], df["bar_index"]
    v = df["volume"].where(~df["missing"] & (df["volume"] > 0))
    med = slot_trailing_median(v, sess, k, 20, min_periods=10)
    f = {"relvol": pd.Series(np.log(v.to_numpy() / med), df.index)}
    pv = (df["vwap"] * df["volume"]).groupby(sess).cumsum()
    cv = df["volume"].groupby(sess).cumsum()
    svwap = pv / cv.where(cv > 0)
    f["vwap_dev"] = df["close"] / svwap - 1
    obv = (np.sign(df["r"].fillna(0)) * df["volume"]).cumsum()
    slope, _ = rolling_slope(obv.to_numpy(), 24)
    f["obv_slope_z"] = rolling_z(pd.Series(slope, df.index), 390)
    illiq = df["r"].abs() / (df["close"] * v)
    f["amihud"] = np.log(illiq.rolling(78, min_periods=60).mean())
    f["bar_index"] = k.astype(float)
    n_bars = np.where(df["is_half_day"], 42, 78)
    f["minutes_to_close"] = pd.Series((n_bars - 1 - k.to_numpy()) * 5.0, df.index)
    return f


def bar_features(df: pd.DataFrame) -> pd.DataFrame:
    atr = wilder(true_range(df), 14)
    f = {}
    for part in (candle_features(df, atr), trend_features(df, atr), intraday_momentum(df),
                 vol_features(df, atr), volume_features(df)):
        f.update(part)
    return pd.DataFrame(f, index=df.index).replace([np.inf, -np.inf], np.nan)
