"""Market (SPY/QQQ) and daily context features (CLAUDE.md 6.7, 6.4 HAR) and event flags.

D12: TSLA rows are kept when QQQ/SPY bars are missing; market features are NaN (= MISSING) on missing
market sessions, on forward-filled market bars, and on the market bar-0 return (and the 12-bar momentum
window) after a missing market session, since those span more than one night.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tslaq.data.calendar import nyse_sessions
from tslaq.features.indicators import true_range, wilder
from tslaq.features.normalize import rolling_z, slot_trailing_median


def market_gap_sessions(mdf: pd.DataFrame) -> set:
    """Market sessions whose previous NYSE session is missing from the market data."""
    s = pd.Index(mdf["session_date"].unique())
    cal = pd.Index(nyse_sessions(str(s[0].date()), str(s[-1].date()))["session_date"])
    prev = pd.Series(cal[:-1], index=cal[1:])
    have = set(s)
    return {d for d in s[1:] if d in prev.index and prev[d] not in have}


def market_features(mdf: pd.DataFrame, name: str) -> pd.DataFrame:
    """Per-bar features of one market symbol on its own series (requires r, sigma_hat). Keyed by
    (session_date, bar_index)."""
    gap_sess = market_gap_sessions(mdf)
    after_gap = mdf["session_date"].isin(gap_sess).to_numpy()
    k = mdf["bar_index"].to_numpy()
    miss = mdf["missing"].to_numpy()
    r = mdf["r"].where(~miss)
    lc = np.log(mdf["close"])
    out = pd.DataFrame({"session_date": mdf["session_date"], "bar_index": mdf["bar_index"]})
    out[f"{name}_r"] = r.where(~(after_gap & (k == 0)))
    out[f"{name}_z"] = out[f"{name}_r"] / mdf["sigma_hat"]
    mom = (lc - lc.shift(12)) / (mdf["sigma_hat"] * np.sqrt(12))
    out[f"{name}_mom12"] = mom.where(~(after_gap & (k < 12)))
    v = mdf["volume"].where(~mdf["missing"] & (mdf["volume"] > 0))
    out[f"{name}_relvol"] = np.log(v.to_numpy() / slot_trailing_median(v, mdf["session_date"], mdf["bar_index"],
                                                                      20, min_periods=10))
    return out


def add_market(df: pd.DataFrame, mkt: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Left-join market features onto TSLA rows; add rolling beta to QQQ and the residual return."""
    f = pd.DataFrame(index=df.index)
    keys = df[["session_date", "bar_index"]]
    for name, mdf in mkt.items():
        m = market_features(mdf, name)
        j = keys.merge(m, on=["session_date", "bar_index"], how="left")
        for c in m.columns[2:]:
            f[c] = j[c].to_numpy()
    if "qqq" in mkt:
        rt, rq = df["r"].where(~df["missing"]), f["qqq_r"]
        beta = rt.rolling(390, min_periods=300).cov(rq) / rq.rolling(390, min_periods=300).var()
        f["beta_qqq"] = beta
        resid = rt - beta.shift(1) * rq
        f["resid_qqq_z"] = rolling_z(resid, 390)
    return f.drop(columns=[c for c in f.columns if c.endswith("_r")])


def daily_context(df: pd.DataFrame) -> pd.DataFrame:
    """Daily features known at the open (from previous completed sessions), the gap, and HAR RV."""
    sess = df["session_date"]
    g = df.groupby("session_date", sort=True)
    d = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "rv": (df["r"] ** 2).groupby(sess).sum(min_count=1)})
    datr = wilder(true_range(d), 14)
    prev = pd.DataFrame({
        "d_prev_ret_atr": (d["close"] - d["close"].shift(1)) / datr,
        "d_atr_c": datr / d["close"],
        "d_sma20_dev": d["close"] / d["close"].rolling(20).mean() - 1,
        "d_mom5": np.log(d["close"] / d["close"].shift(5)),
        "har_rv1": np.log(d["rv"]),
        "har_rv5": np.log(d["rv"].rolling(5).mean()),
        "har_rv22": np.log(d["rv"].rolling(22).mean()),
    }).shift(1)  # known at today's open = values of the previous completed session
    out = pd.DataFrame(prev.reindex(sess).to_numpy(), columns=prev.columns, index=df.index)
    gz = (df["gap"] / df["sigma_hat"]).where(df["bar_index"] == 0)
    out["gap_z"] = gz.groupby(sess).transform("first")
    cr = (df["r"] ** 2).groupby(sess).cumsum()
    cs = (df["seasonal"] ** 2).groupby(sess).cumsum()
    out["rv_today_deseas"] = np.log(cr / cs)
    return out.replace([np.inf, -np.inf], np.nan)


def opex_dates(start, end) -> set:
    """Third Friday of each month; if it is not an NYSE session, the previous session. The exchange
    calendar is published in advance, so it is read past `end` (to the end of end's month)."""
    cal_end = (pd.Timestamp(end) + pd.offsets.MonthEnd(0)).date()
    cal = pd.Index(nyse_sessions(str(pd.Timestamp(start).date()), str(cal_end))["session_date"])
    out = set()
    for m in pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M"):
        fr = pd.date_range(m.start_time, m.end_time, freq="W-FRI")
        if len(fr) < 3:
            continue
        third = fr[2]
        c = cal[cal <= third]
        if len(c) and (third - c[-1]).days < 7:
            out.add(c[-1])
    return out


def event_flags(df: pd.DataFrame, reaction_sessions) -> pd.DataFrame:
    sess = df["session_date"]
    uniq = pd.Index(sess.unique())
    react = set(pd.to_datetime(pd.Series(list(reaction_sessions))))
    prev = pd.Series(uniq[:-1], index=uniq[1:])  # previous session present in the data (known at t)
    after = {s for s in uniq[1:] if prev[s] in react}
    opx = opex_dates(uniq[0], uniq[-1])
    return pd.DataFrame({
        "earn_react": sess.isin(react).astype(float).to_numpy(),
        "earn_after": sess.isin(after).astype(float).to_numpy(),
        "opex": sess.isin(opx).astype(float).to_numpy(),
        "half_day": df["is_half_day"].astype(float).to_numpy(),
        "dow": sess.dt.dayofweek.astype(float).to_numpy(),
    }, index=df.index)
