"""Intraday momentum strategy of Zarattini, Aziz & Barbon (2024) as pre-registered in D29 / D29a.

Indicators (all causal; minute m = the 1-minute bar ending at open + m):
- move_{d,m} = |C_{d,m} / O_d - 1|;
- sigma_{t,m} = mean of move_{d,m} over the 14 most recent previous sessions d that contain minute m (D29a-1);
- UB / LB = max / min(O_t, C_{t-1}) * (1 +/- VM * sigma_{t,m});
- VWAP = cumulative typical price (H+L+C)/3 x volume over the regular session;
- sigma_SPY,t = sd (ddof 1) of the daily close-to-close returns of the 14 previous sessions.
Positions change only at the decision minutes (10:00, 10:30, ..., 15:30; half-days to 12:30) and are flat at the
session's last close. Fills: E0 = open of the next 1-minute bar, E1 = one bar later.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LOOKBACK = 14
VM = 1.0
TARGET_VOL = 0.02
MAX_LEV = 4.0
DECISION_MINUTES = tuple(range(30, 361, 30))  # 10:00 .. 15:30
COMMISSION = 0.0035  # $ per share
MIN_COMMISSION = 0.35  # $ per order (C_real only)
PAPER_SLIPPAGE = 0.001  # $ per share
HALF_TICK = 0.005  # $ per share (D29a-2 floor)
SEC_BP = 0.10  # bp of notional per side (D26)
CONFIGS = ("primary", "opp_band_1x", "vwap_1x")
COSTS = ("gross", "paper", "real", "real2x")


def add_indicators(df: pd.DataFrame, lookback: int = LOOKBACK) -> pd.DataFrame:
    """Per-minute noise area, VWAP and per-session sigma_SPY. `df` is the D29 1-minute frame sorted by
    (session_date, minute); only rows of earlier sessions (and earlier minutes for VWAP) enter each value."""
    df = df.sort_values(["session_date", "minute"]).reset_index(drop=True).copy()
    sess = df.groupby("session_date", sort=True)
    day_open = sess["open"].transform("first")
    df["move"] = (df["close"] / day_open - 1).abs()
    # D29a-1: rolling over the previous sessions that contain this minute (rows of that minute, in session order)
    df["sigma"] = df.groupby("minute", sort=False)["move"].transform(
        lambda s: s.shift(1).rolling(lookback, min_periods=lookback).mean())
    daily = sess.agg(day_open=("open", "first"), day_close=("close", "last"), raw_day_open=("raw_open", "first"))
    daily["prev_close"] = daily["day_close"].shift(1)
    ret = daily["day_close"] / daily["prev_close"] - 1
    daily["sigma_spy"] = ret.shift(1).rolling(lookback, min_periods=lookback).std(ddof=1)
    df = df.join(daily[["day_open", "prev_close", "raw_day_open", "sigma_spy"]], on="session_date")
    hi_ref = np.maximum(df["day_open"], df["prev_close"])
    lo_ref = np.minimum(df["day_open"], df["prev_close"])
    df["ub"] = hi_ref * (1 + VM * df["sigma"])
    df["lb"] = lo_ref * (1 - VM * df["sigma"])
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_pv = (tp * df["volume"]).groupby(df["session_date"]).cumsum()
    cum_v = df["volume"].groupby(df["session_date"]).cumsum()
    df["vwap"] = (cum_pv / cum_v.where(cum_v > 0))
    return df


def decision_signals(day: pd.DataFrame, config: str) -> list[tuple[int, int]]:
    """(decision minute, target sign) for one session. `primary` and `vwap_1x`: the state-free band + VWAP signal;
    `opp_band_1x`: enter on a band cross, hold until the opposite band is crossed (then reverse)."""
    n = int(day["n_minutes"].iloc[0])
    rows = day.set_index("minute")
    out, pos = [], 0
    for m in DECISION_MINUTES:
        if m >= n:
            break
        r = rows.loc[m]
        c, ub, lb, vw = r["close"], r["ub"], r["lb"], r["vwap"]
        if not (np.isfinite(ub) and np.isfinite(lb)):
            out.append((m, 0))
            continue
        if config == "opp_band_1x":
            if pos <= 0 and c > ub:
                pos = 1
            elif pos >= 0 and c < lb:
                pos = -1
        else:
            pos = 1 if (c > ub and c > vw) else (-1 if (c < lb and c < vw) else 0)
        out.append((m, pos))
    return out


@dataclass
class DayPlan:
    session_date: pd.Timestamp
    raw_open: float
    sigma_spy: float
    fills: list  # [(fill raw price, target sign)], the last entry is the close exit with sign 0
    tradable: bool


def day_plans(df: pd.DataFrame, config: str, delay: int = 0) -> list[DayPlan]:
    """Fill schedule per session (independent of AUM). A decision at minute m fills at the open of minute
    m + 1 + delay (raw price); the close exit fills at the raw close of the session's last minute."""
    plans = []
    for d, day in df.groupby("session_date", sort=True):
        day = day.sort_values("minute")
        raw_open = day["raw_open"].to_numpy()
        n = int(day["n_minutes"].iloc[0])
        sig = float(day["sigma_spy"].iloc[0])
        tradable = bool(np.isfinite(day["sigma"].to_numpy()[[m - 1 for m in DECISION_MINUTES if m < n]]).all()
                        and (config != "primary" or np.isfinite(sig)) and np.isfinite(day["prev_close"].iloc[0]))
        fills = []
        if tradable:
            for m, s in decision_signals(day, config):
                fills.append((float(raw_open[m + delay]), s))  # minute m+1+delay has row index m+delay
        fills.append((float(day["raw_close"].to_numpy()[n - 1]), 0))
        plans.append(DayPlan(pd.Timestamp(d), float(day["raw_day_open"].iloc[0]), sig, fills, tradable))
    return plans


def order_cost(q: float, price: float, cost: str, half_spread_bp: float) -> float:
    """Dollar cost of one order of q > 0 shares at `price` (D29 / D29a-2)."""
    if q <= 0 or cost == "gross":
        return 0.0
    if cost == "paper":
        return q * (COMMISSION + PAPER_SLIPPAGE)
    notional = q * price
    c = (max(half_spread_bp * 1e-4 * notional, HALF_TICK * q) + max(MIN_COMMISSION, COMMISSION * q)
         + SEC_BP * 1e-4 * notional)
    return 2 * c if cost == "real2x" else c


def shares_for_day(aum: float, raw_open: float, sigma_spy: float, config: str) -> int:
    lev = min(MAX_LEV, TARGET_VOL / sigma_spy) if config == "primary" else 1.0
    return round(aum / raw_open * lev)


def simulate(plans: list[DayPlan], config: str, cost: str, half_spread_bp: dict,
             aum0: float = 100_000.0, start: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compounding simulation. Returns (daily, trades). daily: session_date, aum_prev, pnl_gross, cost, ret,
    traded_notional, n_orders, long/short gross and net P&L. trades: one row per holding segment (sign, shares,
    entry, exit, gross $ P&L, allocated cost)."""
    aum = aum0
    days, trades = [], []
    for p in plans:
        if start is not None and p.session_date < start:
            continue
        N = shares_for_day(aum, p.raw_open, p.sigma_spy, config) if p.tradable else 0
        pos, entry, gross, cst, notional, n_orders = 0, None, 0.0, 0.0, 0.0, 0
        leg = {1: [0.0, 0.0], -1: [0.0, 0.0]}  # sign -> [gross, cost]
        seg_cost = 0.0
        last_price = None
        hs = half_spread_bp[p.session_date.year]
        for price, s in p.fills:
            if pos != 0 and last_price is not None:
                g = pos * (price - last_price)
                gross += g
                leg[int(np.sign(pos))][0] += g
            target = s * N
            q = abs(target - pos)
            if q > 0:
                c = order_cost(q, price, cost, hs)
                cst += c
                notional += q * price
                n_orders += 1
                # split the order's cost between the segment it closes and the one it opens (by shares)
                if pos != 0 and np.sign(target) != np.sign(pos):
                    close_q, _open_q = abs(pos), abs(target)
                elif pos == 0:
                    close_q, _open_q = 0, abs(target)
                else:  # same sign, size change (cannot happen: N fixed within a day)
                    close_q, _open_q = 0, q
                c_close = c * close_q / q
                c_open = c - c_close
                if pos != 0 and close_q > 0:
                    sgn = int(np.sign(pos))
                    leg[sgn][1] += c_close
                    trades.append({"session_date": p.session_date, "sign": sgn, "shares": abs(pos),
                                   "entry": entry, "exit": price, "gross": sgn * abs(pos) * (price - entry),
                                   "cost": seg_cost + c_close})
                    seg_cost = 0.0
                if target != 0:
                    leg[int(np.sign(target))][1] += c_open
                    seg_cost = c_open
                    entry = price
                pos = target
            last_price = price
        net = gross - cst
        days.append({"session_date": p.session_date, "aum_prev": aum, "pnl_gross": gross, "cost": cst,
                     "ret": net / aum, "ret_gross": gross / aum, "traded_notional": notional, "n_orders": n_orders,
                     "shares": N, "long_gross": leg[1][0], "long_net": leg[1][0] - leg[1][1],
                     "short_gross": leg[-1][0], "short_net": leg[-1][0] - leg[-1][1]})
        aum += net
    return pd.DataFrame(days), pd.DataFrame(trades)
