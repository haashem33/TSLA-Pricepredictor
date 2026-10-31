"""SPY 1-minute regular-session frame (D29 data rules) and the data check against the v2 5-minute bars.

Frame columns: session_date, minute (1..n, bar ENDING at session open + minute), timestamp (bar end, NY),
open/high/low/close (dividend back-adjusted), raw_open/raw_close (unadjusted, for shares and per-share costs),
volume, missing, is_half_day, n_minutes, bad_tick, bad_range.

Stage A (D29): `end_month` must be <= STAGE_A_END; no month after it is read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tslaq.data.calendar import NY, nyse_sessions
from tslaq.data.clean import dividend_adjust, flag_bad_ticks
from tslaq.data.download import month_range
from tslaq.utils.io import ROOT, interim_path, load_config

STAGE_A_END = "2024-02"
RAW_DIR = ROOT / "data" / "raw" / "SPY"


def load_raw_1min(months: list[str], max_month: str = STAGE_A_END) -> pd.DataFrame:
    """Alpaca SIP 1-minute bars from data/raw/SPY/ (data v2 source). Refuses any month after `max_month`."""
    late = [m for m in months if m > max_month]
    if late:
        raise PermissionError(f"months after {max_month} requested: {late[:3]}...; not loading.")
    return pd.concat([pd.read_parquet(RAW_DIR / f"{m}.parquet") for m in months], ignore_index=True)


def minute_frame(m1: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """Regular-session minutes on the full calendar grid; missing minutes forward-filled at the previous close with
    zero volume (flag `missing`). Prices here are as in the source (unadjusted)."""
    m = m1.copy()
    ts = pd.to_datetime(m["timestamp"], utc=True).dt.tz_convert(NY)
    m["session_date"] = ts.dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")
    s = sessions[["session_date", "open", "close"]].rename(columns={"open": "s_open", "close": "s_close"})
    m = m.assign(ts=ts).merge(s, on="session_date", how="inner")
    m = m[(m["ts"] >= m["s_open"]) & (m["ts"] < m["s_close"])]
    m["minute"] = ((m["ts"] - m["s_open"]).dt.total_seconds() // 60).astype(int) + 1
    m = m.drop_duplicates(["session_date", "minute"], keep="last")
    n = ((sessions["close"] - sessions["open"]).dt.total_seconds() // 60).astype(int).to_numpy()
    rep = np.repeat(np.arange(len(sessions)), n)
    g = sessions.iloc[rep].reset_index(drop=True)
    minute = np.concatenate([np.arange(1, k + 1) for k in n])
    grid = pd.DataFrame({"session_date": g["session_date"], "minute": minute,
                         "timestamp": g["open"] + pd.to_timedelta(minute, unit="min"),
                         "is_half_day": g["is_half_day"], "n_minutes": np.repeat(n, n)})
    df = grid.merge(m[["session_date", "minute", "open", "high", "low", "close", "volume"]],
                    on=["session_date", "minute"], how="left")
    df["missing"] = df["close"].isna()
    df["close"] = df["close"].ffill()
    for c in ["open", "high", "low"]:
        df[c] = df[c].fillna(df["close"])
    df["volume"] = df["volume"].fillna(0.0)
    return df.dropna(subset=["close"]).reset_index(drop=True)


def v2_session_set(version: str = "v2") -> pd.DatetimeIndex:
    v = pd.read_parquet(interim_path("SPY", version), columns=["session_date"])
    return pd.DatetimeIndex(v["session_date"].unique())


def build_spy_1min(end_month: str = STAGE_A_END, start_month: str = "2016-01") -> tuple[pd.DataFrame, dict]:
    """D29 cleaning at 1-minute resolution. Sessions dropped: > 5% missing minutes, plus any session absent from the
    SPY v2 5-minute frame (within the same date range). Dividend back-adjustment with dividends up to the last
    session of the frame only."""
    cfg = load_config("data")
    if end_month > STAGE_A_END:
        raise PermissionError(f"Stage A: end_month {end_month} > {STAGE_A_END}")
    months = month_range(start_month, (pd.Period(end_month, "M") + 1).strftime("%Y-%m"))
    m1 = load_raw_1min(months)
    end = pd.Period(end_month, "M").end_time.strftime("%Y-%m-%d")
    sessions = nyse_sessions(f"{start_month}-01", end, cfg["bar_minutes"])
    df = minute_frame(m1, sessions)
    info = {"n_sessions_calendar": len(sessions),
            "sessions_without_data": int(len(sessions) - df["session_date"].nunique()),
            "n_missing_minutes": int(df["missing"].sum())}
    frac = df.groupby("session_date")["missing"].mean()
    bad = frac[frac > cfg["max_missing_frac"]]
    v2s = v2_session_set("v2")
    in_range = df["session_date"].drop_duplicates()
    not_v2 = in_range[~in_range.isin(v2s)]
    drop = set(bad.index) | set(not_v2)
    info["dropped_missing_frac"] = bad.rename("missing_frac").reset_index()
    info["dropped_not_in_v2"] = sorted(set(not_v2) - set(bad.index))
    df = df[~df["session_date"].isin(drop)].reset_index(drop=True)
    df["raw_open"], df["raw_close"] = df["open"], df["close"]
    divs = pd.read_csv(ROOT / cfg["dividends_dir"] / "SPY.csv")
    divs = divs[pd.to_datetime(divs["ex_dividend_date"]) <= df["session_date"].max()]
    df["vwap"] = df["close"]  # placeholder column so dividend_adjust can scale it; not used by the strategy
    df, fac = dividend_adjust(df, divs)
    info["dividends"] = fac
    df["bar_index"] = df["minute"] - 1
    df = flag_bad_ticks(df, cfg["bad_tick_mult"], 5 * 390, cfg["bad_tick_window_sessions"])
    cols = ["session_date", "minute", "timestamp", "open", "high", "low", "close", "raw_open", "raw_close", "volume",
            "missing", "is_half_day", "n_minutes", "bad_tick", "bad_range"]
    return df[cols], info


def data_check(df: pd.DataFrame, version: str = "v2") -> tuple[pd.DataFrame, dict]:
    """(1) 5-minute closes from the 1-minute frame reproduce the v2 SPY 5-minute log returns (same sessions, max
    |difference| <= 1e-10); (2) the source cache is dividend-unadjusted: v2 adjusted close / our raw close is
    constant between ex-dates and jumps by exactly the dividend factor (1 - D / C_prev) at each ex-date."""
    end = df["session_date"].max()
    v = pd.read_parquet(interim_path("SPY", version))
    v = v[v["session_date"] <= end].sort_values("timestamp").reset_index(drop=True)
    five = df[df["minute"] % 5 == 0].copy()
    five["bar_index"] = five["minute"] // 5 - 1
    five = five.sort_values("timestamp").reset_index(drop=True)
    out = {}
    s_ours, s_v2 = set(five["session_date"].unique()), set(v["session_date"].unique())
    out["sessions_ours"], out["sessions_v2"] = len(s_ours), len(s_v2)
    out["sessions_only_ours"], out["sessions_only_v2"] = len(s_ours - s_v2), len(s_v2 - s_ours)
    out["bars_ours"], out["bars_v2"] = len(five), len(v)
    j = v[["session_date", "bar_index", "close"]].merge(
        five[["session_date", "bar_index", "close", "raw_close"]], on=["session_date", "bar_index"],
        suffixes=("_v2", "_ours"), how="outer", indicator=True)
    out["bars_unmatched"] = int((j["_merge"] != "both").sum())
    j = j[j["_merge"] == "both"].reset_index(drop=True)
    r_v2 = np.log(j["close_v2"] / j["close_v2"].shift(1))
    r_ours = np.log(j["close_ours"] / j["close_ours"].shift(1))
    d = (r_v2 - r_ours).abs().iloc[1:]
    out["logret_max_abs_diff"] = float(d.max())
    out["logret_n"] = int(d.notna().sum())
    # dividend check on session-last closes
    last = j.groupby("session_date").last()
    ratio = last["close_v2"] / last["raw_close"]
    cfg = load_config("data")
    divs = pd.read_csv(ROOT / cfg["dividends_dir"] / "SPY.csv")
    divs["ex"] = pd.to_datetime(divs["ex_dividend_date"])
    divs = divs[(divs["ex"] > last.index[0]) & (divs["ex"] <= end)].sort_values("ex")
    rows, prev_ex = [], last.index[0]
    for ex, amt in zip(divs["ex"], divs["amount"]):
        seg = ratio[(ratio.index >= prev_ex) & (ratio.index < ex)]
        before = ratio[ratio.index < ex].iloc[-1]
        after = ratio[ratio.index >= ex].iloc[0]
        c_prev = last["raw_close"][last.index < ex].iloc[-1]
        expected = 1 - amt / c_prev
        rows.append({"ex_date": ex.date(), "amount": amt, "raw_prev_close": c_prev, "expected_factor": expected,
                     "observed_factor": before / after, "abs_diff": abs(before / after - expected),
                     "segment_rel_sd": float(seg.std() / seg.mean()) if len(seg) > 1 else 0.0})
        prev_ex = ex
    seg = ratio[ratio.index >= prev_ex]
    tab = pd.DataFrame(rows)
    out["n_ex_dates"] = len(tab)
    out["div_factor_max_abs_diff"] = float(tab["abs_diff"].max())
    out["ratio_segment_max_rel_sd"] = float(max(tab["segment_rel_sd"].max(), seg.std() / seg.mean()))
    out["min_jump"] = float((1 - tab["observed_factor"]).min())
    out["pass_returns"] = bool(out["logret_max_abs_diff"] <= 1e-10 and out["sessions_only_ours"] == 0
                               and out["sessions_only_v2"] == 0 and out["bars_unmatched"] == 0)
    out["pass_unadjusted"] = bool(out["div_factor_max_abs_diff"] <= 1e-9 and out["ratio_segment_max_rel_sd"] <= 1e-12
                                  and out["min_jump"] > 0)
    return tab, out
