"""1-minute raw bars -> canonical 5-minute regular-session bars (CLAUDE.md 4.2, 4.3).

Canonical schema: timestamp (tz-aware NY, bar END), open, high, low, close, volume, session_date,
bar_index, is_half_day, plus vwap, trade_count, n_minutes, missing, bad_tick, bad_range.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tslaq.data.calendar import NY, bar_grid, nyse_sessions
from tslaq.data.download import load_raw_month, month_range
from tslaq.utils.io import ROOT, interim_path, load_config

PRICE = ["open", "high", "low", "close"]


def aggregate_to_bars(m1: pd.DataFrame, sessions: pd.DataFrame, bar_minutes: int = 5) -> pd.DataFrame:
    """Aggregate start-stamped 1-minute bars into regular-session bars keyed by (session_date, bar_index).

    Minutes outside [open, close) of an NYSE session (pre/post-market, holidays) are dropped.
    """
    m = m1.copy()
    ts = pd.to_datetime(m["timestamp"], utc=True).dt.tz_convert(NY)
    m["session_date"] = ts.dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")
    m["ts"] = ts
    s = sessions[["session_date", "open", "close"]].rename(columns={"open": "s_open", "close": "s_close"})
    m = m.merge(s, on="session_date", how="inner")
    m = m[(m["ts"] >= m["s_open"]) & (m["ts"] < m["s_close"])]
    m["bar_index"] = ((m["ts"] - m["s_open"]).dt.total_seconds() // (60 * bar_minutes)).astype(int)
    m["pv"] = m["vwap"].fillna(m["close"]) * m["volume"]
    m = m.sort_values("ts")
    g = m.groupby(["session_date", "bar_index"], sort=True)
    bars = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                 close=("close", "last"), volume=("volume", "sum"), trade_count=("trade_count", "sum"),
                 pv=("pv", "sum"), n_minutes=("close", "size")).reset_index()
    vol = bars["volume"].where(bars["volume"] > 0)
    bars["vwap"] = (bars["pv"] / vol).fillna(bars["close"])
    return bars.drop(columns="pv")


def fill_missing(bars: pd.DataFrame, sessions: pd.DataFrame, bar_minutes: int = 5) -> pd.DataFrame:
    """Reindex onto the full session grid; forward-fill price with zero volume and flag `missing`."""
    grid = bar_grid(sessions, bar_minutes)
    df = grid.merge(bars, on=["session_date", "bar_index"], how="left")
    df["missing"] = df["close"].isna()
    df["close"] = df["close"].ffill()
    for c in ["open", "high", "low", "vwap"]:
        df[c] = df[c].fillna(df["close"])
    df[["volume", "trade_count", "n_minutes"]] = df[["volume", "trade_count", "n_minutes"]].fillna(0)
    return df.dropna(subset=["close"]).reset_index(drop=True)


def drop_bad_sessions(df: pd.DataFrame, max_missing_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    frac = df.groupby("session_date")["missing"].mean()
    bad = frac[frac > max_missing_frac]
    kept = df[~df["session_date"].isin(bad.index)].reset_index(drop=True)
    return kept, bad.rename("missing_frac").reset_index()


def dividend_adjust(df: pd.DataFrame, divs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Back-adjust prices for cash dividends: bars before each ex-date are scaled by (1 - D / C_prev).

    Intraday returns are unchanged; only the ex-date overnight return changes.
    """
    df = df.copy()
    last_close = df.groupby("session_date")["close"].last()
    sess = last_close.index
    rows = []
    for ex, amt in zip(pd.to_datetime(divs["ex_dividend_date"]), divs["amount"]):
        if ex <= sess[0] or ex > sess[-1]:
            continue
        prev = sess[sess < ex][-1]
        rows.append((ex, float(amt), last_close[prev], 1 - float(amt) / last_close[prev]))
    fac = pd.DataFrame(rows, columns=["ex_date", "amount", "prev_close", "factor"])
    mult = np.ones(len(df))
    sd = df["session_date"].to_numpy()
    for r in fac.itertuples():
        mult[sd < np.datetime64(r.ex_date)] *= r.factor
    for c in PRICE + ["vwap"]:
        df[c] = df[c] * mult
    return df, fac


def flag_bad_ticks(df: pd.DataFrame, mult: float, window_bars: int, window_sessions: int) -> pd.DataFrame:
    """Flag (not delete) bars whose |log return| or log range exceeds `mult` x a trailing median.

    Intraday bars use the previous `window_bars` intraday bars; bar_index 0 (overnight gap) uses the
    previous `window_sessions` opening bars. All windows are trailing (exclude bar t).
    """
    df = df.copy()
    r = np.log(df["close"] / df["close"].shift(1)).abs()
    rng = np.log(df["high"] / df["low"])
    first = df["bar_index"] == 0
    intra = ~first & ~df["missing"]
    flag = pd.Series(False, index=df.index)
    flag_rng = pd.Series(False, index=df.index)
    med = r[intra].shift(1).rolling(window_bars, min_periods=window_bars // 5).median()
    flag[intra] = (r[intra] > mult * med).to_numpy()
    med_rng = rng[intra].shift(1).rolling(window_bars, min_periods=window_bars // 5).median()
    flag_rng[intra] = (rng[intra] > mult * med_rng).to_numpy()
    med0 = r[first].shift(1).rolling(window_sessions, min_periods=5).median()
    flag[first] = (r[first] > mult * med0).to_numpy()
    df["bad_tick"] = flag.astype(bool)
    df["bad_range"] = flag_rng.astype(bool)
    return df


def split_factor(session_date: pd.Series, splits: pd.DataFrame) -> np.ndarray:
    """Cumulative split factor per session: product of the split ratios effective AFTER that session.
    Adjusted price = raw price / factor; adjusted volume = raw volume * factor."""
    f = np.ones(len(session_date))
    sd = session_date.to_numpy()
    for eff, ratio in zip(pd.to_datetime(splits["effective_date"]), splits["split_factor"]):
        f[sd < np.datetime64(eff)] *= float(ratio)
    return f


def split_adjust_bars(bars: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Float (unrounded) split adjustment of bars built from unadjusted prices."""
    bars = bars.copy()
    f = split_factor(bars["session_date"], splits)
    for c in PRICE + ["vwap"]:
        bars[c] = bars[c] / f
    bars["volume"] = bars["volume"] * f
    return bars


def load_raw_v2(symbol: str, months: list[str], cfg: dict) -> pd.DataFrame:
    """v2 TSLA source: unadjusted RAW 1-min files for months before v2.raw_end_month_exclusive,
    the split-adjusted source files afterwards (no splits after 2022-08, so raw == adjusted)."""
    v2 = cfg["v2"]
    parts = []
    for m in months:
        if m < v2["raw_end_month_exclusive"]:
            parts.append(pd.read_parquet(ROOT / v2["raw_dir"] / f"{m}.parquet"))
        else:
            parts.append(load_raw_month(symbol, m, cfg))
    return pd.concat(parts, ignore_index=True)


def build_symbol(symbol: str, cfg: dict | None = None, version: str | None = None,
                 months: list[str] | None = None, end_date: str | None = None,
                 extra_m1: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Canonical bars for `months`. Defaults stop before the holdout. D28 only: `extra_m1` (1-minute bars of later
    months, same columns) is appended and `end_date` replaces the holdout cap (sessions and dividends up to it)."""
    cfg = cfg or load_config("data")
    version = version or cfg.get("data_version", "v1")
    months = months or month_range(cfg["start_month"], cfg["holdout_start"][:7])
    use_raw = version == "v2" and symbol == cfg["target"]
    if use_raw:
        m1 = load_raw_v2(symbol, months, cfg)
    else:
        m1 = pd.concat([load_raw_month(symbol, m, cfg) for m in months], ignore_index=True)
    if extra_m1 is not None:
        m1 = pd.concat([m1, extra_m1], ignore_index=True)
    start = f"{months[0]}-01"
    cap = end_date or (pd.Timestamp(cfg["holdout_start"]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = cap if extra_m1 is not None else min((pd.Period(months[-1], "M").end_time).strftime("%Y-%m-%d"), cap)
    sessions = nyse_sessions(start, end, cfg["bar_minutes"])
    bars = aggregate_to_bars(m1, sessions, cfg["bar_minutes"])
    if use_raw:
        bars = split_adjust_bars(bars, pd.read_csv(ROOT / cfg["v2"]["splits_file"]))
    df = fill_missing(bars, sessions, cfg["bar_minutes"])
    info = {"n_sessions_calendar": len(sessions), "n_missing_bars_before_drop": int(df["missing"].sum()),
            "sessions_without_data": int(len(sessions) - df["session_date"].nunique())}
    df, dropped = drop_bad_sessions(df, cfg["max_missing_frac"])
    info["dropped_sessions"] = dropped
    div_file = ROOT / cfg["dividends_dir"] / f"{symbol}.csv"
    if div_file.exists():
        divs = pd.read_csv(div_file)
        divs = divs[pd.to_datetime(divs["ex_dividend_date"]) <= pd.Timestamp(end)]
        df, fac = dividend_adjust(df, divs)
        info["dividends"] = fac
    df = flag_bad_ticks(df, cfg["bad_tick_mult"], cfg["bad_tick_window_bars"], cfg["bad_tick_window_sessions"])
    cols = ["timestamp", "open", "high", "low", "close", "volume", "session_date", "bar_index", "is_half_day",
            "vwap", "trade_count", "n_minutes", "missing", "bad_tick", "bad_range"]
    return df[cols], info


def daily_from_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Daily bars built from 5-minute bars (CLAUDE.md 4.1)."""
    g = df.groupby("session_date")
    return pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                         "close": g["close"].last(), "volume": g["volume"].sum(),
                         "is_half_day": g["is_half_day"].first()}).reset_index()


def earnings_sessions(cfg: dict | None = None, end_date: str | None = None) -> pd.DataFrame:
    """Map each TSLA report to its reaction session (next session if after close or time unknown). Reports on or
    after the holdout start are skipped; with `end_date` (D28 only) reports up to and including it are kept."""
    cfg = cfg or load_config("data")
    e = pd.read_csv(ROOT / cfg["earnings_file"], comment="#")
    stop = pd.Timestamp(end_date) + pd.Timedelta(days=1) if end_date else pd.Timestamp(cfg["holdout_start"])
    sess = nyse_sessions("2015-12-01", (stop + pd.Timedelta(days=10)).strftime("%Y-%m-%d"))["session_date"]
    out = []
    for d, t in zip(pd.to_datetime(e["reported_date"]), e["report_time"].fillna("unknown")):
        if d >= stop:
            continue
        before_open = str(t).lower().startswith("pre")
        later = sess[sess >= d] if before_open else sess[sess > d]
        if str(t) == "unknown":
            print(f"earnings {d.date()}: report time missing, assuming after close")
        out.append({"reported_date": d, "report_time": t, "assumed_after_close": not before_open,
                    "reaction_session": later.iloc[0]})
    return pd.DataFrame(out)


def main(version: str | None = None) -> None:
    cfg = load_config("data")
    version = version or cfg.get("data_version", "v1")
    interim_path("X", version).parent.mkdir(parents=True, exist_ok=True)
    for s in cfg["symbols"]:
        df, info = build_symbol(s, cfg, version)
        df.to_parquet(interim_path(s, version), index=False)
        daily_from_bars(df).to_parquet(interim_path(s, version).with_name(f"{s}_daily.parquet"), index=False)
        print(version, s, len(df), "bars;", len(info["dropped_sessions"]), "sessions dropped")
    earnings_sessions(cfg).to_parquet(interim_path("X", version).with_name("TSLA_earnings_sessions.parquet"),
                                      index=False)


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
