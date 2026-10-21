"""NYSE regular-session calendar (pandas_market_calendars)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

NY = "America/New_York"


def nyse_sessions(start: str, end: str, bar_minutes: int = 5) -> pd.DataFrame:
    """One row per session: session_date (naive midnight), open/close (tz-aware NY), n_bars, is_half_day."""
    sch = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    out = pd.DataFrame({
        "session_date": pd.DatetimeIndex(sch.index).normalize().astype("datetime64[ns]"),
        "open": pd.DatetimeIndex(sch["market_open"]).tz_convert(NY),
        "close": pd.DatetimeIndex(sch["market_close"]).tz_convert(NY),
    })
    minutes = (out["close"] - out["open"]).dt.total_seconds() / 60
    out["n_bars"] = (minutes // bar_minutes).astype(int)
    out["is_half_day"] = out["n_bars"] < (390 // bar_minutes)
    return out.reset_index(drop=True)


def bar_grid(sessions: pd.DataFrame, bar_minutes: int = 5) -> pd.DataFrame:
    """Expected bars: timestamp = bar END time (tz-aware NY), bar_index 0..n_bars-1."""
    n = sessions["n_bars"].to_numpy()
    rep = np.repeat(np.arange(len(sessions)), n)
    bar_index = np.concatenate([np.arange(k) for k in n])
    s = sessions.iloc[rep].reset_index(drop=True)
    return pd.DataFrame({
        "timestamp": s["open"] + pd.to_timedelta((bar_index + 1) * bar_minutes, unit="min"),
        "session_date": s["session_date"],
        "bar_index": bar_index,
        "is_half_day": s["is_half_day"],
    })
