"""Bar returns (CLAUDE.md 5.1). All causal."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """r = ln(C_t / C_{t-1}) (bar 0 includes the overnight gap); r_intraday = ln(C_t / O_t);
    gap = ln(O_t / C_{t-1}) on bar_index 0, NaN elsewhere."""
    df = df.copy()
    prev_close = df["close"].shift(1)
    df["r"] = np.log(df["close"] / prev_close)
    df["r_intraday"] = np.log(df["close"] / df["open"])
    df["gap"] = np.where(df["bar_index"] == 0, np.log(df["open"] / prev_close), np.nan)
    return df
