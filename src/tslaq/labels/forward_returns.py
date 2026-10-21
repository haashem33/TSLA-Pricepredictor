"""Forward-return and realized-volatility labels (CLAUDE.md 9.1). Labels never cross the session close.

The label at bar t uses bars t+1 .. t+h of the same session only. Negative shifts are allowed here
(and only here), because these are targets, not features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def add_labels(df: pd.DataFrame, horizons=(1, 3, 12)) -> pd.DataFrame:
    """Adds fwd_h (log return), z_h (fwd_h / S_h, if S_h present), rv_h (log realized variance)
    and label_end_h (row position of the last bar used). NaN when t + h is past the session's last bar."""
    df = df.copy()
    g = df.groupby("session_date", sort=False)
    pos = np.arange(len(df))
    # cumulative r^2 within the session; sum over t+1..t+h = csum[t+h] - csum[t]
    csum = (df["r"] ** 2).groupby(df["session_date"], sort=False).cumsum()
    for h in horizons:
        fut_close = g["close"].shift(-h)
        df[f"fwd_{h}"] = np.log(fut_close / df["close"])
        fut_csum = csum.groupby(df["session_date"], sort=False).shift(-h)
        df[f"rv_{h}"] = np.log(fut_csum - csum + EPS)
        df.loc[fut_close.isna(), f"rv_{h}"] = np.nan
        df[f"label_end_{h}"] = np.where(fut_close.notna(), pos + h, -1)
        if f"S_{h}" in df:
            df[f"z_{h}"] = df[f"fwd_{h}"] / df[f"S_{h}"]
    return df
