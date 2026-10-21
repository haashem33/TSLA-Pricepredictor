import numpy as np
import pandas as pd
import pytest

from tslaq.data.calendar import bar_grid, nyse_sessions


def make_bars(start="2019-01-01", end="2019-09-30", seed=0) -> pd.DataFrame:
    """Synthetic canonical 5-minute bars with a U-shaped intraday volatility, real NYSE calendar
    (includes half-days), overnight gaps and a few missing bars."""
    rng = np.random.default_rng(seed)
    g = bar_grid(nyse_sessions(start, end))
    k = g["bar_index"].to_numpy()
    tod = 0.6 + 1.2 * ((k - 38.5) / 38.5) ** 2
    vol = 0.002 * tod * np.exp(0.3 * np.sin(np.arange(len(g)) / 2000))
    vol[k == 0] *= 3
    r = rng.standard_t(5, len(g)) * vol / np.sqrt(5 / 3)
    close = 100 * np.exp(np.cumsum(r))
    open_ = close * np.exp(-r * rng.uniform(0, 1, len(g)))
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.001, len(g))))
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.001, len(g))))
    df = g.assign(open=open_, high=hi, low=lo, close=close, volume=rng.integers(1000, 5000, len(g)).astype(float))
    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
    df["missing"] = False
    miss = rng.choice(len(df), 20, replace=False)
    df.loc[miss, "missing"] = True
    df.loc[miss, "volume"] = 0.0
    return df


@pytest.fixture(scope="session")
def bars():
    return make_bars()
