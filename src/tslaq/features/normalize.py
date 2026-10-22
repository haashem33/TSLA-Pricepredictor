"""Causal normalization helpers (CLAUDE.md 5.3). Every window ends at bar t (or t-1 for the statistics
used to z-score bar t); nothing uses later bars."""
from __future__ import annotations

import numpy as np
import pandas as pd

CLIP = 5.0


def rolling_z(x: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """z_t = (x_t - mean(x_{t-w..t-1})) / std(x_{t-w..t-1})."""
    past = x.shift(1).rolling(window, min_periods=min_periods or window // 2)
    sd = past.std()
    return (x - past.mean()) / sd.where(sd > 0)


def rolling_pct_rank(x: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Percentile rank of x_t among x_{t-w..t} (current value included)."""
    return x.rolling(window + 1, min_periods=min_periods or window // 2).rank(pct=True)


def clip(x, lim: float = CLIP):
    return np.clip(x, -lim, lim)


def slot_trailing_median(values: pd.Series, session: pd.Series, slot: pd.Series, window: int,
                         min_periods: int | None = None) -> np.ndarray:
    """Per row: median of `values` at the same slot over the previous `window` sessions in which that
    slot was observed (current session excluded). NaN values are skipped."""
    mat = pd.DataFrame({"s": session.to_numpy(), "k": slot.to_numpy(), "v": values.to_numpy()}).pivot(
        index="s", columns="k", values="v")
    cols = {}
    for k in mat.columns:
        obs = mat[k].dropna()
        med = obs.rolling(window, min_periods=min_periods or window).median()
        cols[k] = med.reindex(mat.index).ffill().shift(1)
    prof = pd.DataFrame(cols, index=mat.index)
    return prof.to_numpy()[prof.index.get_indexer(session), prof.columns.get_indexer(slot)]
