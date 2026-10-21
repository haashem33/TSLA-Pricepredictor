"""Expanding-window monthly walk-forward with purge and embargo (CLAUDE.md 10.1)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tslaq.cv.purge import purge_embargo_mask


@dataclass
class Fold:
    val_month: str
    train_idx: np.ndarray
    val_idx: np.ndarray


def monthly_folds(session_date: pd.Series, label_end: np.ndarray, first_val_month: str, last_val_month: str,
                  embargo_bars: int = 78) -> list[Fold]:
    """Train on all rows before month M (purged/embargoed), validate on month M; step one month."""
    month = session_date.dt.to_period("M").astype(str).to_numpy()
    pos = np.arange(len(session_date))
    folds = []
    for m in pd.period_range(first_val_month, last_val_month, freq="M").astype(str):
        val = pos[(month == m) & (label_end >= 0)]
        if len(val) == 0:
            continue
        a, b = int(pos[month == m].min()), int(pos[month == m].max())
        cand = pos[month < m]
        keep = purge_embargo_mask(cand, label_end, [(a, b)], embargo_bars)
        folds.append(Fold(m, cand[keep], val))
    return folds
