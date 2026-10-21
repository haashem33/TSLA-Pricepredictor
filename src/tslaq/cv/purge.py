"""Purging and embargo (CLAUDE.md 10.1)."""
from __future__ import annotations

import numpy as np


def purge_embargo_mask(train_idx: np.ndarray, label_end: np.ndarray, val_blocks: list[tuple[int, int]],
                       embargo_bars: int) -> np.ndarray:
    """Boolean mask over `train_idx` of samples to KEEP.

    A training sample t with label window [t, label_end[t]] is purged if that window overlaps any
    validation block [a, b] (row positions, inclusive). Samples in (b, b + embargo_bars] after each
    validation block are dropped (embargo). Samples without a label (label_end < 0) are dropped.
    """
    t = np.asarray(train_idx)
    end = label_end[t]
    keep = end >= 0
    for a, b in val_blocks:
        keep &= ~((t <= b) & (end >= a))           # overlap of [t, end] with [a, b]
        keep &= ~((t > b) & (t <= b + embargo_bars))  # embargo after the block
    return keep
