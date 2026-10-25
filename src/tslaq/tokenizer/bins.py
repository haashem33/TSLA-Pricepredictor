"""Quantile bins fit on training values only (CLAUDE.md 7.1, 7.2)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class QuantileBins:
    """code(x) = number of edges e_k with x >= e_k (after clipping to +-clip), e_k = training quantile k/B.

    Duplicate edges (ties) are removed. Bins with fewer than `min_count` training values are merged into
    their inner neighbour (tails first) until every bin has >= min_count. `means` = mean of the clipped
    training values in each bin (de-tokenization, 7.1). Non-finite x -> code `missing`."""
    n_bins: int
    min_count: int = 50
    clip: float | None = None
    edges: np.ndarray = field(default_factory=lambda: np.empty(0))
    means: np.ndarray = field(default_factory=lambda: np.empty(0))
    counts: np.ndarray = field(default_factory=lambda: np.empty(0))
    merged: list = field(default_factory=list)

    @property
    def missing(self) -> int:
        return self.n_bins

    def _c(self, x):
        x = np.asarray(x, float)
        return np.clip(x, -self.clip, self.clip) if self.clip else x

    def fit(self, x_train) -> QuantileBins:
        x = self._c(x_train)
        x = x[np.isfinite(x)]
        e = np.unique(np.quantile(x, np.arange(1, self.n_bins) / self.n_bins))
        self.merged = []
        while True:
            cnt = np.bincount(np.searchsorted(e, x, side="right"), minlength=len(e) + 1)
            sparse = np.flatnonzero(cnt < self.min_count)
            if not len(sparse) or not len(e):
                break
            # merge the sparse bin farthest out (tails first) into its inner neighbour
            j = sparse[np.argmax(np.abs(sparse - len(e) / 2))]
            drop = j - 1 if j > len(e) / 2 else j  # edge between bin j and its inner neighbour
            self.merged.append({"bin": int(j), "count": int(cnt[j]), "edge": float(e[drop])})
            e = np.delete(e, drop)
        self.edges, self.counts = e, cnt
        code = np.searchsorted(e, x, side="right")
        s = np.bincount(code, weights=x, minlength=len(e) + 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.means = s / cnt
        return self

    def transform(self, x) -> np.ndarray:
        x = self._c(x)
        out = np.searchsorted(self.edges, x, side="right")
        return np.where(np.isfinite(x), out, self.missing).astype(np.int16)

    def detokenize(self, code) -> np.ndarray:
        code = np.asarray(code)
        ok = code < len(self.means)
        return np.where(ok, self.means[np.minimum(code, len(self.means) - 1)], np.nan)
