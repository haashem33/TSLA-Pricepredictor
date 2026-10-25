"""SHAPE field: k-means of the candle-shape vector, fit on training rows only (CLAUDE.md 7.2)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans

CLIP = 5.0


@dataclass
class ShapeKMeans:
    """Standardize each column with training mean / sd, clip to +-5, k-means (k clusters, `n_init` starts,
    fixed seed) on at most `max_rows` training rows drawn with that seed. Rows with any non-finite input ->
    code `missing` (= k)."""
    k: int = 32
    n_init: int = 4
    max_rows: int = 100_000
    seed: int = 0
    mean: np.ndarray = field(default_factory=lambda: np.empty(0))
    sd: np.ndarray = field(default_factory=lambda: np.empty(0))
    centers: np.ndarray = field(default_factory=lambda: np.empty(0))
    inertia: float = np.nan

    @property
    def missing(self) -> int:
        return self.k

    def _std(self, X):
        return np.clip((np.asarray(X, float) - self.mean) / self.sd, -CLIP, CLIP)

    def fit(self, X_train) -> ShapeKMeans:
        X = np.asarray(X_train, float)
        X = X[np.isfinite(X).all(axis=1)]
        self.mean, self.sd = X.mean(axis=0), X.std(axis=0)
        rng = np.random.default_rng(self.seed)
        if len(X) > self.max_rows:
            X = X[np.sort(rng.choice(len(X), self.max_rows, replace=False))]
        km = KMeans(self.k, n_init=self.n_init, random_state=self.seed).fit(self._std(X))
        # order clusters by their mean standardized body (bearish -> bullish), so codes are comparable across folds
        order = np.argsort(km.cluster_centers_[:, 0], kind="stable")
        self.centers, self.inertia = km.cluster_centers_[order], float(km.inertia_)
        return self

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X, float)
        ok = np.isfinite(X).all(axis=1)
        out = np.full(len(X), self.missing, np.int16)
        if ok.any():
            S = self._std(X[ok])
            d = (S**2).sum(1)[:, None] - 2 * S @ self.centers.T + (self.centers**2).sum(1)[None, :]
            out[ok] = np.argmin(d, axis=1)
        return out
