"""Feature clustering by |Spearman correlation| (CLAUDE.md 6.8 step 3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def spearman_matrix(F: pd.DataFrame, min_periods: int = 1000) -> pd.DataFrame:
    R = F.rank()  # NaN stays NaN; pairwise-complete Pearson on ranks ~ Spearman
    C = R.corr(min_periods=min_periods)
    return C.fillna(0.0)


def cluster_features(C: pd.DataFrame, threshold: float = 0.7) -> pd.Series:
    """Average-linkage hierarchical clustering on distance 1 - |rho|; features whose clusters are
    joined at |rho| >= threshold end up together. Returns cluster id per feature."""
    D = 1.0 - C.abs().to_numpy()
    np.fill_diagonal(D, 0.0)
    D = np.clip((D + D.T) / 2, 0, None)
    Z = linkage(squareform(D, checks=False), method="average")
    return pd.Series(fcluster(Z, t=1.0 - threshold, criterion="distance"), index=C.index, name="cluster")
