"""Probabilistic and deflated Sharpe ratio (CLAUDE.md 10.7; Bailey & Lopez de Prado). Per-period Sharpe."""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, norm, skew

EULER_GAMMA = 0.5772156649015329


def psr(sr: float, sr_star: float, T: int, g3: float, g4: float) -> float:
    """P(true SR > sr_star); g3 = skewness, g4 = raw (non-excess) kurtosis of the returns."""
    den = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2
    return float(norm.cdf((sr - sr_star) * np.sqrt(T - 1) / np.sqrt(den)))


def sr_star(V: float, N: int) -> float:
    """Expected maximum Sharpe of N trials with Sharpe variance V under H0 (0 when N = 1)."""
    if N <= 1:
        return 0.0
    g = EULER_GAMMA
    return float(np.sqrt(V) * ((1 - g) * norm.ppf(1 - 1 / N) + g * norm.ppf(1 - 1 / (N * np.e))))


def dsr(returns: np.ndarray, N: int, V: float) -> dict:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    sr = float(r.mean() / r.std(ddof=1))
    g3, g4 = float(skew(r)), float(kurtosis(r, fisher=False))
    s0 = sr_star(V, N)
    return {"sr": sr, "T": len(r), "skew": g3, "kurt": g4, "N": N, "V": V, "sr_star": s0,
            "dsr": psr(sr, s0, len(r), g3, g4), "psr0": psr(sr, 0.0, len(r), g3, g4)}
