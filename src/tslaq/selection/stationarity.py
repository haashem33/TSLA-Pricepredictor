"""ADF / KPSS stationarity checks on training data (CLAUDE.md 5.4)."""
from __future__ import annotations

import warnings

import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss


def stationarity(x: np.ndarray, step: int = 6, max_obs: int = 30000) -> dict:
    """ADF (H0 unit root; pass if p < 0.05) and KPSS (H0 level-stationary; pass if p > 0.05) on the
    series sampled every `step` bars (30 min), last `max_obs` points. KPSS p-values are truncated to
    [0.01, 0.10] by statsmodels."""
    y = np.asarray(x, float)
    y = y[np.isfinite(y)][::step][-max_obs:]
    if len(y) < 500 or np.nanstd(y) == 0:
        return {"adf_p": np.nan, "kpss_p": np.nan, "pass": False, "n": len(y)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = float(adfuller(y, maxlag=12, autolag=None)[1])
        k = float(kpss(y, regression="c", nlags="auto")[1])
    return {"adf_p": a, "kpss_p": k, "pass": bool(a < 0.05 and k > 0.05), "n": len(y)}
