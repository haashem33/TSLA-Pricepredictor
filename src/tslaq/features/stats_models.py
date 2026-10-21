"""Statistical-model features. Phase 1: GARCH(1,1)-t on de-seasonalized returns (CLAUDE.md 6.4, 5.6).

Fit on training data with `arch`, then run the variance recursion forward manually with the trained
parameters held fixed, so sigma_t uses only u_{<t} and the trained parameters.
"""
from __future__ import annotations

import warnings

import numpy as np
from arch import arch_model


def fit_garch_t(u_train: np.ndarray) -> dict:
    u = u_train[np.isfinite(u_train)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(u, mean="Zero", vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off")
    p = res.params
    return {"omega": float(p["omega"]), "alpha": float(p["alpha[1]"]), "beta": float(p["beta[1]"]),
            "nu": float(p["nu"]), "train_var": float(np.mean(u**2))}


def garch_filter(u: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Returns (var_t, var_next): var_t = sigma^2_{t|t-1} from u_{<t}; var_next = sigma^2_{t+1|t}.

    Non-finite u (warm-up, missing bars) do not update the state. The state is initialised at the
    training-sample variance, which does not depend on data after t.
    """
    w, a, b = params["omega"], params["alpha"], params["beta"]
    n = len(u)
    var = np.full(n, np.nan)
    nxt = np.full(n, np.nan)
    cur = np.nan
    for t in range(n):
        ok = np.isfinite(u[t])
        if np.isnan(cur) and ok:
            cur = params["train_var"]
        var[t] = cur
        if ok and not np.isnan(cur):
            cur = w + a * u[t] * u[t] + b * cur
        nxt[t] = cur
    return var, nxt


def garch_multi_step(var_next: np.ndarray, params: dict, h: int) -> np.ndarray:
    """E[sigma^2_{t+i} | F_t] for i = 1..h, shape (n, h)."""
    a, b, w = params["alpha"], params["beta"], params["omega"]
    pers = a + b
    lr = w / (1 - pers) if pers < 1 else params["train_var"]
    i = np.arange(h)
    return lr + (pers ** i)[None, :] * (var_next[:, None] - lr)
