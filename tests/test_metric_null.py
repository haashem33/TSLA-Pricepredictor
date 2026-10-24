"""Metric null test (D18) with the pre-registered D20 criterion 1, fast synthetic version.

A random walk with a U-shaped seasonal volatility has no predictability by construction. Two daily features
known at the open: a persistent level-like feature (previous close / 20-day mean of closes - 1, as d_sma20_dev)
and a 1/price feature (1 / previous close). Both are scored at h = 12 over monthly folds (21 sessions, anchors
from all earlier months, first 12 months training only), 100 seeds (fixed before the run; se of the mean IC
about 0.001):

- the old metric (Spearman pooled within the month, D1) must show the bias: level-like < 0, 1/price > 0;
- the D19 anchored IC v2 must not (D20 criterion 1): mean IC within +-0.003, with zero drift (level_like and
  inv_price) and with a constant drift equal to the real 2016-2025 regular-session mean of z_12 (level_like and
  inv_price in its rolling-z form, 1560 bars as in the pipeline; raw inv_price reported only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tslaq.backtest.metrics import Z_CLIP, fold_ic_stats
from tslaq.features.normalize import rolling_z
from tslaq.selection.ic import feature_scores_v2, score_anchors_v2, spearman_cols, v2_group_ic

N_MONTHS, SPM, NB, H, FIRST = 72, 21, 78, 12, 12
SEEDS = range(100)
REAL_MEAN_Z12 = 0.00192  # TSLA 2016-2025 regular-session mean of z_12 (data v2, lambda 0.97)
TOL = 0.003
RZ_WINDOW = 1560

_k = np.arange(NB)
SEAS = (0.6 + 0.8 * ((_k - 38.5) / 38.5) ** 2) * 1e-3
_cs = np.r_[0, np.cumsum(SEAS**2)]
S_H = np.sqrt(_cs[np.minimum(_k + H + 1, NB)] - _cs[_k + 1])  # exact h-bar scale
VALID = _k < NB - H  # labels never cross the close
# constant per-bar drift that gives E[z_12] = REAL_MEAN_Z12
MU_REAL = REAL_MEAN_Z12 / (H * np.mean(1 / S_H[VALID]))


def _simulate(seed: int, mu: float = 0.0):
    rng = np.random.default_rng(seed)
    nd = N_MONTHS * SPM
    lc = np.log(100.0) + np.cumsum((rng.standard_normal((nd, NB)) * SEAS + mu).ravel()).reshape(nd, NB)
    fwd = np.full((nd, NB), np.nan)
    fwd[:, VALID] = lc[:, H:] - lc[:, : NB - H]
    z = (fwd / S_H).ravel()
    day = np.repeat(np.arange(nd), NB)
    dC = np.exp(lc[:, -1])
    sma = pd.Series(dC).rolling(20).mean().to_numpy()
    level = np.r_[np.nan, (dC / sma - 1)[:-1]][day]
    inv = np.r_[np.nan, 1 / dC[:-1]][day]
    inv_rz = rolling_z(pd.Series(inv), RZ_WINDOW).to_numpy()
    return np.column_stack([level, inv, inv_rz]), z, day // SPM


FEATS = ["level_like", "inv_price", "inv_price_rz"]


def _fold_ics(mu: float):
    old, new = [], []
    for seed in SEEDS:
        X, z, month = _simulate(seed, mu)
        y = np.clip(z, -Z_CLIP, Z_CLIP)
        for M in range(FIRST, N_MONTHS):
            tr, te = month < M, month == M
            old.append(spearman_cols(X[te], z[te]))
            A = score_anchors_v2(X[tr])
            vy = np.nanmean(y[tr] ** 2)
            new.append(v2_group_ic(feature_scores_v2(X[te], A), y[te], month[te], A["vs"], vy).to_numpy()[0])
    return pd.DataFrame(old, columns=FEATS), pd.DataFrame(new, columns=FEATS)


_CACHE: dict = {}


def _ics(mu: float):
    if mu not in _CACHE:
        _CACHE[mu] = _fold_ics(mu)
    return _CACHE[mu]


def _report(tag, D):
    out = {}
    for f in FEATS:
        st = fold_ic_stats(D[f])
        out[f] = st["ic"]
        print(f"{tag} {f}: {st['ic']:+.5f} (se {D[f].std() / np.sqrt(D[f].notna().sum()):.5f}, t {st['t']:+.2f}, "
              f"{st['n_folds']} folds)")
    return out


def test_old_metric_shows_window_bias():
    old, _ = _ics(0.0)
    lv, ip = fold_ic_stats(old["level_like"]), fold_ic_stats(old["inv_price"])
    print(f"old metric: level_like {lv['ic']:+.4f} (t {lv['t']:+.1f}), inv_price {ip['ic']:+.4f} (t {ip['t']:+.1f})")
    assert lv["ic"] < 0 and lv["t"] <= -3
    assert ip["ic"] > 0 and ip["t"] >= 3


def test_v2_no_bias_zero_drift():
    _, new = _ics(0.0)
    m = _report("D19 zero drift", new)
    for f in ("level_like", "inv_price"):
        assert abs(m[f]) <= TOL, f"{f}: {m[f]:+.5f}"


def test_v2_no_bias_real_drift():
    _, new = _ics(MU_REAL)
    m = _report(f"D19 real drift (mu {MU_REAL:.2e}/bar)", new)  # raw inv_price reported only
    for f in ("level_like", "inv_price_rz"):
        assert abs(m[f]) <= TOL, f"{f}: {m[f]:+.5f}"


def test_v2_basic():
    from tslaq.backtest.metrics import fold_ic_v2_model, ic_v2

    rng = np.random.default_rng(0)
    s, y = rng.standard_normal(1000), rng.standard_normal(1000)
    assert np.isclose(ic_v2(s, s, 1.0, 1.0), np.mean(s * s))
    assert np.isnan(ic_v2(s[:5], y[:5], 1.0, 1.0, min_obs=10))
    A = score_anchors_v2(np.column_stack([rng.integers(0, 2, 500), rng.standard_normal(500)]))
    assert A["binary"][0] and 0 < A["mu"][0] < 1  # binary flag as 0/1, centred at its training mean
    sc = feature_scores_v2(np.array([[1.0, 0.0]]), A)
    assert np.isclose(sc[0, 0], 1 - A["mu"][0])
    # model IC: v_s from earlier folds only (first fold from vs_first)
    r = fold_ic_v2_model(np.r_[s, 2 * s], np.r_[s, 2 * s], np.repeat(["a", "b"], 1000), {"a": 0, "b": 0},
                         {"a": 1.0, "b": 1.0}, 1.0)
    assert np.isclose(r["a"], np.mean(np.clip(s, -5, 5) * s))
    assert np.isclose(r["b"], np.mean(np.clip(2 * s, -5, 5) * 2 * s) / np.sqrt(np.mean(s**2)))
