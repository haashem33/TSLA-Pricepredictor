"""Metric null test (D18), fast synthetic version.

A random walk with a U-shaped seasonal volatility has no predictability by construction. Two daily features
known at the open: a persistent level-like feature (previous close / 20-day mean of closes - 1, as d_sma20_dev)
and a 1/price feature (1 / previous close). Both are scored at h = 12 over monthly folds (21 sessions, anchors
from all earlier months, first 12 months training only):

- the old metric (Spearman pooled within the month, D1) must show the bias: level-like < 0, 1/price > 0;
- the D17 anchored IC must not: mean within +-0.005 of 0, or |t| < 2 over the pooled folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tslaq.backtest.metrics import Z_CLIP, fold_ic_stats
from tslaq.selection.ic import anchored_group_ic, feature_scores, score_anchors, spearman_cols

N_MONTHS, SPM, NB, H, FIRST = 72, 21, 78, 12, 12
SEEDS = range(6)
FEATS = ["level_like", "inv_price"]


def _simulate(seed: int):
    rng = np.random.default_rng(seed)
    k = np.arange(NB)
    seas = (0.6 + 0.8 * ((k - 38.5) / 38.5) ** 2) * 1e-3
    nd = N_MONTHS * SPM
    lc = (np.log(100.0) + np.cumsum(rng.standard_normal((nd, NB)) * seas)).reshape(nd, NB)
    fwd = np.full((nd, NB), np.nan)
    fwd[:, : NB - H] = lc[:, H:] - lc[:, : NB - H]  # never crosses the close
    cs = np.r_[0, np.cumsum(seas**2)]
    S = np.sqrt(cs[np.minimum(k + H + 1, NB)] - cs[k + 1])  # exact h-bar scale
    z = (fwd / S).ravel()
    day = np.repeat(np.arange(nd), NB)
    dC = np.exp(lc[:, -1])
    sma = pd.Series(dC).rolling(20).mean().to_numpy()
    level = np.r_[np.nan, (dC / sma - 1)[:-1]]
    inv = np.r_[np.nan, 1 / dC[:-1]]
    return np.column_stack([level[day], inv[day]]), z, day // SPM


def _fold_ics():
    old, new = [], []
    for seed in SEEDS:
        X, z, month = _simulate(seed)
        for M in range(FIRST, N_MONTHS):
            tr, te = month < M, month == M
            old.append(spearman_cols(X[te], z[te]))
            centre, scale = score_anchors(X[tr])
            y = np.clip(z[te], -Z_CLIP, Z_CLIP) - np.nanmean(np.clip(z[tr], -Z_CLIP, Z_CLIP))
            new.append(anchored_group_ic(feature_scores(X[te], centre, scale), y, month[te]).to_numpy()[0])
    return pd.DataFrame(old, columns=FEATS), pd.DataFrame(new, columns=FEATS)


_CACHE: dict = {}


def _ics():
    if not _CACHE:
        _CACHE["old"], _CACHE["new"] = _fold_ics()
    return _CACHE["old"], _CACHE["new"]


def test_old_metric_shows_window_bias():
    old, _ = _ics()
    lv, ip = fold_ic_stats(old["level_like"]), fold_ic_stats(old["inv_price"])
    print(f"old metric: level_like {lv['ic']:+.4f} (t {lv['t']:+.1f}), inv_price {ip['ic']:+.4f} (t {ip['t']:+.1f})")
    assert lv["ic"] < 0 and lv["t"] <= -3
    assert ip["ic"] > 0 and ip["t"] >= 3


def test_anchored_ic_no_bias_on_null():
    _, new = _ics()
    for f in FEATS:
        st = fold_ic_stats(new[f])
        print(f"anchored IC: {f} {st['ic']:+.4f} (t {st['t']:+.1f}, {st['n_folds']} folds)")
        assert abs(st["ic"]) <= 0.005 or abs(st["t"]) < 2, f"{f}: anchored IC {st['ic']:+.4f}, t {st['t']:+.1f}"


def test_anchored_ic_basic():
    from tslaq.backtest.metrics import anchored_ic

    rng = np.random.default_rng(0)
    s, y = rng.standard_normal(1000), rng.standard_normal(1000)
    assert np.isclose(anchored_ic(s, s), 1.0)
    assert np.isclose(anchored_ic(s, -2 * s), -1.0)
    # no centring: adding a constant to the score changes the value (unlike a correlation)
    assert not np.isclose(anchored_ic(s + 3, y), np.corrcoef(s + 3, y)[0, 1])
    assert np.isnan(anchored_ic(np.zeros(10), y[:10]))
    centre, scale = score_anchors(np.column_stack([rng.integers(0, 2, 500), rng.standard_normal(500)]))
    assert scale[0] == 1.0 and 0 < centre[0] < 1  # binary flag: x - training mean
