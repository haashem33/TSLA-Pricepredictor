"""Baseline feature causality and reproducibility."""
import numpy as np

from tslaq.features.returns import add_returns
from tslaq.features.volatility import add_sigma
from tslaq.models.baselines import fit_predict_lgb, simple_return_features


def _X(bars):
    return simple_return_features(add_sigma(add_returns(bars), 0.97, 60, 5))


def test_simple_return_features_no_lookahead(bars):
    full = _X(bars)
    for t in np.random.default_rng(5).integers(70 * 78, len(bars) - 1, 20):
        trunc = _X(bars.iloc[: t + 1])
        np.testing.assert_array_equal(full.to_numpy()[: t + 1], trunc.to_numpy())


def test_lgb_reproducible(bars):
    X = _X(bars).dropna()
    y = np.random.default_rng(0).normal(size=len(X))
    n = len(X) // 2
    p1 = fit_predict_lgb(X.iloc[:n], y[:n], X.iloc[n:])[0]
    p2 = fit_predict_lgb(X.iloc[:n], y[:n], X.iloc[n:])[0]
    np.testing.assert_array_equal(p1, p2)


def test_ic_metric_bias_placebo():
    """On a random walk (no predictability) the per-day IC of a momentum signal is strongly negative;
    the per-slot and per-fold IC are not. Guards the choice of the primary IC metric."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from conftest import make_bars

    from tslaq.backtest.metrics import fold_ic, ic_summary, per_slot_ic
    from tslaq.labels.forward_returns import add_labels

    d = add_labels(add_sigma(add_returns(make_bars("2016-01-01", "2018-12-31", seed=7)), 0.97, 60, 5))
    p = simple_return_features(d)["zcum_78"].to_numpy()
    y = d["fwd_12"].to_numpy()
    assert ic_summary(p, y, d["session_date"].to_numpy())["rank_ic"] < -0.2
    assert abs(per_slot_ic(p, y, d["bar_index"].to_numpy())) < 0.05
    month = d["session_date"].dt.to_period("M").astype(str).to_numpy()
    assert abs(fold_ic(p, y, month).mean()) < 0.05


def test_fold_ic_stats():
    import pandas as pd

    from tslaq.backtest.metrics import fold_ic_stats

    s = fold_ic_stats(pd.Series([0.01, 0.02, 0.03, -0.01]))
    v = np.array([0.01, 0.02, 0.03, -0.01])
    assert np.isclose(s["icir"], v.mean() / v.std(ddof=1))
    assert np.isclose(s["t"], s["icir"] * 2)
    assert s["pos_share"] == 0.75 and np.isclose(s["binom_p"], 5 / 16)
