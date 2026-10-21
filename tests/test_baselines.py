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
