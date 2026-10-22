"""Phase 2 leakage tests (CLAUDE.md 11): test_no_lookahead_features over every Phase 2 feature, Kalman
filtered vs smoothed, D12 market alignment, daily context known at the open, fractional differencing."""
import warnings

import numpy as np
import pandas as pd
import pytest
from conftest import make_bars

from tslaq.features.build import build_features, fit_params
from tslaq.features.normalize import rolling_pct_rank, rolling_z
from tslaq.features.returns import add_returns
from tslaq.features.stats_models import fracdiff, fracdiff_weights, kalman_llt_filter
from tslaq.features.volatility import add_sigma

warnings.filterwarnings("ignore")


def _prep(b):
    return add_sigma(add_returns(b), 0.97, 60, 5, (1, 3, 12))


@pytest.fixture(scope="module")
def setup():
    t, s, q = _prep(make_bars()), _prep(make_bars(seed=1)), _prep(make_bars(seed=2))
    # two cutoffs so the yearly switch of fitted parameters is exercised
    P = [fit_params(t, "2019-05-01"), fit_params(t, "2019-07-01", seed=1)]
    react = ["2019-04-25", "2019-07-25"]
    return t, {"spy": s, "qqq": q}, P, react


def _build(t, mkt, P, react, n=None):
    if n is not None:
        t = t.iloc[:n]
        end = t["timestamp"].iloc[-1]
        mkt = {k: v[v["timestamp"] <= end] for k, v in mkt.items()}  # market data also truncated at t
    return build_features(t, mkt, P, react)


def test_no_lookahead_features(setup):
    t, mkt, P, react = setup
    full = _build(t, mkt, P, react)
    assert full.shape[1] >= 80
    rng = np.random.default_rng(7)
    for i in rng.integers(70 * 78, len(t) - 1, 50):
        tr = _build(t, mkt, P, react, i + 1)
        a, b = full.iloc[: i + 1], tr
        for c in full.columns:
            x, y = a[c].to_numpy(), b[c].to_numpy()
            np.testing.assert_array_equal(np.isnan(x), np.isnan(y), err_msg=f"{c} at t={i}")
            m = ~np.isnan(x)
            np.testing.assert_allclose(x[m], y[m], rtol=1e-9, atol=1e-12, err_msg=f"{c} at t={i}")


def test_features_unchanged_by_future_perturbation(setup):
    t, mkt, P, react = setup
    full = _build(t, mkt, P, react)
    i = 120 * 78 + 30
    t2 = t.copy()
    for c in ["open", "high", "low", "close"]:
        t2.loc[i + 1:, c] *= 1.05
    t2.loc[i + 1:, "volume"] *= 3
    pert = _build(_prep(t2[["timestamp", "session_date", "bar_index", "is_half_day", "open", "high", "low",
                            "close", "volume", "vwap", "missing"]]), mkt, P, react)
    pd.testing.assert_frame_equal(full.iloc[: i + 1], pert.iloc[: i + 1], check_exact=False, rtol=1e-9, atol=1e-12)


def test_kalman_filtered_not_smoothed():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0, 0.01, 3000)) + np.cumsum(np.cumsum(rng.normal(0, 1e-4, 3000)))
    kp = {"params": [1e-5, 1e-4, 1e-8]}
    s_full, i_full = kalman_llt_filter(y, kp)
    for t in rng.integers(100, 2999, 20):
        s, i = kalman_llt_filter(y[: t + 1], kp)
        np.testing.assert_allclose(s_full[: t + 1], s, rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(i_full[: t + 1], i, rtol=1e-10, atol=1e-12, equal_nan=True)


def test_market_missing_session_D12(setup):
    """Dropping a QQQ session keeps the TSLA rows; QQQ features are MISSING (NaN) on that session, on the
    next session's bar 0 return and its first 12 bars of momentum."""
    t, mkt, P, react = setup
    sessions = t["session_date"].unique()
    gone, nxt = sessions[150], sessions[151]
    q2 = mkt["qqq"][mkt["qqq"]["session_date"] != gone]
    q2 = _prep(q2[["timestamp", "session_date", "bar_index", "is_half_day", "open", "high", "low", "close",
                   "volume", "vwap", "missing"]].reset_index(drop=True))
    f = build_features(t, {"spy": mkt["spy"], "qqq": q2}, P, react)
    assert len(f) == len(t)
    on = (t["session_date"] == gone).to_numpy()
    for c in ["qqq_z", "qqq_mom12", "qqq_relvol"]:
        assert f.loc[on, c].isna().all(), c
    after = (t["session_date"] == nxt).to_numpy()
    k = t["bar_index"].to_numpy()
    assert f.loc[after & (k == 0), "qqq_z"].isna().all()
    assert f.loc[after & (k < 12), "qqq_mom12"].isna().all()
    assert f.loc[after & (k >= 12), "qqq_mom12"].notna().all()
    assert f.loc[on, "spy_z"].notna().sum() > 70  # SPY unaffected
    assert f.loc[on, "body"].notna().all()       # TSLA rows kept


def test_daily_context_known_at_open(setup):
    t, mkt, P, react = setup
    f = _build(t, mkt, P, react)
    cols = ["d_prev_ret_atr", "d_atr_c", "d_sma20_dev", "d_mom5", "har_rv1", "har_rv5", "har_rv22"]
    per_sess = f[cols].groupby(t["session_date"]).nunique(dropna=False)
    assert (per_sess <= 1).all().all()  # constant within the session
    sess = t["session_date"].unique()
    d = sess[100]
    prev = t[t["session_date"] == sess[99]]
    assert f.loc[(t["session_date"] == d).to_numpy(), "har_rv1"].iloc[0] == pytest.approx(np.log((prev["r"] ** 2).sum()))


def test_rolling_helpers_past_only():
    x = pd.Series(np.random.default_rng(1).normal(size=500))
    z = rolling_z(x, 50)
    t = 300
    past = x.iloc[t - 50: t]
    assert z.iat[t] == pytest.approx((x.iat[t] - past.mean()) / past.std())
    x2 = x.copy()
    x2.iloc[t + 1:] += 100
    np.testing.assert_array_equal(rolling_z(x2, 50).iloc[: t + 1], z.iloc[: t + 1])
    np.testing.assert_array_equal(rolling_pct_rank(x2, 50).iloc[: t + 1], rolling_pct_rank(x, 50).iloc[: t + 1])


def test_fracdiff():
    w = fracdiff_weights(1.0)
    np.testing.assert_allclose(w, [1.0, -1.0])
    x = np.cumsum(np.random.default_rng(2).normal(size=3000))
    np.testing.assert_allclose(fracdiff(x, 1.0)[1:], np.diff(x))
    w = fracdiff_weights(0.4)
    assert abs(w[-1]) >= 1e-4 and len(w) > 10
    y = fracdiff(x, 0.4)
    t = 2000
    assert y[t] == pytest.approx(np.dot(w, x[t - np.arange(len(w))]))
    x2 = x.copy()
    x2[t + 1:] = 0
    np.testing.assert_array_equal(fracdiff(x2, 0.4)[: t + 1], y[: t + 1])


def test_numba_rolling_stats_match_reference():
    from tslaq.features.stats_models import (
        hurst_rs,
        hurst_rs_reference,
        rolling_entropy,
        rolling_entropy_reference,
    )

    rng = np.random.default_rng(5)
    x = rng.standard_t(4, 6000)
    x[1000:1003] = np.nan
    np.testing.assert_allclose(hurst_rs(x), hurst_rs_reference(x), rtol=1e-10, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(rolling_entropy(x), rolling_entropy_reference(x), rtol=1e-12, atol=1e-12, equal_nan=True)


def _llt_decimal(y, s_irr, s_lvl, s_trd):
    """50-digit reference of the same recursion (exact diffuse start)."""
    from decimal import Decimal as D
    from decimal import getcontext

    getcontext().prec = 50
    si, sv, st = D(s_irr), D(s_lvl), D(s_trd)
    a0, a1 = D(y[1]), D(y[1]) - D(y[0])
    q00, q01, q11 = si, si, 2 * si + sv + st
    out = {}
    a0 = a0 + a1
    p00, p01, p11 = q00 + 2 * q01 + q11 + sv, q01 + q11, q11 + st
    for t in range(2, len(y)):
        v = D(y[t]) - a0
        F = p00 + si
        k0, k1 = p00 / F, p01 / F
        a0 += k0 * v
        a1 += k1 * v
        q00, q01, q11 = p00 - k0 * p00, p01 - k0 * p01, p11 - k1 * p01
        out[t] = (float(a1), float(v / F.sqrt()))
        a0 = a0 + a1
        p00, p01, p11 = q00 + 2 * q01 + q11 + sv, q01 + q11, q11 + st
    return out


def _sim_llt(n, s_irr, s_lvl, s_trd, seed):
    rng = np.random.default_rng(seed)
    slope = np.cumsum(rng.normal(0, np.sqrt(s_trd), n))
    lvl = np.cumsum(slope + rng.normal(0, np.sqrt(s_lvl), n))
    return 5 + lvl + rng.normal(0, np.sqrt(s_irr), n)


def test_kalman_numba_matches_reference_and_statsmodels():
    """The numba filter equals a 50-digit reference of the local-linear-trend recursion (including the
    ill-conditioned case seen on intraday ln C, where statsmodels' float filter drifts), agrees with
    statsmodels UnobservedComponents('lltrend') on well-conditioned parameters, and the MLE recovers the
    simulated variances."""
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    from tslaq.features.stats_models import fit_kalman_llt

    p_ill = [1e-6, 4e-6, 1e-10]
    y = _sim_llt(1500, *p_ill, 6)
    sl, inn = kalman_llt_filter(y, {"params": p_ill})
    ref = _llt_decimal(y, *p_ill)
    t = np.arange(2, len(y))
    np.testing.assert_allclose(sl[t], [ref[i][0] for i in t], rtol=1e-8, atol=1e-15)
    np.testing.assert_allclose(inn[t], [ref[i][1] for i in t], rtol=1e-8, atol=1e-10)
    p_ok = [1e-4, 4e-4, 1e-6]
    y = _sim_llt(5000, *p_ok, 7)
    sl, inn = kalman_llt_filter(y, {"params": p_ok})
    res = UnobservedComponents(y, level="lltrend").filter(np.array(p_ok))
    # statsmodels freezes P once it reaches steady state, so it differs from the exact recursion at ~1e-5
    np.testing.assert_allclose(sl[10:], res.filtered_state[1][10:], rtol=1e-4, atol=1e-8)
    np.testing.assert_allclose(inn[10:], res.standardized_forecasts_error[0][10:], rtol=1e-4, atol=1e-4)
    y = _sim_llt(20000, *p_ill, 8)
    fit = fit_kalman_llt(y)
    np.testing.assert_allclose(fit["params"][:2], p_ill[:2], rtol=0.15)
