"""test_no_lookahead_features / test_horizon_scale / test_garch_fixed_params (CLAUDE.md 11)."""
import numpy as np
import pandas as pd
import pytest

from tslaq.features.returns import add_returns
from tslaq.features.stats_models import fit_garch_t, garch_filter
from tslaq.features.volatility import add_sigma
from tslaq.utils.io import interim_path

FEATURES = ["r", "r_intraday", "gap", "seasonal", "u", "level2", "level2_next", "sigma_hat", "S_1", "S_3", "S_12"]


def _features(df):
    return add_sigma(add_returns(df), lam=0.97, window=60, burn_in_sessions=5)


def _assert_prefix_equal(full, trunc, cols):
    for c in cols:
        a, b = full[c].to_numpy()[: len(trunc)], trunc[c].to_numpy()
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b), err_msg=c)
        m = ~np.isnan(a)
        np.testing.assert_allclose(a[m], b[m], rtol=0, atol=1e-15, err_msg=c)


def test_no_lookahead_features(bars):
    full = _features(bars)
    rng = np.random.default_rng(1)
    ts = rng.integers(60 * 78, len(bars) - 1, 50)  # 50 random t after warm-up
    for t in ts:
        _assert_prefix_equal(full, _features(bars.iloc[: t + 1]), FEATURES)


def test_features_change_only_after_perturbation(bars):
    """Perturbing bar t+1 must not change any feature at bars <= t."""
    full = _features(bars)
    t = 100 * 78 + 30
    b2 = bars.copy()
    b2.loc[t + 1:, ["open", "high", "low", "close"]] *= 1.07
    pert = _features(b2)
    for c in FEATURES:
        np.testing.assert_array_equal(full[c].to_numpy()[: t + 1], pert[c].to_numpy()[: t + 1], err_msg=c)


def test_horizon_scale(bars):
    df = _features(bars)
    prof = df.pivot(index="session_date", columns="bar_index", values="seasonal")
    rng = np.random.default_rng(2)
    checked = 0
    for i in rng.choice(np.flatnonzero(np.isfinite(df["S_3"].to_numpy())), 200, replace=False):
        row = df.iloc[i]
        for h in (1, 3, 12):
            k = int(row["bar_index"])
            n_bars = 42 if row["is_half_day"] else 78
            if k + h >= n_bars:
                assert np.isnan(row[f"S_{h}"])
                continue
            s2 = np.sum(prof.loc[row["session_date"], k + 1: k + h].to_numpy() ** 2)
            expect = np.sqrt(row["level2_next"]) * np.sqrt(s2)
            if np.isfinite(expect):
                assert row[f"S_{h}"] == pytest.approx(expect, rel=1e-12)
                checked += 1
    assert checked > 300
    # level2_next at t uses u_t (known at close of t) but nothing after t
    t = np.flatnonzero(np.isfinite(df["level2"].to_numpy()) & np.isfinite(df["u"].to_numpy()))[500]
    lam = 0.97
    assert df["level2_next"].iat[t] == pytest.approx(lam * df["level2"].iat[t] + (1 - lam) * df["u"].iat[t] ** 2)


def test_seasonal_uses_past_sessions_only(bars):
    df = _features(bars)
    sessions = df["session_date"].unique()
    d = sessions[100]
    k = 10
    past = df[(df["session_date"] < d) & (df["session_date"] >= sessions[40]) & (df["bar_index"] == k)
              & ~df["missing"]]
    expect = past["r"].abs().tail(60).median()
    got = df[(df["session_date"] == d) & (df["bar_index"] == k)]["seasonal"].iloc[0]
    assert got == pytest.approx(expect)


def test_garch_fixed_params(bars):
    df = _features(bars)
    u = df["u"].to_numpy()
    n_train = len(u) // 2
    params = fit_garch_t(u[:n_train])
    assert 0 < params["alpha"] + params["beta"] < 1
    var_full, nxt_full = garch_filter(u, params)
    rng = np.random.default_rng(3)
    for t in rng.integers(n_train, len(u) - 1, 20):
        var_t, nxt_t = garch_filter(u[: t + 1], params)
        np.testing.assert_array_equal(var_full[: t + 1], var_t)
        np.testing.assert_array_equal(nxt_full[: t + 1], nxt_t)


@pytest.mark.skipif(not interim_path("TSLA", "v2").exists(), reason="interim TSLA bars not built")
def test_no_lookahead_features_real_data():
    df = pd.read_parquet(interim_path("TSLA", "v2"))
    df = df[(df["session_date"] >= "2020-01-01") & (df["session_date"] < "2021-01-01")].reset_index(drop=True)
    full = _features(df)
    rng = np.random.default_rng(4)
    for t in rng.integers(70 * 78, len(df) - 1, 10):
        _assert_prefix_equal(full, _features(df.iloc[: t + 1]), FEATURES)
