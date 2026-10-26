"""D27 rules R1 / R2 (fills at t+1, no-trade band, session limits, no dependence on future prices) and the DSR."""
import numpy as np
import pandas as pd
from scipy.stats import norm

from tslaq.backtest import positions as bp
from tslaq.backtest.dsr import dsr, psr, sr_star


def _bars(n_per_day=78, days=3, seed=0):
    rng = np.random.default_rng(seed)
    n = n_per_day * days
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    o = np.r_[100, c[:-1]] * np.exp(rng.normal(0, 2e-4, n))
    sess = np.repeat(pd.date_range("2024-01-02", periods=days, freq="B"), n_per_day).to_numpy()
    return o, c, sess


def _signals(n, seed=1):
    rng = np.random.default_rng(seed)
    return np.arange(n), rng.normal(0, 1, n), np.full(n, 1.2)


def test_r1_fill_timing_band_and_session_limits():
    O, _C, sess = _bars()
    sid, last = bp.session_arrays(sess)
    pos, z, q = _signals(len(O))
    for d in (0, 1):
        F = bp.r1_fills(sid, last, pos, z, q, d, band=0.25)
        assert len(F) > 0
        assert (F["bar"] == F["signal"] + 1 + d).all()
        assert (F["bar"] <= last[F["bar"]] - bp.NO_FILL_LAST).all()
        assert (sid[F["bar"]] == sid[F["signal"]]).all()
        assert (F["w"].abs() <= 1).all()
        # band: every fill moves the position by more than 0.25 from the previous one (flat at session start)
        for _, g in F.groupby(sid[F["bar"].to_numpy()]):
            prev = np.r_[0.0, g["w"].to_numpy()[:-1]]
            assert (np.abs(g["w"].to_numpy() - prev) > 0.25).all()
        # every signal skipped inside the allowed window was within the band of the current position
        cur, cs, fills = 0.0, None, set(F["signal"])
        for t in pos:
            if sid[t] != cs:
                cs, cur = sid[t], 0.0
            if t + 1 + d > last[t] - bp.NO_FILL_LAST:
                assert t not in fills
                continue
            w = np.clip(z[t] / q[t], -1, 1)
            assert (t in fills) == (abs(w - cur) > 0.25)
            if t in fills:
                cur = w


def test_r1_pnl_matches_hand_computation_and_flattens():
    O, C, sess = _bars(days=2)
    sid, last = bp.session_arrays(sess)
    F = pd.DataFrame({"bar": [3, 10, 80], "w": [0.5, -1.0, 1.0], "signal": [2, 9, 79]})
    R = bp.r1_pnl(O, C, sid, last, F)
    exp0 = 0.5 * np.log(O[10] / O[3]) - 1.0 * np.log(C[77] / O[10])
    exp1 = 1.0 * np.log(C[155] / O[80])
    assert np.isclose(R["gross"][:78].sum(), exp0)
    assert np.isclose(R["gross"][78:].sum(), exp1)
    assert np.isclose(R["dw"][:78].sum(), 0.5 + 1.5 + 1.0)  # in, flip, flatten at the close
    assert np.isclose(R["dw"][78:].sum(), 2.0)
    assert R["gross"][:3].sum() == 0 and R["gross"][78:80].sum() == 0  # flat until the first fill
    assert np.allclose(np.sort(R["seg_unit"]), np.sort([np.log(O[10] / O[3]), -np.log(C[77] / O[10]),
                                                        np.log(C[155] / O[80])]))


def test_r2_timing_threshold_overlap():
    O, _C, sess = _bars()
    sid, last = bp.session_arrays(sess)
    pos, z, q = _signals(len(O))
    for d in (0, 1):
        T = bp.r2_trades(O, C, sid, last, pos, z, q, d, 12)
        assert len(T) > 0
        assert (T["entry"] == T["signal"] + 1 + d).all() and (T["exit"] == T["signal"] + 12 + d).all()
        assert (np.abs(z[T["signal"]]) > q[T["signal"]]).all()
        assert (T["exit"] <= last[T["signal"]]).all() and (T["entry"] <= last[T["signal"]] - bp.NO_FILL_LAST).all()
        assert np.allclose(T["gross"], T["side"] * np.log(C[T["exit"]] / O[T["entry"]]))
        for _, g in T.groupby("sid"):
            assert (np.diff(g["signal"].to_numpy()) >= 12).all()


def test_rules_do_not_change_when_future_prices_change():
    O, _C, sess = _bars()
    sid, last = bp.session_arrays(sess)
    pos, z, q = _signals(len(O))
    k = 120
    O2, C2 = O.copy(), C.copy()
    O2[k:] *= 1.07
    C2[k:] *= 0.95
    F = bp.r1_fills(sid, last, pos, z, q, 0)
    assert F.equals(bp.r1_fills(sid, last, pos, z, q, 0))  # fills depend on signals only
    a, c = bp.r1_pnl(O, C, sid, last, F), bp.r1_pnl(O2, C2, sid, last, F)
    assert np.allclose(a["gross"][: k - 1], c["gross"][: k - 1])  # P&L before k unchanged
    T1 = bp.r2_trades(O, C, sid, last, pos, z, q, 0)
    T2 = bp.r2_trades(O2, C2, sid, last, pos, z, q, 0)
    assert T1["signal"].tolist() == T2["signal"].tolist()
    early = T1["exit"] < k
    assert np.allclose(T1.loc[early, "gross"], T2.loc[early, "gross"])


def test_dsr_equals_psr_for_one_trial_and_hand_example():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 1000)
    a = dsr(r, 1, 0.01)
    assert a["sr_star"] == 0 and np.isclose(a["dsr"], a["psr0"])
    # hand example: SR 0.1, SR* 0.05, T 500, skew -0.5, raw kurtosis 5
    den = 1 + 0.5 * 0.1 + (5 - 1) / 4 * 0.01
    assert np.isclose(psr(0.1, 0.05, 500, -0.5, 5), norm.cdf(0.05 * np.sqrt(499) / np.sqrt(den)))
    g = 0.5772156649015329
    assert np.isclose(sr_star(0.04, 10), 0.2 * ((1 - g) * norm.ppf(0.9) + g * norm.ppf(1 - 1 / (10 * np.e))))
