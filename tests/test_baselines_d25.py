"""D25 information-only rule backtest: fills at the next bar's open, non-overlapping trades, session limits."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _bars(n_per_day=78, days=2, seed=0):
    rng = np.random.default_rng(seed)
    n = n_per_day * days
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    o = np.r_[100, c[:-1]] * np.exp(rng.normal(0, 2e-4, n))
    sess = np.repeat(pd.date_range("2024-01-02", periods=days, freq="B"), n_per_day)
    return pd.DataFrame({"open": o, "close": c, "session_date": sess, "bar_index": np.tile(np.arange(n_per_day), days)})


def test_rule_trades_timing_and_overlap():
    import run_phase3_baselines as rb
    L = _bars()
    _, last = rb.session_arrays(L)
    pos = np.arange(len(L))
    pred = np.where(pos % 2 == 0, 1.0, -1.0)
    for h in (1, 3, 12):
        for d in (0, 1):
            T = rb.rule_trades(L, last, pos, pred, h, d)
            t = T["pos"].to_numpy()
            e, x = t + 1 + d, t + h + d
            exp = T["side"] * np.log(L["close"].to_numpy()[x] / L["open"].to_numpy()[e])
            assert np.allclose(T["gross"], exp)
            assert (x <= last[t]).all() and (e <= last[t] - 2).all()
            for _, g in T.groupby("session"):
                assert (np.diff(g["pos"].to_numpy()) >= h).all()
    # a signal at t never depends on prices after t: changing future prices changes P&L, not the trade list
    L2 = L.copy()
    L2.loc[100:, ["open", "close"]] *= 1.1
    a = rb.rule_trades(L, last, pos, pred, 12, 0)
    c = rb.rule_trades(L2, last, pos, pred, 12, 0)
    assert a["pos"].tolist() == c["pos"].tolist()
