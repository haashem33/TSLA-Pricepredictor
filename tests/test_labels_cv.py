"""test_labels_alignment / test_purge_embargo (CLAUDE.md 11)."""
import numpy as np
import pandas as pd

from tslaq.cv.purge import purge_embargo_mask
from tslaq.cv.walk_forward import monthly_folds
from tslaq.features.returns import add_returns
from tslaq.labels.forward_returns import add_labels


def _labels(bars):
    return add_labels(add_returns(bars), (1, 3, 12))


def test_labels_alignment(bars):
    df = _labels(bars)
    c = df["close"].to_numpy()
    sess = df["session_date"].to_numpy()
    r = df["r"].to_numpy()
    for h in (1, 3, 12):
        f = df[f"fwd_{h}"].to_numpy()
        for t in np.random.default_rng(h).integers(0, len(df) - h, 500):
            if sess[t + h] == sess[t]:
                assert np.isclose(f[t], np.log(c[t + h] / c[t]))
                assert np.isclose(df[f"rv_{h}"].iat[t], np.log(np.sum(r[t + 1: t + h + 1] ** 2) + 1e-12))
                assert df[f"label_end_{h}"].iat[t] == t + h
            else:
                assert np.isnan(f[t]) and df[f"label_end_{h}"].iat[t] == -1
        # never crosses the close: last h bars of each session are unlabeled
        n_lab = df.groupby("session_date")[f"fwd_{h}"].apply(lambda s: s.notna().sum())
        n_bar = df.groupby("session_date").size()
        assert ((n_bar - n_lab) == h).all()


def test_labels_use_only_t_to_t_plus_h(bars):
    df = _labels(bars)
    t, h = 80 * 78 + 20, 3
    b2 = bars.copy()
    others = np.r_[np.arange(0, t), np.arange(t + h + 1, len(b2))]
    b2.loc[others, ["open", "high", "low", "close"]] *= np.exp(np.random.default_rng(0).normal(0, .01, len(others)))[:, None]
    d2 = _labels(b2)
    # fwd uses C_t and C_{t+h}; rv uses r_{t+1..t+h}, where r_{t+1} = ln(C_{t+1}/C_t): both only touch t..t+h
    assert d2["fwd_3"].iat[t] == df["fwd_3"].iat[t]
    assert np.isclose(d2["rv_3"].iat[t], df["rv_3"].iat[t])


def test_purge_embargo_generic():
    n = 1000
    h = 12
    label_end = np.arange(n) + h
    val = (400, 499)
    train = np.r_[np.arange(0, 400), np.arange(500, n)]
    keep = purge_embargo_mask(train, label_end, [val], embargo_bars=78)
    kept = train[keep]
    assert not np.any((kept <= val[1]) & (label_end[kept] >= val[0])), "label window overlaps validation"
    assert not np.any((kept > val[1]) & (kept <= val[1] + 78)), "embargo rows present"
    assert set(range(400 - h, 400)).isdisjoint(kept) and 400 - h - 1 in kept
    assert 499 + 79 in kept


def test_purge_embargo_walk_forward(bars):
    df = _labels(bars)
    for h in (1, 3, 12):
        le = df[f"label_end_{h}"].to_numpy()
        folds = monthly_folds(df["session_date"], le, "2019-04", "2019-09", embargo_bars=78)
        assert len(folds) == 6
        for f in folds:
            a = f.val_idx.min()
            assert f.train_idx.max() < a
            assert np.all(le[f.train_idx] < a) and np.all(le[f.train_idx] >= 0)
            assert np.all(le[f.val_idx] >= 0)
            m = pd.Series(df["session_date"].to_numpy()[f.val_idx]).dt.to_period("M").astype(str)
            assert (m == f.val_month).all()
