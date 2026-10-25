"""Tokenizer tests (CLAUDE.md 7.4, 11): test_no_lookahead_tokens, test_tokenizer_roundtrip, bin occupancy and
sparse-bin merging, REL rules, lag-matrix causality."""
import warnings

import numpy as np
import pandas as pd
import pytest
from conftest import make_bars

from tslaq.features.regime import filtered_probs, fit_hmm, hmm_obs
from tslaq.features.returns import add_returns
from tslaq.features.volatility import add_sigma
from tslaq.labels.forward_returns import add_labels
from tslaq.selection.rescore import free_features
from tslaq.tokenizer.bins import QuantileBins
from tslaq.tokenizer.fields import FIELDS, NO_GAP, rel_codes, special, token_inputs
from tslaq.tokenizer.tokenizer import Tokenizer, lag_matrix

warnings.filterwarnings("ignore")
CUTOFF = "2019-06-01"
REACT = ["2019-04-25", "2019-07-25"]


def _prep(b):
    return add_sigma(add_returns(b), 0.97, 60, 5, (1, 3, 12))


def _hmm_state(L, model):
    obs = hmm_obs(L["u"].to_numpy(), L["sigma_hat"].to_numpy())
    obs[L["missing"].to_numpy()] = np.nan
    p = filtered_probs(obs, model)
    return np.where(np.isfinite(p).all(axis=1), np.nanargmax(np.nan_to_num(p, nan=-1), axis=1), np.nan)


def _inputs(t, mkt, model):
    L = _prep(t)
    F, _ = free_features(L, {k: _prep(v) for k, v in mkt.items()}, REACT)
    return L, token_inputs(L, F, _hmm_state(L, model))


@pytest.fixture(scope="module")
def setup():
    t, mkt = make_bars(), {"spy": make_bars(seed=1), "qqq": make_bars(seed=2)}
    L0 = _prep(t)
    tr = (L0["session_date"] < pd.Timestamp(CUTOFF)).to_numpy()
    obs = hmm_obs(L0["u"].to_numpy(), L0["sigma_hat"].to_numpy())
    obs[L0["missing"].to_numpy()] = np.nan
    model, _ = fit_hmm(obs[tr], K=3, n_starts=5, seed=0)
    L, X = _inputs(t, mkt, model)
    L = add_labels(L, (1, 3, 12))
    tok = Tokenizer().fit(X, tr, {h: L[f"z_{h}"].to_numpy() for h in (1, 3, 12)}, CUTOFF)
    return t, mkt, model, L, X, tok, tr


def test_no_lookahead_tokens(setup):
    """Tokens (and their 24-bar lag matrix) at bars <= t are identical when every input is recomputed on
    data truncated at t, with bin edges / k-means / HMM fit once on a fixed training period."""
    t, mkt, model, L, X, tok, tr = setup
    full = tok.transform(X)
    LM, _ = lag_matrix(full)
    rng = np.random.default_rng(3)
    for n in rng.integers(tr.sum(), len(t) - 1, 50):
        tb = t.iloc[: n + 1]
        end = tb["timestamp"].iloc[-1]
        _, Xn = _inputs(tb, {k: v[v["timestamp"] <= end] for k, v in mkt.items()}, model)
        Tn = tok.transform(Xn)
        pd.testing.assert_frame_equal(full.iloc[: n + 1], Tn)
        np.testing.assert_array_equal(LM[: n + 1], lag_matrix(Tn)[0])


def test_tokens_unchanged_by_future_perturbation(setup):
    t, mkt, model, L, X, tok, tr = setup
    n = int(tr.sum()) + 500
    t2 = t.copy()
    t2.loc[n + 1:, ["open", "high", "low", "close", "vwap"]] *= 1.05
    t2.loc[n + 1:, "volume"] *= 3
    _, X2 = _inputs(t2, mkt, model)
    pd.testing.assert_frame_equal(tok.transform(X).iloc[: n + 1], tok.transform(X2).iloc[: n + 1])


def test_tokenizer_roundtrip(setup):
    """De-tokenized RET (training bin means x sigma_hat) vs the true return; output bins vs z_h."""
    _, _, _, L, X, tok, tr = setup
    code = tok.transform(X)["RET"].to_numpy()
    zh = tok.detok_ret(code)
    z = X["ret_z"].to_numpy()
    ok = np.isfinite(zh) & np.isfinite(z)
    r, rh = L["r"].to_numpy()[ok], (zh * L["sigma_hat"].to_numpy())[ok]
    corr = np.corrcoef(r, rh)[0, 1]
    rmse = np.sqrt(np.mean((zh[ok] - np.clip(z[ok], -6, 6)) ** 2))
    print(f"RET round-trip: corr {corr:.4f}, RMSE {rmse:.4f} sigma")
    assert corr > 0.97 and rmse < 0.3  # the two open tail bins dominate the RMSE on 8k training rows
    for h, b in tok.out.items():
        zt = L[f"z_{h}"].to_numpy()
        m = np.isfinite(zt)
        zo = b.detokenize(b.transform(zt[m]))
        c = np.corrcoef(zo, np.clip(zt[m], -6, 6))[0, 1]
        print(f"output h={h} round-trip corr {c:.4f}")
        assert c > 0.97


def test_bins_occupancy_and_merge():
    rng = np.random.default_rng(0)
    b = QuantileBins(64, 50, 6.0).fit(rng.standard_t(3, 2000))  # ~31 per bin -> merges needed
    assert len(b.merged) > 0
    assert b.counts.min() >= 50
    b2 = QuantileBins(64, 50, 6.0).fit(rng.standard_t(3, 200_000))
    assert not b2.merged and len(b2.edges) == 63 and b2.counts.min() >= 50
    x = np.array([-100.0, 0.0, 100.0, np.nan])
    c = b2.transform(x)
    assert c[0] == 0 and c[2] == 63 and c[3] == b2.missing  # frozen edges; clip; NaN -> missing


def test_rel_codes():
    df = pd.DataFrame({"open": [10, 10.2, 9.9, 11.0, 9.0, 9.5, 10.0],
                       "high": [11, 10.8, 11.2, 11.2, 10.0, 11.5, 11.8],
                       "low": [9, 9.5, 8.8, 9.0, 8.5, 8.0, 8.2],
                       "close": [10.5, 10.0, 10.9, 9.2, 9.8, 11.2, 10.5]})
    c = rel_codes(df)
    assert np.isnan(c[0])
    assert c[1] == 0 * 3 + 0  # inside (10.8 < 11, 9.5 > 9); bearish but does not engulf the bullish body
    assert c[2] == 1 * 3 + 1  # outside; bullish 9.9->10.9 engulfs bearish 10.2->10.0
    assert c[3] == 4 * 3 + 2  # equal high -> mixed; bearish 11.0->9.2 engulfs bullish 9.9->10.9
    assert c[4] == 3 * 3 + 0  # lower high, lower low; bullish 9.0->9.8 does not reach 11.0
    assert c[5] == 1 * 3 + 0  # outside; both bullish -> no engulfing
    assert c[6] == 2 * 3 + 0  # higher high, higher low


def test_lag_matrix_causal():
    T = pd.DataFrame({f: np.arange(40) % (FIELDS[f]) for f in FIELDS}).astype(np.int16)
    A, names = lag_matrix(T)
    assert A.shape == (40, len(FIELDS) * 24) and names[1] == "RET_l1"
    assert A[0, 1] == special("RET", "MISSING") and A[5, 1] == T["RET"][4]
    T2 = T.copy()
    T2.iloc[21:] = 0
    np.testing.assert_array_equal(A[:21], lag_matrix(T2)[0][:21])


def test_gap_field(setup):
    _, _, _, L, X, tok, _ = setup
    g = tok.transform(X)["GAP"].to_numpy()
    k = L["bar_index"].to_numpy()
    assert np.all(g[k > 0] == NO_GAP)
    assert np.all((g[k == 0] < NO_GAP) | (g[k == 0] == special("GAP", "MISSING")))
