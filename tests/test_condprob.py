"""test_condprob (CLAUDE.md 11) and selection helpers."""
import numpy as np
import pandas as pd

from tslaq.selection.condprob import benjamini_hochberg, bucket_table, fit_edges, wilson
from tslaq.selection.ic import signed_walk_forward, spearman_cols
from tslaq.selection.redundancy import cluster_features, spearman_matrix


def test_condprob_known_probabilities():
    """Three groups with P(up) = 0.45 / 0.50 / 0.55: each estimate falls inside its Wilson interval and
    the intervals cover the truth; the outer groups are flagged after BH."""
    rng = np.random.default_rng(0)
    n = 30000
    g = rng.integers(0, 3, n)
    p_true = np.array([0.45, 0.50, 0.55])[g]
    fwd = np.where(rng.random(n) < p_true, 1.0, -1.0) * rng.uniform(0.001, 0.01, n)
    x = g.astype(float)
    tab = bucket_table(x, fwd, fwd, fit_edges(x, categorical=True), h=1, cost=0.0)
    assert len(tab) == 3
    for _, r in tab.iterrows():
        assert r["wilson_lo"] <= r["p_up"] <= r["wilson_hi"]
        truth = {"v=0": 0.45, "v=1": 0.50, "v=2": 0.55}[r["bucket"]]
        assert r["wilson_lo"] <= truth <= r["wilson_hi"]
    rej = dict(zip(tab["bucket"], benjamini_hochberg(tab["pval"].to_numpy(), 0.10)))
    assert rej["v=0"] and rej["v=2"]


def test_condprob_noise_flags_roughly_nothing():
    rng = np.random.default_rng(1)
    pv = []
    for f in range(40):
        x = rng.normal(size=20000)
        fwd = rng.normal(size=20000)
        for h in (1, 3, 12):
            tab = bucket_table(x, fwd, fwd, fit_edges(x, categorical=False), h=h, cost=0.0)
            pv.extend(tab["pval"].tolist())
    rej = benjamini_hochberg(np.array(pv), 0.10)
    assert len(pv) == 40 * 3 * 14
    assert rej.sum() <= 2


def test_n_eff_widens_interval():
    lo1, hi1 = wilson(0.5, 10000)
    lo12, hi12 = wilson(0.5, 10000 / 12)
    assert (hi12 - lo12) > 3 * (hi1 - lo1)
    assert abs((hi1 - lo1) / 2 - 0.0098) < 5e-4  # ~ +-1 pp at n = 10,000 (6.9 interpretation rule)


def test_bh_matches_definition():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    # q = 0.05: thresholds 0.005, 0.01, 0.015, ... -> first two rejected
    assert benjamini_hochberg(p, 0.05).tolist() == [True, True] + [False] * 8


def test_spearman_cols_pairwise():
    rng = np.random.default_rng(2)
    y = rng.normal(size=1000)
    X = np.column_stack([y + rng.normal(size=1000), rng.normal(size=1000)])
    X[:100, 0] = np.nan
    from scipy.stats import spearmanr
    got = spearman_cols(X, y, 50)
    assert np.isclose(got[0], spearmanr(X[100:, 0], y[100:]).statistic)
    assert np.isclose(got[1], spearmanr(X[:, 1], y).statistic)


def test_signed_walk_forward_uses_past_only():
    months = [f"20{y}-{m:02d}" for y in (16, 17, 18) for m in range(1, 13)]
    ic = pd.Series(-0.02, index=months)
    ic["2018-06"] = 0.5  # a future spike must not change earlier signs
    wf = signed_walk_forward(ic, months[24:], min_past=12)
    assert np.isclose(wf["2018-01"], 0.02) and np.isclose(wf["2018-06"], -0.5)


def test_clusters_group_correlated_features():
    rng = np.random.default_rng(3)
    a = rng.normal(size=5000)
    F = pd.DataFrame({"a": a, "a2": a + 0.1 * rng.normal(size=5000), "b": rng.normal(size=5000)})
    cl = cluster_features(spearman_matrix(F, 100), 0.7)
    assert cl["a"] == cl["a2"] != cl["b"]
