"""test_hmm_filtered (CLAUDE.md 11) and EM checks for the numpy Gaussian HMM."""
import numpy as np

from tslaq.features.regime import (
    GaussianHMM,
    backward,
    backward_reference,
    em_fit,
    filtered_probs,
    fit_hmm,
    forward,
    forward_reference,
    order_states,
    random_init,
    sample_hmm,
    smoothed_probs,
)


def _true_model() -> GaussianHMM:
    A = np.array([[0.97, 0.02, 0.01], [0.03, 0.94, 0.03], [0.02, 0.06, 0.92]])
    means = np.array([[0.0, -1.0], [0.2, 0.0], [-0.3, 1.0]])
    covs = np.array([[[0.25, 0.02], [0.02, 0.05]],
                     [[1.0, 0.05], [0.05, 0.08]],
                     [[4.0, -0.2], [-0.2, 0.1]]])
    return GaussianHMM(np.array([0.5, 0.3, 0.2]), A, means, covs)


def _sample(T=20000, seed=0):
    return sample_hmm(_true_model(), T, np.random.default_rng(seed))


def test_em_loglik_never_decreases():
    X, _ = _sample(5000, 1)
    for seed in range(3):
        init = random_init(X, 3, np.random.default_rng(seed))
        _, trace = em_fit(X, init, max_iter=60, tol=0.0)
        d = np.diff(trace)
        assert np.all(d >= -1e-8 * np.abs(np.asarray(trace[1:]))), d.min()


def test_em_recovers_known_parameters():
    X, s = _sample(20000, 2)
    m, runs = fit_hmm(X, K=3, n_starts=5, seed=0)
    assert len(runs) == 5 and m.loglik == max(r["loglik"] for r in runs)
    t = _true_model()
    np.testing.assert_allclose(m.means, t.means, atol=0.06)
    np.testing.assert_allclose(m.covs[:, 0, 0], t.covs[:, 0, 0], rtol=0.08)
    np.testing.assert_allclose(m.covs[:, 1, 1], t.covs[:, 1, 1], rtol=0.08)
    np.testing.assert_allclose(np.diag(m.transmat), np.diag(t.transmat), atol=0.02)
    # states are ordered by variance of dimension 0 (low -> high)
    assert np.all(np.diff(m.covs[:, 0, 0]) > 0)


def test_order_states_by_variance():
    t = _true_model()
    p = [2, 0, 1]
    shuffled = GaussianHMM(t.startprob[p], t.transmat[np.ix_(p, p)], t.means[p], t.covs[p])
    o = order_states(shuffled)
    np.testing.assert_allclose(o.transmat, t.transmat)
    np.testing.assert_allclose(o.means, t.means)


def test_variance_floor_prevents_degenerate_state():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(3000, 2))
    X[:40] = [5.0, 5.0]  # a cluster of identical points that would give a zero-variance state
    m, _ = fit_hmm(X, K=3, n_starts=5, seed=1, var_floor_frac=1e-2)
    floor = 1e-2 * X.var(axis=0)
    for k in range(3):
        assert np.all(np.diag(m.covs[k]) >= floor * (1 - 1e-9))
        assert np.linalg.eigvalsh(m.covs[k]).min() > 0


def test_numba_forward_backward_match_numpy_reference():
    rng = np.random.default_rng(4)
    for K in (2, 3, 4):
        b = rng.uniform(0.01, 1, (500, K))
        A = rng.dirichlet(np.ones(K), size=K)
        sp = rng.dirichlet(np.ones(K))
        a1, c1 = forward(b, sp, A)
        a2, c2 = forward_reference(b, sp, A)
        np.testing.assert_allclose(a1, a2, rtol=1e-12)
        np.testing.assert_allclose(c1, c2, rtol=1e-12)
        np.testing.assert_allclose(backward(b, A, c1), backward_reference(b, A, c2), rtol=1e-10)


def test_hmm_filtered():
    """Filtered probabilities at t are unchanged when data after t is deleted; smoothed ones change
    (so this test would fail if `predict_proba`-style smoothing were used)."""
    X, _ = _sample(4000, 5)
    X[100:105] = np.nan  # gaps are skipped, not filled
    m, _ = fit_hmm(X[:2000], K=3, n_starts=5, seed=0)
    full = filtered_probs(X, m)
    sm_full = smoothed_probs(X, m)
    rng = np.random.default_rng(6)
    changed = 0
    for t in rng.integers(2000, len(X) - 50, 50):
        trunc = filtered_probs(X[: t + 1], m)
        np.testing.assert_array_equal(full[: t + 1], trunc)
        sm = smoothed_probs(X[: t + 1], m)
        changed += not np.allclose(sm_full[t], sm[t], atol=1e-6)
    assert changed > 25
    assert np.all(np.isnan(full[100:105]))
