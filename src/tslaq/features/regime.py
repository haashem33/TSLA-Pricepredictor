"""Gaussian HMM regime features (CLAUDE.md 6.6, 5.6), implemented in numpy.

`hmmlearn` has no wheel for Python 3.14, so the model is fitted here with Baum-Welch EM. Design:

* Scaled forward/backward recursions (numba-compiled; numpy reference versions kept for tests). Emission log-densities are shifted by their per-row maximum
  before exponentiation (the shift is added back to the log-likelihood), so outliers cannot underflow.
* EM from `n_starts` (>= 5) random starts on the training data; the start with the best final
  log-likelihood is kept.
* Covariances are floored (diagonal >= `var_floor_frac` x the data variance of that dimension, plus an
  eigenvalue floor) so no state can collapse onto a few points.
* After every fit the states are ordered by the variance of the first observation dimension (u, the
  de-seasonalized return), low to high, so state 0 means the calm regime in every fold.
* Features use FILTERED probabilities only: p_t ∝ (p_{t-1} @ A) * b(x_t). No backward pass, no Viterbi.
  `smoothed_probs` exists only so the tests can show that smoothing would leak.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

LOG2PI = np.log(2 * np.pi)


@dataclass
class GaussianHMM:
    startprob: np.ndarray   # (K,)
    transmat: np.ndarray    # (K, K), rows sum to 1
    means: np.ndarray       # (K, D)
    covs: np.ndarray        # (K, D, D)
    loglik: float = np.nan
    n_iter: int = 0

    @property
    def K(self) -> int:
        return len(self.startprob)

    def as_dict(self) -> dict:
        return {"startprob": self.startprob.tolist(), "transmat": self.transmat.tolist(),
                "means": self.means.tolist(), "covs": self.covs.tolist(), "loglik": self.loglik,
                "n_iter": self.n_iter}


def log_emission(X: np.ndarray, means: np.ndarray, covs: np.ndarray) -> np.ndarray:
    """(T, K) Gaussian log-densities."""
    T, D = X.shape
    out = np.empty((T, len(means)))
    for k, (m, c) in enumerate(zip(means, covs)):
        L = np.linalg.cholesky(c)
        sol = np.linalg.solve(L, (X - m).T)
        out[:, k] = -0.5 * (D * LOG2PI + np.sum(sol**2, axis=0)) - np.sum(np.log(np.diag(L)))
    return out


def _scaled_emission(logb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shift = logb.max(axis=1)
    return np.exp(logb - shift[:, None]), shift


@njit(cache=True)
def _forward_nb(b, startprob, A):
    T, K = b.shape
    alpha = np.empty((T, K))
    c = np.empty(T)
    s = 0.0
    for j in range(K):
        alpha[0, j] = startprob[j] * b[0, j]
        s += alpha[0, j]
    c[0] = s
    for j in range(K):
        alpha[0, j] /= s
    for t in range(1, T):
        s = 0.0
        for j in range(K):
            acc = 0.0
            for i in range(K):
                acc += alpha[t - 1, i] * A[i, j]
            alpha[t, j] = acc * b[t, j]
            s += alpha[t, j]
        c[t] = s
        for j in range(K):
            alpha[t, j] /= s
    return alpha, c


@njit(cache=True)
def _backward_nb(b, A, c):
    T, K = b.shape
    beta = np.empty((T, K))
    y = np.empty(K)
    for j in range(K):
        beta[T - 1, j] = 1.0
    for t in range(T - 2, -1, -1):
        for j in range(K):
            y[j] = b[t + 1, j] * beta[t + 1, j]
        for i in range(K):
            acc = 0.0
            for j in range(K):
                acc += A[i, j] * y[j]
            beta[t, i] = acc / c[t + 1]
    return beta


def forward(b: np.ndarray, startprob: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scaled forward pass. Returns alpha (T, K), rows normalised to 1 (= filtered probabilities), and
    the scaling constants c (T,), with log P(x_1..T) = sum log c + sum of emission shifts."""
    return _forward_nb(np.ascontiguousarray(b, dtype=np.float64), np.asarray(startprob, np.float64),
                       np.ascontiguousarray(A, dtype=np.float64))


def backward(b: np.ndarray, A: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Scaled backward pass consistent with `forward` (beta_T = 1; beta_t = A (b_{t+1} beta_{t+1}) / c_{t+1})."""
    return _backward_nb(np.ascontiguousarray(b, dtype=np.float64), np.ascontiguousarray(A, dtype=np.float64),
                        np.asarray(c, np.float64))


def forward_reference(b, startprob, A):
    """Plain numpy forward pass (reference implementation for tests)."""
    T, K = b.shape
    alpha = np.empty((T, K))
    c = np.empty(T)
    a = startprob * b[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * b[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    return alpha, c


def backward_reference(b, A, c):
    T, K = b.shape
    beta = np.empty((T, K))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (b[t + 1] * beta[t + 1])) / c[t + 1]
    return beta


def _floor_covs(covs: np.ndarray, floor: np.ndarray) -> np.ndarray:
    out = covs.copy()
    for k in range(len(out)):
        c = 0.5 * (out[k] + out[k].T)
        d = np.diag(c).copy()
        np.fill_diagonal(c, np.maximum(d, floor))
        w, V = np.linalg.eigh(c)
        out[k] = (V * np.maximum(w, floor.min())) @ V.T
    return out


def order_states(m: GaussianHMM) -> GaussianHMM:
    """Relabel states by the variance of observation dimension 0 (low -> high)."""
    o = np.argsort(m.covs[:, 0, 0], kind="stable")
    return GaussianHMM(m.startprob[o], m.transmat[np.ix_(o, o)], m.means[o], m.covs[o], m.loglik, m.n_iter)


def em_fit(X: np.ndarray, init: GaussianHMM, max_iter: int = 200, tol: float = 1e-6,
           var_floor: np.ndarray | None = None) -> tuple[GaussianHMM, list[float]]:
    """Baum-Welch EM. Returns the fitted model and the log-likelihood trace (one value per E-step,
    evaluated at the parameters entering that iteration). Stops when the per-observation gain < tol."""
    T, D = X.shape
    floor = var_floor if var_floor is not None else np.zeros(D)
    sp, A, mu, cov = init.startprob.copy(), init.transmat.copy(), init.means.copy(), init.covs.copy()
    trace: list[float] = []
    for it in range(max_iter):
        b, shift = _scaled_emission(log_emission(X, mu, cov))
        alpha, c = forward(b, sp, A)
        ll = float(np.sum(np.log(c)) + shift.sum())
        trace.append(ll)
        if it > 0 and (ll - trace[-2]) / T < tol:
            break
        beta = backward(b, A, c)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        # xi summed over t: alpha_t[i] A[i,j] b_{t+1}[j] beta_{t+1}[j] / c_{t+1}
        nb = (b[1:] * beta[1:]) / c[1:, None]
        xi = A * (alpha[:-1].T @ nb)
        sp = gamma[0] + 1e-12
        sp /= sp.sum()
        A = xi / xi.sum(axis=1, keepdims=True)
        w = gamma.sum(axis=0)
        mu = (gamma.T @ X) / w[:, None]
        cov = np.empty((len(w), D, D))
        for k in range(len(w)):
            d = X - mu[k]
            cov[k] = (gamma[:, k, None] * d).T @ d / w[k]
        cov = _floor_covs(cov, floor)
    m = GaussianHMM(sp, A, mu, cov, trace[-1], len(trace))
    return m, trace


def random_init(X: np.ndarray, K: int, rng: np.random.Generator) -> GaussianHMM:
    """Means at K random observations; covariances = data covariance scaled by random factors;
    sticky random transition matrix."""
    D = X.shape[1]
    base = np.cov(X.T).reshape(D, D)
    means = X[rng.choice(len(X), K, replace=False)]
    covs = np.stack([base * rng.uniform(0.3, 2.0) for _ in range(K)])
    A = 0.8 * np.eye(K) + 0.2 * rng.dirichlet(np.ones(K), size=K)
    return GaussianHMM(np.full(K, 1.0 / K), A / A.sum(axis=1, keepdims=True), means, covs)


def fit_hmm(X: np.ndarray, K: int = 3, n_starts: int = 5, seed: int = 0, max_iter: int = 200,
            tol: float = 1e-6, var_floor_frac: float = 1e-3) -> tuple[GaussianHMM, list[dict]]:
    """Multi-start EM on TRAINING observations (rows with NaN are skipped). Returns the best model
    (states ordered by variance of dimension 0) and a per-start summary."""
    if n_starts < 5:
        raise ValueError("n_starts must be >= 5")
    X = X[np.isfinite(X).all(axis=1)]
    floor = var_floor_frac * X.var(axis=0)
    rng = np.random.default_rng(seed)
    best, runs = None, []
    for s in range(n_starts):
        m, trace = em_fit(X, random_init(X, K, rng), max_iter, tol, floor)
        runs.append({"start": s, "loglik": m.loglik, "n_iter": m.n_iter})
        if best is None or m.loglik > best.loglik:
            best = m
    return order_states(best), runs


def filtered_probs(X: np.ndarray, m: GaussianHMM) -> np.ndarray:
    """Causal filtered state probabilities P(s_t | x_1..x_t) with FIXED parameters.

    Rows with non-finite observations get NaN and do not update the state (the prediction step is
    skipped too, matching how EM treated the training sequence as contiguous valid rows)."""
    out = np.full((len(X), m.K), np.nan)
    ok = np.isfinite(X).all(axis=1)
    if ok.any():
        b, _ = _scaled_emission(log_emission(X[ok], m.means, m.covs))
        alpha, _ = forward(b, m.startprob, m.transmat)
        out[ok] = alpha
    return out


def smoothed_probs(X: np.ndarray, m: GaussianHMM) -> np.ndarray:
    """P(s_t | x_1..x_T): uses FUTURE observations. Never use as a feature (tests only)."""
    out = np.full((len(X), m.K), np.nan)
    ok = np.isfinite(X).all(axis=1)
    b, _ = _scaled_emission(log_emission(X[ok], m.means, m.covs))
    alpha, c = forward(b, m.startprob, m.transmat)
    g = alpha * backward(b, m.transmat, c)
    out[ok] = g / g.sum(axis=1, keepdims=True)
    return out


def sample_hmm(m: GaussianHMM, T: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Simulate T observations (for tests)."""
    s = np.empty(T, int)
    s[0] = rng.choice(m.K, p=m.startprob)
    u = rng.random(T)
    cum = np.cumsum(m.transmat, axis=1)
    for t in range(1, T):
        s[t] = min(int(np.searchsorted(cum[s[t - 1]], u[t])), m.K - 1)
    X = np.empty((T, m.means.shape[1]))
    for k in range(m.K):
        idx = s == k
        X[idx] = rng.multivariate_normal(m.means[k], m.covs[k], idx.sum())
    return X, s


def hmm_obs(u: np.ndarray, sigma_hat: np.ndarray) -> np.ndarray:
    """Observation vector [u_t, ln sigma_hat_t] (CLAUDE.md 6.6); u clipped to +-10."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.column_stack([np.clip(u, -10, 10), np.log(sigma_hat)])
