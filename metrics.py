"""
metrics.py
==========
Performance metrics of Section 4.4: entropy gap, sliced W2, objective J.

Conventions
-----------
H(p0)               : Monte Carlo entropy of the exact GMM (2e5 draws).
H(rho0)             : Monte Carlo entropy of the diagonal GMM refit to the
                      terminal particle cloud (centers fixed at mu_k),
                      evaluated on 2e5 fresh samples from the fitted mixture.
entropy gap         : H(p0) - H(rho0^nu).
sliced W2           : RMS over 256 random unit projections of exact 1-D W2
                      between 2e4 generated and 2e4 true samples.
objective J         : H(rho0^nu) - lam * int_0^T R(nu) dt,
                      R(nu) = (nu^2 - g^2)^2 / (2 g^2).

Common random numbers: all terminal clouds compared in one table come from
forward simulations with the same seed (identical prior and Brownian draws),
and all sliced-W2 evaluations share one fixed true sample and one fixed
projection set.
"""

import numpy as np
import gmm_control as gc
import particle_solver as ps


def H_gmm_mc(centers, w, v, n=200_000, seed=11):
    """Monte Carlo entropy of a fitted diagonal GMM via fresh samples."""
    rng = np.random.default_rng(seed)
    K, d = centers.shape
    ks = rng.choice(K, size=n, p=w)
    X = centers[ks] + np.sqrt(v[ks])[:, None] * rng.standard_normal((n, d))
    lp = ps.logpdf_gmm_fit(X, centers, w, v)
    return -lp.mean(), lp.std() / np.sqrt(n)


def H_particles_gmm(X0, MU, n_em=25, n_mc=200_000, seed=11):
    """Fit the controlled-score GMM to terminal particles, MC entropy."""
    w, v = ps.fit_gmm(np.asarray(X0), MU.copy(), n_em=n_em)
    h, se = H_gmm_mc(MU, w, v, n=n_mc, seed=seed)
    return h, se, w, v


def sliced_w2(X_gen, X_true, n_proj=256, seed=123):
    """RMS of exact 1-D W2 over random unit projections (equal sample sizes)."""
    rng = np.random.default_rng(seed)
    d = X_true.shape[1]
    U = rng.standard_normal((n_proj, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    pg = np.sort(np.asarray(X_gen) @ U.T, axis=0)
    pt = np.sort(np.asarray(X_true) @ U.T, axis=0)
    w2 = np.sqrt(((pg - pt) ** 2).mean(axis=0))
    return float(np.sqrt((w2**2).mean()))


def objective(H, nu2, tg, lam):
    """J = H - lam * int R(nu) dt."""
    return H - lam * gc.running_cost(nu2, tg)
