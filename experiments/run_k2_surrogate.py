"""
run_k2_surrogate.py
===================
Two-mode (K=2, d=1) surrogate-kernel sign study behind SI Figure
fig:particle-kernel (SI Appendix, app:exact).

Three curves per mismatch scenario:
  1. exact surrogate kernel S_ref(t) by numerical quadrature of the
     Feynman-Kac costate on the reference law;
  2. closed form eq:k2kernel (first order in delta, no overlap terms);
  3. particle kernel estimator of Algorithm alg:particle in the surrogate
     configuration (exact forward-noised particles, first iterate,
     mixture-of-quadratics costate), mean +/- sd over seeds.

Setup: d=1, sig0=0.5, modes at -1 and 3, pi=(0.75, 0.25), VP schedule
linear beta on [0.1, 20], T=1. The sub-unit base variance makes
Sig2(t) = m(t)^2 sig0^2 + 1 - m(t)^2 genuinely t-dependent, which is
what tilts the per-mode weights pi_k delta_k / (Sig2 + delta_k) of
eq:k2kernel along t and allows a sign change under competing mismatches.

Scenarios: competing mismatch delta=(+0.02, -0.05) (left panel) and
aligned mismatch delta=(+0.02, +0.05) (right panel).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion_control import gmm_control as gc
from diffusion_control import particle_solver as ps

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "k2_surrogate")
os.makedirs(RES, exist_ok=True)

# ---- two-mode testbed ------------------------------------------------------
MU = np.array([[-1.0], [3.0]])          # (K, 1)
WTS = np.array([0.75, 0.25])
SIG0L = 0.5
K, D = 2, 1
M = 40                                   # coarse kernel grid
N_PART = 20_000
SEEDS = range(12)

SCENARIOS = {
    "competing": np.array([+0.02, -0.05]),
    "aligned": np.array([+0.02, +0.05]),
}


def norm_pdf(x, mu, var):
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


def gmm_pdf_and_score(x, means, vars_, wts):
    """1-D mixture pdf and score with per-mode variances. x: (G,)."""
    r_un = np.stack([w * norm_pdf(x, m0, v)
                     for w, m0, v in zip(wts, means, vars_)], axis=1)
    p = np.maximum(r_un.sum(axis=1), 1e-300)
    num = ((r_un / np.asarray(vars_)[None, :])
           * (np.asarray(means)[None, :] - x[:, None])).sum(axis=1)
    return p, num / p


def score_particles(X, t, vars_):
    """Per-mode-variance mixture score on particle array X (N,1)."""
    x = X[:, 0]
    means = gc.m(t) * MU[:, 0]
    r_un = np.stack([w * norm_pdf(x, m0, v)
                     for w, m0, v in zip(WTS, means, vars_)], axis=1)
    p = np.maximum(r_un.sum(axis=1), 1e-300)
    s = ((r_un / np.asarray(vars_)[None, :])
         * (means[None, :] - x[:, None])).sum(axis=1) / p
    return s[:, None]


def exact_surrogate_kernel(tg, delta, n_grid=4001, gh_n=64):
    """S_ref(t) = E_{x~p_t}[ d/dx psi(t,x) * (s_theta - s_true)(x) ]
    with psi(t,x) = E[-1 - log p0(X0) | X_t = x] under the true channel.
    Gauss-Hermite quadrature for the inner conditional expectation.
    """
    x = np.linspace(-9.0, 9.0, n_grid)
    dx = x[1] - x[0]
    xi, wi = np.polynomial.hermite_e.hermegauss(gh_n)   # for N(0,1)
    S = np.zeros_like(tg)
    s02 = SIG0L**2
    for j, t in enumerate(tg):
        mt = gc.m(t)
        sig2 = gc.Sig2(t, sig0=SIG0L)
        means_t = mt * MU[:, 0]
        p_t, s_true = gmm_pdf_and_score(x, means_t, np.full(K, sig2), WTS)
        _, s_frozen = gmm_pdf_and_score(x, means_t, sig2 + delta, WTS)

        # responsibilities under the true law
        r_true = np.stack([w * norm_pdf(x, m0, sig2)
                           for w, m0 in zip(WTS, means_t)], axis=1)
        r_true /= r_true.sum(axis=1, keepdims=True)

        # psi(t,x) = sum_k r_k E[phi(X0) | X_t=x, k],  phi = -1 - log p0
        # X0 | X_t=x, k ~ N(mu_post, var_post)
        var_post = s02 * (1.0 - mt**2 * s02 / sig2)
        psi = np.zeros_like(x)
        for k in range(K):
            mu_post = MU[k, 0] + (mt * s02 / sig2) * (x - means_t[k])
            x0 = mu_post[:, None] + np.sqrt(var_post) * xi[None, :]   # (G,gh)
            logp0_vals = np.log(np.maximum(
                sum(w * norm_pdf(x0, m0, s02) for w, m0 in zip(WTS, MU[:, 0])),
                1e-300))
            Elogp0 = (logp0_vals * wi[None, :]).sum(axis=1) / np.sqrt(np.pi)
            psi += r_true[:, k] * (-1.0 - Elogp0)
        dpsi = np.gradient(psi, dx)
        S[j] = np.sum(dpsi * (s_frozen - s_true) * p_t) * dx
    return S


def closed_form_kernel(tg, delta):
    """eq:k2kernel without the overlap remainder R_ref."""
    mt2 = gc.m(tg) ** 2
    sig2 = gc.Sig2(tg, sig0=SIG0L)
    wsum = sum(w * dk / (sig2**2 * (sig2 + dk))        # Sig^4 = (Sig^2)^2
               for w, dk in zip(WTS, delta))
    return D * mt2 * s02_const * wsum


s02_const = SIG0L**2


def particle_surrogate_kernel(tg, delta, seed, n=N_PART):
    """Algorithm alg:particle in the surrogate configuration: exact
    forward-noised particles, first iterate, true score as the reference
    law (s_ctrl -> nabla log p_t)."""
    rng = np.random.default_rng(seed)
    Xs = np.empty(M + 1, dtype=object)
    ks = rng.choice(K, size=n, p=WTS)
    X0 = MU[ks] + SIG0L * rng.standard_normal((n, D))
    for j, t in enumerate(tg):
        mt = gc.m(t)
        sig2 = gc.Sig2(t, sig0=SIG0L)
        sd = np.sqrt(max(sig2 - mt**2 * SIG0L**2, 0.0))
        Xs[j] = mt * X0 + sd * rng.standard_normal((n, D))
    # cross-fitted costate regression as in kernel_from_particles, but
    # with the exact reference score replacing the fitted controlled score
    half = n // 2
    folds = [(np.arange(0, half), np.arange(half, n)),
             (np.arange(half, n), np.arange(0, half))]
    S = np.zeros(M + 1)
    for fit_idx, ev_idx in folds:
        w_prev = None
        for j, t in enumerate(tg):
            Xf, Xe = Xs[j][fit_idx], Xs[j][ev_idx]
            cen = gc.m(t) * MU
            w, v = ps.fit_gmm(Xf, cen, n_em=15, w0=w_prev)
            w_prev = w
            if j == 0:
                Y0f = -(1.0 + ps.logpdf_gmm_fit(Xf, cen, w, v))
            coef = ps.fit_costate(Xf, Y0f, cen, w, v)
            gp = ps.grad_psi_mix(Xe, cen, w, v, coef)
            sig2 = gc.Sig2(t, sig0=SIG0L)
            s_th = score_particles(Xe, t, sig2 + delta)
            s_true = score_particles(Xe, t, np.full(K, sig2))
            S[j] += 0.5 * np.mean((gp * (s_th - s_true)).sum(axis=1))
    return S


def main():
    tg = np.linspace(0.0, gc.T, M + 1)
    out = {"tg": tg, "MU": MU, "WTS": WTS, "sig0": SIG0L}
    for name, delta in SCENARIOS.items():
        print(f"[{name}] delta = {delta}", flush=True)
        S_exact = exact_surrogate_kernel(tg, delta)
        S_cf = closed_form_kernel(tg, delta)
        S_part = np.stack([particle_surrogate_kernel(tg, delta, seed=s)
                           for s in SEEDS])
        out[f"S_exact_{name}"] = S_exact
        out[f"S_cf_{name}"] = S_cf
        out[f"S_part_mean_{name}"] = S_part.mean(axis=0)
        out[f"S_part_sd_{name}"] = S_part.std(axis=0)
        print(f"  exact range [{S_exact.min():+.4f}, {S_exact.max():+.4f}]  "
              f"sign changes: {int(np.sum(np.diff(np.sign(S_exact)) != 0))}",
              flush=True)
        np.savez(os.path.join(RES, f"k2_{name}.npz"),
                 tg=tg, delta=delta, S_exact=S_exact, S_cf=S_cf,
                 S_part=S_part)
    np.savez(os.path.join(RES, "k2_surrogate_all.npz"), **out)
    print("saved to", RES)


if __name__ == "__main__":
    main()
