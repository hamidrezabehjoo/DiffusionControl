"""
gmm_control.py
==============
Shared synthetic testbed for the diffusion noise-control experiments of
Section 4.4 ("Synthetic verification of the exact theory").

Model
-----
Data distribution: K-component Gaussian mixture
    p0 = sum_k w_k N(mu_k, sig0^2 I_d),      sig0 = 1.
Centers mu_k ~ Uniform[-4,4]^d drawn once (seed 0) and frozen; the K-mode
experiment family is nested: modes for K are the first K rows of the K=32
draw, weights follow the period-8 pattern PI8 (renormalized).

Forward noising: variance-preserving (VP) schedule with beta linear on
[B0, B1] = [0.1, 20], horizon T = 1:
    f(x,t) = -1/2 beta(t) x,   g(t) = sqrt(beta(t)),
    m(t)   = exp(-1/2 int_0^t beta(s) ds),
    Sig2(t) = m(t)^2 sig0^2 + 1 - m(t)^2   (= 1 identically when sig0 = 1).

Frozen score: exact score of the same mixture with every modal variance
offset by the constant delta (delta = -0.1 boost scenario, +0.1 damp):
    s_theta(., t) = score of sum_k w_k N(m(t) mu_k, (Sig2(t) + delta) I_d).

Controlled reverse SDE (integrated from t = T to t = 0):
    dx = b_nu(x,t) dt + nu(t) dW,   b_nu = f - (g^2 + nu^2)/2 * s_theta,
with terminal rho_T = N(0, I_d).
"""

import numpy as np

# ---- global constants -----------------------------------------------------
T = 1.0
B0 = 0.1
B1 = 20.0
SIG0 = 1.0
PI8 = np.array([0.20, 0.20, 0.15, 0.15, 0.10, 0.10, 0.07, 0.03])
NU_MIN, NU_MAX = 0.05, 8.0          # admissible set for nu(t)


# ---- VP schedule ----------------------------------------------------------
def beta(t):
    return B0 + (B1 - B0) * np.asarray(t, dtype=float)


def g(t):
    return np.sqrt(beta(t))


def int_beta(t):
    """int_0^t beta(s) ds"""
    t = np.asarray(t, dtype=float)
    return B0 * t + 0.5 * (B1 - B0) * t**2


def m(t):
    return np.exp(-0.5 * int_beta(t))


def Sig2(t, sig0=SIG0):
    """exact marginal variance of one mode under the forward VP channel"""
    m2 = m(t) ** 2
    return m2 * sig0**2 + 1.0 - m2


# ---- modes and weights ----------------------------------------------------
def make_modes(d, K=8, seed=0):
    """First K rows of the K=32 draw (nested family)."""
    rng = np.random.default_rng(seed)
    MU32 = rng.uniform(-4.0, 4.0, size=(32, d))
    return MU32[:K].copy()


def make_weights(K):
    """Period-8 weight pattern PI8, renormalized."""
    w = PI8[np.arange(K) % 8].astype(float)
    return w / w.sum()


# ---- Gaussian-mixture scores and densities --------------------------------
def pairwise_d2(x, centers):
    """Squared distances ||x_i - c_k||^2, (N, d) x (K, d) -> (N, K).

    Computed as ||x||^2 + ||c||^2 - 2 x.c via BLAS matmul, which avoids
    allocating (N, K, d) temporaries (the memory-bandwidth bottleneck of
    the broadcast formulation for large N and d).
    """
    x2 = np.einsum("ij,ij->i", x, x)
    c2 = np.einsum("ij,ij->i", centers, centers)
    d2 = x2[:, None] + c2[None, :] - 2.0 * (x @ centers.T)
    return np.maximum(d2, 0.0)          # guard against roundoff negatives


def log_comp(x, centers, var):
    """log N(x; c_k, var I_d) up to the constant, for each component k.

    x: (N, d), centers: (K, d). Returns (N, K) log-density (with constant).
    """
    d = x.shape[1]
    d2 = pairwise_d2(x, centers)                            # (N, K)
    return -0.5 * d2 / var - 0.5 * d * np.log(2.0 * np.pi * var)


def responsibilities(x, centers, var, wts):
    """Posterior component probabilities r_k(x), (N, K)."""
    lg = log_comp(x, centers, var) + np.log(wts)[None, :]
    lg -= lg.max(axis=1, keepdims=True)
    r = np.exp(lg)
    return r / r.sum(axis=1, keepdims=True)


def score_gmm(x, centers, var, wts):
    """Score of sum_k w_k N(c_k, var I_d) at x. x: (N,d) -> (N,d).

    Uses sum_k r_k (c_k - x) / var = (r @ centers - x) / var, avoiding
    (N, K, d) temporaries.
    """
    r = responsibilities(x, centers, var, wts)                # (N,K)
    return (r @ centers - x) / var


def logpdf_gmm(x, centers, var, wts):
    lg = log_comp(x, centers, var) + np.log(wts)[None, :]
    mx = lg.max(axis=1, keepdims=True)
    return (mx[:, 0] + np.log(np.exp(lg - mx).sum(axis=1)))


def score_frozen(x, t, MU, wts, delta):
    """Frozen (mismatched) score s_theta(x, t)."""
    c = m(t) * MU
    return score_gmm(x, c, Sig2(t) + delta, wts)


def score_true(x, t, MU, wts):
    """Exact score of p_t (delta = 0); used only for sanity checks."""
    c = m(t) * MU
    return score_gmm(x, c, Sig2(t), wts)


def logp0(x, MU, wts, sig0=SIG0):
    return logpdf_gmm(x, MU, sig0**2, wts)


def sample_p0(n, MU, wts, sig0=SIG0, seed=0):
    """Exact samples from p0. Returns (X, ks)."""
    rng = np.random.default_rng(seed)
    K, d = MU.shape
    ks = rng.choice(K, size=n, p=wts)
    X = MU[ks] + sig0 * rng.standard_normal((n, d))
    return X, ks


# ---- controlled dynamics ----------------------------------------------------
def drift_reverse(x, t, nu2, MU, wts, delta):
    """b_nu(x,t) = f - (g^2+nu^2)/2 s_theta.  nu2: scalar (schedule value)."""
    return -0.5 * beta(t) * x - 0.5 * (beta(t) + nu2) * score_frozen(
        x, t, MU, wts, delta)


def H_p0_exact(MU, wts, n=200_000, sig0=SIG0, seed=7):
    """Monte Carlo Shannon entropy H(p0) = E[-log p0]."""
    X, _ = sample_p0(n, MU, wts, sig0, seed)
    lp = logp0(X, MU, wts, sig0)
    return -lp.mean(), lp.std() / np.sqrt(n)


def running_cost(nu2, tg):
    """R(nu) = (nu^2 - g^2)^2 / (2 g^2), trapezoid-integrated on grid tg."""
    g2 = g(tg) ** 2
    R = 0.5 * (nu2 - g2) ** 2 / g2
    return np.trapezoid(R, tg)


# ---- exact per-mode relaxation (validation aid, K = 1 or separated modes) --
def terminal_variance_const(mbar, delta, nu2_const, sig0=SIG0):
    """Analytic terminal variance of one mode at t=0 under a constant
    schedule nu^2 = const, using Sig2(t) = 1 (sig0 = 1), so the frozen
    per-mode variance is w = 1 + delta for all t. Solves
        dv/dtau = beta v - (beta + nu2) v / (1 + delta) + nu2,
    tau = T - t, v(0) = 1, with beta averaged (constant-coefficient approx).
    """
    w = 1.0 + delta
    bbar = 0.5 * (B0 + B1)
    a = bbar - (bbar + nu2_const) / w
    v_eq = -nu2_const / a if abs(a) > 1e-14 else np.inf
    return v_eq + (1.0 - v_eq) * np.exp(a * T)
