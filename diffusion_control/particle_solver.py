"""
particle_solver.py
==================
Particle fixed-point solver of SI Appendix (algorithm: iterative particle
fixed point / PMP shooting), specialized to the GMM testbed of
gmm_control.py.

One Picard iteration:
  1. Forward pass: simulate N trajectories of the controlled reverse SDE
     from t = T to t = 0 with the current schedule nu^2(t_j) (piecewise
     constant on a uniform grid of M cells, Nf fine Euler--Maruyama steps).
  2. Controlled score: diagonal-GMM fit at each grid time with centers
     FIXED at m(t) mu_k (EM over weights and variances only).
  3. Backward regression: costate psi(x, t_j) = E[Y0 | X_{t_j} = x],
     Y0 = -(1 + log rho0_hat(X0)), as a mixture of per-mode quadratics
     (responsibility-weighted least squares on features [1, x, x^2]).
  4. Kernel Monte Carlo with two-fold cross-fitting:
     S_hat(t_j) = (1/N) sum_i grad psi(X^i, t_j) . (s_theta - s_ctrl)(X^i, t_j).
  5. Stationarity update: nu^2 <- g^2 (1 + S_hat / (2 lam)),
     projected to [NU_MIN^2, NU_MAX^2].

Common random numbers: for a fixed seed the Brownian increments and prior
draws are identical for every schedule, so schedule comparisons are free of
sampling noise.
"""

import numpy as np
from . import gmm_control as gc


# ---------------------------------------------------------------------------
# 1. forward pass
# ---------------------------------------------------------------------------
def forward_particles(nu2, MU, wts, delta, N=20_000, M=40, Nf=800, seed=0,
                      store_slices=True):
    """Simulate the controlled reverse SDE t: T -> 0.

    nu2 : (M+1,) schedule values on the uniform coarse grid t_j = j*T/M
          (value at cell [t_j, t_{j+1}) is nu2[j]; nu2[M] unused).
    Returns (Xs, t_slices): Xs is (M+1, N, d) float32 if store_slices,
    otherwise only the terminal cloud (N, d) float32.
    """
    assert Nf % M == 0, "Nf must be a multiple of M"
    N = int(N)
    K, d = MU.shape
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, d))                       # X_T ~ N(0, I)
    dt = gc.T / Nf
    dtc = gc.T / M
    t_slices = np.linspace(0.0, gc.T, M + 1)
    Xs = np.empty((M + 1, N, d), dtype=np.float32) if store_slices else None
    if store_slices:
        Xs[M] = X
    for i in range(Nf):                                   # fine step t -> t-dt
        t = gc.T - i * dt
        j = min(int(t / dtc), M - 1)                      # active coarse cell
        n2 = nu2[j]
        b = gc.drift_reverse(X, t, n2, MU, wts, delta)
        X = X - b * dt + np.sqrt(n2 * dt) * rng.standard_normal((N, d))
        if store_slices and (i + 1) % (Nf // M) == 0:
            Xs[M - (i + 1) // (Nf // M)] = X
    return (Xs if store_slices else X.astype(np.float32)), t_slices


# ---------------------------------------------------------------------------
# 2. controlled score: diagonal GMM with fixed centers
# ---------------------------------------------------------------------------
def fit_gmm(X, centers, n_em=15, var_floor=1e-3, w0=None):
    """EM over weights w_k and isotropic variances v_k, centers fixed.

    Returns (w, v): weights (K,), variances (K,).
    """
    N, d = X.shape
    K = centers.shape[0]
    w = np.full(K, 1.0 / K) if w0 is None else w0.copy()
    d2 = gc.pairwise_d2(X, centers)                             # (N,K)
    v = np.full(K, d2.mean() / 1.0)
    for _ in range(n_em):
        lg = -0.5 * d2 / v[None, :] - 0.5 * d * np.log(v)[None, :] \
             + np.log(w)[None, :]
        lg -= lg.max(axis=1, keepdims=True)
        r = np.exp(lg)
        r /= r.sum(axis=1, keepdims=True)
        Nk = r.sum(axis=0) + 1e-12
        w = Nk / N
        v = (r * d2).sum(axis=0) / (d * Nk)
        v = np.maximum(v, var_floor)
    return w, np.maximum(v, var_floor)


def s_ctrl_gmm(x, centers, w, v):
    """Score of the fitted diagonal GMM (per-mode isotropic var v_k).

    Uses sum_k r_k (c_k - x)/v_k = (r/v) @ centers - x * (r @ (1/v)),
    avoiding (N, K, d) temporaries.
    """
    d2 = gc.pairwise_d2(x, centers)
    lg = -0.5 * d2 / v[None, :] - 0.5 * x.shape[1] * np.log(v)[None, :] \
         + np.log(w)[None, :]
    lg -= lg.max(axis=1, keepdims=True)
    r = np.exp(lg)
    r /= r.sum(axis=1, keepdims=True)
    rv = r / v[None, :]
    return rv @ centers - x * rv.sum(axis=1, keepdims=True), r


def logpdf_gmm_fit(x, centers, w, v):
    d2 = gc.pairwise_d2(x, centers)
    lg = -0.5 * d2 / v[None, :] - 0.5 * x.shape[1] * np.log(2 * np.pi * v)[None, :] \
         + np.log(w)[None, :]
    mx = lg.max(axis=1, keepdims=True)
    return mx[:, 0] + np.log(np.exp(lg - mx).sum(axis=1))


# ---------------------------------------------------------------------------
# 3. mixture-of-quadratics costate
# ---------------------------------------------------------------------------
def fit_costate(X, Y0, centers, w, v, ridge=1e-8):
    """Per-mode responsibility-weighted WLS on features [1, x, x^2].

    Returns coef (K, 1+2d): [a_k, b_k (d), c_k (d)] with
    q_k(x) = a_k + b_k.x + sum_i c_ki x_i^2.
    """
    N, d = X.shape
    K = centers.shape[0]
    F = np.concatenate([np.ones((N, 1)), X, X**2], axis=1)      # (N,1+2d)
    _, r = s_ctrl_gmm(X, centers, w, v)
    coef = np.empty((K, 1 + 2 * d))
    for k in range(K):
        wk = r[:, k]
        Fw = F * wk[:, None]
        A = F.T @ Fw + ridge * np.eye(1 + 2 * d)
        coef[k] = np.linalg.solve(A, Fw.T @ Y0)
    return coef


def psi_mix(X, centers, w, v, coef):
    """psi(x) = sum_k r_k(x) q_k(x)."""
    _, r = s_ctrl_gmm(X, centers, w, v)
    d = X.shape[1]
    F = np.concatenate([np.ones((X.shape[0], 1)), X, X**2], axis=1)
    q = F @ coef.T                                            # (N,K)
    return (r * q).sum(axis=1)


def grad_psi_mix(X, centers, w, v, coef):
    """grad psi = sum_k [ q_k grad r_k + r_k (b_k + 2 c_k x) ].

    With pull_k = (c_k - x)/v_k and grad r_k = r_k (pull_k - s):
        sum_k q_k r_k pull_k = ((q r)/v) @ centers - x * sum_k (q r)_k / v_k
        sum_k r_k (b_k + 2 c_k x) = r @ b + 2 x (r @ c)
    so no (N, K, d) temporary is ever formed.
    """
    N, d = X.shape
    s, r = s_ctrl_gmm(X, centers, w, v)                        # score & resp
    F = np.concatenate([np.ones((N, 1)), X, X**2], axis=1)
    q = F @ coef.T                                             # (N,K)
    qr_v = (q * r) / v[None, :]                                # (N,K)
    termA = qr_v @ centers - X * qr_v.sum(axis=1, keepdims=True)
    termB = -s * (q * r).sum(axis=1, keepdims=True)
    bcoef = coef[:, 1:1 + d]
    ccoef = coef[:, 1 + d:]
    termC = r @ bcoef + 2.0 * X * (r @ ccoef)
    return termA + termB + termC


# ---------------------------------------------------------------------------
# 4. kernel with two-fold cross-fitting
# ---------------------------------------------------------------------------
def kernel_from_particles(Xs, t_slices, MU, wts, delta, M, n_em=15):
    """Estimate S(t_j) on the coarse grid from a stored trajectory cloud.

    Xs: (M+1, N, d). Two-fold cross-fitting: fit (GMM, costate) on one half,
    evaluate the kernel on the other, average both directions.
    Returns S (M+1,).
    """
    N = Xs.shape[1]
    half = N // 2
    folds = [(np.arange(0, half), np.arange(half, N)),
             (np.arange(half, N), np.arange(0, half))]
    S = np.zeros(M + 1)
    for fit_idx, ev_idx in folds:
        w_prev = None
        for j in range(M + 1):
            Xf = np.asarray(Xs[j][fit_idx])
            Xe = np.asarray(Xs[j][ev_idx])
            cen = gc.m(t_slices[j]) * MU
            w, v = fit_gmm(Xf, cen, n_em=n_em, w0=w_prev)
            w_prev = w                                    # warm start next slice
            # terminal targets for the costate (fold's own t=0 fit)
            if j == 0:
                Y0f = -(1.0 + logpdf_gmm_fit(Xf, cen, w, v))
            coef = fit_costate(Xf, Y0f, cen, w, v)
            gp = grad_psi_mix(Xe, cen, w, v, coef)
            s_th = gc.score_frozen(Xe, t_slices[j], MU, wts, delta)
            s_ct, _ = s_ctrl_gmm(Xe, cen, w, v)
            S[j] += 0.5 * np.mean((gp * (s_th - s_ct)).sum(axis=1))
    return S


# ---------------------------------------------------------------------------
# 5. Picard fixed point
# ---------------------------------------------------------------------------
def picard_particle(MU, wts, delta, lam, N=20_000, M=40, Nf=800, seed=0,
                    n_iter=12, tol_rel=1e-3, verbose=True):
    """Run the fixed-point iteration. Returns dict with histories."""
    tg = np.linspace(0.0, gc.T, M + 1)
    g2 = gc.g(tg) ** 2
    nu2 = g2.copy()
    hist_nu2 = [nu2.copy()]
    hist_S, hist_res = [], []
    Xs_last = None
    for n in range(n_iter):
        Xs, t_slices = forward_particles(nu2, MU, wts, delta, N=N, M=M,
                                         Nf=Nf, seed=seed)
        S = kernel_from_particles(Xs, t_slices, MU, wts, delta, M)
        nu2_new = g2 * (1.0 + S / (2.0 * lam))
        nu2_new = np.clip(nu2_new, gc.NU_MIN**2, gc.NU_MAX**2)
        res = np.sqrt(np.trapezoid((np.sqrt(nu2_new) - np.sqrt(nu2))**2, tg))
        hist_S.append(S.copy())
        hist_res.append(res)
        nu2 = nu2_new
        hist_nu2.append(nu2.copy())
        Xs_last = Xs
        if verbose:
            print(f"[picard] iter {n}: res={res:.4e}  "
                  f"nu*/g(t=0)={np.sqrt(nu2[0]/g2[0]):.3f}", flush=True)
        if res < tol_rel * np.sqrt(np.trapezoid(g2, tg)):
            if verbose:
                print("[picard] converged", flush=True)
            break
    return dict(tg=tg, nu2=np.array(hist_nu2), S=np.array(hist_S),
                res=np.array(hist_res), Xs_last=Xs_last)
