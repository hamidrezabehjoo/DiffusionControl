"""
exact_k1.py
===========
Exact K=1 Gaussian solution of the fixed-point problem (Riccati form).

For p0 = N(mu, sig0^2 I_d) and frozen per-mode variance w(t) = Sig2(t)+delta,
the controlled law remains Gaussian, N(m(t) mu, v(t) I_d), with

    dv/dtau  = beta v - (beta + nu^2) v / w + nu^2,   tau = T - t,  v(0) = 1,

and the costate is quadratic, psi_t(x) = a(t) + q(t) ||x - m(t) mu||^2 / 2,
with q solving (forward in t from the terminal condition psi_0 = -(1+log rho_0))

    dq/dt = (beta - (beta + nu^2)/w) q,               q(0) = 1/v(t=0).

The exact PMP kernel is

    S(t) = d * q(t) * (w(t) - v(t)) / w(t).

Sign check: at the uncontrolled fixed law, w - v has the sign of -delta, so
sign S = -sign(delta): boost for over-sharp scores, damp for over-diffuse.
"""
import numpy as np
import gmm_control as gc


def solve_v(nu2, tg, delta, sig0=gc.SIG0):
    """v(t) on the grid tg (ascending in t), v(T) = 1. RK4 in tau = T - t."""
    w = gc.Sig2(tg, sig0) + delta
    bta = gc.beta(tg)
    n = len(tg)
    v = np.empty(n)
    v[-1] = 1.0
    # integrate from t = T down to 0: dv/dt = -[beta v - (beta+nu2) v/w + nu2]
    for i in range(n - 2, -1, -1):
        h = tg[i + 1] - tg[i]
        # RK4 with coefficients interpolated at fractional grid index
        def f(idx_f, vv):
            # interpolate coefficients at fractional index
            i0 = int(np.floor(idx_f))
            i0 = min(max(i0, 0), n - 2)
            fr = idx_f - i0
            bi = bta[i0] * (1 - fr) + bta[i0 + 1] * fr
            wi = w[i0] * (1 - fr) + w[i0 + 1] * fr
            ui = nu2[i0] * (1 - fr) + nu2[i0 + 1] * fr
            return -(bi * vv - (bi + ui) * vv / wi + ui)
        x0 = i + 1.0
        k1 = f(x0, v[i + 1])
        k2 = f(x0 - 0.5, v[i + 1] - 0.5 * h * k1)
        k3 = f(x0 - 0.5, v[i + 1] - 0.5 * h * k2)
        k4 = f(x0 - 1.0, v[i + 1] - h * k3)
        v[i] = v[i + 1] - h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return v


def solve_q(nu2, tg, v0, delta, sig0=gc.SIG0):
    """q(t) forward in t from q(0) = 1/v0. Exponential Euler (exact for
    piecewise-constant coefficients)."""
    w = gc.Sig2(tg, sig0) + delta
    bta = gc.beta(tg)
    rate = bta - (bta + nu2) / w
    q = np.empty(len(tg))
    q[0] = 1.0 / v0
    for i in range(1, len(tg)):
        h = tg[i] - tg[i - 1]
        r = 0.5 * (rate[i] + rate[i - 1])
        q[i] = q[i - 1] * np.exp(r * h)
    return q


def kernel_exact(nu2, tg, delta, d=32, sig0=gc.SIG0):
    """Exact PMP kernel S(t) on the grid tg. Returns (S, v, q)."""
    v = solve_v(nu2, tg, delta, sig0)
    q = solve_q(nu2, tg, v[0], delta, sig0)
    w = gc.Sig2(tg, sig0) + delta
    S = d * q * (w - v) / w
    return S, v, q


def fixed_point_exact(lam, delta, d=32, M=40, n_iter=500, tol=1e-12,
                      sig0=gc.SIG0):
    """Exact Picard fixed point on the coarse grid. Returns (tg, nu2, res)."""
    tg = np.linspace(0.0, gc.T, M + 1)
    g2 = gc.g(tg) ** 2
    nu2 = g2.copy()
    res = []
    for _ in range(n_iter):
        S, v, q = kernel_exact(nu2, tg, delta, d, sig0)
        nu2_new = np.clip(g2 * (1.0 + S / (2.0 * lam)), gc.NU_MIN**2,
                          gc.NU_MAX**2)
        r = np.sqrt(np.trapezoid((np.sqrt(nu2_new) - np.sqrt(nu2))**2, tg))
        res.append(r)
        nu2 = nu2_new
        if r < tol:
            break
    return tg, nu2, np.array(res)
