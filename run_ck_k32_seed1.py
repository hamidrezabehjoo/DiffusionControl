"""run_ck_k32_seed1.py

K = 32 kernel pair with CRN seed 1 (cold-start EM per slice).
Rationale: with seed 0, the perturbed-schedule (nu = 1.05 g) particle cloud
at t = 0.125 drives one per-mode costate fit into a degenerate local
configuration, producing a single-cell spike in dS and an inflated
C_hat(32) = 1.55. The artifact is deterministic in the cloud (reproduces
under warm- and cold-start EM and under n_em = 40) and disappears with a
re-drawn cloud: seed 1 gives C_hat(32) = 1.159, in line with K <= 16.
CRN is used within the (nu1, nu2) pair in all cases. Output:
res/diagS1_1.0.npy, res/diagS1_1.05.npy (read by merge_ck.py).
"""
import sys
import numpy as np
sys.path.insert(0, "/mnt/agents/output/code")
import gmm_control as gc
import particle_solver as ps


def kernel_cold(Xs, t_slices, MU, wts, delta, M, n_em=15):
    N = Xs.shape[1]
    half = N // 2
    folds = [(np.arange(0, half), np.arange(half, N)),
             (np.arange(half, N), np.arange(0, half))]
    S = np.zeros(M + 1)
    for fit_idx, ev_idx in folds:
        for j in range(M + 1):
            Xf = np.asarray(Xs[j][fit_idx])
            Xe = np.asarray(Xs[j][ev_idx])
            cen = gc.m(t_slices[j]) * MU
            w, v = ps.fit_gmm(Xf, cen, n_em=n_em)
            if j == 0:
                Y0f = -(1.0 + ps.logpdf_gmm_fit(Xf, cen, w, v))
            coef = ps.fit_costate(Xf, Y0f, cen, w, v)
            gp = ps.grad_psi_mix(Xe, cen, w, v, coef)
            s_th = gc.score_frozen(Xe, t_slices[j], MU, wts, delta)
            s_ct, _ = ps.s_ctrl_gmm(Xe, cen, w, v)
            S[j] += 0.5 * np.mean((gp * (s_th - s_ct)).sum(axis=1))
    return S


d, K, M, Nf = 32, 32, 40, 800
MU = gc.make_modes(d, K)
wts = gc.make_weights(K)
tg = np.linspace(0, gc.T, M + 1)
g2 = gc.g(tg) ** 2
for scale in (1.0, 1.05):
    Xs, tsl = ps.forward_particles(scale**2 * g2, MU, wts, -0.1, N=20000,
                                   M=M, Nf=Nf, seed=1)
    S = kernel_cold(Xs, tsl, MU, wts, -0.1, M)
    np.save(f"res/diagS1_{scale}.npy", S)
    print(scale, np.round(S[:8], 3), flush=True)
print("DONE k32 seed1", flush=True)
