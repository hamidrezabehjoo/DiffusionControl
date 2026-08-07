"""run_ck.py

Kernel Lipschitz constant vs number of modes K in d = 32:
    C_hat(K) = ||S_{nu1} - S_{nu2}||_L2 / ||nu1 - nu2||_L2,
    nu1 = g, nu2 = 1.05 g   (so ||nu1 - nu2||_L2 = 0.05 ||g||_L2),
on nested K-mode subsets of the testbed mixture (first K modes of the frozen
center draw, period-8 weights renormalized), boost scenario delta = -0.1,
particle estimator with common random numbers (identical seed for nu1, nu2).
Saves res/ck_d32.npz.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gmm_control as gc
import particle_solver as ps

d = 64
N, M, Nf, SEED = 20_000, 40, 800, 0
Ks = [int(k) for k in sys.argv[1].split(",")] if len(sys.argv) > 1 else [2, 4, 8, 16, 32]
tag = sys.argv[2] if len(sys.argv) > 2 else "d32"
n_em = int(sys.argv[3]) if len(sys.argv) > 3 else 15
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
tg = np.linspace(0.0, gc.T, M + 1)
g2 = gc.g(tg) ** 2

Chat = np.empty(len(Ks))
S_all = []
for i, K in enumerate(Ks):
    MU = gc.make_modes(d, K)
    wts = gc.make_weights(K)
    Sk = []
    for scale in (1.0, 1.05):
        Xs, tsl = ps.forward_particles(scale**2 * g2, MU, wts, -0.1, N=N,
                                       M=M, Nf=Nf, seed=SEED)
        S = ps.kernel_from_particles(Xs, tsl, MU, wts, -0.1, M, n_em=n_em)
        Sk.append(S)
        print(f"K={K} scale={scale}: done, S(0)={S[0]:+.3f}", flush=True)
    dS = Sk[1] - Sk[0]
    Chat[i] = np.sqrt(np.trapezoid(dS**2, tg)) / (0.05 * np.sqrt(np.trapezoid(g2, tg)))
    S_all.append(np.array(Sk))
    print(f"K={K}: C_hat={Chat[i]:.4f}", flush=True)

np.savez(f"res/ck_{tag}.npz", Ks=np.array(Ks), Chat=Chat, S=np.array(S_all), tg=tg)
print("DONE ck", flush=True)
