"""run_reduced.py SEED

Grid-searches the reduced controller nu = alpha g on [0, t_w] (boost
scenario, lambda = 0.8) on the objective J, with common random numbers,
and saves the terminal cloud of the best configuration for metrics.
Writes res/reduced_sSEED.npz and res/X0_reduced_sSEED.npy.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gmm_control as gc
import particle_solver as ps
import metrics as mt

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
d, K, LAM = 32, 8, 0.8
N, M, Nf = 20_000, 40, 800
MU = gc.make_modes(d, K)
wts = gc.make_weights(K)
tg = np.linspace(0.0, gc.T, M + 1)
g2 = gc.g(tg) ** 2

ALPHAS = [1.2, 1.35, 1.5, 1.65]
TWS = [0.1, 0.2, 0.3, 0.45]

if SEED == 0:
    best = None
    log = []
    for alpha in ALPHAS:
        for tw in TWS:
            nu2 = np.where(tg <= tw, alpha**2 * g2, g2)
            X, _ = ps.forward_particles(nu2, MU, wts, -0.1, N=N, M=M, Nf=Nf,
                                        seed=SEED, store_slices=False)
            H, _, _, _ = mt.H_particles_gmm(X, MU)
            J = mt.objective(H, nu2, tg, LAM)
            log.append((alpha, tw, H, J))
            print(f"alpha={alpha} tw={tw}: H={H:.3f} J={J:.4f}", flush=True)
            if best is None or J > best[3]:
                best = (alpha, tw, H, J)
else:
    z0 = np.load("res/reduced_s0.npz")
    best = (float(z0["alpha"]), float(z0["tw"]), None, None)
    log = []

alpha, tw = best[0], best[1]
print(f"BEST: alpha={alpha} tw={tw}", flush=True)
nu2 = np.where(tg <= tw, alpha**2 * g2, g2)
X, _ = ps.forward_particles(nu2, MU, wts, -0.1, N=N, M=M, Nf=Nf, seed=SEED,
                            store_slices=False)
np.save(f"res/X0_reduced_s{SEED}.npy", X)
if SEED == 0:
    np.savez(f"res/reduced_s{SEED}.npz", alpha=alpha, tw=tw, log=np.array(log),
             tg=tg, nu2=nu2)
print("DONE reduced", SEED, flush=True)
