"""run_lam_clouds.py

Terminal clouds (seed 0) for the converged boost schedules at
lambda = 0.4 and 1.6 (histories already in res/picard_boost_l*.npz);
used for the SI lambda-sensitivity table.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gmm_control as gc
import particle_solver as ps

d, K = 32, 8
N, M, Nf, SEED = 20_000, 40, 800, 0
MU = gc.make_modes(d, K)
wts = gc.make_weights(K)

for lam in ("0.4", "1.6"):
    z = np.load(f"res/picard_boost_l{lam}.npz")
    nu2 = z["nu2"][-1]
    X, _ = ps.forward_particles(nu2, MU, wts, -0.1, N=N, M=M, Nf=Nf,
                                seed=SEED, store_slices=False)
    np.save(f"res/X0_boost_l{lam}_star.npy", X)
    print("done", lam, flush=True)
print("DONE lam clouds", flush=True)
