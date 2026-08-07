"""run_picard.py DELTA LAMBDA N_ITER TAG [save_clouds]

Picard fixed-point for one scenario. Saves res/picard_TAG.npz with the
schedule history, kernel history, residuals; optionally the terminal clouds
of the final iterate and of the uncontrolled sampler (same CRN seed).
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gmm_control as gc
import particle_solver as ps

delta = float(sys.argv[1])
lam = float(sys.argv[2])
n_iter = int(sys.argv[3])
tag = sys.argv[4]
save_clouds = len(sys.argv) > 5 and sys.argv[5] == "save_clouds"
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 0

d, K = 32, 8
N, M, Nf = 20_000, 40, 800
MU = gc.make_modes(d, K)
wts = gc.make_weights(K)

out = ps.picard_particle(MU, wts, delta, lam, N=N, M=M, Nf=Nf, seed=SEED,
                         n_iter=n_iter)
os.makedirs("res", exist_ok=True)
np.savez(f"res/picard_{tag}.npz", tg=out["tg"], nu2=out["nu2"],
         S=out["S"], res=out["res"], delta=delta, lam=lam)

if save_clouds:
    np.save(f"res/X0_{tag}_star.npy", np.asarray(out["Xs_last"][0]))
    tg = out["tg"]
    Xunc, _ = ps.forward_particles(gc.g(tg) ** 2, MU, wts, delta, N=N, M=M,
                                   Nf=Nf, seed=SEED, store_slices=False)
    np.save(f"res/X0_{tag}_unc.npy", Xunc)
print("DONE", tag, flush=True)
