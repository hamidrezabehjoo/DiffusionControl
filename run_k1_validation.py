"""run_k1_validation.py

Exact K=1 ground-truth validation of the particle solver (d = 32):
  (i) first-iterate particle kernel S_hat at nu = g for delta in
      {-0.1, 0, +0.1}, three seeds -> mean +/- sd vs the exact Riccati
      kernel (sign check + exact-score triviality);
  (ii) particle Picard fixed point for delta = -0.1, lambda = 0.8 vs the
      exact fixed point;
  (iii) exact fixed point for delta = +0.1, lambda = 0.8 (boundary collapse)
      and lambda = 16 (interior), for reference.
Writes res/k1_validation.npz.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gmm_control as gc
import particle_solver as ps
import exact_k1 as ek

d = 32
N, M, Nf = 20_000, 40, 800
MU = gc.make_modes(d, 1)
wts = gc.make_weights(1)
tg = np.linspace(0.0, gc.T, M + 1)
g2 = gc.g(tg) ** 2

out = {"tg": tg}

# (i) first-iterate kernels, 3 seeds
for delta, name in [(-0.1, "boost"), (0.0, "triv"), (+0.1, "damp")]:
    S_seeds = []
    for seed in (0, 1, 2):
        Xs, tsl = ps.forward_particles(g2, MU, wts, delta, N=N, M=M, Nf=Nf,
                                       seed=seed)
        S = ps.kernel_from_particles(Xs, tsl, MU, wts, delta, M)
        S_seeds.append(S)
        print(f"first-iterate K=1 delta={delta:+.1f} seed={seed}: "
              f"S(0)={S[0]:+.4f}", flush=True)
    out[f"S_part_{name}"] = np.array(S_seeds)
    S_ex, v, q = ek.kernel_exact(g2, tg, delta, d)
    out[f"S_exact_{name}"] = S_ex

# (ii) particle fixed point, boost
res = ps.picard_particle(MU, wts, -0.1, 0.8, N=N, M=M, Nf=Nf, seed=0,
                         n_iter=12)
out["nu2_part_boost"] = res["nu2"][-1]
out["res_part_boost"] = res["res"]
tg_e, nu2_e, res_e = ek.fixed_point_exact(0.8, -0.1, d, M)
out["nu2_exact_boost"] = nu2_e
print("boost fixed point: particle ratio(0) =",
      round(np.sqrt(res['nu2'][-1][0] / g2[0]), 3), " exact =",
      round(np.sqrt(nu2_e[0] / g2[0]), 3), flush=True)

# (iii) exact damp fixed points
_, nu2_d8, _ = ek.fixed_point_exact(0.8, +0.1, d, M)
_, nu2_d16, _ = ek.fixed_point_exact(16.0, +0.1, d, M)
out["nu2_exact_damp_l0.8"] = nu2_d8
out["nu2_exact_damp_l16"] = nu2_d16
print("exact damp l=0.8: nu(0) =", round(np.sqrt(nu2_d8[0]), 4),
      " l=16: ratio(0) =", round(np.sqrt(nu2_d16[0] / g2[0]), 3), flush=True)

np.savez("res/k1_validation.npz", **out)
print("DONE k1", flush=True)
