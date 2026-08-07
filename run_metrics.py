"""run_metrics.py

Builds the performance table (d = 32, K = 8): for each scenario
(boost delta=-0.1, damp delta=+0.1) compare the uncontrolled sampler and
the converged particle schedule on entropy gap, sliced W2, objective J.
All clouds compared within a scenario share CRN seeds; the true sample and
projection set are fixed across everything.
Saves res/table_d32.json.
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gmm_control as gc
import metrics as mt

d, K = 32, 8
MU = gc.make_modes(d, K)
wts = gc.make_weights(K)

H0, H0se = gc.H_p0_exact(MU, wts)
print(f"H(p0) = {H0:.4f} +- {H0se:.4f}", flush=True)

X_true, _ = gc.sample_p0(20_000, MU, wts, seed=12345)

out = {"H_p0": H0, "H_p0_se": H0se, "scenarios": {}}
for tag, delta, LAM in [("boost_l0.8", -0.1, 0.8), ("damp_l16", +0.1, 16.0)]:
    sc = {}
    z = np.load(f"res/picard_{tag}.npz")
    tg, nu2_star = z["tg"], z["nu2"][-1]
    for name, Xf, nu2 in [("unc", f"res/X0_{tag}_unc.npy", gc.g(tg) ** 2),
                          ("star", f"res/X0_{tag}_star.npy", nu2_star)]:
        X = np.load(Xf)
        H, Hse, w, v = mt.H_particles_gmm(X, MU)
        sw2 = mt.sliced_w2(X, X_true)
        J = mt.objective(H, nu2, tg, LAM)
        sc[name] = dict(H=H, H_se=Hse, gap=H0 - H, sw2=sw2, J=J)
        print(f"{tag}/{name}: H={H:.3f} gap={H0-H:.3f} sw2={sw2:.4f} J={J:.3f}",
              flush=True)
    out["scenarios"][tag] = sc

with open("res/table_d32.json", "w") as f:
    json.dump(out, f, indent=2)
print("DONE metrics", flush=True)
