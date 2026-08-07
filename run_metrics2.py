"""run_metrics2.py

Multi-seed metrics for the revised Table 2, the lambda-sensitivity table,
and the mode-occupancy table (Section 4.4 / SI Appendix).

Reads terminal clouds produced by run_picard.py (seeds 0-3), run_reduced.py
(seeds 0-3) and run_lam_clouds.py (seed 0) and writes res/table2_multi.json
with per-seed and mean +/- std values of
  H(rho0), entropy gap, sliced W2, objective J, mode occupancies.

Common random numbers: within each seed, the uncontrolled and controlled
clouds share prior and Brownian draws, so paired differences are meaningful.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gmm_control as gc
import particle_solver as ps
import metrics as mt

d, K = 32, 8
MU = gc.make_modes(d, K)
wts = gc.make_weights(K)
SEEDS = [0, 1, 2]

# fixed reference sample of p0 for sliced W2 (shared by all evaluations)
X_true, ks_true = gc.sample_p0(20_000, MU, wts, seed=12345)


def tag_for(base, seed):
    return base if seed == 0 else f"{base}_s{seed}"


def occupancy(X):
    """Fraction of samples assigned to each true mode (argmax posterior)."""
    r = gc.responsibilities(X, MU, gc.SIG0 ** 2, wts)
    ks = r.argmax(axis=1)
    return np.array([(ks == k).mean() for k in range(K)])


def cloud_metrics(X, nu2, tg, lam):
    H, se, w, v = mt.H_particles_gmm(X, MU)
    sw2 = mt.sliced_w2(np.asarray(X), X_true)
    J = mt.objective(H, nu2, tg, lam)
    return dict(H=H, H_se=se, sw2=sw2, J=J, occ=occupancy(X).tolist())


def boundary_fraction(nu2):
    lo = np.isclose(np.sqrt(nu2), gc.NU_MIN, atol=1e-6)
    hi = np.isclose(np.sqrt(nu2), gc.NU_MAX, atol=1e-6)
    return float((lo | hi).mean())


out = {"H_p0": None, "scenarios": {}}
Hp0, Hp0_se = gc.H_p0_exact(MU, wts, n=200_000, seed=11)
out["H_p0"] = dict(H=Hp0, se=Hp0_se)
print(f"H(p0) = {Hp0:.4f} +/- {Hp0_se:.4f}", flush=True)

# ---------------------------------------------------------------- main table
for name, base, delta, lam in [("boost", "boost_l0.8", -0.1, 0.8),
                               ("damp", "damp_l16", +0.1, 16.0)]:
    rows = []
    for s in SEEDS:
        tag = tag_for(base, s)
        z = np.load(f"res/picard_{tag}.npz")
        tg, nu2 = z["tg"], z["nu2"][-1]
        star = cloud_metrics(np.load(f"res/X0_{tag}_star.npy"), nu2, tg, lam)
        unc = cloud_metrics(np.load(f"res/X0_{tag}_unc.npy"),
                            gc.g(tg) ** 2, tg, lam)
        star["gap"] = Hp0 - star["H"]
        unc["gap"] = Hp0 - unc["H"]
        star["dJ"] = star["J"] - unc["J"]
        star["boundary"] = boundary_fraction(nu2)
        rows.append(dict(seed=s, unc=unc, star=star))
        print(f"{name} seed {s}: H_unc={unc['H']:.3f} H_star={star['H']:.3f} "
              f"dJ={star['dJ']:+.4f}", flush=True)
    out["scenarios"][name] = dict(delta=delta, lam=lam, rows=rows)

# ------------------------------------------------------- reduced controller
z0 = np.load("res/reduced_s0.npz")
alpha, tw = float(z0["alpha"]), float(z0["tw"])
tg = z0["tg"]
nu2_red = np.where(tg <= tw, alpha ** 2 * gc.g(tg) ** 2, gc.g(tg) ** 2)
rows = []
for s in SEEDS:
    red = cloud_metrics(np.load(f"res/X0_reduced_s{s}.npy"), nu2_red, tg, 0.8)
    red["gap"] = Hp0 - red["H"]
    rows.append(dict(seed=s, red=red))
    print(f"reduced seed {s}: H={red['H']:.3f} J={red['J']:.4f}", flush=True)
out["reduced"] = dict(alpha=alpha, tw=tw, rows=rows)

# ------------------------------------------------------ lambda sensitivity
lam_rows = []
for name, tag, delta, lam in [("Boost", "boost_l0.4", -0.1, 0.4),
                              ("Boost", "boost_l0.8", -0.1, 0.8),
                              ("Boost", "boost_l1.6", -0.1, 1.6),
                              ("Damp", "damp_l0.8", +0.1, 0.8),
                              ("Damp", "damp_l16", +0.1, 16.0)]:
    z = np.load(f"res/picard_{tag}.npz")
    tg, nu2 = z["tg"], z["nu2"][-1]
    m = cloud_metrics(np.load(f"res/X0_{tag}_star.npy"), nu2, tg, lam)
    m["gap"] = Hp0 - m["H"]
    m["boundary"] = boundary_fraction(nu2)
    lam_rows.append(dict(scenario=name, tag=tag, lam=lam, **m))
    print(f"lam-table {tag}: H={m['H']:.3f} sw2={m['sw2']:.4f} "
          f"J={m['J']:.4f} bnd={m['boundary']:.2f}", flush=True)
out["lam_table"] = lam_rows

with open("res/table2_multi.json", "w") as f:
    json.dump(out, f, indent=1)
print("DONE metrics2", flush=True)
