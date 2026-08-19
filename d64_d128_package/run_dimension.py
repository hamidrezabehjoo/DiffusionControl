"""run_dimension.py — dimension-scaling experiments (d = 64, 128, ...)

Runs the Section 4.4 protocol of the paper at dimension D on the shared
K = 8 Gaussian-mixture testbed, writing one self-contained results folder
res_d{D}/ that can be sent back for analysis.

Usage:
    python3 run_dimension.py D [STAGES] [SEED]

    D      ambient dimension (e.g. 64 or 128)
    STAGES comma-separated subset of
               k1kernel  first-iterate kernel, K = 1, delta in {-0.1,0,+0.1}
                         (+ exact Riccati reference; dimension-free check)
               k8kernel  first-iterate kernel, K = 8 testbed, delta = -/+0.1
               boost     Picard fixed point, delta = -0.1, lam = 0.8 * D/32
               damp      Picard fixed point, delta = +0.1, lam = 16 * D/32
               metrics   H(p0), entropy gap, sliced W2, objective J,
                         mode occupancies (needs boost + damp done)
           default: k1kernel,k8kernel,boost,damp,metrics
    SEED   common-random-number seed (default 0)

Penalty scaling: the entropy kernel scales linearly with d (Section 3 of
the paper), so the dimension-appropriate penalties are lam = 0.8 * (D/32)
(boost) and lam = 16 * (D/32) (damp). With the unscaled d = 32 penalties
the boost update would simply be D/32 times stronger and run into the
upper admissible boundary.

Checkpoints: every stage writes its output file as soon as it finishes and
is skipped on re-run if the file exists (delete the file to force a
re-run); the Picard stages additionally checkpoint after EVERY iteration,
so an interrupted run resumes where it stopped.

Requirements: Python 3.10+, NumPy (>= 2.0 preferred; older versions are
handled by the np.trapezoid shim below). No other dependencies.
"""
import os
import sys
import json
import time

import numpy as np

if not hasattr(np, "trapezoid"):          # NumPy < 2.0 compatibility
    np.trapezoid = np.trapz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gmm_control as gc
import particle_solver as ps
import exact_k1 as ek
import metrics as mt

# --------------------------------------------------------------------- setup
D = int(sys.argv[1]) if len(sys.argv) > 1 else 64
STAGES = (sys.argv[2].split(",") if len(sys.argv) > 2
          else ["k1kernel", "k8kernel", "boost", "damp", "metrics"])
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0

K = 8
N, M, Nf = 20_000, 40, 800
N_ITER = 12
LAM_BOOST = 0.8 * D / 32.0
LAM_DAMP = 16.0 * D / 32.0
DELTA_BOOST, DELTA_DAMP = -0.1, +0.1

MU = gc.make_modes(D, K)            # centers redrawn in dimension D, seed 0
wts = gc.make_weights(K)
tg = np.linspace(0.0, gc.T, M + 1)
g2 = gc.g(tg) ** 2

RES = f"res_d{D}"
os.makedirs(RES, exist_ok=True)


def done(fname):
    return os.path.exists(os.path.join(RES, fname))


def tic(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    return time.time()


def toc(t0):
    print(f"    ({(time.time() - t0) / 60.0:.1f} min)", flush=True)


# ------------------------------------------------------------------ k1kernel
def stage_k1kernel():
    if done("k1_kernel.npz"):
        print("k1kernel: already done, skipping", flush=True)
        return
    MU1 = gc.make_modes(D, 1)
    w1 = gc.make_weights(1)
    out = {"tg": tg, "d": D}
    for delta, name in [(DELTA_BOOST, "boost"), (0.0, "triv"),
                        (DELTA_DAMP, "damp")]:
        t0 = tic(f"k1kernel: forward+kernel, delta={delta:+.1f}")
        Xs, tsl = ps.forward_particles(g2, MU1, w1, delta, N=N, M=M, Nf=Nf,
                                       seed=SEED)
        S = ps.kernel_from_particles(Xs, tsl, MU1, w1, delta, M)
        out[f"S_part_{name}"] = S
        S_ex, _, _ = ek.kernel_exact(g2, tg, delta, D)
        out[f"S_exact_{name}"] = S_ex
        tic(f"    S(0) particle {S[0]:+.3f} vs exact {S_ex[0]:+.3f} "
            f"(S/d: {S[0] / D:+.5f} vs {S_ex[0] / D:+.5f})")
        toc(t0)
    np.savez(os.path.join(RES, "k1_kernel.npz"), **out)


# ------------------------------------------------------------------ k8kernel
def stage_k8kernel():
    if done("k8_kernel.npz"):
        print("k8kernel: already done, skipping", flush=True)
        return
    out = {"tg": tg, "d": D}
    for delta, name in [(DELTA_BOOST, "boost"), (DELTA_DAMP, "damp")]:
        t0 = tic(f"k8kernel: forward+kernel, delta={delta:+.1f}")
        Xs, tsl = ps.forward_particles(g2, MU, wts, delta, N=N, M=M, Nf=Nf,
                                       seed=SEED)
        S = ps.kernel_from_particles(Xs, tsl, MU, wts, delta, M)
        out[f"S_{name}"] = S
        tic(f"    S(0) = {S[0]:+.3f}  (S/d = {S[0] / D:+.5f})")
        toc(t0)
    np.savez(os.path.join(RES, "k8_kernel.npz"), **out)


# -------------------------------------------------------------- picard stage
def picard_stage(tag, delta, lam):
    """Resumable Picard fixed point with per-iteration checkpointing."""
    ckpt = os.path.join(RES, f"picard_{tag}.npz")
    if done(f"X0_{tag}_star.npy"):
        print(f"{tag}: already done, skipping", flush=True)
        return
    if os.path.exists(ckpt):
        z = np.load(ckpt)
        nu2 = z["nu2"][-1].copy()
        hist_nu2 = [a for a in z["nu2"]]
        hist_S = [a for a in z["S"]]
        hist_res = [r for r in z["res"]]
        tic(f"{tag}: resuming from iteration {len(hist_res)}")
    else:
        nu2 = g2.copy()
        hist_nu2, hist_S, hist_res = [nu2.copy()], [], []
    tol = 1e-3 * np.sqrt(np.trapezoid(g2, tg))
    Xs_last = None
    while len(hist_res) < N_ITER:
        t0 = tic(f"{tag}: iteration {len(hist_res)} "
                 f"(lam={lam:.2f}, delta={delta:+.1f})")
        Xs, tsl = ps.forward_particles(nu2, MU, wts, delta, N=N, M=M, Nf=Nf,
                                       seed=SEED)
        S = ps.kernel_from_particles(Xs, tsl, MU, wts, delta, M)
        nu2_new = np.clip(g2 * (1.0 + S / (2.0 * lam)), gc.NU_MIN ** 2,
                          gc.NU_MAX ** 2)
        res = float(np.sqrt(np.trapezoid((np.sqrt(nu2_new)
                                          - np.sqrt(nu2)) ** 2, tg)))
        hist_S.append(S.copy())
        hist_res.append(res)
        nu2 = nu2_new
        hist_nu2.append(nu2.copy())
        Xs_last = Xs
        tic(f"    res={res:.4e}  nu*/g(t=0)={np.sqrt(nu2[0] / g2[0]):.3f}")
        np.savez(ckpt, tg=tg, nu2=np.array(hist_nu2), S=np.array(hist_S),
                 res=np.array(hist_res), delta=delta, lam=lam, d=D)
        toc(t0)
        if res < tol:
            tic(f"{tag}: converged")
            break
    np.save(os.path.join(RES, f"X0_{tag}_star.npy"),
            np.asarray(Xs_last[0]))
    t0 = tic(f"{tag}: uncontrolled cloud (same CRN seed)")
    Xunc, _ = ps.forward_particles(g2, MU, wts, delta, N=N, M=M, Nf=Nf,
                                   seed=SEED, store_slices=False)
    np.save(os.path.join(RES, f"X0_{tag}_unc.npy"), Xunc)
    toc(t0)


# ------------------------------------------------------------------- metrics
def stage_metrics():
    if done("metrics.json"):
        print("metrics: already done, skipping", flush=True)
        return
    t0 = tic("metrics: evaluating")
    Hp0, Hp0_se = gc.H_p0_exact(MU, wts, n=200_000, seed=11)
    X_true, _ = gc.sample_p0(20_000, MU, wts, seed=12345)
    out = {"d": D, "H_p0": Hp0, "H_p0_se": Hp0_se,
           "R0": float(np.linalg.norm(MU, axis=1).max()),
           "lam_boost": LAM_BOOST, "lam_damp": LAM_DAMP, "scenarios": {}}

    def occupancy(X):
        r = gc.responsibilities(X, MU, gc.SIG0 ** 2, wts)
        ks = r.argmax(axis=1)
        return [float((ks == k).mean()) for k in range(K)]

    def cloud(X, nu2v, lam):
        H, se, w, v = mt.H_particles_gmm(X, MU)
        return dict(H=H, H_se=se, gap=Hp0 - H,
                    sw2=mt.sliced_w2(np.asarray(X), X_true),
                    J=mt.objective(H, nu2v, tg, lam),
                    occ=occupancy(X))

    for name, tag, lam in [("boost", "boost", LAM_BOOST),
                           ("damp", "damp", LAM_DAMP)]:
        z = np.load(os.path.join(RES, f"picard_{tag}.npz"))
        nu2_star = z["nu2"][-1]
        star = cloud(np.load(os.path.join(RES, f"X0_{tag}_star.npy")),
                     nu2_star, lam)
        unc = cloud(np.load(os.path.join(RES, f"X0_{tag}_unc.npy")), g2, lam)
        star["dJ"] = star["J"] - unc["J"]
        out["scenarios"][name] = dict(unc=unc, star=star)
        tic(f"metrics {name}: gap {unc['gap']:.3f} -> {star['gap']:.3f}, "
            f"sw2 {unc['sw2']:.4f} -> {star['sw2']:.4f}, "
            f"dJ={star['dJ']:+.4f}")
    with open(os.path.join(RES, "metrics.json"), "w") as f:
        json.dump(out, f, indent=1)
    toc(t0)


# ---------------------------------------------------------------------- main
print(f"=== dimension D = {D}, seed {SEED}, stages {STAGES} ===", flush=True)
print(f"    lam_boost = {LAM_BOOST}, lam_damp = {LAM_DAMP} "
      f"(scaled by D/32 from the d = 32 values 0.8 / 16)", flush=True)
for st in STAGES:
    if st == "k1kernel":
        stage_k1kernel()
    elif st == "k8kernel":
        stage_k8kernel()
    elif st == "boost":
        picard_stage("boost", DELTA_BOOST, LAM_BOOST)
    elif st == "damp":
        picard_stage("damp", DELTA_DAMP, LAM_DAMP)
    elif st == "metrics":
        stage_metrics()
    else:
        raise ValueError(f"unknown stage {st!r}")
print(f"=== D = {D} done; results in {RES}/ ===", flush=True)
