#!/usr/bin/env python3
"""run_synthetic.py — synthetic validation suite (Gaussian / GMM testbed).

Runs the full synthetic verification protocol of the paper at a chosen
ambient dimension D (headline: D = 128) on the shared K-mode Gaussian
mixture testbed (diffusion_control.gmm_control).

Usage:
    python3 run_synthetic.py [--dim 128] [--seed 0] [--stages a,b,c]

Stages (each checkpoints to results/d{D}/seed{S}/ and skips if done):
    k1kernel      first-iterate kernel, K = 1, delta in {-0.1, 0, +0.1},
                  against the exact Riccati kernel (sign check, exact-score
                  triviality; dimension-free collapse of S/d)
    k1fixed       K = 1 particle Picard fixed point vs exact fixed point
                  (boost), plus exact damp fixed points (boundary/interior)
    k8kernel      first-iterate kernel, K = 8 testbed, delta = -/+0.1
    boost         Picard fixed point, delta = -0.1, lam = 0.8*(D/32)
    damp          Picard fixed point, delta = +0.1, lam = 16*(D/32)
    damp_boundary Picard fixed point, delta = +0.1, lam = 0.8*(D/32)
                  (degenerate: schedule pins to the lower boundary)
    reduced       reduced controller nu = alpha g on [0, t_w]: seed 0
                  grid-searches (alpha, t_w) on the objective J; other
                  seeds re-evaluate the seed-0 optimum under CRN
    metrics       entropy gap, sliced W2, objective J, mode occupancies
                  for every terminal cloud present in the results folder

Penalty scaling: the entropy kernel scales linearly with d, so the
dimension-appropriate penalties are lam_boost = 0.8*(D/32) and
lam_damp = 16*(D/32), making the update g^2 S/(2 lam) dimension-free.

Requirements: Python 3.10+, NumPy >= 1.24. Pure CPU, no GPU.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

if not hasattr(np, "trapezoid"):            # NumPy < 2.0 compatibility
    np.trapezoid = np.trapz

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from diffusion_control import gmm_control as gc          # noqa: E402
from diffusion_control import particle_solver as ps      # noqa: E402
from diffusion_control import exact_k1 as ek             # noqa: E402
from diffusion_control import metrics as mt              # noqa: E402

# --------------------------------------------------------------------- args
P = argparse.ArgumentParser()
P.add_argument("--dim", type=int, default=128)
P.add_argument("--seed", type=int, default=0)
P.add_argument("--stages", type=str,
               default="k1kernel,k1fixed,k8kernel,boost,damp,reduced,metrics")
P.add_argument("--n-iter", type=int, default=12)
ARGS = P.parse_args()

D, SEED = ARGS.dim, ARGS.seed
STAGES = ARGS.stages.split(",")
N_ITER = ARGS.n_iter

K = 8
N, M, Nf = 20_000, 40, 800
LAM_BOOST = 0.8 * D / 32.0
LAM_DAMP = 16.0 * D / 32.0
DELTA_BOOST, DELTA_DAMP = -0.1, +0.1

MU = gc.make_modes(D, K)               # centers Uniform[-4,4]^D, seed 0
wts = gc.make_weights(K)
MU1 = gc.make_modes(D, 1)
w1 = gc.make_weights(1)
tg = np.linspace(0.0, gc.T, M + 1)
g2 = gc.g(tg) ** 2

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", f"d{D}", f"seed{SEED}")
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


# ------------------------------------------------------------------- k1fixed
def stage_k1fixed():
    if done("k1_fixed.npz"):
        print("k1fixed: already done, skipping", flush=True)
        return
    out = {"tg": tg, "d": D}
    t0 = tic("k1fixed: exact fixed points (Riccati Picard)")
    _, nu2_eb, res_eb = ek.fixed_point_exact(LAM_BOOST, DELTA_BOOST, D, M)
    _, nu2_ed_bnd, _ = ek.fixed_point_exact(LAM_BOOST, DELTA_DAMP, D, M)
    _, nu2_ed_int, _ = ek.fixed_point_exact(LAM_DAMP, DELTA_DAMP, D, M)
    out["nu2_exact_boost"] = nu2_eb
    out["res_exact_boost"] = res_eb
    out["nu2_exact_damp_boundary"] = nu2_ed_bnd
    out["nu2_exact_damp_interior"] = nu2_ed_int
    tic(f"    exact boost: nu*/g(0) = {np.sqrt(nu2_eb[0] / g2[0]):.3f}; "
        f"damp lam={LAM_BOOST:.1f}: nu(0) = {np.sqrt(nu2_ed_bnd[0]):.4f} "
        f"(boundary); damp lam={LAM_DAMP:.1f}: "
        f"nu*/g(0) = {np.sqrt(nu2_ed_int[0] / g2[0]):.3f} (interior)")
    toc(t0)
    ckpt = os.path.join(RES, "k1_picard_boost.npz")
    if os.path.exists(ckpt):
        z = np.load(ckpt)
        out["nu2_part_boost"] = z["nu2"][-1]
        out["res_part_boost"] = z["res"]
    else:
        t0 = tic("k1fixed: particle Picard, boost scenario")
        r = ps.picard_particle(MU1, w1, DELTA_BOOST, LAM_BOOST, N=N, M=M,
                               Nf=Nf, seed=SEED, n_iter=N_ITER)
        out["nu2_part_boost"] = r["nu2"][-1]
        out["res_part_boost"] = r["res"]
        np.savez(ckpt, tg=tg, nu2=r["nu2"], S=r["S"], res=r["res"])
        tic(f"    particle boost: nu*/g(0) = "
            f"{np.sqrt(r['nu2'][-1][0] / g2[0]):.3f}")
        toc(t0)
    np.savez(os.path.join(RES, "k1_fixed.npz"), **out)


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
def picard_stage(tag, delta, lam, n_iter=None):
    """Resumable Picard fixed point with per-iteration checkpointing."""
    n_iter = n_iter or N_ITER
    ckpt = os.path.join(RES, f"picard_{tag}.npz")
    if done(f"X0_{tag}_star.npy"):
        print(f"{tag}: already done, skipping", flush=True)
        return
    if os.path.exists(ckpt):
        z = np.load(ckpt)
        nu2 = z["nu2"][-1].copy()
        hist_nu2 = [a for a in z["nu2"]]
        hist_S = [a for a in z["S"]]
        hist_res = [float(r) for r in z["res"]]
        tic(f"{tag}: resuming from iteration {len(hist_res)}")
    else:
        nu2 = g2.copy()
        hist_nu2, hist_S, hist_res = [nu2.copy()], [], []
    tol = 1e-3 * np.sqrt(np.trapezoid(g2, tg))
    Xs_last = None
    while len(hist_res) < n_iter:
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
    # Always evaluate the terminal cloud at the FINAL schedule nu2 (the
    # converged fixed point), not at the pre-update schedule of the last
    # Picard pass.
    t0 = tic(f"{tag}: terminal cloud at the fixed point")
    Xstar, _ = ps.forward_particles(nu2, MU, wts, delta, N=N, M=M, Nf=Nf,
                                    seed=SEED, store_slices=False)
    np.save(os.path.join(RES, f"X0_{tag}_star.npy"), np.asarray(Xstar))
    toc(t0)
    t0 = tic(f"{tag}: uncontrolled cloud (same CRN seed)")
    Xunc, _ = ps.forward_particles(g2, MU, wts, delta, N=N, M=M, Nf=Nf,
                                   seed=SEED, store_slices=False)
    np.save(os.path.join(RES, f"X0_{tag}_unc.npy"), Xunc)
    toc(t0)


# ------------------------------------------------------------------- reduced
def stage_reduced():
    """Reduced controller nu = alpha g on [0, t_w], boost scenario."""
    if done(f"X0_reduced.npy"):
        print("reduced: already done, skipping", flush=True)
        return
    ALPHAS = [1.2, 1.35, 1.5, 1.65]
    TWS = [0.1, 0.2, 0.3, 0.45]
    grid_file = os.path.join(RES, "..", "seed0", "reduced_grid.npz")
    if SEED == 0:
        t0 = tic("reduced: grid search over (alpha, t_w)")
        log = []
        for alpha in ALPHAS:
            for tw in TWS:
                nu2 = np.where(tg <= tw, alpha ** 2 * g2, g2)
                X, _ = ps.forward_particles(nu2, MU, wts, DELTA_BOOST, N=N,
                                            M=M, Nf=Nf, seed=SEED,
                                            store_slices=False)
                H, _, _, _ = mt.H_particles_gmm(X, MU)
                J = mt.objective(H, nu2, tg, LAM_BOOST)
                log.append((alpha, tw, H, J))
                print(f"    alpha={alpha} tw={tw}: H={H:.3f} J={J:.4f}",
                      flush=True)
        log = np.array(log)
        best = log[np.argmax(log[:, 3])]
        np.savez(os.path.join(RES, "reduced_grid.npz"), log=log,
                 alpha=best[0], tw=best[1])
        tic(f"    best: alpha={best[0]} tw={best[1]} J={best[3]:.4f}")
        toc(t0)
        alpha, tw = float(best[0]), float(best[1])
    else:
        z = np.load(grid_file)
        alpha, tw = float(z["alpha"]), float(z["tw"])
        tic(f"reduced: evaluating seed-0 optimum alpha={alpha}, tw={tw}")
    nu2 = np.where(tg <= tw, alpha ** 2 * g2, g2)
    X, _ = ps.forward_particles(nu2, MU, wts, DELTA_BOOST, N=N, M=M, Nf=Nf,
                                seed=SEED, store_slices=False)
    np.save(os.path.join(RES, "X0_reduced.npy"), X)
    np.savez(os.path.join(RES, "reduced_best.npz"), alpha=alpha, tw=tw,
             nu2=nu2, tg=tg)
    print("    done", flush=True)


# ------------------------------------------------------------------- metrics
def stage_metrics():
    if done("metrics.json"):
        print("metrics: already done, skipping", flush=True)
        return
    t0 = tic("metrics: evaluating")
    Hp0, Hp0_se = gc.H_p0_exact(MU, wts, n=200_000, seed=11)
    X_true, _ = gc.sample_p0(20_000, MU, wts, seed=12345)
    out = {"d": D, "seed": SEED, "H_p0": Hp0, "H_p0_se": Hp0_se,
           "R0": float(np.linalg.norm(MU, axis=1).max()),
           "lam_boost": LAM_BOOST, "lam_damp": LAM_DAMP, "entries": {}}

    def occupancy(X):
        r = gc.responsibilities(X, MU, gc.SIG0 ** 2, wts)
        ks = r.argmax(axis=1)
        return [float((ks == k).mean()) for k in range(K)]

    def cloud(X, nu2v, lam):
        X = np.asarray(X)
        H, se, w, v = mt.H_particles_gmm(X, MU)
        return dict(H=H, H_se=se, gap=Hp0 - H,
                    sw2=mt.sliced_w2(X, X_true),
                    J=mt.objective(H, nu2v, tg, lam),
                    occ=occupancy(X))

    entries = []
    for tag, lam in [("boost", LAM_BOOST), ("damp", LAM_DAMP)]:
        star_f = os.path.join(RES, f"X0_{tag}_star.npy")
        if not os.path.exists(star_f):
            continue
        z = np.load(os.path.join(RES, f"picard_{tag}.npz"))
        nu2_star = z["nu2"][-1]
        unc = cloud(np.load(os.path.join(RES, f"X0_{tag}_unc.npy")), g2, lam)
        star = cloud(np.load(star_f), nu2_star, lam)
        star["dJ"] = star["J"] - unc["J"]
        entries += [(f"{tag}_unc", unc), (f"{tag}_star", star)]
        tic(f"metrics {tag}: gap {unc['gap']:.3f} -> {star['gap']:.3f}, "
            f"sw2 {unc['sw2']:.4f} -> {star['sw2']:.4f}, "
            f"dJ={star['dJ']:+.4f}")
    red_f = os.path.join(RES, "X0_reduced.npy")
    if os.path.exists(red_f):
        z = np.load(os.path.join(RES, "reduced_best.npz"))
        red = cloud(np.load(red_f), z["nu2"], LAM_BOOST)
        red["alpha"], red["tw"] = float(z["alpha"]), float(z["tw"])
        entries.append(("reduced", red))
        tic(f"metrics reduced (alpha={red['alpha']}, tw={red['tw']}): "
            f"gap {red['gap']:.3f}, J {red['J']:.4f}")
    out["entries"] = dict(entries)
    with open(os.path.join(RES, "metrics.json"), "w") as f:
        json.dump(out, f, indent=1)
    toc(t0)


# ---------------------------------------------------------------------- main
print(f"=== synthetic suite: D = {D}, seed {SEED}, stages {STAGES} ===",
      flush=True)
print(f"    lam_boost = {LAM_BOOST}, lam_damp = {LAM_DAMP} "
      f"(scaled by D/32 from the d = 32 values 0.8 / 16)", flush=True)
for st in STAGES:
    if st == "k1kernel":
        stage_k1kernel()
    elif st == "k1fixed":
        stage_k1fixed()
    elif st == "k8kernel":
        stage_k8kernel()
    elif st == "boost":
        picard_stage("boost", DELTA_BOOST, LAM_BOOST)
    elif st == "damp":
        picard_stage("damp", DELTA_DAMP, LAM_DAMP)
    elif st == "damp_boundary":
        picard_stage("damp_boundary", DELTA_DAMP, LAM_BOOST, n_iter=8)
    elif st == "reduced":
        stage_reduced()
    elif st == "metrics":
        stage_metrics()
    else:
        raise ValueError(f"unknown stage {st!r}")
print(f"=== D = {D}, seed {SEED} done; results in {RES}/ ===", flush=True)
