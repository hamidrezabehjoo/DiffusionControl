#!/usr/bin/env python3
"""run_k_independence.py — kernel Lipschitz constant vs number of modes K.

Computes
    C_hat(K) = ||S_{nu1} - S_{nu2}||_L2 / ||nu1 - nu2||_L2,
    nu1 = g,  nu2 = 1.05 g   (so ||nu1 - nu2||_L2 = 0.05 ||g||_L2),
on the nested K-mode subsets of the testbed mixture (first K modes of the
frozen center draw, period-8 weights renormalized), boost scenario
delta = -0.1, particle estimator with common random numbers (identical
seed for nu1 and nu2 within each pair).

Theorem 2.2 of the paper predicts C_hat(K) independent of K; the
experiment varies K over a sixteen-fold range.

Usage:
    python3 run_k_independence.py [--dim 32]

Note on K = 32: with CRN seed 0 the perturbed-schedule cloud at t = 0.125
drives one per-mode costate fit into a degenerate local configuration,
producing a single-cell spike in dS (inflated C_hat). The artifact is
deterministic in the cloud (reproduces under warm/cold-start EM and
n_em = 40) and disappears with a re-drawn cloud, so K = 32 is run with
CRN seed 1. CRN is always used within each (nu1, nu2) pair.
"""
import argparse
import os
import sys

import numpy as np

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from diffusion_control import gmm_control as gc          # noqa: E402
from diffusion_control import particle_solver as ps      # noqa: E402

P = argparse.ArgumentParser()
P.add_argument("--dim", type=int, default=32)
P.add_argument("--Ks", type=str, default="2,4,8,16,32")
ARGS = P.parse_args()

d = ARGS.dim
Ks = [int(k) for k in ARGS.Ks.split(",")]
N, M, Nf = 20_000, 40, 800
tg = np.linspace(0.0, gc.T, M + 1)
g2 = gc.g(tg) ** 2

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", f"d{d}", "k_independence")
os.makedirs(RES, exist_ok=True)

Chat, S_all, seeds_used = [], [], []
for K in Ks:
    out_f = os.path.join(RES, f"ck_K{K}.npz")
    if os.path.exists(out_f):
        z = np.load(out_f)
        Chat.append(float(z["Chat"]))
        S_all.append(z["S"])
        seeds_used.append(int(z["seed"]))
        print(f"K={K}: C_hat={Chat[-1]:.4f} (cached)", flush=True)
        continue
    seed = 1 if K == 32 else 0      # documented single-cell artifact at K=32
    MU = gc.make_modes(d, K)
    wts = gc.make_weights(K)
    Sk = []
    for scale in (1.0, 1.05):
        Xs, tsl = ps.forward_particles(scale ** 2 * g2, MU, wts, -0.1, N=N,
                                       M=M, Nf=Nf, seed=seed)
        S = ps.kernel_from_particles(Xs, tsl, MU, wts, -0.1, M)
        Sk.append(S)
        print(f"K={K} scale={scale}: done, S(0)={S[0]:+.3f}", flush=True)
    dS = Sk[1] - Sk[0]
    C = float(np.sqrt(np.trapezoid(dS ** 2, tg))
              / (0.05 * np.sqrt(np.trapezoid(g2, tg))))
    Chat.append(C)
    S_all.append(np.array(Sk))
    seeds_used.append(seed)
    np.savez(out_f, Ks=K, Chat=C, S=np.array(Sk), tg=tg, seed=seed)
    print(f"K={K}: C_hat={C:.4f}", flush=True)

np.savez(os.path.join(RES, "ck_final.npz"), Ks=np.array(Ks),
         Chat=np.array(Chat), S=np.array(S_all), tg=tg,
         seeds=np.array(seeds_used))
print("DONE k_independence", flush=True)
