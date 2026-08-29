"""
make_figures.py
===============
Generates every figure referenced by pnas.tex / si.tex into
../paper/figs/. Tolerant of missing inputs (stages still running):
skips what it cannot build and reports what it built.

Figures:
  fig_schedule_boost.png      main, fig:boost (a)   [picard_boost.npz]
  fig_convergence_boost.png   main, fig:boost (b)   [picard_boost.npz]
  fig_schedule_damp.png       main, fig:damp (a)    [picard_damp.npz]
  fig_convergence_damp.png    main, fig:damp (b)    [picard_damp.npz]
  fig_k_independence.png      main, fig:kindependence [d32 k_independence]
  fig_k1_validation.png       SI, fig:k1validation  [k1_kernel/k1_fixed, seeds]
  fig_kernel_particle.png     SI, fig:particle-kernel [k2_surrogate]
  fig_dimension.png           SI, fig:dimension     [d32/d64/d128 suites]
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RES = os.path.join(ROOT, "results")
FIGS = os.path.abspath(os.path.join(ROOT, "..", "paper", "figs"))
os.makedirs(FIGS, exist_ok=True)

sys.path.insert(0, ROOT)
from diffusion_control import gmm_control as gc

TG = np.linspace(0.0, gc.T, 41)
G = gc.g(TG)

BUILT, SKIPPED = [], []


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    BUILT.append(name)
    print("built", name, flush=True)


def skip(name, why):
    SKIPPED.append((name, why))
    print("SKIP", name, "-", why, flush=True)


def load(path):
    return np.load(path) if os.path.exists(path) else None


# ---------------------------------------------------------------- boost figs
def boost_figs():
    z = load(os.path.join(RES, "d128", "seed0", "picard_boost.npz"))
    if z is None:
        for n in ("fig_schedule_boost.png", "fig_convergence_boost.png"):
            skip(n, "picard_boost.npz missing")
        return
    tg, nu2h, res = z["tg"], z["nu2"], z["res"]
    g = gc.g(tg)
    nu_star = np.sqrt(nu2h[-1])

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(tg, nu_star, color="C0", lw=2, label=r"$\nu^*$ (fixed point)")
    ax.plot(tg, g, color="k", ls="--", lw=1.5, label=r"$g$ (reference)")
    ax.set_xlabel("forward time $t$")
    ax.set_ylabel(r"noise level")
    ax2 = ax.twinx()
    ax2.plot(tg, nu_star / g, color="C3", ls=":", lw=1.8,
             label=r"$\nu^*/g$")
    ax2.set_ylabel(r"ratio $\nu^*/g$", color="C3")
    ax2.tick_params(axis="y", colors="C3")
    ax2.axhline(1.0, color="C3", lw=0.5, alpha=0.4)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    save(fig, "fig_schedule_boost.png")

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.semilogy(np.arange(1, len(res) + 1), res, "o-", color="C0", lw=1.8)
    ax.set_xlabel("Picard iteration $n$")
    ax.set_ylabel(r"$\|\nu^{(n+1)}-\nu^{(n)}\|_{L^2}$")
    ax.grid(alpha=0.3, which="both")
    save(fig, "fig_convergence_boost.png")


# ----------------------------------------------------------------- damp figs
def damp_figs():
    z = load(os.path.join(RES, "d128", "seed0", "picard_damp.npz"))
    if z is None:
        for n in ("fig_schedule_damp.png", "fig_convergence_damp.png"):
            skip(n, "picard_damp.npz missing")
        return
    tg, nu2h, res = z["tg"], z["nu2"], z["res"]
    g = gc.g(tg)
    nu_star = np.sqrt(nu2h[-1])

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(tg, nu_star, color="C3", lw=2, label=r"$\nu^*$ (fixed point)")
    ax.plot(tg, g, color="k", ls="--", lw=1.5, label=r"$g$ (reference)")
    ax.set_xlabel("forward time $t$")
    ax.set_ylabel(r"noise level")
    ax2 = ax.twinx()
    ax2.plot(tg, nu_star / g, color="C0", ls=":", lw=1.8,
             label=r"$\nu^*/g$")
    ax2.set_ylabel(r"ratio $\nu^*/g$", color="C0")
    ax2.tick_params(axis="y", colors="C0")
    ax2.axhline(1.0, color="C0", lw=0.5, alpha=0.4)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
    save(fig, "fig_schedule_damp.png")

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.semilogy(np.arange(1, len(res) + 1), res, "o-", color="C3", lw=1.8)
    ax.set_xlabel("Picard iteration $n$")
    ax.set_ylabel(r"$\|\nu^{(n+1)}-\nu^{(n)}\|_{L^2}$")
    ax.grid(alpha=0.3, which="both")
    save(fig, "fig_convergence_damp.png")


# ---------------------------------------------------------- k-independence
def k_independence_fig():
    z = load(os.path.join(RES, "d32", "k_independence", "ck_final.npz"))
    if z is None:
        skip("fig_k_independence.png", "ck_final.npz missing")
        return
    Ks, Chat = z["Ks"], z["Chat"]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(Ks, Chat, "o-", color="C0", lw=1.8, ms=6)
    m = Chat.mean()
    ax.axhline(m, color="k", ls="--", lw=1)
    ax.fill_between([Ks.min(), Ks.max()], m * 0.95, m * 1.05,
                    color="gray", alpha=0.2)
    ax.set_xlabel("number of modes $K$")
    ax.set_ylabel(r"$\widehat C(K)$")
    ax.set_xticks(Ks)
    save(fig, "fig_k_independence.png")


# ------------------------------------------------------------ k1 validation
def k1_validation_fig():
    kernels, dims = [], []
    for seed in (0, 1, 2):
        z = load(os.path.join(RES, "d128", f"seed{seed}", "k1_kernel.npz"))
        if z is not None:
            kernels.append(z)
            dims.append(int(z["d"]))
    zf = load(os.path.join(RES, "d128", "seed0", "k1_fixed.npz"))
    if not kernels or zf is None:
        skip("fig_k1_validation.png", "k1 kernel/fixed results missing")
        return
    tg = kernels[0]["tg"]
    g = gc.g(tg)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))

    ax = axes[0]
    for name, col, lab in [("boost", "C0", r"$\delta=-0.1$"),
                           ("damp", "C3", r"$\delta=+0.1$")]:
        Sp = np.stack([z[f"S_part_{name}"] for z in kernels])
        Se = kernels[0][f"S_exact_{name}"]
        ax.plot(tg, Sp.mean(0), color=col, lw=2,
                label=lab + r" particle")
        if len(kernels) > 1:
            ax.fill_between(tg, Sp.mean(0) - Sp.std(0),
                            Sp.mean(0) + Sp.std(0), color=col, alpha=0.2)
        ax.plot(tg, Se, color=col, ls="--", lw=1.5, label=lab + " exact")
    Striv = np.stack([z["S_part_triv"] for z in kernels])
    ax.fill_between(tg, Striv.mean(0) - 2 * Striv.std(0),
                    Striv.mean(0) + 2 * Striv.std(0),
                    color="gray", alpha=0.3, label=r"$\delta=0$ ($\pm2$ sd)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("forward time $t$")
    ax.set_ylabel(r"kernel $\hat S(t)$ at $\nu^{(0)}=g$")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("(a) First-iterate kernel, $K=1$", fontsize=10)

    ax = axes[1]
    ax.plot(tg, np.sqrt(zf["nu2_part_boost"] / g**2), color="C0", lw=2,
            label="particle Picard")
    ax.plot(tg, np.sqrt(zf["nu2_exact_boost"] / g**2), color="k", ls="--",
            lw=1.5, label="exact Riccati")
    ax.axhline(1.0, color="gray", lw=0.7, ls=":")
    ax.set_xlabel("forward time $t$")
    ax.set_ylabel(r"$\nu^*/g$")
    ax.legend(fontsize=8)
    ax.set_title(r"(b) Boost fixed point ($\delta=-0.1$)", fontsize=10)
    save(fig, "fig_k1_validation.png")


# -------------------------------------------------------- k2 surrogate fig
def k2_fig():
    zc = load(os.path.join(RES, "k2_surrogate", "k2_competing.npz"))
    za = load(os.path.join(RES, "k2_surrogate", "k2_aligned.npz"))
    if zc is None or za is None:
        skip("fig_kernel_particle.png", "k2_surrogate results missing")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=False)
    for ax, z, title in [
        (axes[0], zc, r"(a) competing $\delta=(+0.02,-0.05)$"),
        (axes[1], za, r"(b) aligned $\delta=(+0.02,+0.05)$"),
    ]:
        tg = z["tg"]
        Sp = z["S_part"]
        ax.plot(tg, z["S_exact"], color="k", lw=2, label="exact (quadrature)")
        ax.plot(tg, z["S_cf"], color="C3", ls="--", lw=1.6,
                label="closed form (eq:k2kernel)")
        ax.plot(tg, Sp.mean(0), color="C0", lw=1.8, label="particle estimator")
        ax.fill_between(tg, Sp.mean(0) - Sp.std(0), Sp.mean(0) + Sp.std(0),
                        color="C0", alpha=0.2)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xlabel("forward time $t$")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel(r"surrogate kernel $S_{\rm ref}(t)$")
    axes[0].legend(fontsize=8)
    save(fig, "fig_kernel_particle.png")


# ------------------------------------------------------------- dimension fig
def dimension_fig():
    suites = {}
    for d in (32, 64, 128):
        pb = load(os.path.join(RES, f"d{d}", "seed0", "picard_boost.npz"))
        pd_ = load(os.path.join(RES, f"d{d}", "seed0", "picard_damp.npz"))
        kk = load(os.path.join(RES, f"d{d}", "seed0", "k8_kernel.npz"))
        if pb is not None and kk is not None:
            suites[d] = (pb, pd_, kk)
    if len(suites) < 2:
        skip("fig_dimension.png", f"only {len(suites)} dims available")
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4))
    for d, (pb, pd_, kk) in suites.items():
        tg = pb["tg"]
        g = gc.g(tg)
        col = {32: "C0", 64: "C2", 128: "C3"}[d]
        axes[0].plot(tg, np.sqrt(pb["nu2"][-1]) / g, color=col, lw=1.8,
                     label=f"$d={d}$")
        if pd_ is not None:
            axes[1].plot(tg, np.sqrt(pd_["nu2"][-1]) / g, color=col, lw=1.8,
                         label=f"$d={d}$")
        axes[2].plot(tg, kk["S_boost"] / d, color=col, lw=1.8)
        axes[2].plot(tg, kk["S_damp"] / d, color=col, lw=1.8, ls="--")
    for ax, ttl, ylab in [
        (axes[0], "(a) boost fixed point", r"$\nu^*/g$"),
        (axes[1], "(b) damp fixed point", r"$\nu^*/g$"),
        (axes[2], "(c) first-iterate kernel$/d$", r"$\hat S(t)/d$"),
    ]:
        ax.set_xlabel("forward time $t$")
        ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=10)
        ax.axhline(1.0 if ax is not axes[2] else 0.0, color="gray",
                   lw=0.6, ls=":")
    axes[0].legend(fontsize=8)
    axes[2].plot([], [], color="k", lw=1.8, label=r"$\delta=-0.1$")
    axes[2].plot([], [], color="k", lw=1.8, ls="--", label=r"$\delta=+0.1$")
    axes[2].legend(fontsize=8)
    save(fig, "fig_dimension.png")


if __name__ == "__main__":
    boost_figs()
    damp_figs()
    k_independence_fig()
    k1_validation_fig()
    k2_fig()
    dimension_fig()
    print(f"\nbuilt {len(BUILT)} figures -> {FIGS}")
    if SKIPPED:
        print("skipped:", ", ".join(n for n, _ in SKIPPED))
