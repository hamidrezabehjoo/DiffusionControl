"""make_fig_k1.py

SI figure for the K=1 exact ground-truth validation
(SI Appendix, Section "Exact K=1 ground truth").

Panel (a): first-iterate kernel S_hat at nu = g for delta = -0.1 and
           delta = +0.1 (mean +/- sd over 3 seeds) vs the exact Riccati
           kernel; the delta = 0 run is shown as a band around zero
           (exact-score triviality, Observation 3.7).
Panel (b): boost fixed point nu*/g: particle Picard vs exact Riccati
           fixed point (delta = -0.1, lambda = 0.8).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

z = np.load("res/k1_validation.npz")
tg = z["tg"]

fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))

ax = axes[0]
for name, col, lab, ls in [("boost", "C0", r"$\delta=-0.1$ (boost)", "-"),
                           ("damp", "C3", r"$\delta=+0.1$ (damp)", "-")]:
    Sp = z[f"S_part_{name}"]
    ax.plot(tg, z[f"S_exact_{name}"], color=col, ls="--", lw=1.4,
            label=lab + " exact")
    ax.plot(tg, Sp.mean(0), color=col, ls=ls, lw=1.8,
            label=lab + " particle")
    ax.fill_between(tg, Sp.mean(0) - Sp.std(0), Sp.mean(0) + Sp.std(0),
                    color=col, alpha=0.25)
St = z["S_part_triv"]
ax.fill_between(tg, St.mean(0) - 2 * St.std(0), St.mean(0) + 2 * St.std(0),
                color="0.4", alpha=0.3,
                label=r"$\delta=0$ particle ($\pm2\,$sd)")
ax.axhline(0.0, color="k", lw=0.6)
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"kernel $S(t)$ at $\nu=g$")
ax.set_title(r"(a) First-iterate kernel, $K=1$, $d=32$", fontsize=10)
ax.legend(fontsize=7, loc="center right", framealpha=0.9)

ax = axes[1]
g2 = 0.1 + 19.9 * tg
ax.plot(tg, np.sqrt(z["nu2_exact_boost"] / g2), "k--", lw=1.4,
        label="exact Riccati fixed point")
ax.plot(tg, np.sqrt(z["nu2_part_boost"] / g2), "C0", lw=1.8,
        label="particle Picard fixed point")
ax.axhline(1.0, color="0.5", lw=0.6)
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\nu^*(t)/g(t)$")
ax.set_title(r"(b) Boost fixed point, $\delta=-0.1$, $\lambda=0.8$",
             fontsize=10)
ax.legend(fontsize=8, loc="upper right")

fig.tight_layout()
fig.savefig("../figs/fig_k1_validation.png", dpi=200, bbox_inches="tight")
print("wrote figs/fig_k1_validation.png")
