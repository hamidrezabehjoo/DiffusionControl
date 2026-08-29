#!/usr/bin/env python3
"""SI figure: ADK ensemble control.
A: basin occupancy, stock vs damped (nu = g/2 on [0, 0.5]).
B: damp dose-response on [0, 0.5] (alpha = 1, 0.75, 0.5, 0.25).
C: boost arms are null (seed 42).
"""
import matplotlib.pyplot as plt
import numpy as np

# per-seed basin counts (open, intermediate, closed), n=100 per arm per seed
seeds = (42, 1, 2)
stock = {42: (18, 55, 27), 1: (13, 55, 32), 2: (14, 55, 31)}
damp50 = {42: (31, 48, 21), 1: (27, 52, 21), 2: (29, 46, 25)}
damp75 = {42: (18, 54, 28), 1: (16, 49, 35), 2: (18, 51, 31)}
damp25 = {42: (50, 38, 12), 1: (41, 46, 13), 2: (40, 45, 15)}
basins = ["open\n(4AKE)", "intermediate", "closed\n(1AKE)"]
colors = ["#2166ac", "#999999", "#b2182b"]

fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9),
                         gridspec_kw={"width_ratios": [1.15, 1, 1]})

# --- panel A: pooled occupancy, stock vs damp, per-seed points -------------
ax = axes[0]
S = np.array([stock[s] for s in seeds], dtype=float)
D = np.array([damp50[s] for s in seeds], dtype=float)
x = np.arange(3)
w = 0.36
for j, (data, label, hatch) in enumerate([(S, "stock", ""), (D, r"damp ($\nu=g/2$)", "//")]):
    means = data.mean(axis=0)
    ax.bar(x + (j - 0.5) * w, means, w, label=label, color=colors,
           alpha=0.45 if j == 0 else 0.95, edgecolor="k", linewidth=0.5, hatch=hatch)
    for i in range(3):  # per-seed scatter
        ax.scatter(np.full(3, x[i] + (j - 0.5) * w), data[:, i],
                   color="k", s=7, zorder=5)
ax.set_xticks(x)
ax.set_xticklabels(basins)
ax.set_ylabel("occupancy (%)")
ax.set_title("A  Basin occupancy (100 samples $\\times$ 3 seeds)", fontsize=9)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.annotate("p=5$\\times$10$^{-5}$", xy=(0, 33), ha="center", fontsize=7.5)
ax.annotate("p=0.04", xy=(2, 36.5), ha="center", fontsize=7.5)
ax.annotate("n.s.", xy=(1, 58), ha="center", fontsize=7.5)
ax.set_ylim(0, 70)

# --- panel B: damp dose-response on [0, 0.5] --------------------------------
ax = axes[1]
doses = [(1.00, stock), (0.75, damp75), (0.50, damp50), (0.25, damp25)]
xs = np.arange(len(doses))
open_m, open_sd, nat_m, nat_sd = [], [], [], []
for a, d in doses:
    arr = np.array([d[s] for s in d], dtype=float)
    open_m.append(arr[:, 0].mean()); open_sd.append(arr[:, 0].std(ddof=1))
    nat = arr[:, 0] + arr[:, 2]
    nat_m.append(nat.mean()); nat_sd.append(nat.std(ddof=1))
ax.errorbar(xs - 0.05, open_m, yerr=open_sd, fmt="o-", color="#2166ac",
            capsize=3, lw=1.4, ms=5, label="open basin")
ax.errorbar(xs + 0.05, nat_m, yerr=nat_sd, fmt="s--", color="#555555",
            capsize=3, lw=1.4, ms=5, label="native (open+closed)")
for i, (a, d) in enumerate(doses):
    arr = np.array([d[s] for s in d], dtype=float)
    ax.scatter(np.full(3, xs[i] - 0.05), arr[:, 0], color="#2166ac", s=6,
               alpha=0.5, zorder=1)
ax.set_xticks(xs)
ax.set_xticklabels([r"1 (stock)", "0.75", "0.5", "0.25"], fontsize=8)
ax.set_xlabel(r"damping gain $\alpha$  ($t_w=0.5$)", fontsize=8.5)
ax.set_ylabel("occupancy (%)")
ax.set_title("B  Dose response (trend p$<$10$^{-16}$)", fontsize=9)
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(0, 70)

# --- panel C: boost arms are null (seed 42) ---------------------------------
ax = axes[2]
arms = ["stock", r"$\alpha$2.5" + "\n[0,.2]", r"$\alpha$3" + "\n[0,.3]",
        r"$\alpha$3" + "\n[0,.5]", r"$\alpha$0.5" + "\n[0,.5]"]
native = [45, 46, 43, 49, 52]
open_f = [18, 20, 16, 18, 31]
cols = ["#999999", "#7570b3", "#7570b3", "#7570b3", "#e08214"]
ax.bar(np.arange(5), native, 0.62, color=cols, edgecolor="k", linewidth=0.5)
ax.bar(np.arange(5), open_f, 0.62, color="white", edgecolor="k", linewidth=0.5,
       hatch="..", label="open fraction")
ax.axhline(45, color="k", ls=":", lw=0.8)
ax.text(-0.42, 46, "stock", fontsize=7.5, ha="left")
ax.set_xticks(np.arange(5))
ax.set_xticklabels(arms, fontsize=7.5)
ax.set_ylabel("native-basin occupancy (%)")
ax.set_title("C  Boost arms are null (seed 42)", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
ax.set_ylim(0, 70)

fig.tight_layout()
fig.savefig("figs/fig_adk_damp.png", dpi=300)
print("wrote figs/fig_adk_damp.png")
