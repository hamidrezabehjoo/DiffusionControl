"""Build fig_starling_saxs.pdf/.png: 3-panel IDP/STARLING dose-response figure.

Panel A: stock replication scatter (133-protein SAXS Rg benchmark).
Panel B: median per-protein mean-Rg shift (ctrl - stock) vs controller gain alpha.
Panel C: per-protein SAXS chi2_r trajectories across alpha for 6 showcase proteins.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

U = "/mnt/agents/upload/"
OUT = "/mnt/agents/output/paper/figs/fig_starling_saxs"

ALPHAS = [0.5, 1.0, 2.5, 4.0]
PP_FILES = {0.5: "stage2_per_protein(3).csv",
            1.0: "stage2_per_protein(5).csv",
            2.5: "stage2_per_protein(1).csv",
            4.0: "stage2_per_protein(4).csv"}
SHOWCASE = ["mbp", "anac046", "hev_pnt3", "rs", "gon7", "pnt"]

pp = {a: pd.read_csv(U + f).set_index("name") for a, f in PP_FILES.items()}
rg133 = pd.read_csv(U + "rg_133_rg_comparison.csv")

plt.rcParams.update({"font.size": 9})
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))

# --- Panel A: stock replication ------------------------------------------------
ax = axes[0]
x, y = rg133["saxs_rg_A"], rg133["starling_rg_A"]
rmse = float(np.sqrt(np.mean((y - x) ** 2)))
r2 = float(np.corrcoef(x, y)[0, 1] ** 2)  # squared Pearson correlation, as in the STARLING paper
ax.scatter(x, y, s=9, c="0.35", alpha=0.65, edgecolors="none")
lim = [0, 78]
ax.plot(lim, lim, "k--", lw=0.8)
ax.text(0.04, 0.96, f"RMSE = {rmse:.2f} $\\AA$\n$R^2$ = {r2:.2f},  n = {len(x)}",
        transform=ax.transAxes, va="top", ha="left", fontsize=8.5)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("$R_g$ (SAXS) [$\\AA$]")
ax.set_ylabel("$R_g$ (STARLING DDPM) [$\\AA$]")
ax.set_title("A  Stock replication", fontsize=9, loc="left")

# --- Panel B: mean-Rg dose response --------------------------------------------
ax = axes[1]
meds, q1s, q3s = [], [], []
for a in ALPHAS:
    d = pp[a]["rg_ctrl"] - pp[a]["rg_stock"]
    meds.append(float(d.median()))
    q1s.append(float(d.quantile(0.25)))
    q3s.append(float(d.quantile(0.75)))
meds, q1s, q3s = map(np.array, (meds, q1s, q3s))
ax.axhline(0, color="k", ls="--", lw=0.8)
ax.errorbar(ALPHAS, meds, yerr=[meds - q1s, q3s - meds],
            fmt="o-", color="k", lw=1.4, ms=5, capsize=3)
ax.text(0.45, 0.06, "median, IQR; n = 53", transform=ax.transAxes, fontsize=8, color="0.3")
ax.set_ylim(-1.75, 0.25)
ax.set_xticks(ALPHAS)
ax.set_xlabel(r"controller strength $\alpha$")
ax.set_ylabel("$\\Delta$ mean $R_g$ (ctrl $-$ stock) [$\\AA$]")
ax.set_title("B  Mean-$R_g$ dose response", fontsize=9, loc="left")

# --- Panel C: per-protein chi2_r trajectories -----------------------------------
ax = axes[2]
colors = plt.cm.tab10.colors
for i, name in enumerate(SHOWCASE):
    xs, ys = [], []
    for a in ALPHAS:
        df = pp[a]
        if name in df.index and np.isfinite(df.loc[name, "chi2_ctrl"]):
            xs.append(a); ys.append(df.loc[name, "chi2_ctrl"])
    stock = pp[ALPHAS[0]].loc[name, "chi2_stock"] if name in pp[ALPHAS[0]].index else np.nan
    ax.plot(xs, ys, "o-", color=colors[i], lw=1.3, ms=4, label=name)
    if np.isfinite(stock):
        ax.axhline(stock, color=colors[i], ls=":", lw=0.9)
ax.set_yscale("log")
ax.set_xticks(ALPHAS)
ax.set_xlabel(r"controller strength $\alpha$")
ax.set_ylabel("$\\chi^2_r$ vs SAXS experiment")
ax.set_title("C  Per-protein $\\chi^2_r$", fontsize=9, loc="left")
ax.legend(fontsize=7, loc="center right", framealpha=0.9, borderpad=0.3,
          handlelength=1.2, labelspacing=0.25)
ax.text(0.03, 0.04, "dotted: stock", transform=ax.transAxes, fontsize=7, color="0.35")

for ax in axes:
    ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.tight_layout(w_pad=2.0)
fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".png", dpi=300)
print("saved", OUT, "| panel A: RMSE", round(rmse, 2), "R2", round(r2, 3),
      "| panel B medians:", np.round(meds, 3))
