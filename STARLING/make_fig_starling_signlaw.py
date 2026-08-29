"""Build fig_starling_signlaw.pdf/.png: sign predictability of the STARLING response.

4 panels (alpha = 0.5, 1, 2.5, 4): per-protein Delta log chi2_r (ctrl/stock) vs
stock Rg bias (rg_stock - rg_exp). Spearman r and p annotated per panel.
n = 48 (proteins with experimental error bars; chi2_stock > 0).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

U = "/mnt/agents/upload/"
OUT = "/mnt/agents/output/paper/figs/fig_starling_signlaw"

ALPHAS = [0.5, 1.0, 2.5, 4.0]
PP_FILES = {0.5: "stage2_per_protein(3).csv",
            1.0: "stage2_per_protein(5).csv",
            2.5: "stage2_per_protein(1).csv",
            4.0: "stage2_per_protein(4).csv"}

plt.rcParams.update({"font.size": 9})
fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.8), sharex=True, sharey=True)

for ax, a in zip(axes, ALPHAS):
    df = pd.read_csv(U + PP_FILES[a])
    df = df[df["chi2_stock"] > 0].copy()
    df["logr"] = np.log(df["chi2_ctrl"] / df["chi2_stock"])
    bias = df["rg_stock"] - df["rg_exp"]
    r, p = stats.spearmanr(bias, df["logr"])

    ax.axhline(0, color="k", ls="--", lw=0.7)
    ax.axvline(0, color="0.6", ls=":", lw=0.7)
    ax.scatter(bias, df["logr"], s=11, c="0.35", alpha=0.7, edgecolors="none")
    # OLS trend line for visual guidance
    m, b = np.polyfit(bias, df["logr"], 1)
    xs = np.array([bias.min(), bias.max()])
    ax.plot(xs, m * xs + b, color="tab:red", lw=1.2)
    ax.set_title(rf"$\alpha = {a}$", fontsize=9)
    ptxt = f"$p = {p:.3f}$" if p >= 1e-3 else f"$p = {p:.0e}$".replace("e-0", "\\times 10^{-").replace("e-", "\\times 10^{-") + "}$"
    if p < 1e-3:
        import math
        expo = math.floor(math.log10(p))
        mant = p / 10**expo
        ptxt = f"$p = {mant:.1f} \\times 10^{{{expo}}}$"
    ax.text(0.04, 0.05, f"Spearman $r = {r:+.2f}$\n{ptxt}",
            transform=ax.transAxes, fontsize=8, va="bottom")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)

axes[0].set_ylabel(r"$\Delta\log\chi^2_r$ (ctrl / stock)")
for ax in axes:
    ax.set_xlabel(r"stock $R_g$ bias ($R_g^{\rm stock} - R_g^{\rm exp}$) [$\AA$]")

fig.tight_layout(w_pad=1.6)
fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".png", dpi=300)

# print stats for the paper text
for a in ALPHAS:
    df = pd.read_csv(U + PP_FILES[a])
    df = df[df["chi2_stock"] > 0].copy()
    df["logr"] = np.log(df["chi2_ctrl"] / df["chi2_stock"])
    r, p = stats.spearmanr(df["rg_stock"] - df["rg_exp"], df["logr"])
    print(f"alpha={a}: n={len(df)}, spearman r={r:+.3f}, p={p:.2e}")
print("saved", OUT)
