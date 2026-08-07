"""make_figs.py

Figures for Section 4.4 (d = 32 only):
  fig_convergence.png   Picard residuals: lambda sweep (boost) + scenario comparison
  fig_schedules.png     optimal vs reference schedule, boost and damp scenarios
  fig_k_independence.png  kernel Lipschitz constant vs K = 2,4,8,16,32
Reads res/*.npz; writes PNGs (300 DPI) to ../figs/.
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gmm_control as gc

C_BLUE, C_TEAL, C_RED, C_BROWN = "#1a1ab2", "#0f7b6c", "#c02828", "#7a5c17"
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
os.makedirs(FIGDIR, exist_ok=True)


def rate(res):
    """log-slope over the longest decreasing prefix (>=4 points)."""
    r = np.asarray(res)
    n = 2
    while n < len(r) and r[n] < r[n - 1]:
        n += 1
    n = max(n, 4)
    n = min(n, len(r))
    a = np.polyfit(np.arange(n), np.log(r[:n]), 1)[0]
    return np.exp(a), n


# ---------------------------------------------------------------- convergence
def fig_convergence():
    zb = {l: np.load(f"res/picard_boost_l{l}.npz") for l in ("0.4", "0.8", "1.6")}
    zd = np.load("res/picard_damp_l0.8.npz")
    zd16 = np.load("res/picard_damp_l16.npz")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax = axes[0]
    rates = {}
    for l, c in [("0.4", C_BROWN), ("0.8", C_BLUE), ("1.6", C_TEAL)]:
        r = zb[l]["res"]
        k, _ = rate(r)
        rates[l] = k
        ax.semilogy(np.arange(len(r)), r, "o-", ms=3.5, lw=1.4, color=c,
                    label=rf"$\lambda={l}$, $\hat\kappa\approx{k:.2f}$")
    ax.set_xlabel("iteration $n$")
    ax.set_ylabel(r"$\|\nu_{n+1}-\nu_n\|_{L^2}$")
    ax.set_title(r"boost scenario ($\bar\delta=-0.1$)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_xticks(np.arange(0, max(len(zb[l]["res"]) for l in zb), 2))
    ax = axes[1]
    for z, c, lab in [(zb["0.8"], C_BLUE, r"boost ($\bar\delta=-0.1$, $\lambda=0.8$)"),
                      (zd, C_RED, r"damp ($\bar\delta=+0.1$, $\lambda=0.8$)"),
                      (zd16, C_BROWN, r"damp ($\bar\delta=+0.1$, $\lambda=16$)")]:
        r = z["res"]
        ax.semilogy(np.arange(len(r)), r, "o-", ms=3.5, lw=1.4, color=c, label=lab)
    ax.set_xlabel("iteration $n$")
    ax.set_title("both scenarios")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_convergence.png"), dpi=300)
    plt.close(fig)
    return rates


# ------------------------------------------------------------------ schedules
def fig_schedules():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for ax, tag, col, tit in [
            (axes[0], "boost_l0.8", C_BLUE, r"boost scenario ($\bar\delta=-0.1$)"),
            (axes[1], "damp_l16", C_RED, r"damp scenario ($\bar\delta=+0.1$)")]:
        z = np.load(f"res/picard_{tag}.npz")
        tg, nu2 = z["tg"], z["nu2"][-1]
        gt, nu = gc.g(tg), np.sqrt(nu2)
        ax.plot(tg, gt, "--", color="0.35", lw=1.4, label=r"reference $g(t)$")
        ax.plot(tg, nu, "-", color=col, lw=1.8, label=r"optimal $\nu^*(t)$")
        ax.fill_between(tg, gt, nu, where=nu >= gt, color=col, alpha=0.15)
        ax.fill_between(tg, gt, nu, where=nu < gt, color=col, alpha=0.15)
        ax2 = ax.twinx()
        ax2.plot(tg, nu / gt, ":", color=col, lw=1.2, alpha=0.8)
        ax2.axhline(1.0, color="0.7", lw=0.7)
        ax2.set_ylabel(r"ratio $\nu^*/g$", color=col)
        ax2.tick_params(axis="y", colors=col)
        ax.set_xlabel("reverse time $t$")
        ax.set_title(tit)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    axes[0].set_ylabel(r"schedule")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_schedules.png"), dpi=300)
    plt.close(fig)


# ------------------------------------------------------------- k-independence
def fig_kindependence():
    z = np.load("res/ck_final.npz")
    Ks, Chat = z["Ks"], z["Chat"]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(Ks, Chat, "o-", color=C_BLUE, ms=5, lw=1.4)
    mean = Chat.mean()
    ax.axhline(mean, color="0.4", lw=0.9, ls="--")
    ax.fill_between([Ks[0], Ks[-1]], mean * 0.95, mean * 1.05, color="0.4",
                    alpha=0.12)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks])
    ax.set_xlabel("number of modes $K$")
    ax.set_ylabel(r"$\widehat C(K)$")
    spread = (Chat.max() - Chat.min()) / mean * 100
    ax.set_title(f"$d=32$: spread {spread:.0f}% about the mean")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_k_independence.png"), dpi=300)
    plt.close(fig)
    return dict(Ks=Ks.tolist(), Chat=Chat.tolist(), spread=spread)


if __name__ == "__main__":
    rates = fig_convergence()
    fig_schedules()
    ck = fig_kindependence()
    with open("res/fig_data.json", "w") as f:
        json.dump({"rates": rates, "ck": ck}, f, indent=2)
    print("rates:", rates)
    print("ck:", ck)
    print("DONE figs")
