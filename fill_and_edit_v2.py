"""fill_and_edit_v2.py

Splices the rewritten Section 4.4 (d=32 only, boost+damp scenarios, no
oracle) into pnas.tex, updates the contributions item and the SI
implementation-details sentence, and fills all numeric placeholders from
res/*. Reads pnas.tex/si.tex in place (backs up to *.bak_v2 first).
"""
import json, re, sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import gmm_control as gc

FLOATS = r"""
    \begin{figure}[h]
        \centering
        \includegraphics[width=0.95\textwidth]{figs/fig_convergence.png}
        \caption{Fixed-point iteration of \eqref{eq:fixed_point_map} on the $K=8$ testbed in $d=32$: residuals $\|\nu_{n+1}-\nu_n\|_{L^2}$ versus iteration. Left: boost scenario ($\delta=-0.1$) at three penalties; the decay is geometric and the empirical rate improves with $\lambda$ ($\hat\kappa\approx@@RATE04@@/@@RATE08@@/@@RATE16@@$ at $\lambda=0.4/0.8/1.6$), consistent with the $\kappa=O(1/\lambda)$ law of Theorem~\ref{thm:gaussian}; the residuals plateau at the Monte Carlo noise floor of the kernel estimator. Right: both scenarios. At $\lambda=0.8$ the damping iteration drifts toward the lower admissible boundary (slow initial decay) and locks onto it once the projection engages; at the reported interior damping configuration ($\lambda=@@LAMDAMP@@$) convergence takes two iterations.}
        \label{fig:convergence}
    \end{figure}

    \begin{table}[h]
        \centering
        \caption{Performance of the optimal schedules on the $K=8$ testbed of Section~\ref{sec:synthetic} in $d=32$ (boost: $\delta=-0.1$, $\lambda=0.8$; damp: $\delta=+0.1$, $\lambda=@@LAMDAMP@@$). Entropy gap $H(p_0)-H(\rho_0^\nu)$; sliced $W_2$ (root-mean-square of exact one-dimensional $W_2$ over $256$ random unit projections, $2\times10^4$ samples each side, common random numbers across schedules); objective $J$ of \eqref{eq:objective}. Parentheses give the change relative to the uncontrolled sampler of the same scenario. In the boost scenario the optimal schedule improves all three metrics; in the damping scenario the objective improves while the fidelity metrics worsen---the entropy gain is bought at the expense of $p_0$ (see text).}
        \label{tab:highd}
        \begin{tabular}{@{}llccc@{}}
            \toprule
            Scenario & Schedule & Entropy gap & Sliced $W_2$ to $p_0$ & Objective $J$ \\
            \midrule
            Boost ($\delta=-0.1$) & Uncontrolled $\nu=g$ & @@B_UNC_GAP@@ & @@B_UNC_SW2@@ & @@B_UNC_J@@ \\
            & Particle $\nu^*$ & @@B_ST_GAP@@ ($@@B_ST_GAP_PC@@$) & @@B_ST_SW2@@ ($@@B_ST_SW2_PC@@$) & @@B_ST_J@@ ($+@@B_ST_J_D@@$) \\
            \midrule
            Damp ($\delta=+0.1$) & Uncontrolled $\nu=g$ & @@D_UNC_GAP@@ & @@D_UNC_SW2@@ & @@D_UNC_J@@ \\
            & Particle $\nu^*$ & @@D_ST_GAP@@ ($@@D_ST_GAP_PC@@$) & @@D_ST_SW2@@ ($@@D_ST_SW2_PC@@$) & @@D_ST_J@@ ($+@@D_ST_J_D@@$) \\
            \bottomrule
        \end{tabular}
    \end{table}

    \begin{figure}[h]
        \centering
        \includegraphics[width=0.6\textwidth]{figs/fig_k_independence.png}
        \caption{Kernel Lipschitz constant versus the number of modes in $d=32$: $\widehat C(K)=\|S_{\nu_1}-S_{\nu_2}\|_{L^2([0,T])}/\|\nu_1-\nu_2\|_{L^2([0,T])}$ between the reference schedule $\nu_1=g$ and the perturbed schedule $\nu_2=1.05\,g$, computed with the particle estimator under common random numbers on nested $K$-mode subsets of the testbed mixture (first $K$ modes, weights renormalized), $K\in\{2,4,8,16,32\}$. $\widehat C(K)$ varies by less than @@CKVAR@@\% about its mean over a sixteen-fold range of $K$, consistent with the $K$-independent contraction constant of Theorem~\ref{thm:kgmm}; the empirical Picard rates of Figure~\ref{fig:convergence} track $\widehat C/(2\lambda)$.}
        \label{fig:kindependence}
    \end{figure}

    \begin{figure}[h]
        \centering
        \includegraphics[width=0.95\textwidth]{figs/fig_schedules.png}
        \caption{Optimal versus reference schedule on the $K=8$ testbed in $d=32$ (solid: $\nu^*(t)$; dashed: $g(t)$; dotted, right axis: ratio $\nu^*/g$). Left: boost scenario ($\delta=-0.1$, $\lambda=0.8$): $\nu^*$ exceeds $g$ on the whole productive window, with peak ratio $\nu^*/g\approx@@RATIOBOOST@@$ at $t=0$ decaying to one by $t\approx@@BOOSTWINDOW@@$. Right: damping scenario ($\delta=+0.1$, $\lambda=@@LAMDAMP@@$): $\nu^*$ lies below $g$ on the same window with the mirror-image profile (minimum ratio $\approx@@RATIODAMP@@$ at $t=0$). The two scenarios bracket the reference schedule with opposite signs, exactly as the kernel sign analysis of Section~\ref{sec:structural} predicts.}
        \label{fig:schedules}
    \end{figure}

"""


def rate(res):
    r = np.asarray(res)
    n = 2
    while n < len(r) and r[n] < r[n - 1]:
        n += 1
    n = max(n, 4)
    n = min(n, len(r))
    return float(np.exp(np.polyfit(np.arange(n), np.log(r[:n]), 1)[0]))


def main():
    tab = json.load(open(f"{HERE}/res/table_d32.json"))
    figd = json.load(open(f"{HERE}/res/fig_data.json"))

    zb = np.load(f"{HERE}/res/picard_boost_l0.8.npz")
    zd = np.load(f"{HERE}/res/picard_damp_l16.npz")
    zd8 = np.load(f"{HERE}/res/picard_damp_l0.8.npz")
    tg = zb["tg"]
    g2 = gc.g(tg) ** 2

    rb = np.sqrt(zb["nu2"][-1] / g2)
    rd = np.sqrt(zd["nu2"][-1] / g2)
    ratio_boost = rb.max()
    boost_window = tg[np.where(rb > 1.05)[0]].max()
    ratio_damp = rd.min()
    pinned = np.isclose(np.sqrt(zd8["nu2"][-1]), 0.05, atol=1e-6)
    pinned_frac = 100 * tg[pinned].max() / gc.T

    rates = {l: rate(np.load(f"{HERE}/res/picard_boost_l{l}.npz")["res"])
             for l in ("0.4", "0.8", "1.6")}

    B = tab["scenarios"]["boost_l0.8"]
    D = tab["scenarios"]["damp_l16"]
    ck = figd["ck"]

    def pc(st, unc, key):
        return 100 * (st[key] - unc[key]) / abs(unc[key])

    rep = {
        "@@RATIOBOOST@@": f"{ratio_boost:.2f}",
        "@@BOOSTWINDOW@@": f"{boost_window:.2f}".rstrip("0").rstrip("."),
        "@@RATIODAMP@@": f"{ratio_damp:.2f}",
        "@@PINNEDFRAC@@": f"{pinned_frac:.0f}",
        "@@LAMDAMP@@": "16",
        "@@RATE04@@": f"{rates['0.4']:.2f}",
        "@@RATE08@@": f"{rates['0.8']:.2f}",
        "@@RATE16@@": f"{rates['1.6']:.2f}",
        "@@GAPCLOSE@@": f"{-pc(B['star'], B['unc'], 'gap'):.0f}",
        "@@W2CLOSE@@": f"{-pc(B['star'], B['unc'], 'sw2'):.0f}",
        "@@JDAMPGAIN@@": f"{D['star']['J'] - D['unc']['J']:.3f}",
        "@@CKVAR@@": f"{ck['spread']:.0f}",
        # table
        "@@B_UNC_GAP@@": f"{B['unc']['gap']:.3f}", "@@B_UNC_SW2@@": f"{B['unc']['sw2']:.4f}", "@@B_UNC_J@@": f"{B['unc']['J']:.3f}",
        "@@B_ST_GAP@@": f"{B['star']['gap']:.3f}", "@@B_ST_SW2@@": f"{B['star']['sw2']:.4f}", "@@B_ST_J@@": f"{B['star']['J']:.3f}",
        "@@B_ST_GAP_PC@@": f"{pc(B['star'], B['unc'], 'gap'):+.0f}\\%", "@@B_ST_SW2_PC@@": f"{pc(B['star'], B['unc'], 'sw2'):+.0f}\\%",
        "@@B_ST_J_D@@": f"{B['star']['J'] - B['unc']['J']:.3f}",
        "@@D_UNC_GAP@@": f"{D['unc']['gap']:.3f}", "@@D_UNC_SW2@@": f"{D['unc']['sw2']:.4f}", "@@D_UNC_J@@": f"{D['unc']['J']:.3f}",
        "@@D_ST_GAP@@": f"{D['star']['gap']:.3f}", "@@D_ST_SW2@@": f"{D['star']['sw2']:.4f}", "@@D_ST_J@@": f"{D['star']['J']:.3f}",
        "@@D_ST_GAP_PC@@": f"{pc(D['star'], D['unc'], 'gap'):+.0f}\\%", "@@D_ST_SW2_PC@@": f"{pc(D['star'], D['unc'], 'sw2'):+.0f}\\%",
        "@@D_ST_J_D@@": f"{D['star']['J'] - D['unc']['J']:.3f}",
    }

    body = open(f"{HERE}/sec44_v2.tex").read() + FLOATS
    for k, v in rep.items():
        assert k in body, k
        body = body.replace(k, v)
    assert "@@" not in body, [m for m in re.findall(r"@@[A-Z0-9_]+@@", body)]

    pnas = open(f"{OUT}/pnas.tex").read()
    open(f"{OUT}/pnas.tex.bak_v2", "w").write(pnas)
    i0 = pnas.index(r"\subsection{Synthetic verification of the exact theory}")
    i1 = pnas.index(r"\section{Reduced-Order Controller and Protein Applications}")
    pnas = pnas[:i0] + body + "\n" + pnas[i1:]

    old_co = (r"\item We develop an exact particle-based forward--backward solver "
              r"and verify its convergence against PDE ground truth.")
    new_co = (r"\item We develop an exact particle-based forward--backward solver "
              r"and verify its convergence rate, sign structure, and $K$-independence "
              r"on a controlled synthetic testbed.")
    assert old_co in pnas
    pnas = pnas.replace(old_co, new_co)
    open(f"{OUT}/pnas.tex", "w").write(pnas)

    si = open(f"{OUT}/si.tex").read()
    open(f"{OUT}/si.tex.bak_v2", "w").write(si)
    old_si1 = (r"and the update $\nu^{(n+1)}(t_j)^2=g(t_j)^2+\frac{g(t_j)^2}{2\lambda}\hat S_{\nu^{(n)}}(t_j)$ "
               r"without projection (the reported configurations satisfy the admissible bounds).")
    new_si1 = (r"and the update $\nu^{(n+1)}(t_j)^2=\Pi_{[\nu_{\min}^2,\nu_{\max}^2]}\bigl[g(t_j)^2+\frac{g(t_j)^2}{2\lambda}\hat S_{\nu^{(n)}}(t_j)\bigr]$.")
    assert old_si1 in si
    si = si.replace(old_si1, new_si1)
    old_si2 = (r"The $d=2$ ground truth is an independently computed Fokker--Planck solution on a fine grid; "
               r"the oracle schedule in $d=32$ is obtained by exact Gaussian-moment propagation of the fixed-point map.")
    new_si2 = (r"The projection is inactive in the boost scenario of Section~\ref{sec:synthetic} of the main text "
               r"and active in the damping scenario at small penalties, where the iteration runs to the lower boundary. "
               r"The complete Python implementation of the algorithm accompanies this submission.")
    assert old_si2 in si
    si = si.replace(old_si2, new_si2)
    open(f"{OUT}/si.tex", "w").write(si)
    print("spliced OK")
    print(json.dumps({k.strip("@"): v for k, v in rep.items()}, indent=1))


if __name__ == "__main__":
    main()
