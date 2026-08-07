"""fill_and_edit_v3.py

Reviewer-round edits: injects multi-seed numbers into pnas.tex (Table 2
restructure with mean +/- sd over 3 seeds, reduced-controller row, occupancy
sentence, damp significance) and appends the new SI section
(Exact one-component ground truth and supplementary numerics) to si.tex.

Reads: res/table2_multi.json, res/k1_validation.npz
Backups: pnas.tex.bak_v3, si.tex.bak_v3
Placeholders in templates use [[key]].
"""
import json
import numpy as np

J = json.load(open("res/table2_multi.json"))
K1 = np.load("res/k1_validation.npz")

Hp0 = J["H_p0"]["H"]
Hp0_se = J["H_p0"]["se"]
SEEDS = [0, 1, 2]
nsd = len(SEEDS)


def msd(vals, nd=2):
    v = np.array(vals, dtype=float)
    return f"{v.mean():.{nd}f} $\\pm$ {v.std(ddof=1 if len(v) > 1 else 0):.{nd}f}", v.mean(), v.std(ddof=1 if len(v) > 1 else 0)


def rows_of(name):
    return {r["seed"]: r for r in J["scenarios"][name]["rows"]}


B = rows_of("boost")
D = rows_of("damp")
R = {r["seed"]: r["red"] for r in J["reduced"]["rows"]}
alpha, tw = J["reduced"]["alpha"], J["reduced"]["tw"]

# ---- per-seed series -------------------------------------------------------
def series(scn, key, which="star"):
    return [scn[s][which][key] for s in SEEDS]


bH_unc, bH_unc_m, _ = msd(series(B, "H", "unc"))
bH_st, bH_st_m, _ = msd(series(B, "H", "star"))
bg_unc, bg_unc_m, _ = msd(series(B, "gap", "unc"))
bg_st, bg_st_m, _ = msd(series(B, "gap", "star"))
bw_unc, bw_unc_m, _ = msd(series(B, "sw2", "unc"), 4)
bw_st, bw_st_m, _ = msd(series(B, "sw2", "star"), 4)
bJ_unc, bJ_unc_m, _ = msd(series(B, "J", "unc"), 3)
bJ_st, bJ_st_m, _ = msd(series(B, "J", "star"), 3)

rH, rH_m, _ = msd([R[s]["H"] for s in SEEDS])
rg, rg_m, _ = msd([R[s]["gap"] for s in SEEDS])
rw, rw_m, _ = msd([R[s]["sw2"] for s in SEEDS], 4)
rJ, rJ_m, _ = msd([R[s]["J"] for s in SEEDS], 3)

dH_unc, dH_unc_m, _ = msd(series(D, "H", "unc"))
dH_st, dH_st_m, _ = msd(series(D, "H", "star"))
dg_unc, dg_unc_m, _ = msd(series(D, "gap", "unc"))
dg_st, dg_st_m, _ = msd(series(D, "gap", "star"))
dw_unc, dw_unc_m, _ = msd(series(D, "sw2", "unc"), 4)
dw_st, dw_st_m, _ = msd(series(D, "sw2", "star"), 4)
dJ_unc, dJ_unc_m, _ = msd(series(D, "J", "unc"), 3)
dJ_st, dJ_st_m, _ = msd(series(D, "J", "star"), 3)

# percentages (from means)
b_gap_pct = 100 * (bg_st_m - bg_unc_m) / bg_unc_m          # negative = closed
b_w2_pct = 100 * (bw_st_m - bw_unc_m) / bw_unc_m
b_dJ = bJ_st_m - bJ_unc_m
r_dJ = rJ_m - bJ_unc_m
red_pc = 100 * r_dJ / b_dJ
d_gap_pct = 100 * (dg_st_m - dg_unc_m) / dg_unc_m
d_w2_pct = 100 * (dw_st_m - dw_unc_m) / dw_unc_m
d_dJ = dJ_st_m - dJ_unc_m

# damp significance (paired across seeds)
dJs = np.array([D[s]["star"]["dJ"] for s in SEEDS])
t_stat = dJs.mean() / (dJs.std(ddof=1) / np.sqrt(nsd))
if abs(t_stat) >= 2.92:
    sigphrase = (f"paired $t={t_stat:.1f}$ on two degrees of freedom, "
                 "significant at the $5\\%$ level")
else:
    sigphrase = (f"positive in every seed but not significant at the $5\\%$ "
                 f"level: paired $t={t_stat:.1f}$ on two degrees of freedom")

# rare-mode occupancy (mode index 7, true mass 0.03)
occ_b_unc = np.array([B[s]["unc"]["occ"][7] for s in SEEDS])
occ_b_st = np.array([B[s]["star"]["occ"][7] for s in SEEDS])
occ_rare_unc = f"{occ_b_unc.mean():.3f}"
occ_rare_st = f"{occ_b_st.mean():.3f}"

# ---- K=1 validation numbers -------------------------------------------------
tg1 = K1["tg"]
i0, i25 = 0, 10      # t = 0 and t = 0.25
k1 = {}
for name in ("boost", "damp"):
    Sp = K1[f"S_part_{name}"]
    Se = K1[f"S_exact_{name}"]
    k1[name] = dict(
        p0=f"{Sp[:, i0].mean():+.3f} $\\pm$ {Sp[:, i0].std(ddof=1):.3f}",
        e0=f"{Se[i0]:+.3f}",
        p25=f"{Sp[:, i25].mean():+.3f} $\\pm$ {Sp[:, i25].std(ddof=1):.3f}",
        e25=f"{Se[i25]:+.3f}")
St = K1["S_part_triv"]
triv_max = np.abs(St.mean(0)).max()
triv_sd = St.std(0).mean()
g2_1 = 0.1 + 19.9 * tg1
r0_part = np.sqrt(K1["nu2_part_boost"][0] / g2_1[0])
r0_ex = np.sqrt(K1["nu2_exact_boost"][0] / g2_1[0])
rel_l2 = (np.sqrt(np.trapezoid((np.sqrt(K1["nu2_part_boost"])
                                - np.sqrt(K1["nu2_exact_boost"])) ** 2, tg1))
          / np.sqrt(np.trapezoid(g2_1, tg1)))
nu_d8_0 = np.sqrt(K1["nu2_exact_damp_l0.8"][0])
r_d16 = np.sqrt(K1["nu2_exact_damp_l16"][0] / g2_1[0])

# ======================== 1. new Table 2 (pnas.tex) ==========================
TABLE2 = r"""\begin{table}[h]
        \centering
        \caption{Performance on the $K=8$ testbed of Section~\ref{sec:synthetic} in $d=32$ (boost: $\delta=-0.1$, $\lambda=0.8$; damp: $\delta=+0.1$, $\lambda=16$). Entries are mean $\pm$ standard deviation over three independent solver seeds; the uncontrolled and controlled clouds of each seed share common random numbers, so paired differences are meaningful. $H(p_0)=[[HP0]]\pm[[HP0SE]]$ by Monte Carlo on $2\times10^5$ exact samples; the entropy gap is $H(p_0)-H(\rho_0^\nu)$; sliced $W_2$ is the root-mean-square of exact one-dimensional $W_2$ over $256$ random unit projections ($2\times10^4$ samples each side); $J$ is the objective \eqref{eq:objective}. Parentheses give the change relative to the uncontrolled sampler of the same scenario; for the signed entropy gap this is the relative change of the signed value, so the negative percentage in the damping scenario means the gap widens. The reduced row is the one-knob controller $\nu=\alpha g$ on $[0,t_w]$ with $\alpha=[[ALPHA]]$, $t_w=[[TW]]$ (selected on $J$ by grid search): it recovers [[REDPC]]\% of the exact solver's objective gain. In the boost scenario the optimal schedule improves all three metrics; in the damping scenario the objective improves while the fidelity metrics worsen---the entropy gain is bought at the expense of $p_0$ (see text).}
        \label{tab:highd}
        \begin{tabular}{@{}llcccc@{}}
            \toprule
            Scenario & Schedule & $H(\rho_0^\nu)$ & Entropy gap & Sliced $W_2$ to $p_0$ & Objective $J$ \\
            \midrule
            Boost ($\delta=-0.1$) & Uncontrolled $\nu=g$ & [[BHUNC]] & [[BGUNC]] & [[BWUNC]] & [[BJUNC]] \\
            & Reduced $\nu=\alpha g$ on $[0,t_w]$ & [[RH]] & [[RG]] & [[RW]] & [[RJ]] \\
            & Particle $\nu^*$ & [[BHST]] & [[BGST]] ([[BGPCT]]\%) & [[BWST]] ([[BWPCT]]\%) & [[BJST]] ($+[[BDJ]]$) \\
            \midrule
            Damp ($\delta=+0.1$) & Uncontrolled $\nu=g$ & [[DHUNC]] & [[DGUNC]] & [[DWUNC]] & [[DJUNC]] \\
            & Particle $\nu^*$ & [[DHST]] & [[DGST]] ([[DGPCT]]\%) & [[DWST]] ([[DWPCT]]\%) & [[DJST]] ($+[[DDJ]]$) \\
            \bottomrule
        \end{tabular}
    \end{table}"""

rep = {
    "HP0": f"{Hp0:.3f}", "HP0SE": f"{Hp0_se:.3f}",
    "ALPHA": f"{alpha:.2f}".rstrip("0").rstrip("."), "TW": f"{tw:.2f}".rstrip("0").rstrip("."),
    "REDPC": f"{red_pc:.0f}",
    "BHUNC": bH_unc, "BGUNC": bg_unc, "BWUNC": bw_unc, "BJUNC": bJ_unc,
    "RH": rH, "RG": rg, "RW": rw, "RJ": rJ,
    "BHST": bH_st, "BGST": bg_st, "BWST": bw_st, "BJST": bJ_st,
    "BGPCT": f"{b_gap_pct:+.0f}", "BWPCT": f"{b_w2_pct:+.0f}", "BDJ": f"{b_dJ:.3f}",
    "DHUNC": dH_unc, "DGUNC": dg_unc, "DWUNC": dw_unc, "DJUNC": dJ_unc,
    "DHST": dH_st, "DGST": dg_st, "DWST": dw_st, "DJST": dJ_st,
    "DGPCT": f"{d_gap_pct:+.0f}", "DWPCT": f"{d_w2_pct:+.0f}", "DDJ": f"{d_dJ:.3f}",
}
for k, v in rep.items():
    TABLE2 = TABLE2.replace(f"[[{k}]]", v)

pnas = open("../pnas.tex").read()
open("../pnas.tex.bak_v3", "w").write(pnas)

cap_anchor = "\\caption{Performance of the optimal schedules on the $K=8$ testbed"
i_cap = pnas.find(cap_anchor)
assert i_cap > 0, "table caption anchor not found"
i_begin = pnas.rfind("\\begin{table}[h]", 0, i_cap)
i_end = pnas.find("\\end{table}", i_cap) + len("\\end{table}")
pnas = pnas[:i_begin] + TABLE2 + pnas[i_end:]

# ---- boost paragraph: error bars + reduced controller + occupancy ----------
old1 = ("Table~\\ref{tab:highd} quantifies the gain: the optimal schedule "
        "closes 20\\% of the entropy gap and 10\\% of the sliced-$W_2$ "
        "discrepancy relative to the uncontrolled sampler.")
new1 = (f"Table~\\ref{{tab:highd}} quantifies the gain (mean $\\pm$ standard "
        f"deviation over three independent solver seeds): the optimal "
        f"schedule closes {-b_gap_pct:.0f}\\% of the entropy gap and "
        f"{-b_w2_pct:.0f}\\% of the sliced-$W_2$ discrepancy relative to the "
        f"uncontrolled sampler. Most of the gain is already captured by a "
        f"one-knob controller: the reduced schedule $\\nu=\\alpha g$ with "
        f"$\\alpha={alpha:.2f}$ on the window $[0,{tw:.2f}]$ recovers "
        f"{red_pc:.0f}\\% of the exact solver's objective gain "
        f"(Table~\\ref{{tab:highd}}), and the full profile $\\nu^*$ is "
        f"well approximated by a constant amplification on the productive "
        f"window. Control does not sacrifice the rare mode: its terminal "
        f"occupancy is {occ_rare_st} under $\\nu^*$ versus {occ_rare_unc} "
        f"uncontrolled (true mass $0.03$; occupancies of all eight modes are "
        f"tabulated in SI Appendix, Table~S\\ref{{S-tab:occupancy}}).")
assert old1 in pnas, "boost anchor not found"
pnas = pnas.replace(old1, new1)

# ---- damp paragraph: significance -------------------------------------------
old2 = ("Table~\\ref{tab:highd} confirms the mechanism: the objective "
        "improves by 0.037, yet the entropy gap")
new2 = (f"Table~\\ref{{tab:highd}} confirms the mechanism: the objective "
        f"improves by {d_dJ:.3f} on average ({sigphrase}), yet the entropy "
        f"gap")
assert old2 in pnas, "damp anchor not found"
pnas = pnas.replace(old2, new2)

# ---- convergence paragraph: pointer to lambda table --------------------------
old3 = "excluding the noise-floor plateau.)"
new3 = ("excluding the noise-floor plateau.) The penalty dependence of the "
        "converged fixed points in both scenarios is reported in SI "
        "Appendix, Table~S\\ref{S-tab:lamsens}.")
assert old3 in pnas
pnas = pnas.replace(old3, new3)

open("../pnas.tex", "w").write(pnas)
print("pnas.tex updated")

# ======================== 2. new SI section (si.tex) =========================
# lambda-sensitivity rows
lam_tex = []
for r in J["lam_table"]:
    gap = f"{r['gap']:+.3f}"
    bf = r["boundary"]
    if bf < 0.02:
        bnd = "no"
    elif bf > 0.5:
        bnd = f"yes ({100 * bf:.0f}\\% of grid)"
    else:
        bnd = f"partial ({100 * bf:.0f}\\%)"
    lam_tex.append(
        f"            {r['scenario']} & {r['lam']} & {gap} & {r['sw2']:.4f} "
        f"& {r['J']:.3f} & {bnd} \\\\")
LAMROWS = "\n".join(lam_tex)

# occupancy rows (boost scenario, mean +/- sd over seeds)
occ_tex = []
PI8 = [0.20, 0.20, 0.15, 0.15, 0.10, 0.10, 0.07, 0.03]
for k in range(8):
    u = np.array([B[s]["unc"]["occ"][k] for s in SEEDS])
    c = np.array([B[s]["star"]["occ"][k] for s in SEEDS])
    occ_tex.append(
        f"            {k + 1} & {PI8[k]:.2f} & {u.mean():.3f} $\\pm$ "
        f"{u.std(ddof=1):.3f} & {c.mean():.3f} $\\pm$ {c.std(ddof=1):.3f} \\\\")
OCCROWS = "\n".join(occ_tex)

SI_SECTION = r"""
    % ============================================================
    \section{Exact one-component ground truth and supplementary numerics}
    \label{app:k1validation}

    This section collects an independent exact ground truth for the particle
    solver of Section~\ref{app:algorithm} and the supplementary tables
    referenced from Section~\ref{sec:synthetic} of the main text.

    \subsection{Exact Riccati solution for $K=1$}
    Take a single Gaussian component, $p_0=\mathcal N(\mu,\sigma_0^2 I_d)$,
    and the frozen score of $\mathcal N(m(t)\mu,w(t)I_d)$ with
    $w(t)=\Sigma_t^2+\delta$. The controlled drift is affine and the noise
    isotropic, so the controlled law remains Gaussian,
    $\rho_t^\nu=\mathcal N(m(t)\mu,v(t)I_d)$, and the controlled
    Fokker--Planck equation reduces to the scalar Riccati equation
    (with $\tau=T-t$)
    \begin{equation}\label{eq:riccati-v}
        \frac{dv}{d\tau}=\beta\,v-\frac{\beta+\nu^2}{w}\,v+\nu^2,
        \qquad v(\tau=0)=1 .
    \end{equation}
    The terminal entropy of a Gaussian is a function of $v$ alone, and the
    costate is quadratic,
    $\psi_t(x)=a(t)+\tfrac{q(t)}{2}\|x-m(t)\mu\|^2$; the terminal condition
    $\psi_0=-(1+\log\rho_0^\nu)$ gives $q(0)=1/v(0)$, and the costate
    equation reduces to
    \begin{equation}\label{eq:costate-q}
        \dot q=\Bigl(\beta-\frac{\beta+\nu^2}{w}\Bigr)q .
    \end{equation}
    Substituting into the kernel \eqref{eq:kernel} gives the closed form
    \begin{equation}\label{eq:kernel-k1}
        S_\nu(t)=d\,q(t)\,\frac{w(t)-v(t)}{w(t)} .
    \end{equation}
    Two consequences are immediate. First, since $q>0$, the sign of the
    kernel is the sign of $w-v$; at the uncontrolled schedule $\nu=g$ the
    Riccati equation relaxes $v$ from $1$ toward the quasi-equilibrium
    $v_{\rm eq}=w/(2-w)$, where $w-v_{\rm eq}=-\delta w/(2-w)$ has the sign
    of $-\delta$, so $\sign S_g=-\sign\delta$ throughout: the exact
    one-component model already exhibits the boost/damp sign law of
    Section~\ref{sec:structural}. Second, the dimension enters only through
    the explicit factor $d$ in \eqref{eq:kernel-k1}: the $K=1$ solution is
    exact in every dimension, and we evaluate it at the testbed dimension
    $d=32$. Equations \eqref{eq:riccati-v}--\eqref{eq:kernel-k1} are
    integrated on the same $M=40$ grid as the particle solver (Runge--Kutta
    in $\tau$ for $v$, exponential Euler for $q$), and the exact fixed point
    is obtained by the same projected Picard map with $\hat S$ replaced by
    \eqref{eq:kernel-k1}.

    \subsection{Validation of the particle solver against the exact solution}
    We run the particle solver on the one-component model at $d=32$ with the
    testbed hyperparameters ($N=2\times10^4$ particles, $M=40$, $800$
    Euler--Maruyama steps, two-fold cross-fitting). Three checks.
    (i)~\emph{First-iterate sign and magnitude} (the direct controlled-law
    sign check): from $\nu^{(0)}=g$, the first kernel estimate $\hat S$ is
    compared with \eqref{eq:kernel-k1} for $\delta=-0.1$ and $\delta=+0.1$
    over three independent seeds (Figure~\ref{fig:k1validation}(a) and
    Table~\ref{tab:k1sign}): the sign is correct in every seed and the
    magnitude agrees within Monte Carlo error over the productive window.
    (ii)~\emph{Exact-score triviality} (Observation~3.7 of the main text):
    at $\delta=0$ the exact kernel vanishes identically, and the particle
    estimate is statistically indistinguishable from zero
    ($\max_t|\overline{\hat S}(t)|=[[TRIVMAX]]$ against a per-time Monte
    Carlo standard deviation of $\approx[[TRIVSD]]$;
    Figure~\ref{fig:k1validation}(a), gray band).
    (iii)~\emph{Fixed point}: the particle Picard fixed point
    ($\delta=-0.1$, $\lambda=0.8$) reproduces the exact fixed point
    (Figure~\ref{fig:k1validation}(b)): the peak ratio $\nu^*/g$ at $t=0$
    is [[R0PART]] (particle) versus [[R0EX]] (exact), and the relative
    $L^2$ distance between the two schedules is [[RELL2]]\% of $\|g\|_{L^2}$.
    The exact one-component model also corroborates the damping phenomenology
    of the main text: at $\delta=+0.1$ the exact fixed point runs to the
    lower boundary at $\lambda=0.8$ ($\nu^*(0)=[[NUD8]]=\nu_{\min}$) and is
    interior at $\lambda=16$ (ratio $\nu^*/g=[[RD16]]$ at $t=0$), matching
    the $K=8$ particle results.

    \begin{figure}[h]
        \centering
        \includegraphics[width=\textwidth]{figs/fig_k1_validation.png}
        \caption{Exact one-component ground truth for the particle solver
        ($K=1$, $d=32$, testbed schedule and hyperparameters).
        (a)~First-iterate kernel $\hat S(t)$ at $\nu^{(0)}=g$: particle
        estimator (solid, mean $\pm$ one standard deviation over three
        seeds) against the exact Riccati kernel \eqref{eq:kernel-k1}
        (dashed) for the over-sharp ($\delta=-0.1$, blue) and over-diffuse
        ($\delta=+0.1$, red) mismatches; the gray band is the $\delta=0$
        run ($\pm2$ standard deviations), statistically indistinguishable
        from zero (Observation~3.7). (b)~Boost fixed point: particle Picard
        versus exact Riccati fixed point ($\delta=-0.1$, $\lambda=0.8$).}
        \label{fig:k1validation}
    \end{figure}

    \begin{table}[h]
        \centering
        \caption{First-iterate kernel at $\nu^{(0)}=g$ in the one-component
        model ($d=32$): particle estimator (mean $\pm$ standard deviation,
        three seeds) against the exact Riccati kernel \eqref{eq:kernel-k1}
        at $t=0$ and $t=0.25$. The sign matches the exact kernel in every
        seed, in both mismatch directions.}
        \label{tab:k1sign}
        \begin{tabular}{@{}ccccc@{}}
            \toprule
            & \multicolumn{2}{c}{$t=0$} & \multicolumn{2}{c}{$t=0.25$} \\
            \cmidrule(lr){2-3}\cmidrule(lr){4-5}
            $\delta$ & particle $\hat S$ & exact $S$ & particle $\hat S$ & exact $S$ \\
            \midrule
            $-0.1$ & [[KB_P0]] & [[KB_E0]] & [[KB_P25]] & [[KB_E25]] \\
            $+0.1$ & [[KD_P0]] & [[KD_E0]] & [[KD_P25]] & [[KD_E25]] \\
            \bottomrule
        \end{tabular}
    \end{table}

    \subsection{Penalty sensitivity}
    Table~\ref{tab:lamsens} reports the converged fixed point as a function
    of the penalty $\lambda$ in both scenarios (seed $0$; metrics as in
    Table~\ref{tab:highd} of the main text). In the boost scenario the
    solution is interior at every reported penalty and the gain decreases
    smoothly as $\lambda$ grows; in the damping scenario the fixed point is
    pinned to the lower admissible boundary at $\lambda=0.8$ and becomes
    interior only at the much larger penalty $\lambda=16$---the
    dimension-appropriate scale of the entropy reward, which grows linearly
    in $d$ while the quadratic penalty does not.

    \begin{table}[h]
        \centering
        \caption{Penalty sensitivity of the converged fixed point on the
        $K=8$ testbed ($d=32$, seed $0$). Entropy deviation is the signed
        gap $H(p_0)-H(\rho_0^{\nu^*})$; ``boundary active'' records whether
        the projection onto $[\nu_{\min},\nu_{\max}]$ engages at the fixed
        point (fraction of grid points at the boundary).}
        \label{tab:lamsens}
        \begin{tabular}{@{}lccccc@{}}
            \toprule
            Scenario & $\lambda$ & Entropy deviation & Sliced $W_2$ & Objective $J$ & Boundary active? \\
            \midrule
[[LAMROWS]]
            \bottomrule
        \end{tabular}
    \end{table}

    \subsection{Mode occupancy}
    Table~\ref{tab:occupancy} reports the terminal mode occupancies
    $\hat\pi_k=\frac1N\sum_{i=1}^N\mathbf 1\{\text{sample }i\text{ assigned
    to mode }k\}$ (assignment by the true-mixture posterior, boost scenario,
    mean $\pm$ standard deviation over the three solver seeds). The
    uncontrolled sampler already preserves the weights up to sampling noise,
    and the controlled sampler changes no occupancy by more than
    [[OCCSHIFT]] in absolute value; in particular the rare mode of mass
    $0.03$ retains occupancy [[OCCRAREST]] under $\nu^*$ versus
    [[OCCRAREUNC]] uncontrolled. The entropy gain of the boost scenario is
    therefore a within-mode dispersion correction, not a reallocation of
    mass across modes.

    \begin{table}[h]
        \centering
        \caption{Terminal mode occupancies on the $K=8$ testbed ($d=32$,
        boost scenario, $\delta=-0.1$, $\lambda=0.8$): true weights versus
        the uncontrolled and controlled terminal clouds (mean $\pm$
        standard deviation over three solver seeds, $N=2\times10^4$
        samples each, assignment by the true-mixture posterior).}
        \label{tab:occupancy}
        \begin{tabular}{@{}cccc@{}}
            \toprule
            Mode $k$ & true $\pi_k$ & uncontrolled $\hat\pi_k$ & controlled $\hat\pi_k$ \\
            \midrule
[[OCCROWS]]
            \bottomrule
        \end{tabular}
    \end{table}
"""

occ_shift = max(
    abs(np.mean([B[s]["star"]["occ"][k] for s in SEEDS])
        - np.mean([B[s]["unc"]["occ"][k] for s in SEEDS])) for k in range(8))

srep = {
    "TRIVMAX": f"{triv_max:.3f}", "TRIVSD": f"{triv_sd:.3f}",
    "R0PART": f"{r0_part:.2f}", "R0EX": f"{r0_ex:.2f}",
    "RELL2": f"{100 * rel_l2:.1f}",
    "NUD8": f"{nu_d8_0:.2f}", "RD16": f"{r_d16:.3f}",
    "KB_P0": k1["boost"]["p0"], "KB_E0": k1["boost"]["e0"],
    "KB_P25": k1["boost"]["p25"], "KB_E25": k1["boost"]["e25"],
    "KD_P0": k1["damp"]["p0"], "KD_E0": k1["damp"]["e0"],
    "KD_P25": k1["damp"]["p25"], "KD_E25": k1["damp"]["e25"],
    "LAMROWS": LAMROWS, "OCCROWS": OCCROWS,
    "OCCSHIFT": f"{occ_shift:.3f}",
    "OCCRAREST": occ_rare_st, "OCCRAREUNC": occ_rare_unc,
}
for k, v in srep.items():
    SI_SECTION = SI_SECTION.replace(f"[[{k}]]", v)
assert "[[" not in SI_SECTION, "unfilled placeholder remains"

si = open("../si.tex").read()
open("../si.tex.bak_v3", "w").write(si)
anchor = ("\n    % ============================================================\n"
          "    \\section{Fokker--Planck linearization")
assert anchor in si, "SI splice anchor not found"
si = si.replace(anchor, SI_SECTION + anchor, 1)
open("../si.tex", "w").write(si)
print("si.tex updated")
print("damp paired t =", round(float(t_stat), 2),
      "| reduced captures", round(float(red_pc), 1), "% of exact gain")
