#!/usr/bin/env python3
"""
curve_benchmark.py -- Stage 1 metric: fit ensemble SAXS curves to experiment.

Replicates the STARLING paper's curve-comparison protocol (their Fig. 3E and
Figs. S13-S14, computed with the internal `housetools` helper
saxs.fit_scattering_curves):

  1. scale-align the simulated curve to experiment over q in [0.01, 0.25] A^-1
       c* = argmin_c sum_q w(q) [ I_exp(q) - c I_sim(q) ]^2,   w = 1/sigma^2
     (closed form: c* = sum(w I_exp I_sim) / sum(w I_sim^2))
  2. goodness of fit over the fitting-relevant range q <= 0.15 A^-1:
       chi2_r = (1/(n-1)) sum_q [ (I_exp - c* I_sim) / sigma ]^2

The script compares, for every protein:
  * experiment  vs OUR Debye curve from the locally generated stock ensemble
  * experiment  vs the paper's REFERENCE FoXS curve (downloaded; replication
    check that requires no generation at all)
  * OUR curve   vs the reference FoXS curve (forward-model sanity check)

Usage
-----
    python curve_benchmark.py \
        --exp-dir data/saxs_scattering/experiment \
        --our-curves curves \
        --ref-dir data/saxs_scattering/ensembles \
        --seq-fasta data/saxs_scattering/experiment/sequences.fasta \
        --out results/curve_benchmark

Outputs: curve_metrics.csv, one overlay PDF per protein, summary printout.
"""

import argparse
import glob
import os

import numpy as np

QMIN_FIT, QMAX_FIT, QMAX_CHI2 = 0.01, 0.25, 0.15


def read_fasta(path):
    seqs, name, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf)
                name = line[1:].split()[0]
                buf = []
            elif line:
                buf.append(line)
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def load_curve(path):
    """Load q, I, sigma from a whitespace/tab separated 3-col file."""
    q, I, e = [], [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                qi, Ii = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            ei = float(parts[2]) if len(parts) > 2 else np.nan
            q.append(qi)
            I.append(Ii)
            e.append(ei)
    q, I, e = np.array(q), np.array(I), np.array(e)
    if np.all(np.isnan(e)) or np.any(e <= 0):
        e = np.full_like(q, np.nan)
    return q, I, e


def experimental_file(exp_dir, name):
    """Prefer <name>_clean.dat, else <name>.dat (paper's own convention).

    Case-insensitive: the fasta keys and folder names disagree in case for a
    few proteins (e.g. fasta 'nhE6cmdd' vs folder 'nhe6cmdd',
    'hev_pnt3_YYY_AAA' vs 'hev_pnt3_yyy_aaa').
    """
    if not os.path.isdir(exp_dir):
        return None
    folders = {f.lower(): f for f in os.listdir(exp_dir)
               if os.path.isdir(os.path.join(exp_dir, f))}
    folder = folders.get(name.lower())
    if folder is None:
        return None
    d = os.path.join(exp_dir, folder)
    files = {f.lower(): f for f in os.listdir(d)}
    for cand in (f"{name}_clean.dat", f"{folder}_clean.dat",
                 f"{name}.dat", f"{folder}.dat"):
        hit = files.get(cand.lower())
        if hit is not None:
            return os.path.join(d, hit)
    return None


def fit_curves(q_exp, I_exp, s_exp, q_sim, I_sim,
               qmin=QMIN_FIT, qmax=QMAX_FIT, qmax_chi2=QMAX_CHI2):
    """Scale-align sim->exp on [qmin,qmax]; chi2_r on q<=qmax_chi2."""
    Is = np.interp(q_exp, q_sim, I_sim)
    ok = np.isfinite(s_exp) & (s_exp > 0)
    w = np.where(ok, 1.0 / np.where(ok, s_exp, 1) ** 2, 0.0)

    mfit = (q_exp >= qmin) & (q_exp <= qmax) & (w > 0)
    denom = np.sum(w[mfit] * Is[mfit] ** 2)
    if denom <= 0 or not np.isfinite(denom):
        # some experimental files lack usable error bars in the fit range;
        # fall back to unweighted least squares for the scale
        mfit2 = (q_exp >= qmin) & (q_exp <= qmax)
        denom2 = np.sum(Is[mfit2] ** 2)
        c = (np.sum(I_exp[mfit2] * Is[mfit2]) / denom2
             if denom2 > 0 else np.nan)
    else:
        c = np.sum(w[mfit] * I_exp[mfit] * Is[mfit]) / denom

    mchi = (q_exp <= qmax_chi2) & (w > 0)
    n = int(mchi.sum())
    chi2 = np.sum(((I_exp[mchi] - c * Is[mchi]) / s_exp[mchi]) ** 2)
    chi2_r = chi2 / max(n - 1, 1)
    # full-range chi2 for reference
    mall = ok
    chi2_full = np.sum(((I_exp[mall] - c * Is[mall]) / s_exp[mall]) ** 2)
    chi2_full_r = chi2_full / max(int(mall.sum()) - 1, 1)
    return dict(scale=c, chi2_r=chi2_r, n_chi2=n, chi2_full_r=chi2_full_r)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--our-curves", default=None,
                    help="dir with <name>.dat curves from saxs_debye.py")
    ap.add_argument("--ref-dir", default=None,
                    help="downloaded reference ensembles/ dir with "
                         "<name>/average_curve.dat")
    ap.add_argument("--seq-fasta", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)
    fig_dir = os.path.join(args.out, "overlays")
    os.makedirs(fig_dir, exist_ok=True)

    seqs = read_fasta(args.seq_fasta)
    rows = []
    for name in sorted(seqs):
        expf = experimental_file(args.exp_dir, name)
        if expf is None:
            print(f"[skip] {name}: no experimental curve found")
            continue
        qe, Ie, se = load_curve(expf)

        res = {"name": name, "L": len(seqs[name])}

        ours = None
        if args.our_curves:
            p = os.path.join(args.our_curves, f"{name}.dat")
            if os.path.isfile(p):
                qo, Io, _ = load_curve(p)
                r = fit_curves(qe, Ie, se, qo, Io)
                res.update({f"ours_{k}": v for k, v in r.items()})
                ours = (qo, Io, r["scale"])

        ref = None
        if args.ref_dir:
            p = None
            if os.path.isdir(args.ref_dir):
                ref_folders = {f.lower(): f for f in os.listdir(args.ref_dir)
                               if os.path.isdir(os.path.join(args.ref_dir, f))}
                rf = ref_folders.get(name.lower())
                if rf is not None:
                    cand = os.path.join(args.ref_dir, rf, "average_curve.dat")
                    if os.path.isfile(cand):
                        p = cand
            if p is not None:
                qr, Ir, _ = load_curve(p)
                r = fit_curves(qe, Ie, se, qr, Ir)
                res.update({f"ref_{k}": v for k, v in r.items()})
                ref = (qr, Ir, r["scale"])

        if ours is not None and ref is not None:
            # ours vs reference, aligned to each other (uniform weights)
            qr, Ir, _ = ref
            qo, Io, _ = ours
            Io_i = np.interp(qr, qo, Io)
            c = np.sum(Ir * Io_i) / np.sum(Io_i**2)
            res["ours_vs_ref_rmsd_log"] = float(
                np.sqrt(np.mean((np.log10(Ir[Ir > 0])
                                 - np.log10((c * Io_i)[Ir > 0])) ** 2)))

        rows.append(res)

        # overlay figure
        fig, ax = plt.subplots(figsize=(3, 2.2), dpi=150)
        ax.errorbar(qe, Ie, yerr=se if np.all(np.isfinite(se)) else None,
                    fmt="o", ms=1.5, lw=0.3, color="k", alpha=0.5,
                    label="experiment")
        if ref is not None:
            ax.plot(ref[0], ref[2] * ref[1], "-", lw=0.8, color="tab:blue",
                    label=f"paper FoXS ($\\chi^2_r$={res.get('ref_chi2_r', np.nan):.1f})")
        if ours is not None:
            ax.plot(ours[0], ours[2] * ours[1], "-", lw=0.8, color="tab:red",
                    label=f"ours Debye ($\\chi^2_r$={res.get('ours_chi2_r', np.nan):.1f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("q ($\\AA^{-1}$)", fontsize=7)
        ax.set_ylabel("I(q)", fontsize=7)
        ax.set_title(name, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{name}.pdf"))
        plt.close(fig)

    # ------------------------------------------------------------- summary
    import csv
    keys = sorted({k for r in rows for k in r})
    csv_path = os.path.join(args.out, "curve_metrics.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {csv_path}")

    def summarize(tag):
        vals = np.array([r[f"{tag}_chi2_r"] for r in rows if f"{tag}_chi2_r" in r])
        if len(vals):
            print(f"{tag}: n={len(vals)}  median chi2_r={np.median(vals):.2f}  "
                  f"mean={vals.mean():.2f}  frac<5={np.mean(vals < 5):.2f}")

    summarize("ref")
    summarize("ours")
    print("\nOverlays in", fig_dir)


if __name__ == "__main__":
    main()
