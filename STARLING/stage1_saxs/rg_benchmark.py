#!/usr/bin/env python3
"""
rg_benchmark.py -- Stage 1 metric: ensemble Rg vs SAXS-derived Rg.

Two benchmarks:

A) 53-protein curve set (paper Fig. 3E/S13-S14 systems):
   experimental Rg from the MFF re-analysis (mff_analysis_all.csv)
   vs mean Rg of the locally generated stock STARLING ensemble.

B) 133-sequence Rg set (paper Fig. 3B headline: RMSE ~ 4.7 A, R^2 ~ 0.9):
   experimental Rg from saxs_rg/all_comparison_data.csv (sequences filtered
   to <=384 residues) vs locally generated stock STARLING ensembles.
   NOTE: this requires having run generate_ensembles.py on
   all_comparison_seqs.fasta with --max-len 384.

Rg is computed from distance maps exactly as STARLING does:
    Rg^2 = (1 / 2 L^2) sum_ij d_ij^2

Usage
-----
    # benchmark A
    python rg_benchmark.py --mode mff \
        --mff-csv data/saxs_scattering/experiment/mff_analysis_all.csv \
        --rg-dir runs/stage1_scattering/rg --out results/rg_mff

    # benchmark B
    python rg_benchmark.py --mode comparison \
        --comparison-csv data/saxs_rg/all_comparison_data.csv \
        --rg-dir runs/stage1_rg/rg --out results/rg_133
"""

import argparse
import csv
import os

import numpy as np


def read_mff_csv(path):
    """name -> (Rg, err); robust to the known malformed line in the file."""
    out = {}
    with open(path) as fh:
        for line in fh:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 2:
                continue
            name = parts[0].lower()
            try:
                rg = float(parts[1])
            except ValueError:
                continue
            err = None
            if len(parts) > 2:
                try:
                    err = float(parts[2])
                except ValueError:
                    err = None
            out[name] = (rg, err)
    return out


def read_comparison_csv(path):
    """name -> (SAXS Rg, sequence); only rows with a numeric SAXS Rg."""
    out = {}
    with open(path) as fh:
        rdr = csv.reader(fh)
        header = next(rdr)
        for row in rdr:
            if len(row) < 3:
                continue
            name = row[1].strip()
            try:
                rg = float(row[2])
            except ValueError:
                continue
            seq = row[-1].strip()
            out[name] = (rg, seq)
    return out


def collect_our_rg(rg_dir):
    out = {}
    for f in os.listdir(rg_dir):
        if f.endswith(".npy"):
            out[f[:-4]] = float(np.mean(np.load(os.path.join(rg_dir, f))))
    return out


def report(pairs, out_prefix, label):
    """pairs: list of (name, exp_rg, our_rg)."""
    exp = np.array([p[1] for p in pairs])
    our = np.array([p[2] for p in pairs])
    rmse = float(np.sqrt(np.mean((exp - our) ** 2)))
    r2 = float(np.corrcoef(exp, our)[0, 1] ** 2)
    bias = float(np.mean(our - exp))
    print(f"\n[{label}] n={len(pairs)}")
    print(f"  RMSE  = {rmse:.2f} A")
    print(f"  R^2   = {r2:.3f}")
    print(f"  bias  = {bias:+.2f} A (ours - experiment)")

    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    with open(out_prefix + "_rg_comparison.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "saxs_rg_A", "starling_rg_A", "abs_err_A"])
        for n_, e_, o_ in pairs:
            w.writerow([n_, f"{e_:.2f}", f"{o_:.2f}", f"{abs(e_ - o_):.2f}"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3, 3), dpi=150)
    ax.plot(exp, our, "ok", ms=3, alpha=0.6)
    lim = [0, max(exp.max(), our.max()) * 1.1]
    ax.plot(lim, lim, "k--", lw=0.5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"$R_g$ (SAXS) [$\AA$]", fontsize=8)
    ax.set_ylabel(r"$R_g$ (STARLING, stock) [$\AA$]", fontsize=8)
    ax.text(0.05, 0.92, f"RMSE = {rmse:.2f} $\\AA$", transform=ax.transAxes,
            fontsize=8)
    ax.text(0.05, 0.84, f"$R^2$ = {r2:.2f},  n = {len(pairs)}",
            transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_prefix + "_rg_comparison.pdf")
    fig.savefig(out_prefix + "_rg_comparison.png", dpi=300)
    print(f"  wrote {out_prefix}_rg_comparison.pdf/.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["mff", "comparison"], required=True)
    ap.add_argument("--mff-csv", default=None)
    ap.add_argument("--comparison-csv", default=None)
    ap.add_argument("--rg-dir", required=True,
                    help="dir with <name>.npy per-conformer Rg arrays")
    ap.add_argument("--max-len", type=int, default=384,
                    help="length filter for --mode comparison (paper used 384)")
    ap.add_argument("--out", required=True, help="output path prefix")
    args = ap.parse_args()

    ours = collect_our_rg(args.rg_dir)
    ours_lower = {k.lower(): v for k, v in ours.items()}
    print(f"[setup] found local Rg for {len(ours)} sequences")

    pairs, missing = [], []
    if args.mode == "mff":
        exp = read_mff_csv(args.mff_csv)
        for name, (rg, _err) in exp.items():
            if name in ours_lower:
                pairs.append((name, rg, ours_lower[name]))
            else:
                missing.append(name)
        label = "MFF re-analysed SAXS Rg (53-protein curve set)"
    else:
        exp = read_comparison_csv(args.comparison_csv)
        for name, (rg, seq) in exp.items():
            if len(seq) > args.max_len:
                continue
            if name.lower() in ours_lower:
                pairs.append((name, rg, ours_lower[name.lower()]))
            else:
                missing.append(name)
        label = f"SAXS Rg comparison set (L<={args.max_len})"

    if missing:
        print(f"[warn] {len(missing)} sequences missing local ensembles, e.g. "
              f"{missing[:5]}")
    report(pairs, args.out, label)


if __name__ == "__main__":
    main()
