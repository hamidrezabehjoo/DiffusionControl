#!/usr/bin/env python3
"""
compare_stage2.py -- Stage 2 analysis: controlled vs stock DDPM on the SAXS
benchmark.

Merges:
  * curve metrics for the controlled arm (curve_benchmark.py output on the
    Stage-2 curves) and the stock arm (Stage-1 results, or curve_benchmark.py
    output on the paired stock curves)
  * per-conformer Rg arrays from both arms

Reports per protein:
  * chi2_r(controlled) vs chi2_r(stock) and the difference
  * mean Rg(controlled) vs mean Rg(stock) vs experimental Rg
  * |Rg - Rg_exp| change (did the controller move TOWARD experiment?)

Plus summary statistics (median delta, sign test, Wilcoxon signed-rank) and
figures.

Usage
-----
    python compare_stage2.py \
        --stock-metrics results/curve_benchmark/curve_metrics.csv \
        --ctrl-metrics  results/curve_benchmark_stage2/curve_metrics.csv \
        --stock-rg runs/stage1_scattering/rg \
        --ctrl-rg  runs/stage2_alpha2.5_tw0.2/rg \
        --mff-csv data/saxs_scattering/experiment/mff_analysis_all.csv \
        --out results/stage2_comparison
"""

import argparse
import csv
import os

import numpy as np


def read_metrics(path):
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            name = r["name"]
            try:
                chi2 = float(r["ours_chi2_r"])
                n = int(r["ours_n_chi2"])
            except (ValueError, KeyError):
                continue
            out[name.lower()] = dict(name=name, chi2_r=chi2, n_chi2=n, L=int(r["L"]))
    return out


def read_rg_dir(path):
    out = {}
    for f in os.listdir(path):
        if f.endswith(".npy"):
            out[f[:-4].lower()] = np.load(os.path.join(path, f))
    return out


def read_mff(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            p = [x.strip() for x in line.strip().split(",")]
            if len(p) >= 2:
                try:
                    out[p[0].lower()] = float(p[1])
                except ValueError:
                    pass
    return out


def sign_test(deltas):
    """Two-sided sign test p-value (exact, binomial)."""
    from math import comb
    d = np.asarray(deltas)
    d = d[d != 0]
    n = len(d)
    k = int(np.sum(d > 0))
    if n == 0:
        return np.nan, 0, 0
    p = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2**n * 2
    return min(p, 1.0), k, n


def wilcoxon(deltas):
    from scipy.stats import wilcoxon as _w
    d = np.asarray(deltas)
    d = d[d != 0]
    if len(d) < 5:
        return np.nan
    return _w(d).pvalue


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stock-metrics", required=True)
    ap.add_argument("--ctrl-metrics", required=True)
    ap.add_argument("--stock-rg", required=True)
    ap.add_argument("--ctrl-rg", required=True)
    ap.add_argument("--mff-csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)
    stock = read_metrics(args.stock_metrics)
    ctrl = read_metrics(args.ctrl_metrics)
    rg_stock = read_rg_dir(args.stock_rg)
    rg_ctrl = read_rg_dir(args.ctrl_rg)
    exp_rg = read_mff(args.mff_csv)

    common = sorted(set(stock) & set(ctrl))
    print(f"[setup] {len(common)} proteins in both metric files")

    rows = []
    for k in common:
        r = dict(name=stock[k]["name"], L=stock[k]["L"],
                 chi2_stock=stock[k]["chi2_r"], chi2_ctrl=ctrl[k]["chi2_r"],
                 n_chi2=stock[k]["n_chi2"])
        if k in rg_stock and k in rg_ctrl:
            r["rg_stock"] = float(np.mean(rg_stock[k]))
            r["rg_ctrl"] = float(np.mean(rg_ctrl[k]))
            r["rg_std_stock"] = float(np.std(rg_stock[k]))
            r["rg_std_ctrl"] = float(np.std(rg_ctrl[k]))
            if k in exp_rg:
                r["rg_exp"] = exp_rg[k]
                r["abs_err_stock"] = abs(r["rg_stock"] - exp_rg[k])
                r["abs_err_ctrl"] = abs(r["rg_ctrl"] - exp_rg[k])
        rows.append(r)

    # ------------------------------------------------------- write table
    keys = ["name", "L", "chi2_stock", "chi2_ctrl", "n_chi2",
            "rg_exp", "rg_stock", "rg_ctrl", "rg_std_stock", "rg_std_ctrl",
            "abs_err_stock", "abs_err_ctrl"]
    with open(os.path.join(args.out, "stage2_per_protein.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

    # ------------------------------------------------------- chi2 analysis
    valid = [r for r in rows if r["n_chi2"] > 0]
    d_chi2 = np.array([r["chi2_ctrl"] - r["chi2_stock"] for r in valid])
    cs = np.array([r["chi2_stock"] for r in valid])
    cc = np.array([r["chi2_ctrl"] for r in valid])
    p_sign, k_pos, n_sign = sign_test(d_chi2)
    p_wil = wilcoxon(d_chi2)
    print(f"\n[SAXS chi2_r] n={len(valid)} proteins with experimental errors")
    print(f"  stock median={np.median(cs):.2f}  controlled median={np.median(cc):.2f}")
    print(f"  median delta (ctrl - stock) = {np.median(d_chi2):+.3f}")
    print(f"  improved (delta<0): {np.sum(d_chi2<0)}/{len(d_chi2)}")
    print(f"  sign test p = {p_sign:.4f}   Wilcoxon p = {p_wil:.4f}")

    # --------------------------------------------------------- Rg analysis
    rg_rows = [r for r in rows if "rg_exp" in r]
    if rg_rows:
        de = np.array([r["abs_err_ctrl"] - r["abs_err_stock"] for r in rg_rows])
        rmse_s = np.sqrt(np.mean([(r["rg_stock"]-r["rg_exp"])**2 for r in rg_rows]))
        rmse_c = np.sqrt(np.mean([(r["rg_ctrl"]-r["rg_exp"])**2 for r in rg_rows]))
        p_sign_rg, _, _ = sign_test(de)
        p_wil_rg = wilcoxon(de)
        print(f"\n[Rg vs MFF experiment] n={len(rg_rows)}")
        print(f"  RMSE stock = {rmse_s:.2f} A   controlled = {rmse_c:.2f} A")
        print(f"  median delta |Rg-Rg_exp| = {np.median(de):+.3f} A")
        print(f"  improved: {np.sum(de<0)}/{len(de)}   "
              f"sign p = {p_sign_rg:.4f}   Wilcoxon p = {p_wil_rg:.4f}")
        # ensemble spread: the controller's primary predicted effect
        ss = np.array([r["rg_std_stock"] for r in rg_rows])
        sc_ = np.array([r["rg_std_ctrl"] for r in rg_rows])
        dstd = sc_ - ss
        print(f"\n[Rg spread] median std stock = {np.median(ss):.2f} A   "
              f"controlled = {np.median(sc_):.2f} A")
        print(f"  median delta std = {np.median(dstd):+.3f} A   "
              f"Wilcoxon p = {wilcoxon(dstd):.4f}")

    # ----------------------------------------------------------- figures
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=150)
    ax = axes[0]
    ax.plot(cs, cc, "ok", ms=3, alpha=0.6)
    lim = [0, max(cs.max(), cc.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=0.5)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"$\chi^2_r$ stock DDPM", fontsize=8)
    ax.set_ylabel(r"$\chi^2_r$ controlled", fontsize=8)
    ax.set_title(f"median $\\Delta$ = {np.median(d_chi2):+.2f} "
                 f"(Wilcoxon p = {p_wil:.3f})", fontsize=8)
    ax.tick_params(labelsize=7)

    ax = axes[1]
    if rg_rows:
        es = [r["abs_err_stock"] for r in rg_rows]
        ec = [r["abs_err_ctrl"] for r in rg_rows]
        ax.plot(es, ec, "ok", ms=3, alpha=0.6)
        lim = [0, max(max(es), max(ec)) * 1.05]
        ax.plot(lim, lim, "k--", lw=0.5)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(r"|$R_g$ - $R_g^{exp}$| stock [$\AA$]", fontsize=8)
        ax.set_ylabel(r"|$R_g$ - $R_g^{exp}$| controlled [$\AA$]", fontsize=8)
        ax.set_title(f"RMSE {rmse_s:.2f} -> {rmse_c:.2f} $\\AA$ "
                     f"(p = {p_wil_rg:.3f})", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "stage2_comparison.pdf"))
    fig.savefig(os.path.join(args.out, "stage2_comparison.png"), dpi=300)
    print(f"\nwrote {args.out}/stage2_per_protein.csv and "
          f"stage2_comparison.pdf/.png")


if __name__ == "__main__":
    main()
