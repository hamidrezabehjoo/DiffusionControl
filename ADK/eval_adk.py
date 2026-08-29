#!/usr/bin/env python3
"""Evaluate ADK predictions against the open (4AKE) and closed (1AKE) states.

For every predicted model (PDB) in a Boltz output directory, computes
C-alpha RMSD (Kabsch) and TM-score against both reference structures,
assigns each model to a basin, and writes a per-model CSV plus a summary.

TM-score uses d0 = 1.24 * (L - 15)^(1/3) - 1.8 and a few iterations of
weighted Kabsch superposition (weights 1/(1+(d_i/d0)^2)), which matches
TM-align closely for identical-sequence pairs.

Usage:
    python3 eval_adk.py --pred_dir results_stock/predictions/adk \
        --tag stock --out eval_stock.csv
"""

import argparse
import glob
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# E. coli adenylate kinase, 214 residues; both references share this sequence.
ADK_SEQ = (
    "MRIILLGAPGAGKGTQAQFIMEKYGIPQISTGDMLRAAVKSGSELGKQAKDIMDAGKLVTDELVIALV"
    "KERIAQEDCRNGFLLDGFPRTIPQADAMKEAGINVDYVLEFDVPDELIVDRIVGRRVHAPSGRVYHVK"
    "FNPPKVEGKDDVTGEELTTRKDDQEETVRKRLVEYHQMTAPLIGYYSKEAEAGNTKYAKVDGTKPVAE"
    "VRADLEKILG"
)


def read_ca_coords(path):
    """Read C-alpha coordinates of chain A, residues 1..214, from a PDB file."""
    coords = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            chain = line[21].strip()
            if chain and chain != "A":
                continue
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords[resseq] = (x, y, z)
    if not coords:
        raise ValueError(f"no CA atoms found in {path}")
    idx = sorted(coords)
    return idx, np.array([coords[i] for i in idx], dtype=float)


def kabsch_superpose(mobile, ref, weights=None):
    """Return mobile coordinates optimally superposed onto ref."""
    if weights is None:
        weights = np.ones(len(ref))
    w = weights / weights.sum()
    cm = (mobile * w[:, None]).sum(axis=0)
    cr = (ref * w[:, None]).sum(axis=0)
    m = mobile - cm
    r = ref - cr
    cov = (m * w[:, None]).T @ r
    v, _, wt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(v @ wt))
    corr = np.diag([1.0, 1.0, d])
    rot = v @ corr @ wt
    return m @ rot + cr, rot, cm, cr


def rmsd(a, b):
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def tm_score(mobile, ref, n_iter=5):
    """TM-score of mobile onto ref (normalized by len(ref))."""
    L = len(ref)
    d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
    d0 = max(d0, 0.5)
    aln, _, _, _ = kabsch_superpose(mobile, ref)
    for _ in range(n_iter):
        d = np.sqrt(((aln - ref) ** 2).sum(axis=1))
        w = 1.0 / (1.0 + (d / d0) ** 2)
        aln, _, _, _ = kabsch_superpose(mobile, ref, weights=w)
    d = np.sqrt(((aln - ref) ** 2).sum(axis=1))
    tm = float((1.0 / (1.0 + (d / d0) ** 2)).sum() / L)
    return tm, rmsd(aln, ref), d0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True,
                    help="Directory with predicted *_model_*.pdb files")
    ap.add_argument("--ref_open", default=None,
                    help="Open-state reference (default: data/4AKE.pdb next to "
                         "the script or under the current directory)")
    ap.add_argument("--ref_closed", default=None,
                    help="Closed-state reference (default: data/1AKE.pdb, same rule)")
    ap.add_argument("--tag", default="run", help="Label for this ensemble")
    ap.add_argument("--out", default=None, help="CSV output path")
    ap.add_argument("--escape_rmsd", type=float, default=3.5,
                    help="RMSD threshold (A) to count a model as reaching a native basin")
    args = ap.parse_args()

    def find_ref(given, name):
        if given:
            return given
        for base in (SCRIPT_DIR, os.getcwd()):
            cand = os.path.join(base, "data", name)
            if os.path.isfile(cand):
                return cand
        sys.exit(f"reference {name} not found in data/ next to the script or "
                 f"under the current directory; pass --ref_open/--ref_closed")

    ref_open_path = find_ref(args.ref_open, "4AKE.pdb")
    ref_closed_path = find_ref(args.ref_closed, "1AKE.pdb")
    _, ref_open = read_ca_coords(ref_open_path)
    _, ref_closed = read_ca_coords(ref_closed_path)
    assert len(ref_open) == len(ref_closed) == 214, "reference length mismatch"

    if not os.path.isdir(args.pred_dir):
        sys.exit(f"prediction directory not found: {args.pred_dir}")
    # Boltz nests outputs under <out_dir>/boltz_results_<name>/predictions/<name>/;
    # search recursively so any parent directory works.
    files = sorted(glob.glob(
        os.path.join(args.pred_dir, "**", "*_model_*.pdb"), recursive=True))
    if not files:
        any_pdb = glob.glob(os.path.join(args.pred_dir, "**", "*.pdb"),
                            recursive=True)
        hint = (f" (found {len(any_pdb)} other .pdb files, none matching "
                f"*_model_*.pdb, e.g. {os.path.basename(any_pdb[0])})"
                if any_pdb else " (no .pdb files at all)")
        sys.exit(f"no *_model_*.pdb files under {args.pred_dir}{hint}")

    rows = []
    for path in files:
        idx, pred = read_ca_coords(path)
        if len(pred) != 214:
            print(f"WARNING: {os.path.basename(path)} has {len(pred)} CA atoms, skipped")
            continue
        tm_o, r_o, _ = tm_score(pred, ref_open)
        tm_c, r_c, _ = tm_score(pred, ref_closed)
        if r_o <= args.escape_rmsd and r_o < r_c:
            basin = "open"
        elif r_c <= args.escape_rmsd and r_c < r_o:
            basin = "closed"
        else:
            basin = "intermediate/other"
        rows.append(dict(
            model=os.path.basename(path),
            rmsd_open=r_o, tm_open=tm_o,
            rmsd_closed=r_c, tm_closed=tm_c,
            basin=basin,
        ))

    out = args.out or f"eval_{args.tag}.csv"
    with open(out, "w") as f:
        f.write("tag,model,rmsd_open,tm_open,rmsd_closed,tm_closed,basin\n")
        for r in rows:
            f.write(f"{args.tag},{r['model']},{r['rmsd_open']:.3f},{r['tm_open']:.4f},"
                    f"{r['rmsd_closed']:.3f},{r['tm_closed']:.4f},{r['basin']}\n")

    n = len(rows)
    basins = {}
    for r in rows:
        basins[r["basin"]] = basins.get(r["basin"], 0) + 1
    best_open = min(rows, key=lambda r: r["rmsd_open"])
    best_closed = min(rows, key=lambda r: r["rmsd_closed"])
    print(f"\n=== {args.tag}: {n} models ===")
    print(f"  basin counts: {basins}")
    print(f"  best to open   (4AKE): {best_open['model']}  "
          f"RMSD {best_open['rmsd_open']:.2f} A, TM {best_open['tm_open']:.3f}")
    print(f"  best to closed (1AKE): {best_closed['model']}  "
          f"RMSD {best_closed['rmsd_closed']:.2f} A, TM {best_closed['tm_closed']:.3f}")
    mo = np.mean([r["rmsd_open"] for r in rows])
    mc = np.mean([r["rmsd_closed"] for r in rows])
    print(f"  mean RMSD to open {mo:.2f} A | to closed {mc:.2f} A")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
