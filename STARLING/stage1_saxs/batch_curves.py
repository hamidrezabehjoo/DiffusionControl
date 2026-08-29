#!/usr/bin/env python3
"""
batch_curves.py -- compute Debye SAXS curves for all generated ensembles.

    python batch_curves.py \
        --seq-fasta data/saxs_scattering/experiment/sequences.fasta \
        --dm-dir runs/stage1_scattering/distance_maps \
        --out curves [--ff uniform]

Skips sequences whose curve already exists (resumable).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saxs_debye import saxs_from_distance_maps, sequence_mean_ff  # noqa: E402
from curve_benchmark import read_fasta  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-fasta", required=True)
    ap.add_argument("--dm-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ff", choices=["uniform", "exact"], default="uniform")
    ap.add_argument("--qmax", type=float, default=0.4)
    ap.add_argument("--dq", type=float, default=0.001)
    ap.add_argument("--dr", type=float, default=0.25)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seqs = read_fasta(args.seq_fasta)
    q = np.arange(0.0, args.qmax + 0.5 * args.dq, args.dq)
    q[0] = 1e-6

    for idx, (name, seq) in enumerate(sorted(seqs.items()), 1):
        out_path = os.path.join(args.out, f"{name}.dat")
        dm_path = os.path.join(args.dm_dir, f"{name}.npy")
        if os.path.exists(out_path):
            print(f"[{idx}/{len(seqs)}] {name}: curve exists, skipping")
            continue
        if not os.path.isfile(dm_path):
            print(f"[{idx}/{len(seqs)}] {name}: no distance maps, skipping")
            continue
        dms = np.load(dm_path).astype(np.float64)
        I, I_err = saxs_from_distance_maps(dms, seq, q, dr=args.dr,
                                           ff_mode=args.ff)
        if args.ff == "uniform":
            f0 = sequence_mean_ff(np.array([0.0]), seq)[0]
            I[0] = (len(seq) * f0) ** 2
        np.savetxt(out_path, np.column_stack([q, I, I_err]),
                   fmt="%.18e", delimiter="\t")
        print(f"[{idx}/{len(seqs)}] {name}: curve written "
              f"(M={dms.shape[0]}, L={dms.shape[1]})")


if __name__ == "__main__":
    main()
