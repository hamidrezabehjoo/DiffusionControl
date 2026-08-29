#!/usr/bin/env python3
"""
Compare a controlled STARLING ensemble against a baseline ensemble.

Reports the quantities the noise controller is designed to improve:
ensemble diversity (pairwise distance-map RMSD spread, Rg distribution
width) while leaving local structure intact. Usage:

    python compare_ensembles.py baseline.npy controlled.npy
"""

import sys

import numpy as np


def rg_from_dm(dm):
    """Rg per conformation from a distance map, Rg^2 = sum d_ij^2 / (2L^2)."""
    L = dm.shape[-1]
    rg2 = (dm**2).sum(axis=(-2, -1)) / (2.0 * L * L)
    return np.sqrt(np.clip(rg2, 0.0, None))


def pairwise_rmsd(dm, max_pairs=20000, seed=0):
    """RMSD between pairs of distance maps (a diversity proxy)."""
    n = len(dm)
    rng = np.random.default_rng(seed)
    n_pairs = min(max_pairs, n * (n - 1) // 2)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    diff = dm[i] - dm[j]
    return np.sqrt((diff**2).mean(axis=(-2, -1)))


def main():
    base = np.load(sys.argv[1])
    ctrl = np.load(sys.argv[2])

    for label, dm in [("baseline ", base), ("controlled", ctrl)]:
        rg = rg_from_dm(dm)
        pr = pairwise_rmsd(dm)
        print(f"{label}: N={len(dm)}")
        print(f"  Rg               : {rg.mean():8.3f} +/- {rg.std():.3f} A")
        print(f"  Rg spread (IQR)  : {np.percentile(rg, 75) - np.percentile(rg, 25):8.3f} A")
        print(f"  pairwise DM RMSD : {pr.mean():8.3f} +/- {pr.std():.3f} A")

    rg_b, rg_c = rg_from_dm(base), rg_from_dm(ctrl)
    pr_b, pr_c = pairwise_rmsd(base), pairwise_rmsd(ctrl)
    print("\ncontrolled vs baseline:")
    print(f"  pairwise RMSD    : {100*(pr_c.mean()/pr_b.mean()-1):+6.1f}%")
    print(f"  Rg std           : {100*(rg_c.std()/rg_b.std()-1):+6.1f}%")
    print(f"  Rg IQR           : "
          f"{100*((np.percentile(rg_c,75)-np.percentile(rg_c,25))/(np.percentile(rg_b,75)-np.percentile(rg_b,25))-1):+6.1f}%")


if __name__ == "__main__":
    main()
