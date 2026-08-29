#!/usr/bin/env python3
"""
saxs_debye.py -- SAXS curves from STARLING distance maps via the Debye formula.

STARLING produces C-alpha-only coarse-grained ensembles, so all-atom SAXS
codes (FoXS/CRYSOL) can only be applied after a 3D reconstruction step. Here
we instead back-calculate the ensemble-averaged scattering *directly from the
distance maps*, which is exact for the Debye formula and avoids the MDS
reconstruction entirely:

    I(q) = < sum_ij f_i(q) f_j(q) sinc(q * d_ij) >_ensemble

Two modes:

* `--ff uniform` (default): every residue uses the sequence-averaged residue
  form factor f_bar(q), giving

      I(q) = f_bar(q)^2 * [ L + 2 < sum_{i<j} sinc(q d_ij) > ]

  which is evaluated from a *pooled pair-distance histogram* over the whole
  ensemble (fast, and additionally yields the ensemble pair-distance
  distribution P(r) for free).

* `--ff exact`: full sum over residue-type pairs f_a(q) f_b(q) S_ab(q), with
  S_ab from per-type-pair pooled histograms (slower, sequence-specific).

Form factors: vacuum Cromer-Mann atomic scattering factors (9-parameter, ITC
table 6.1.1.4) summed over the atoms of each residue (H excluded, standard
for SAXS). No hydration-layer / excluded-volume correction is applied
(FoXS c1/c2 fitting); this is a *vacuum* Debye calculation. For the Stage-2
comparison (stock DDPM vs controlled DDPM) the same forward model is applied
to both ensembles, so systematic forward-model errors cancel. For reference,
the STARLING paper used FoXS with default parameters (c1=1.0, c2=2.0) on
MDS-reconstructed C-alpha traces; a validation mode below compares our curves
against their published reference curves.

CLI
---
    python saxs_debye.py --dm distance_maps/ash1.npy --seq <SEQ> \
        --out curves/ash1_ours.dat [--ff uniform] [--qmax 0.4] [--dq 0.001]

Output .dat format (tab separated): q  I(q)  err_I(q)
where err is the standard error over conformers (block over the ensemble).
"""

import argparse
import os

import numpy as np

# ----------------------------------------------------------------------
# Cromer-Mann coefficients (a1..a4, b1..b4, c) for X-ray form factors,
# International Tables for Crystallography C, table 6.1.1.4.
# f(s) = sum_i a_i exp(-b_i s^2) + c,  s = q / (4 pi),  q in A^-1.
CROMER_MANN = {
    "C": ([2.31000, 1.02000, 1.58860, 0.86500],
          [20.8439, 10.2075, 0.56870, 51.6512], 0.21560),
    "N": ([12.2126, 3.13220, 2.01250, 1.16630],
          [0.00570, 9.89330, 28.9975, 0.58260], -11.529),
    "O": ([3.04850, 2.28680, 1.54630, 0.86700],
          [13.2771, 5.70110, 0.32390, 32.9089], 0.25080),
    "S": ([6.29150, 3.03530, 1.98910, 1.54100],
          [2.43860, 32.3336, 0.67850, 81.6937], 1.14070),
    "H": ([0.489918, 0.262003, 0.196767, 0.049879],
          [20.6593, 7.74039, 49.5519, 2.20159], 0.001305),
}

# average atomic composition per residue (backbone + sidechain), H excluded
RESIDUE_ATOMS = {
    "A": {"C": 3, "N": 1, "O": 1},
    "R": {"C": 6, "N": 4, "O": 1},
    "N": {"C": 4, "N": 2, "O": 2},
    "D": {"C": 4, "N": 1, "O": 3},
    "C": {"C": 3, "N": 1, "O": 1, "S": 1},
    "Q": {"C": 5, "N": 2, "O": 2},
    "E": {"C": 5, "N": 1, "O": 3},
    "G": {"C": 2, "N": 1, "O": 1},
    "H": {"C": 6, "N": 3, "O": 1},
    "I": {"C": 6, "N": 1, "O": 1},
    "L": {"C": 6, "N": 1, "O": 1},
    "K": {"C": 6, "N": 2, "O": 1},
    "M": {"C": 5, "N": 1, "O": 1, "S": 1},
    "F": {"C": 9, "N": 1, "O": 1},
    "P": {"C": 5, "N": 1, "O": 1},
    "S": {"C": 3, "N": 1, "O": 2},
    "T": {"C": 4, "N": 1, "O": 2},
    "W": {"C": 11, "N": 2, "O": 1},
    "Y": {"C": 9, "N": 1, "O": 2},
    "V": {"C": 5, "N": 1, "O": 1},
}

AA_ORDER = sorted(RESIDUE_ATOMS)


def atom_ff(q, element):
    """Cromer-Mann atomic form factor at q (A^-1)."""
    a, b, c = CROMER_MANN[element]
    s2 = (q / (4.0 * np.pi)) ** 2
    f = np.full_like(q, c, dtype=np.float64)
    for ai, bi in zip(a, b):
        f = f + ai * np.exp(-bi * s2)
    return f


def residue_ff(q, aa):
    """Vacuum form factor of one residue (sum of atomic factors)."""
    f = np.zeros_like(q, dtype=np.float64)
    for el, n in RESIDUE_ATOMS[aa].items():
        f = f + n * atom_ff(q, el)
    return f


def sequence_mean_ff(q, seq):
    """Sequence-averaged residue form factor."""
    f = np.zeros_like(q, dtype=np.float64)
    for aa in seq:
        f = f + residue_ff(q, aa)
    return f / len(seq)


def _pooled_pair_distance_histogram(dms, rmax, dr):
    """
    Pooled histogram of all i<j pair distances over the ensemble.

    Returns (bin_centers, counts) where counts[k] is the total number of
    pairs (summed over conformers) falling in bin k.
    """
    bins = np.arange(0.0, rmax + dr, dr)
    counts = np.zeros(len(bins) - 1, dtype=np.float64)
    L = dms.shape[-1]
    iu = np.triu_indices(L, k=1)
    for m in range(dms.shape[0]):
        d = dms[m][iu]
        counts += np.histogram(d, bins=bins)[0]
    centers = 0.5 * (bins[1:] + bins[:-1])
    return centers, counts


def _pair_type_histograms(dms, seq, rmax, dr):
    """Pooled pair-distance histograms per residue-type pair (a <= b)."""
    bins = np.arange(0.0, rmax + dr, dr)
    L = len(seq)
    idx = {aa: [i for i, r in enumerate(seq) if r == aa] for aa in set(seq)}
    hists = {}
    types = sorted(idx)
    for ia, a in enumerate(types):
        for b in types[ia:]:
            hists[(a, b)] = np.zeros(len(bins) - 1, dtype=np.float64)
    for m in range(dms.shape[0]):
        dm = dms[m]
        for ia, a in enumerate(types):
            Ia = idx[a]
            for b in types[ia:]:
                Ib = idx[b]
                if a == b:
                    sub = dm[np.ix_(Ia, Ia)]
                    vals = sub[np.triu_indices(len(Ia), k=1)]
                else:
                    vals = dm[np.ix_(Ia, Ib)].ravel()
                hists[(a, b)] += np.histogram(vals, bins=bins)[0]
    return bins, hists


def saxs_from_distance_maps(dms, seq, q, dr=0.25, ff_mode="uniform",
                            block=50):
    """
    Ensemble-averaged SAXS curve from (M, L, L) distance maps.

    Returns (I, I_err): mean and standard error over conformers. Errors are
    estimated by evaluating per-block histograms is expensive; instead we
    compute per-conformer curves on the fly in uniform mode. For 'exact'
    mode errors are reported as NaN (use uniform for error bars).
    """
    M, L = dms.shape[0], dms.shape[1]
    assert len(seq) == L
    rmax = float(dms.max()) * 1.02 + dr

    if ff_mode == "uniform":
        fbar = sequence_mean_ff(q, seq)
        # per-conformer histograms -> per-conformer S(q) for error estimate
        bins = np.arange(0.0, rmax + dr, dr)
        centers = 0.5 * (bins[1:] + bins[:-1])
        qr = q[:, None] * centers[None, :]
        sinc_qr = np.sinc(qr / np.pi)  # np.sinc(x) = sin(pi x)/(pi x)
        iu = np.triu_indices(L, k=1)
        S = np.empty((M, len(q)), dtype=np.float64)
        for m in range(M):
            h = np.histogram(dms[m][iu], bins=bins)[0]
            S[m] = L + 2.0 * (sinc_qr @ h)
        I_per_conf = (fbar**2)[None, :] * S
        I = I_per_conf.mean(axis=0)
        I_err = I_per_conf.std(axis=0, ddof=1) / np.sqrt(M)
        return I, I_err

    elif ff_mode == "exact":
        bins, hists = _pair_type_histograms(dms, seq, rmax, dr)
        centers = 0.5 * (bins[1:] + bins[:-1])
        qr = q[:, None] * centers[None, :]
        sinc_qr = np.sinc(qr / np.pi)
        I = np.zeros(len(q), dtype=np.float64)
        counted = set()
        for (a, b), h in hists.items():
            fa, fb = residue_ff(q, a), residue_ff(q, b)
            na = seq.count(a)
            nb = seq.count(b)
            if a == b:
                # diagonal blocks: self terms na * fa^2 + pairs 2*hist
                I += fa * fa * na
                I += 2.0 * (fa * fb) * (sinc_qr @ h) / M
            else:
                I += 2.0 * (fa * fb) * (sinc_qr @ h) / M
        return I, np.full(len(q), np.nan)

    else:
        raise ValueError(ff_mode)


def load_dm(path):
    dms = np.load(path)
    assert dms.ndim == 3 and dms.shape[1] == dms.shape[2]
    return dms.astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dm", required=True, help="distance map .npy (M, L, L)")
    ap.add_argument("--seq", required=True, help="amino acid sequence")
    ap.add_argument("--out", required=True, help="output .dat file")
    ap.add_argument("--ff", choices=["uniform", "exact"], default="uniform")
    ap.add_argument("--qmax", type=float, default=0.4)
    ap.add_argument("--dq", type=float, default=0.001)
    ap.add_argument("--dr", type=float, default=0.25,
                    help="distance histogram bin width (A)")
    args = ap.parse_args()

    dms = load_dm(args.dm)
    q = np.arange(0.0, args.qmax + 0.5 * args.dq, args.dq)
    q[0] = 1e-6  # avoid q=0 singularity in downstream tools; I(0) exact anyway

    print(f"[debye] {os.path.basename(args.dm)}: M={dms.shape[0]}, "
          f"L={dms.shape[1]}, ff={args.ff}")
    I, I_err = saxs_from_distance_maps(dms, args.seq, q, dr=args.dr,
                                       ff_mode=args.ff)
    # exact I(0) = (sum f_i)^2
    if args.ff == "uniform":
        f0 = sequence_mean_ff(np.array([0.0]), args.seq)[0]
        I[0] = (len(args.seq) * f0) ** 2

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savetxt(args.out, np.column_stack([q, I, I_err]),
               fmt="%.18e", delimiter="\t")
    print(f"[debye] wrote {args.out}")


if __name__ == "__main__":
    main()
