#!/usr/bin/env python3
"""
generate_ensembles.py -- Stage 1: generate STOCK STARLING ensembles for the
SAXS benchmark sequences.

This is the unmodified STARLING forward pass (no controller). It writes, per
sequence:

    <out>/<name>.starling          full STARLING ensemble archive
    <out>/distance_maps/<name>.npy (M, L, L) distance maps (Angstrom)
    <out>/rg/<name>.npy            (M,) per-conformer Rg (Angstrom)

Usage
-----
    # the 53-protein SAXS-curve benchmark (paper Fig. 3E / S13-S14)
    python generate_ensembles.py \
        --fasta data/saxs_scattering/experiment/sequences.fasta \
        --out runs/stage1_scattering --conformations 400 --sampler ddpm

    # the 133-sequence Rg benchmark (paper Fig. 3B); sequences are
    # pre-filtered to <384 residues with --max-len
    python generate_ensembles.py \
        --fasta data/saxs_rg/all_comparison_seqs.fasta \
        --out runs/stage1_rg --conformations 600 --sampler ddpm --max-len 384

Notes
-----
* Sampler choice: the STARLING release defaults to DDIM with 30 steps (what
  the paper's high-throughput runs used). We use `--sampler ddpm` here because
  the Stage 2 controller modifies the DDPM reverse SDE; DDPM runs the FULL
  trained Markov chain (no step subsampling), so it is ~num_timesteps/30x
  slower than DDIM. For a quick pipeline smoke test use `--sampler ddim`.
* Distance maps are extracted with starling's own symmetrization and the Rg
  is computed exactly as starling.structure.Ensemble.radius_of_gyration does:
      Rg^2 = (1 / 2 L^2) * sum_ij d_ij^2
* Resumable: sequences with an existing distance-map .npy are skipped.
"""

import argparse
import os
import sys
import time

import numpy as np


def read_fasta(path):
    """Minimal FASTA reader -> dict name -> sequence (no extra dependency)."""
    seqs, name, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf)
                name = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conformations", type=int, default=400)
    ap.add_argument("--sampler", default="ddpm",
                    choices=["ddpm", "ddim", "plms"])
    ap.add_argument("--steps", type=int, default=30,
                    help="denoising steps (only used by ddim/plms; ddpm always "
                         "runs the full trained chain)")
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--ionic-strength", type=float, default=150)
    ap.add_argument("--device", default=None, help="cpu / cuda / cuda:0 / mps")
    ap.add_argument("--max-len", type=int, default=None,
                    help="keep only sequences with <= this many residues "
                         "(use 384 for the 133-sequence Rg benchmark)")
    ap.add_argument("--only", default=None,
                    help="comma-separated list of sequence names to run")
    args = ap.parse_args()

    seqs = read_fasta(args.fasta)
    if args.max_len is not None:
        seqs = {k: v for k, v in seqs.items() if len(v) <= args.max_len}
    if args.only:
        keep = set(args.only.split(","))
        seqs = {k: v for k, v in seqs.items() if k in keep}
    print(f"[setup] {len(seqs)} sequences to generate")

    os.makedirs(args.out, exist_ok=True)
    dm_dir = os.path.join(args.out, "distance_maps")
    rg_dir = os.path.join(args.out, "rg")
    os.makedirs(dm_dir, exist_ok=True)
    os.makedirs(rg_dir, exist_ok=True)

    from starling import generate  # lazy import so --help works w/o starling

    for idx, (name, seq) in enumerate(seqs.items(), 1):
        dm_path = os.path.join(dm_dir, f"{name}.npy")
        if os.path.exists(dm_path):
            print(f"[{idx}/{len(seqs)}] {name}: already done, skipping")
            continue
        print(f"[{idx}/{len(seqs)}] {name}: L={len(seq)}, "
              f"{args.conformations} conformers, sampler={args.sampler}")
        t0 = time.time()
        ens = generate(
            {name: seq},
            conformations=args.conformations,
            ionic_strength=args.ionic_strength,
            device=args.device,
            steps=args.steps,
            sampler=args.sampler,
            batch_size=args.batch_size,
            output_directory=args.out,       # writes <name>.starling
            return_data=True,
            return_single_ensemble=True,
            show_progress_bar=False,
            show_per_step_progress_bar=False,
            verbose=True,
        )
        # distance maps: (M, L, L); Rg per conformer as in Ensemble
        dms = np.asarray(ens.distance_maps(), dtype=np.float32)
        L = dms.shape[-1]
        rg = np.sqrt((dms.astype(np.float64) ** 2).sum(axis=(1, 2)) / (2.0 * L * L))
        np.save(dm_path, dms)
        np.save(os.path.join(rg_dir, f"{name}.npy"), rg.astype(np.float32))
        print(f"        done in {time.time() - t0:.1f}s; "
              f"mean Rg = {rg.mean():.2f} A")

    print("\nAll ensembles generated.")


if __name__ == "__main__":
    main()
