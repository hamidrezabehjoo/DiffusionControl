#!/usr/bin/env python3
"""
Generate IDP ensembles with STARLING under the reduced noise controller.

Runs the ControlledDDPMSampler (nu(t) = alpha*g(t) on [0, t_w]) on one or
more sequences and saves the resulting distance maps as .npy files, in the
same format STARLING itself writes. Optionally also runs the stock DDPM
sampler (--baseline) so both ensembles can be compared directly with
compare_ensembles.py.

Examples
--------
# controlled ensemble, validated protein defaults (alpha=2.5, t_w=0.2)
python generate_controlled.py --sequence "MEEPQSDPSVEPPLSQETFSDLWKLLPEN" \
    --conformations 400 --alpha 2.5 --window 0.2 --output runs/p53_tad

# controlled + baseline for a fasta file of sequences
python generate_controlled.py --fasta my_idps.fasta --conformations 400 \
    --alpha 2.5 --window 0.2 --baseline --output runs/benchmark

# plug in a schedule from the exact fixed-point solver of the paper
python generate_controlled.py --sequence "..." --conformations 400 \
    --nu-schedule nu_star.npy --output runs/exact
"""

import argparse
import os
import time

import numpy as np
import torch

from starling.inference.model_loading import ModelManager
from starling.inference.generation import symmetrize_distance_map
from starling.samplers.ddpm_sampler import DDPMSampler

from controlled_sampler import ControlledDDPMSampler


def parse_args():
    p = argparse.ArgumentParser(
        description="STARLING ensemble generation with the reduced "
                    "inference-time noise controller."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sequence", type=str, help="Single amino-acid sequence.")
    src.add_argument("--fasta", type=str, help="Path to a FASTA file.")
    p.add_argument("--conformations", type=int, default=400,
                   help="Number of conformations per sequence (default 400).")
    p.add_argument("--alpha", type=float, default=2.5,
                   help="Noise amplification on the window; >1 boost, <1 damp "
                        "(default 2.5, the validated protein setting).")
    p.add_argument("--window", type=float, default=0.2,
                   help="Control window t_w as a fraction of forward time "
                        "(default 0.2, the validated protein setting).")
    p.add_argument("--nu-min", type=float, default=0.0,
                   help="Admissible floor for nu_t (default 0).")
    p.add_argument("--nu-schedule", type=str, default=None,
                   help="Optional .npy file with a full per-timestep nu "
                        "schedule (overrides --alpha/--window).")
    p.add_argument("--baseline", action="store_true",
                   help="Also generate an ensemble with the stock DDPM "
                        "sampler for comparison.")
    p.add_argument("--batch-size", type=int, default=100,
                   help="Conformations per sampling batch (default 100).")
    p.add_argument("--ionic-strength", type=float, default=150.0,
                   help="Ionic strength in mM for conditioning (default 150).")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="cpu or cuda (default: cuda if available).")
    p.add_argument("--output", type=str, default="starling_control_output",
                   help="Output directory.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed.")
    return p.parse_args()


def read_sequences(args):
    if args.sequence is not None:
        seq = args.sequence.upper()
        return {"sequence_1": seq}
    import protfasta
    return protfasta.read_fasta(args.fasta, invalid_sequence_action="convert")


def run_ensemble(sampler, sequence, conformations, batch_size):
    """Sample `conformations` distance maps in batches; return (N, L, L)."""
    maps = []
    done = 0
    while done < conformations:
        b = min(batch_size, conformations - done)
        dm = sampler.sample(b, labels=sequence,
                            show_per_step_progress_bar=False)
        L = len(sequence)
        maps.append(torch.stack(
            [symmetrize_distance_map(m[:, :L, :L]) for m in dm]
        ))
        done += b
    return torch.cat(maps, dim=0)


def rg_summary(distance_maps):
    """Radius of gyration per conformation from a distance map.

    Rg^2 = (1 / 2L^2) * sum_{ij} d_ij^2   (exact for a distance matrix
    realizable in Euclidean space; a standard estimator otherwise).
    """
    L = distance_maps.shape[-1]
    rg2 = (distance_maps**2).sum(dim=(-2, -1)) / (2.0 * L * L)
    return torch.sqrt(torch.clamp(rg2, min=0.0)).numpy()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sequences = read_sequences(args)

    mm = ModelManager()
    encoder_model, diffusion = mm.get_models(device=args.device)

    nu_schedule = None
    tag = f"alpha{args.alpha:g}_tw{args.window:g}"
    if args.nu_schedule is not None:
        nu_schedule = np.load(args.nu_schedule)
        tag = "exact-schedule"

    controlled = ControlledDDPMSampler(
        ddpm_model=diffusion,
        encoder_model=encoder_model,
        alpha=args.alpha,
        window=args.window,
        nu_min=args.nu_min,
        nu_schedule=nu_schedule,
        ionic_strength=args.ionic_strength,
    )
    stock = DDPMSampler(
        ddpm_model=diffusion,
        encoder_model=encoder_model,
        ionic_strength=args.ionic_strength,
    ) if args.baseline else None

    print(f"device={args.device}  T={controlled.n_steps}  "
          f"controller: nu = {args.alpha:g}*g on [0,{args.window:g}] "
          f"({int(round(args.window * controlled.n_steps))} of "
          f"{controlled.n_steps} steps)")

    for name, sequence in sequences.items():
        t0 = time.time()
        dm_ctrl = run_ensemble(controlled, sequence,
                               args.conformations, args.batch_size)
        path = os.path.join(args.output, f"{name}_STARLING_DM_{tag}.npy")
        np.save(path, dm_ctrl.numpy())
        rg = rg_summary(dm_ctrl)
        print(f"[{name}] controlled: {len(dm_ctrl)} conformations "
              f"({time.time()-t0:.0f}s)  Rg = {rg.mean():.2f} +/- "
              f"{rg.std():.2f} A  ->  {path}")

        if stock is not None:
            torch.manual_seed(args.seed)
            t0 = time.time()
            dm_base = run_ensemble(stock, sequence,
                                   args.conformations, args.batch_size)
            path = os.path.join(args.output, f"{name}_STARLING_DM_baseline.npy")
            np.save(path, dm_base.numpy())
            rg = rg_summary(dm_base)
            print(f"[{name}] baseline:   {len(dm_base)} conformations "
                  f"({time.time()-t0:.0f}s)  Rg = {rg.mean():.2f} +/- "
                  f"{rg.std():.2f} A  ->  {path}")

    print("Done.")


if __name__ == "__main__":
    main()
