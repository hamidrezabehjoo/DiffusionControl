#!/usr/bin/env python3
"""
generate_stage2.py -- Stage 2: generate STARLING ensembles with the reduced
noise controller (ControlledDDPMSampler) for the SAXS benchmark.

Mirrors generate_ensembles.py exactly, but the reverse SDE carries the
controller nu(t) = alpha*g(t) on [0, t_w] (two-line DDPM modification:

    score coefficient beta_t -> (beta_t + nu_t^2)/2
    injected noise std       -> nu_t   on control-window steps,

out-of-window steps bit-identical to stock).

Output layout is identical to Stage 1, so batch_curves.py /
curve_benchmark.py / rg_benchmark.py run unchanged:

    <out>/distance_maps/<name>.npy   (M, L, L)
    <out>/rg/<name>.npy              (M,)

Common random numbers (CRN): with --paired (default) the stock arm is
regenerated with the SAME per-protein seed, so both arms consume identical
Gaussian draws step-by-step (both samplers draw one randn_like per reverse
step). This makes the comparison pairwise at the level of noise realizations
and sharply reduces sampling noise in per-protein differences. The Stage-1
stock ensembles remain as an independent, larger replication of the stock arm.

Usage
-----
    # paired comparison, validated protein defaults (alpha=2.5, t_w=0.2)
    python generate_stage2.py \
        --fasta data/saxs_scattering/experiment/sequences.fasta \
        --out runs/stage2_alpha2.5_tw0.2 \
        --conformations 400 --alpha 2.5 --window 0.2

    # controlled arm only (compare against Stage-1 stock statistically)
    python generate_stage2.py --fasta ... --out runs/stage2_... \
        --conformations 400 --alpha 2.5 --window 0.2 --no-stock

Requires controlled_sampler.py (one directory up by default, or pass
--controller-module).
"""

import argparse
import os
import sys
import time
import zlib

import numpy as np
import torch


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


def protein_seed(base_seed, name):
    """Deterministic per-protein seed (stable across runs/machines)."""
    return (base_seed + zlib.crc32(name.encode())) % (2**31)


def run_ensemble(sampler, sequence, conformations, batch_size):
    """Sample `conformations` distance maps in batches -> (M, L, L) numpy."""
    from starling.inference.generation import symmetrize_distance_map
    maps = []
    done = 0
    while done < conformations:
        b = min(batch_size, conformations - done)
        dm = sampler.sample(b, labels=sequence, show_per_step_progress_bar=False)
        L = len(sequence)
        maps.append(torch.stack(
            [symmetrize_distance_map(m[:, :L, :L]) for m in dm]
        ))
        done += b
    return torch.cat(maps, dim=0).numpy()


def rg_from_dms(dms):
    L = dms.shape[-1]
    return np.sqrt((dms.astype(np.float64) ** 2).sum(axis=(1, 2))
                   / (2.0 * L * L))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conformations", type=int, default=400)
    ap.add_argument("--alpha", type=float, default=2.5,
                    help="noise amplification on the window (>1 boost, <1 damp)")
    ap.add_argument("--window", type=float, default=0.2,
                    help="control window t_w as fraction of forward time")
    ap.add_argument("--nu-min", type=float, default=0.0)
    ap.add_argument("--nu-schedule", default=None,
                    help="optional .npy full per-timestep nu schedule "
                         "(overrides --alpha/--window)")
    ap.add_argument("--no-stock", action="store_true",
                    help="skip the paired stock arm (compare against Stage-1 "
                         "stock statistically instead)")
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--ionic-strength", type=float, default=150.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0,
                    help="base seed; per-protein seed = seed + crc32(name)")
    ap.add_argument("--only", default=None,
                    help="comma-separated sequence names to run")
    ap.add_argument("--controller-module", default=None,
                    help="path to controlled_sampler.py (default: one "
                         "directory up from this script)")
    args = ap.parse_args()

    # locate the controller module
    ctrl_dir = args.controller_module or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir)
    ctrl_dir = os.path.abspath(ctrl_dir)
    if not os.path.isfile(os.path.join(ctrl_dir, "controlled_sampler.py")):
        sys.exit(f"controlled_sampler.py not found in {ctrl_dir}; "
                 f"pass --controller-module")
    sys.path.insert(0, ctrl_dir)
    from controlled_sampler import ControlledDDPMSampler
    from starling.samplers.ddpm_sampler import DDPMSampler
    from starling.inference.model_loading import ModelManager

    seqs = read_fasta(args.fasta)
    if args.only:
        keep = set(args.only.split(","))
        seqs = {k: v for k, v in seqs.items() if k in keep}
    print(f"[setup] {len(seqs)} sequences | alpha={args.alpha} "
          f"window={args.window} | paired stock: {not args.no_stock}")

    dm_dir = os.path.join(args.out, "distance_maps")
    rg_dir = os.path.join(args.out, "rg")
    stock_dm_dir = os.path.join(args.out, "stock_paired", "distance_maps")
    stock_rg_dir = os.path.join(args.out, "stock_paired", "rg")
    for d in (dm_dir, rg_dir):
        os.makedirs(d, exist_ok=True)
    if not args.no_stock:
        for d in (stock_dm_dir, stock_rg_dir):
            os.makedirs(d, exist_ok=True)

    mm = ModelManager()
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder_model, diffusion = mm.get_models(device=device)
    print(f"[setup] device={device}, T={diffusion.num_timesteps} trained steps")

    nu_schedule = None
    if args.nu_schedule is not None:
        nu_schedule = np.load(args.nu_schedule)

    controlled = ControlledDDPMSampler(
        ddpm_model=diffusion, encoder_model=encoder_model,
        alpha=args.alpha, window=args.window, nu_min=args.nu_min,
        nu_schedule=nu_schedule, ionic_strength=args.ionic_strength,
    )
    stock = None if args.no_stock else DDPMSampler(
        ddpm_model=diffusion, encoder_model=encoder_model,
        ionic_strength=args.ionic_strength,
    )
    n_win = int(round(args.window * controlled.n_steps))
    print(f"[setup] controller active on {n_win}/{controlled.n_steps} "
          f"reverse steps (late-reverse window)")

    for idx, (name, seq) in enumerate(seqs.items(), 1):
        seed = protein_seed(args.seed, name)

        dm_path = os.path.join(dm_dir, f"{name}.npy")
        if not os.path.exists(dm_path):
            t0 = time.time()
            torch.manual_seed(seed)
            np.random.seed(seed)
            dms = run_ensemble(controlled, seq, args.conformations,
                               args.batch_size)
            np.save(dm_path, dms.astype(np.float32))
            np.save(os.path.join(rg_dir, f"{name}.npy"),
                    rg_from_dms(dms).astype(np.float32))
            print(f"[{idx}/{len(seqs)}] {name}: controlled done in "
                  f"{time.time()-t0:.0f}s, mean Rg = "
                  f"{rg_from_dms(dms).mean():.2f} A")
        else:
            print(f"[{idx}/{len(seqs)}] {name}: controlled exists, skipping")

        if stock is not None:
            s_path = os.path.join(stock_dm_dir, f"{name}.npy")
            if not os.path.exists(s_path):
                t0 = time.time()
                torch.manual_seed(seed)   # CRN: identical draws as controlled
                np.random.seed(seed)
                dms = run_ensemble(stock, seq, args.conformations,
                                   args.batch_size)
                np.save(s_path, dms.astype(np.float32))
                np.save(os.path.join(stock_rg_dir, f"{name}.npy"),
                        rg_from_dms(dms).astype(np.float32))
                print(f"[{idx}/{len(seqs)}] {name}: paired stock done in "
                      f"{time.time()-t0:.0f}s, mean Rg = "
                      f"{rg_from_dms(dms).mean():.2f} A")
            else:
                print(f"[{idx}/{len(seqs)}] {name}: paired stock exists, "
                      f"skipping")

    print("\nAll Stage-2 ensembles generated.")


if __name__ == "__main__":
    main()
