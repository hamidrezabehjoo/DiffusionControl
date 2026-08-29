# Stage 1 — Replicate STARLING's SAXS validation (stock model)

Goal: generate ensembles with the **unmodified** STARLING diffusion sampler and
quantify agreement with experimental SAXS data, reproducing the two headline
validations of the STARLING paper (Novak, Lotthammer, Emenecker & Holehouse,
bioRxiv 2025.02.14.638373):

| benchmark | data | metric | paper value |
|---|---|---|---|
| **A. Rg benchmark** | 133 IDR sequences (ALBATROSS calibration set, <384 aa) | RMSE, R² of ⟨Rg⟩ vs SAXS Rg | RMSE ≈ 4.72 Å, R² ≈ 0.9 |
| **B. Curve benchmark** | 53 proteins with high-quality SAXS curves | χ²ᵣ after scale alignment | "excellent agreement" for the vast majority |

All experimental data comes from the authors' own supporting-data repository
(`holehouse-lab/supportingdata/2026/starling_2026`), so **no SASBDB hunting is
needed** — `download_data.py` fetches exactly the files the authors used.

Directory layout after this stage:

```
stage1_saxs/
├── download_data.py         # step 0
├── generate_ensembles.py    # step 2
├── saxs_debye.py            # step 3 (single curve)
├── batch_curves.py          # step 3 (all curves)
├── rg_benchmark.py          # step 4 (metric A)
├── curve_benchmark.py       # step 4 (metric B)
├── data/                    # downloaded experimental data
├── runs/                    # generated ensembles (distance maps + Rg)
├── curves/                  # back-calculated SAXS curves
└── results/                 # metrics + figures
```

---

## Step 0 — Environment

```bash
# STARLING itself (as before)
git clone https://github.com/idptools/starling.git
cd starling && pip install . && cd ..

# only numpy/scipy/matplotlib/requests are needed beyond starling
pip install requests
```

The first STARLING run downloads the VAE + DDPM weights automatically
(one-time, cached in `~/.cache/torch/hub/`).

## Step 1 — Download the experimental data

```bash
cd stage1_saxs
python download_data.py --out data
```

This fetches:

* `data/saxs_rg/all_comparison_data.csv` + `all_comparison_seqs.fasta`
  (137 sequences; we filter to ≤384 aa → the paper's 133)
* `data/saxs_scattering/experiment/sequences.fasta` (53 proteins)
* `data/saxs_scattering/experiment/mff_analysis_all.csv`
  (experimental Rg from the Molecular-Form-Factor re-analysis of each curve)
* `data/saxs_scattering/experiment/<name>/<name>[ _clean].dat`
  (experimental scattering curves; `_clean.dat` preferred, matching the
  authors' convention)
* `data/saxs_scattering/ensembles/<name>/average_curve.dat`
  (the **paper's own reference STARLING→FoXS curves** — used as a replication
  anchor in step 4b before you generate anything)

### Step 1b — Zero-generation replication check (do this first!)

Before running any sampling, verify the metric machinery reproduces the
paper's agreement using *their* reference curves:

```bash
python curve_benchmark.py \
    --exp-dir data/saxs_scattering/experiment \
    --ref-dir data/saxs_scattering/ensembles \
    --seq-fasta data/saxs_scattering/experiment/sequences.fasta \
    --out results/reference_check
```

Expected: low χ²ᵣ for most proteins (e.g. `ash1` gives χ²ᵣ ≈ 0.66 over
q ≤ 0.15 Å⁻¹). The summary line `ref: median chi2_r=...` is the paper's own
agreement level — this is the number your generated ensembles must match.
**If this check fails, stop: the data download or metric is broken, not
STARLING.**

## Step 2 — Generate stock STARLING ensembles

Curve benchmark (53 proteins, 400 conformers — the paper's FoXS setting):

```bash
python generate_ensembles.py \
    --fasta data/saxs_scattering/experiment/sequences.fasta \
    --out runs/stage1_scattering \
    --conformations 400 --sampler ddpm
```

Rg benchmark (133 sequences, 600 conformers — the paper's setting
`starling ... -c 600`):

```bash
python generate_ensembles.py \
    --fasta data/saxs_rg/all_comparison_seqs.fasta \
    --out runs/stage1_rg \
    --conformations 600 --sampler ddpm --max-len 384
```

Notes:

* `--sampler ddpm` is deliberate: the Stage-2 controller modifies the DDPM
  reverse SDE, so the Stage-1 baseline must be the stock DDPM. DDPM runs the
  **full** trained Markov chain (no step subsampling), so it is much slower
  than the default DDIM/30-steps. For a quick pipeline smoke test use
  `--sampler ddim --conformations 100 --only ash1,p53,sic1`.
* Runtime: GPU → seconds/minutes per sequence (DDIM) to ~minutes (DDPM);
  CPU → feasible but slow. The script is resumable (skips finished sequences).
* Outputs per sequence: `runs/.../<name>.starling`,
  `runs/.../distance_maps/<name>.npy`, `runs/.../rg/<name>.npy`.

## Step 3 — Back-calculate SAXS curves from the ensembles

We compute the ensemble-averaged Debye scattering **directly from the distance
maps** (no 3D reconstruction, no FoXS installation):

```bash
python batch_curves.py \
    --seq-fasta data/saxs_scattering/experiment/sequences.fasta \
    --dm-dir runs/stage1_scattering/distance_maps \
    --out curves/stage1_scattering
```

The calculator (`saxs_debye.py`) uses vacuum Cromer–Mann residue form factors
and a pooled pair-distance histogram — exact for the ensemble Debye formula.
The paper used FoXS on MDS-reconstructed Cα traces instead; the two forward
models differ mainly at high q (hydration shell). This does **not** affect the
Stage-2 comparison, where the same calculator is applied to stock and
controlled ensembles. (If you want the paper-exact route: generate with
`return_structures`, run `starling2xtc`, then FoXS via IMP
(`conda install -c salilab imp`), 400 conformers, default c1=1.0/c2=2.0.)

## Step 4 — Compute the metrics

### A. Rg benchmark (headline number)

```bash
python rg_benchmark.py --mode comparison \
    --comparison-csv data/saxs_rg/all_comparison_data.csv \
    --rg-dir runs/stage1_rg/rg \
    --out results/rg_133
```

→ prints RMSE / R² / bias and writes `results/rg_133_rg_comparison.pdf` +
`.csv`. **Target: RMSE ≈ 4.7 Å, R² ≈ 0.9** (preprint values with the UNet
weights; the current ViT weights may shift these slightly — that is expected
and is itself the replication result).

Also run the 53-protein MFF Rg check:

```bash
python rg_benchmark.py --mode mff \
    --mff-csv data/saxs_scattering/experiment/mff_analysis_all.csv \
    --rg-dir runs/stage1_scattering/rg \
    --out results/rg_mff
```

### B. Curve benchmark

```bash
python curve_benchmark.py \
    --exp-dir data/saxs_scattering/experiment \
    --our-curves curves/stage1_scattering \
    --ref-dir data/saxs_scattering/ensembles \
    --seq-fasta data/saxs_scattering/experiment/sequences.fasta \
    --out results/curve_benchmark
```

Protocol (same as the paper's `housetools` fit): scale-align simulated →
experimental over q ∈ [0.01, 0.25] Å⁻¹ (weighted least squares), then
χ²ᵣ over q ≤ 0.15 Å⁻¹. Outputs:

* `curve_metrics.csv` — per-protein χ²ᵣ for our curve vs experiment **and**
  for the paper's reference curve vs experiment, plus a log-space RMSD between
  our curve and the reference curve (forward-model sanity check)
* `overlays/<name>.pdf` — experiment (black) vs reference (blue) vs ours (red)

## What "Stage 1 replicated" means

1. Step 1b: reference curves reproduce the paper-level χ²ᵣ (metric OK).
2. Step 4A: your stock-DDPM Rg RMSE/R² lands near 4.7 Å / 0.9 (within
   sampling noise and model-version drift).
3. Step 4B: per-protein χ²ᵣ distribution of *your* stock-DDPM curves is
   comparable to the reference column in `curve_metrics.csv`.

Once these hold, Stage 2 replaces the sampler with `ControlledDDPMSampler`
(one-line change in `generate_ensembles.py` via the `controlled_sampler`
module) and re-runs steps 2–4 — identical data, identical metric, only the
reverse-SDE control changes.
