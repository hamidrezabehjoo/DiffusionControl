# Applying the reduced noise controller to STARLING

This folder contains a drop-in application of the reduced two-parameter
noise controller from *Inference-time noise control of diffusion models*
(Behjoo et al.) to [STARLING](https://github.com/idptools/starling), the
VAE + latent-DDPM ensemble generator for intrinsically disordered
proteins. No retraining, no changes to STARLING's models — the control
is a two-line modification of the reverse DDPM step in latent space
(SI Appendix, app:ddpm of the paper):

1. score coefficient `beta_t` → `(beta_t + nu_t^2) / 2`
2. injected noise std → `nu_t`, with `nu_t = alpha * sqrt(beta_t)` on
   the control window `[0, t_w]`

`t` is forward (noising) time, so the window `[0, t_w]` is the
**late-reverse / low-noise** part of sampling — the last `t_w` fraction
of denoising steps. This is where the theory (kernel envelope
`1/Sigma_t^4`) and the protein experiments both localize productive
control action.

## Files

| file | purpose |
|---|---|
| `controlled_sampler.py` | `ControlledDDPMSampler` — subclasses STARLING's `DDPMSampler`, overriding only `p_sample`. Out-of-window steps are bit-identical to the stock sampler. |
| `generate_controlled.py` | CLI: load STARLING's pretrained VAE + DDPM, generate controlled (and optionally baseline) ensembles, save distance maps as `.npy`. |
| `compare_ensembles.py` | Diversity comparison (Rg distribution, pairwise distance-map RMSD) between baseline and controlled ensembles. |
| `stage1_saxs/` | **Stage 1 SAXS replication pipeline** (experimental data download, stock ensemble generation, Debye SAXS back-calculation, Rg + χ² metrics). See `stage1_saxs/STAGE1_INSTRUCTIONS.md`. |
| `stage1_saxs/generate_stage2.py` | **Stage 2**: ensemble generation with `ControlledDDPMSampler` (paired CRN stock arm optional). See `stage1_saxs/STAGE2_INSTRUCTIONS.md`. |
| `stage1_saxs/compare_stage2.py` | Controlled-vs-stock analysis: per-protein Δχ²ᵣ, Rg RMSE shift, sign/Wilcoxon tests. |

## Install (on your local machine)

```bash
git clone https://github.com/idptools/starling
cd starling
pip install .          # installs STARLING and its dependencies
# copy this folder next to your clone, or anywhere on your PYTHONPATH
```

Model weights are downloaded automatically on first use (STARLING's
`ModelManager` pulls them into your torch hub cache).

## Run

```bash
# Controlled ensemble, validated protein defaults (alpha=2.5, t_w=0.2):
python generate_controlled.py --sequence "MEEPQSDPSVEPPLSQETFSDLWKLLPEN" \
    --conformations 400 --alpha 2.5 --window 0.2 --output runs/p53

# Controlled + baseline side by side (for a FASTA of many sequences):
python generate_controlled.py --fasta my_idps.fasta --conformations 400 \
    --alpha 2.5 --window 0.2 --baseline --output runs/benchmark

# Compare:
python compare_ensembles.py runs/p53/sequence_1_STARLING_DM_baseline.npy \
                            runs/p53/sequence_1_STARLING_DM_alpha2.5_tw0.2.npy
```

On a GPU this takes minutes for a few hundred conformations; on CPU it
works but is slow — use `--device cuda` where possible and tune
`--batch-size` to your VRAM.

## Choosing (alpha, t_w)

- The paper's protein experiments grid-searched (alpha, t_w) once on a
  single validation system (Chignolin) and transferred the result
  **zero-shot** to all other systems: **alpha = 2.5, t_w = 0.2** for the
  over-sharp-score regime (variance underestimated), which is the regime
  of all protein systems studied. These are the defaults here.
- If your target behaves differently (e.g. ensembles that are too
  heterogeneous), you may be in the over-smoothed regime: use
  `0 < alpha < 1` (damping).
- A small grid over `alpha in {1.5, 2.5, 4}` x `window in {0.1, 0.2, 0.4}`
  on one sequence, scored by `compare_ensembles.py`, is a cheap way to
  confirm the regime for your system before large runs.
- `alpha = 1` gives the continuous-time reference sampler `nu = g`
  (noise `sqrt(beta_t)`); note STARLING's stock sampler uses the exact
  posterior variance `beta_tilde_t`, which agrees to first order in
  `beta_t`. For a strict baseline, use `--baseline` (stock sampler).

## Using a schedule from the exact solver

If you run the paper's exact fixed-point solver
(`DiffusionControl/experiments/run_synthetic.py`, stage `reduced` or the
Picard stages) and export a per-timestep `nu*` schedule with
`T = starling` timesteps (e.g. `np.save("nu_star.npy", nu_star)`), you
can plug it in directly — it overrides `alpha`/`window`:

```bash
python generate_controlled.py --sequence "..." --conformations 400 \
    --nu-schedule nu_star.npy --output runs/exact
```

## How it maps onto STARLING

STARLING diffuses in the VAE latent space: latents of shape
`(N, 1, 24, 24)` denoised over `T` steps, then decoded to C-alpha
distance maps. The controller modifies only the latent reverse steps
inside the window; the VAE encoder/decoder, the sequence conditioning
(PLM embeddings + ionic strength), and the optional constraint
machinery (`RgConstraint`, `DistanceConstraint`, `HelicityConstraint`,
which compose with this sampler unchanged) are all untouched.
