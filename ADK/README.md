# Controlled Boltz-2 for ADK basin escape

PMP inference-time noise control applied to the Boltz-2 diffusion sampler,
for the adenylate kinase (ADK) open/closed conformational experiment.

## Contents

| File | What it is |
|---|---|
| `diffusionv2.py` | Drop-in replacement for Boltz 2.2.1's `boltz/model/modules/diffusionv2.py`, carrying the two-line controller modification (env-gated; stock behavior by default) |
| `adk.yaml` | Boltz input: E. coli ADK, 214 aa (sequence identical in 1AKE and 4AKE) |
| `run_adk_experiment.sh` | End-to-end driver: stock run + controlled run + evaluation |
| `eval_adk.py` | CA-RMSD and TM-score of every predicted model vs 1AKE (closed) and 4AKE (open), basin assignment, CSV + summary |
| `data/1AKE.pdb` | Closed-state reference (holo, Ap5A-bound) |
| `data/4AKE.pdb` | Open-state reference (apo) |

## Setup

```bash
pip install "boltz[cuda]==2.2.1"        # GPU environment
BOLTZ_PKG=$(python -c "import boltz, os; print(os.path.dirname(boltz.__file__))")
cp diffusionv2.py "$BOLTZ_PKG/model/modules/diffusionv2.py"
```

Model weights + CCD data (~8 GB) download automatically to the cache dir
(default `~/.boltz`, override with the `$BOLTZ_CACHE` env var or `--cache`).
If `$HOME` has a disk quota, point the cache at a bigger filesystem first:

```bash
export BOLTZ_CACHE=/scratch/$USER/boltz_cache   # needs ~8 GB free
```

**After an interrupted download**, delete the partial checkpoint before
rerunning — Boltz only checks file existence, so a truncated
`boltz2_conf.ckpt` would be picked up as if complete:

```bash
rm -f "$BOLTZ_CACHE"/boltz2_conf.ckpt ~/.boltz/boltz2_conf.ckpt
```

**`OSError: [Errno 122] Disk quota exceeded` during prediction** (not during
download) usually comes from Triton's JIT kernel cache at `~/.triton`, not
from Boltz. The run script redirects it together with the other caches:

```bash
export BIGCACHE=/scratch/$USER/caches   # one root for all caches
sh run_adk_experiment.sh
```

**Warning:** `BIGCACHE` must be on a filesystem outside your home quota.
`~/scratch` is just a subfolder of home and does NOT help (unless
`readlink -f ~/scratch` shows it is a symlink to another filesystem).
Find real bulk storage with `df -h ~ /scratch /work /data /lustre 2>/dev/null`
and pick a path with enough free space; the script preflights this and stops
early with a clear message. The output dir (`results_*/`) is written next to
`adk.yaml`, so run from a location with space too.

If it still fails, check what is eating the home quota:
`du -sh ~/.triton ~/.cache ~/.boltz ~/.nv 2>/dev/null` and clean up.

## Run

```bash
bash run_adk_experiment.sh
```

or manually:

```bash
# Stock baseline (NU_ALPHA unset or 1.0 = exact stock sampler)
boltz predict adk.yaml --use_msa_server --use_potentials \
    --diffusion_samples 100 --sampling_steps 200 \
    --output_format pdb --seed 42 --out_dir results_stock --override

# Controlled: nu(t) = 2.5 * g(t) on the late-reverse window [0, 0.2]
NU_ALPHA=2.5 NU_TW=0.2 \
boltz predict adk.yaml --use_msa_server --use_potentials \
    --diffusion_samples 20 --sampling_steps 200 \
    --output_format pdb --seed 42 --out_dir results_ctrl --override

# Evaluate
python3 eval_adk.py --pred_dir results_stock/predictions/adk --tag stock --out eval_stock.csv
python3 eval_adk.py --pred_dir results_ctrl/predictions/adk  --tag ctrl  --out eval_ctrl.csv
```

After the first run you can drop `--use_msa_server` and point the YAML at the
cached MSA (`msa: <path>.a3m`) to avoid repeated server queries.

## What the patch does

Boltz-2's sampler is EDM-style: each step optionally adds churn noise
(`gamma` schedule, active only above `sigma = 1 A`) and then takes an
Euler denoising step. Two facts matter here:

1. With the default 200-step schedule, **churn is off in the entire late
   window** `t in [0, 0.2]` (steps 160-199, `sigma < 1 A`) — the stock
   sampler is fully deterministic there.
2. The controller `nu(t) = alpha * g(t)` therefore uses as the per-step
   reference `g^2` the stock churn variance where churn exists, and the
   variance-exploding increment `sigma_tm^2 - sigma_t^2` (the EDM analog
   of DDPM's `beta_t`) where the stock sampler is deterministic.

On in-window steps (normalized forward time `t = 1 - step/N <= NU_TW`,
final landing step excluded), the patch changes exactly one thing:

```
injected noise std :  g   ->   nu = alpha * g
```

The injected noise raises the sample to level `t_hat_c = sqrt(sigma_tm^2 +
nu^2)`, the denoiser is queried at that true level, and the standard
(unscaled) Euler descent then traverses the full gap `t_hat_c -> sigma_t`.
Because the descent removes exactly the injected noise, the marginal noise
schedule is preserved by construction: control only widens the exploratory
detour, it never leaves the trajectory off-schedule.

The DDPM form of the controller also carries a score-coefficient correction
`g^2 -> (g^2 + nu^2)/2`. That term exists to compensate injected noise
inside fine-grained Euler-Maruyama steps; inside Boltz's churn+full-descent
splitting it double-counts the compensation and overshoots the schedule
(empirically: alpha=3, t_w=0.3 with the drift factor unfolded all 100 ADK
models, mean RMSD ~25 A). It is therefore OFF by default; `NU_DRIFT=1`
restores it for comparison.

`NU_ALPHA=1` skips the block entirely and the sampler is bit-identical to
stock Boltz-2, so stock baselines can be run from the same patched
installation.

Physical-potential guidance (`--use_potentials`) is untouched and composes
with the control, since it acts on the denoised prediction downstream.

## Output interpretation

`eval_adk.py` assigns each model to a basin by nearest-reference RMSD with
a 3.5 A threshold (`--escape_rmsd`). The quantities for the paper:

- **Stock**: fraction of models in `open` / `closed` /
  `intermediate/other` — the collapse statistic (previously: 100/100
  collapsed to an intermediate, RMSD ~4 A / TM ~0.57 to both states).
- **Controlled**: escape fraction = models landing in a native basin.
  Report as "k/N controlled trajectories escaped, vs 0/100 stock".

Model files are confidence-ranked (`_model_0` = top confidence); all ranks
are evaluated.
