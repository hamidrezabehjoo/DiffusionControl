# Dimension-scaling runs (d = 64 and d = 128)

This package runs the Section 4.4 protocol of the paper ("Synthetic
verification of the exact theory") at ambient dimensions d = 64 and d = 128
on the same K = 8 Gaussian-mixture testbed design (weights
pi = (0.20, 0.20, 0.15, 0.15, 0.10, 0.10, 0.07, 0.03), sigma0 = 1, VP
schedule beta linear on [0.1, 20], T = 1, admissible nu in [0.05, 8]).
Centers are drawn uniformly in [-4, 4]^d with seed 0 in the respective
dimension (they cannot be shared across dimensions).

## Requirements

- Python 3.10+ and NumPy (>= 2.0 preferred; older NumPy works through a
  compatibility shim in `run_dimension.py`). Nothing else is required.
- Memory: under 1 GB per run at d = 128 (one stored trajectory ensemble is
  about 420 MB).
- No GPU needed; the solver is pure NumPy and uses one core per run.

## What is computed

`run_dimension.py D` executes five stages, writing `res_d{D}/`:

1. `k1kernel` — first-iterate kernel at nu = g for the one-component model,
   delta = -0.1 / 0 / +0.1, against the exact Riccati kernel. Because the
   exact kernel is S(t) = d * q(t) * (w(t) - v(t)) / w(t), the rescaled
   kernel S/d is dimension-free: this stage checks that the particle
   estimator collapses onto the same S/d curve at every d.
2. `k8kernel` — first-iterate kernel on the K = 8 testbed (delta = -/+ 0.1).
3. `boost` — Picard fixed point, delta = -0.1, 12 iterations, with a
   checkpoint saved after every iteration.
4. `damp` — Picard fixed point, delta = +0.1 (converges in ~2-3 iterations).
5. `metrics` — H(p0), entropy gap, sliced W2, objective J, mode occupancies
   for the uncontrolled and controlled terminal clouds, and the support
   radius R0 = max_k ||mu_k|| of the drawn centers.

Penalty scaling (important): the entropy kernel scales linearly with d, so
the dimension-appropriate penalties are

    lambda_boost = 0.8 * (d/32)   ->  1.6 (d=64),  3.2 (d=128)
    lambda_damp  = 16  * (d/32)   -> 32.0 (d=64), 64.0 (d=128)

which the script sets automatically. (At the unscaled d = 32 penalty the
boost update would be d/32 times stronger and would press against the
upper admissible boundary.)

## How to run

Everything for one dimension (single core):

    python3 run_dimension.py 64
    python3 run_dimension.py 128

or, to use two cores, run the two dimensions in parallel in separate
terminals. Stages can also be run individually, e.g.

    python3 run_dimension.py 128 boost
    python3 run_dimension.py 128 metrics

Runs are resumable: each stage is skipped if its output file exists, and
the Picard stages checkpoint after every iteration, so an interrupted run
continues where it stopped (just start the same command again). Delete an
output file to force re-computation of that stage.

## Expected wall time (per single core, N = 2 x 10^4 particles)

Measured on a 2-core cloud CPU (one forward pass + kernel evaluation takes
~7 min at d = 64 and ~8-17 min at d = 128); scale by your machine's speed.

| stage              | d = 64     | d = 128    |
|--------------------|-----------|------------|
| k1kernel (3 runs)  | ~5 min    | ~12 min    |
| k8kernel (2 runs)  | ~15 min   | ~25 min    |
| boost (12 iters)   | ~1.5 h    | ~3.5 h     |
| damp (~3 iters)    | ~30 min   | ~1 h       |
| metrics            | ~15 min   | ~30 min    |
| **total**          | **~2.5 h**| **~5.5 h** |

An overnight run on two cores (one dimension per core) completes both.

## Sanity check: expected first-iterate kernels

Your installation should reproduce the following smoke-test numbers
(CRN seed 0) before you start the long runs — run `k8kernel` first and
compare (`res_d{D}/k8_kernel.npz`, first entry of `S_boost` / `S_damp`):

| d   | boost S(0) | boost S(0)/d | damp S(0) | damp S(0)/d |
|-----|-----------|--------------|-----------|-------------|
| 32  | +3.611    | +0.1128      | -3.219    | -0.1006     |
| 64  | +7.411    | +0.1158      | -6.342    | -0.0991     |
| 128 | +14.867   | +0.1162      | -12.588   | -0.0983     |

The rescaled kernel S/d is dimension-free to within ~3% over a four-fold
dimension range, exactly the linear-in-d scaling of the theory; small
positive drift with d comes from the centers being redrawn in each
dimension (larger separation at larger d). The K = 1 particle kernel
should match the exact Riccati kernel to within ~10% (d = 64 check:
+7.413 vs +7.111 for delta = -0.1; -6.351 vs -5.816 for delta = +0.1;
-0.205 vs 0 for delta = 0, the same small systematic fit bias documented
in the SI for d = 32). Reference npz files from our smoke test are in
`expected_smoke/`.

## What to send back

Zip the two results folders:

    zip -r results_dimensions.zip res_d64 res_d128

The folders contain `k1_kernel.npz`, `k8_kernel.npz`, `picard_boost.npz`,
`picard_damp.npz` (schedule/kernel/residual histories), the terminal clouds
`X0_*.npy`, and `metrics.json`. If file size is a concern, the `X0_*.npy`
clouds (about 10 MB each at d = 128) may be omitted after `metrics.json`
has been produced.

## Files

- `run_dimension.py` — the driver (all stages, checkpointing, summaries).
- `gmm_control.py`, `particle_solver.py`, `exact_k1.py`, `metrics.py` —
  the solver modules, identical to the d = 32 experiments of the paper
  (dimension is a parameter; nothing dimension-specific is hard-coded).
