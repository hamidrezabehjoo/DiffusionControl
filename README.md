# DiffusionControl

Code for *Inference-time noise control of diffusion models* (Behjoo et al.).

The paper formulates the inference-time diffusion coefficient `nu(t)` of a
pretrained diffusion model as an optimal control variable for terminal
entropy, derives its Pontryagin characterization, and shows the optimal
controller reduces to a **two-line change of any DDPM sampler**:

1. score coefficient `beta_t` → `(beta_t + nu_t^2) / 2`
2. injected noise std `sqrt(beta_t)` → `nu_t`

with the reduced two-parameter family `nu(t) = alpha * g(t)` on the
low-noise window `[0, t_w]`.

## Layout

```
diffusion_control/        the package
  gmm_control.py          K-GMM synthetic testbed (model, VP schedule,
                          exact mixture scores/densities)
  particle_solver.py      forward particle solver, costate regression,
                          kernel estimator (two-fold cross-fitting),
                          Picard fixed-point iteration
  exact_k1.py             exact one-component ground truth (Riccati costate,
                          exact kernel and fixed point)
  metrics.py              terminal entropy, objective, sliced W2, occupancy
experiments/
  run_synthetic.py        unified checkpointed driver; all headline numbers
                          (default: d = 128, seed 0)
  run_k_independence.py   kernel Lipschitz constant vs number of modes K
  run_k2_surrogate.py     two-mode surrogate-kernel sign study (SI)
  make_figures.py         builds every paper figure into ../paper/figs/
results/                  all run outputs (checkpointed, resumable)
```

## Install

```bash
pip install numpy scipy matplotlib   # that's it
```

## Reproduce the paper's synthetic results

Headline suite (K=8 mixture, d=128, seed 0; boost + damp + reduced
controller + metrics). Stages checkpoint after every Picard iteration and
resume automatically if interrupted:

```bash
python3 experiments/run_synthetic.py --dim 128 --seed 0
```

Individual stages:

```bash
python3 experiments/run_synthetic.py --dim 128 --stages k8kernel
python3 experiments/run_synthetic.py --dim 128 --stages boost
python3 experiments/run_synthetic.py --dim 128 --stages damp
python3 experiments/run_synthetic.py --dim 128 --stages damp_boundary
python3 experiments/run_synthetic.py --dim 128 --stages reduced,metrics
```

Supporting studies:

```bash
# dimension robustness (SI app:dimensions), ~same cost per dim
python3 experiments/run_synthetic.py --dim 32 --stages k8kernel,boost,damp,reduced,metrics
python3 experiments/run_synthetic.py --dim 64 --stages k8kernel,boost,damp,reduced,metrics

# K-independence of the kernel Lipschitz constant (Thm kgmm, d=32)
python3 experiments/run_k_independence.py

# two-mode surrogate-kernel sign study (SI app:exact, fig:particle-kernel)
python3 experiments/run_k2_surrogate.py

# one-component exact validation, seeds for error bars (SI app:k1validation)
python3 experiments/run_synthetic.py --dim 128 --seed 1 --stages k1kernel
python3 experiments/run_synthetic.py --dim 128 --seed 2 --stages k1kernel
```

Figures (after the runs above):

```bash
python3 experiments/make_figures.py
```

## Notes

- All heavy linear algebra is matmul-based (no `(N, K, d)` broadcast
  temporaries), so d=128 with N=2×10^4 particles runs on a CPU workstation.
- Common random numbers are used across compared runs (same CRN seed), and
  the kernel estimator uses two-fold cross-fitting with a
  mixture-of-quadratures costate regression — see
  `particle_solver.kernel_from_particles`.
- The reduced controller applied to a real protein latent-diffusion model
  (STARLING) lives in the companion folder `starling_control/`.
