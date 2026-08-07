# Reproduction code: Section 4.4 (Synthetic verification of the exact theory)

Particle fixed-point solver for inference-time noise control of diffusion
models, applied to the K-mode Gaussian-mixture testbed of the paper
(d = 32). Pure NumPy; Python 3.10+.

## Files

- `gmm_control.py` — testbed definition: VP schedule (beta linear on
  [0.1, 20], T = 1), nested K-mode GMM family (centers Uniform[-4,4]^d,
  seed 0; period-8 weights renormalized), frozen score with per-mode
  variance offset delta, controlled reverse-SDE drift, entropy of p0,
  quadratic running cost.
- `particle_solver.py` — the particle fixed-point algorithm (SI Appendix,
  Algorithm 1): Euler--Maruyama forward passes with common random numbers,
  diagonal-GMM controlled score (centers fixed at m(t) mu_k, EM over
  weights/variances), mixture-of-quadratics costate regression
  (responsibility-weighted least squares), two-fold cross-fitted kernel
  Monte Carlo, projected stationarity update, Picard iteration.
- `metrics.py` — entropy gap (MC entropy of p0 vs. refitted controlled-score
  mixture), sliced W2 (256 random projections, exact 1-D W2), objective J.
- `run_picard.py DELTA LAMBDA N_ITER TAG [save_clouds] [SEED]` — one
  fixed-point run. The boost (lambda = 0.8) and interior damp (lambda = 16)
  runs are repeated with CRN seeds 1 and 2 for the mean +/- sd error bars
  of Table 2. Reported runs:
  - boost scenario: `python3 run_picard.py -0.1 0.8 12 boost_l0.8 save_clouds`
    (also lambda = 0.4, 1.6 for the rate law; no clouds needed there)
  - damping scenario: `python3 run_picard.py 0.1 0.8 12 damp_l0.8 save_clouds`
    (runs to the lower admissible boundary) and
    `python3 run_picard.py 0.1 16 14 damp_l16 save_clouds` (interior solution)
- `run_ck.py K1,K2,... TAG [N_EM] [SEED]` — kernel Lipschitz constant
  C_hat(K) between nu1 = g and nu2 = 1.05 g under common random numbers
  (`python3 run_ck.py 2,4,8 a` and `python3 run_ck.py 16,32 b`).
  K = 32 is computed by `run_ck_k32_seed1.py` (CRN seed 1; the seed-0 cloud
  produces a deterministic single-cell costate-fit artifact at t = 0.125,
  documented in that file). `merge_ck.py` assembles res/ck_final.npz.
- `run_metrics.py` — performance table (both scenarios, common random
  numbers), writes `res/table_d32.json`.
- `exact_k1.py` — exact K = 1 Gaussian ground truth: Riccati variance ODE,
  quadratic-costate ODE, closed-form kernel S(t) = d q (w - v)/w, exact
  projected Picard fixed point (SI Appendix, Section "Exact one-component
  ground truth").
- `run_k1_validation.py` — validates the particle solver against the exact
  K = 1 solution: first-iterate kernel sign/magnitude for delta = -0.1,
  0, +0.1 over three seeds (the delta = 0 run is the exact-score
  triviality check), particle vs. exact boost fixed point, exact damping
  fixed points at lambda = 0.8 (boundary) and 16 (interior). Writes
  `res/k1_validation.npz`.
- `run_reduced.py SEED` — grid search of the reduced controller
  nu = alpha g on [0, t_w] (alpha in {1.2,...,1.65}, t_w in {0.1,...,0.45})
  on the objective J under common random numbers; seed 0 selects the best
  configuration, seeds > 0 re-evaluate it. Writes `res/reduced_s0.npz` and
  `res/X0_reduced_sSEED.npy`.
- `run_lam_clouds.py` — terminal clouds of the converged boost schedules at
  lambda = 0.4 and 1.6 (for the SI penalty-sensitivity table).
- `run_metrics2.py` — multi-seed metrics (3 independent solver seeds):
  mean +/- sd of H(rho0), entropy gap, sliced W2, objective J for both
  scenarios; reduced-controller gain; per-mode terminal occupancies;
  penalty-sensitivity rows. Writes `res/table2_multi.json`.
- `make_figs.py` — figures `fig_convergence.png`, `fig_schedules.png`,
  `fig_k_independence.png` (300 DPI, written to `../figs/`).
- `make_fig_k1.py` — SI figure `fig_k1_validation.png` (first-iterate
  kernels vs. exact Riccati kernel; particle vs. exact fixed point).
- `fill_and_edit_v2.py`, `fill_and_edit_v3.py` — splice the computed
  numbers into the manuscript (not needed to reproduce the experiments).

## Conventions

- Reverse time t in [0, T]: rho_T = N(0, I), rho_0 = generated law.
- Schedule nu^2 piecewise constant on a uniform grid of M = 40 cells;
  N = 2 x 10^4 particles; 800 fine steps per forward pass.
- Common random numbers: a fixed seed makes the prior and Brownian draws
  identical across schedules, so schedule comparisons are noise-free.
- Admissible set nu in [0.05, 8]; the projection is inactive in the boost
  scenario and active (lower bound) in the damping scenario at lambda = 0.8.
