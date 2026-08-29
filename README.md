# Optimal Inference-Time Noise Control for Diffusion Models

Code for the paper *Optimal Inference-Time Noise Control for Diffusion Models*
(Behjoo & Zhang).

The paper formulates the inference-time diffusion coefficient `nu(t)` of a
pretrained diffusion model as an optimal control variable for terminal
entropy, derives its Wasserstein Pontryagin characterization, and shows the
optimal controller reduces to a **two-line change of any DDPM sampler**:

1. score coefficient `beta_t` &rarr; `(beta_t + nu_t^2) / 2`
2. injected noise std `sqrt(beta_t)` &rarr; `nu_t`

with the reduced two-parameter family `nu(t) = alpha * g(t)` on the
low-noise window `[0, t_w]`. The theory predicts where control acts (late
reverse time), in which direction (sign of the score's variance mismatch),
and that it is inert when the score is exact.

## Repository layout

| Folder | Contents | Paper section |
|---|---|---|
| `Synthetic/` | K-GMM testbed, exact K=1 ground truth (Riccati), particle fixed-point solver, K-independence and dimension-scaling suites, all paper figures | Synthetic verification of the exact theory |
| `ALDP/` | Alanine dipeptide null-control experiment (Prediction 1: near-exact score &rArr; controller inactive) | Protein Applications, Prediction 1 |
| `FastFolding/` | Arts et al. fast-folding benchmark with the DFF sampler: Chignolin temporal ablation and per-system gains (Predictions 2, 3) | Protein Applications, Predictions 2–3 |
| `ADK/` | Boltz-2 port of the controller; adenylate kinase open/closed basin redistribution, dose–response, and bit-identical no-churn null control (Prediction 4) | Protein Applications, Prediction 4 |
| `STARLING/` | Latent-diffusion IDP ensembles: SAXS benchmark replication, dose response, bidirectional per-protein control, sign-law diagnosis (Prediction 5) | Protein Applications, Prediction 5 |

Each folder has its own README with exact run instructions.

## Controller interface

All protein experiments use the same two knobs, exposed as environment
variables in the patched samplers:

| Variable | Meaning | Paper symbol |
|---|---|---|
| `NU_ALPHA` | noise gain on the control window (1.0 = stock) | `alpha` |
| `NU_TW` | late-reverse window `[0, t_w]` in normalized time | `t_w` |

Out-of-window steps are bit-identical to the stock sampler, and paired
stock/controlled runs consume identical Gaussian draws (common random
numbers).

## Citation

If you use this code, please cite the paper (citation block to be added on
publication).
