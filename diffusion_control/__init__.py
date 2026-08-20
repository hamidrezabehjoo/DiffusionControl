"""diffusion_control: inference-time noise control for diffusion models.

Synthetic Gaussian / Gaussian-mixture testbed, exact K=1 solution,
particle fixed-point solver, and metrics, as described in
"Optimal Inference-Time Noise Control for Diffusion Models".
"""
from . import gmm_control, particle_solver, exact_k1, metrics

__all__ = ["gmm_control", "particle_solver", "exact_k1", "metrics"]
