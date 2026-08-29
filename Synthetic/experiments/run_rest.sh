#!/bin/sh
cd /mnt/agents/output/DiffusionControl
R=results
# reduced cloud at the user's grid optimum, then metrics
python3 - <<'PY'
import numpy as np, sys
sys.path.insert(0, ".")
from diffusion_control import gmm_control as gc
from diffusion_control import particle_solver as ps
D, K, SEED = 128, 8, 0
MU = gc.make_modes(D, K); wts = gc.make_weights(K)
tg = np.linspace(0, gc.T, 41); g2 = gc.g(tg)**2
alpha, tw = 1.35, 0.3
nu2 = np.where(tg <= tw, alpha**2 * g2, g2)
X, _ = ps.forward_particles(nu2, MU, wts, -0.1, N=20_000, M=40, Nf=800,
                            seed=SEED, store_slices=False)
np.save("results/d128/seed0/X0_reduced.npy", X)
print("reduced cloud saved")
PY
python3 experiments/run_synthetic.py --dim 128 --stages metrics > $R/d128_metrics.log 2>&1
# k1 kernel error-bar seeds
python3 experiments/run_synthetic.py --dim 128 --seed 1 --stages k1kernel > $R/d128_seed1_k1.log 2>&1
python3 experiments/run_synthetic.py --dim 128 --seed 2 --stages k1kernel > $R/d128_seed2_k1.log 2>&1
# dimension suites
python3 experiments/run_synthetic.py --dim 32 --stages k8kernel,boost,damp,reduced,metrics > $R/d32_seed0.log 2>&1
python3 experiments/run_synthetic.py --dim 64 --stages k8kernel,boost,damp,reduced,metrics > $R/d64_seed0.log 2>&1
python3 experiments/run_synthetic.py --dim 128 --stages damp_boundary > $R/d128_damp_boundary.log 2>&1
python3 experiments/make_figures.py > $R/figures.log 2>&1
echo ALLDONE > $R/all_done.txt
