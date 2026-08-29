#!/bin/sh
cd /mnt/agents/output/DiffusionControl
R=results
python3 experiments/run_synthetic.py --dim 128 --seed 0 > $R/d128_seed0.log 2>&1
python3 experiments/run_synthetic.py --dim 32 --stages k8kernel,boost,damp,reduced,metrics > $R/d32_seed0.log 2>&1
python3 experiments/run_synthetic.py --dim 64 --stages k8kernel,boost,damp,reduced,metrics > $R/d64_seed0.log 2>&1
python3 experiments/run_synthetic.py --dim 128 --stages damp_boundary > $R/d128_damp_boundary.log 2>&1
python3 experiments/run_synthetic.py --dim 128 --seed 1 --stages k1kernel > $R/d128_seed1_k1.log 2>&1
python3 experiments/run_synthetic.py --dim 128 --seed 2 --stages k1kernel > $R/d128_seed2_k1.log 2>&1
echo ALLDONE > $R/all_done.txt
