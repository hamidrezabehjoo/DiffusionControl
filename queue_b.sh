#!/bin/bash
# queue_b.sh — core-B chain after boost s2 finishes:
# damp l16 seeds 1,2 (fast, early stop) -> lam clouds -> reduced seeds 0-2
cd /mnt/agents/output/code || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

for S in 1 2; do
  python3 run_picard.py 0.1 16 14 "damp_l16_s${S}" save_clouds "${S}" \
    > "res_damp_s${S}.log" 2>&1
done

python3 run_lam_clouds.py > res_lam_clouds.log 2>&1

for S in 0 1 2; do
  python3 run_reduced.py "${S}" > "res_reduced_s${S}.log" 2>&1
done

echo ALL_DONE > res_queue_b.done
