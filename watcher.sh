#!/bin/bash
# watcher.sh — chain follow-up jobs when the boost seed runs finish.
cd /mnt/agents/output/code || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# core A: when boost s1 (PID $1) exits -> K=1 validation
(
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
  python3 run_k1_validation.py > res_k1.log 2>&1
  echo K1_DONE > res_k1.done
) &

# core B: when boost s2 (PID $2) exits -> damp seeds + lam clouds + reduced
(
  while kill -0 "$2" 2>/dev/null; do sleep 60; done
  bash queue_b.sh > res_queue_b.log 2>&1
) &

wait
echo WATCHER_DONE > res_watcher.done
