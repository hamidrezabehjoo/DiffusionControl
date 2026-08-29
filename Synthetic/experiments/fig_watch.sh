#!/bin/sh
cd /mnt/agents/output/DiffusionControl
R=results
# regenerate as soon as the d=128 damp fixed point exists
while [ ! -f $R/d128/seed0/X0_damp_star.npy ]; do sleep 120; done
python3 experiments/make_figures.py > $R/figures.log 2>&1
# and again when the whole queue (incl. d32/d64) is done
while [ ! -f $R/all_done.txt ]; do sleep 180; done
python3 experiments/make_figures.py >> $R/figures.log 2>&1
echo FIGS_DONE > $R/figs_done.txt
