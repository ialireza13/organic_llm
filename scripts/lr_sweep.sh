#!/bin/bash
# LR sweep for the from-scratch uniform S baseline (seed 0), 4 runs concurrently.
cd ~/organic-growth && source .venv/bin/activate
for lr in 1e-3 2e-3 4e-3 8e-3; do
  python train.py --scale S --arm scratch --seed 0 --lr $lr --run_name S_lrsweep_$lr --results_dir results/lrsweep > runs/S_lrsweep_$lr.log 2>&1 &
done
wait
echo SWEEP_DONE
