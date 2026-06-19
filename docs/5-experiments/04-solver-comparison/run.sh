#!/bin/bash
# Experiment 04 — solver comparison (convergence half), 1024³.
# {kdk,dkd,bf} × nb-steps {10,18,30,50} = 12 sims. float64. Speed half lives in Exp 11 (--perf).
source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 04 — solver comparison (convergence)  (MODE=$MODE)"

NB_SHELLS="${NB_SHELLS:-10}"   # lightcone shells (box/2 / N ≥ min_width 50 Mpc/h)
COMMON="--sim-mode pm --mesh-size 1024 1024 1024 --box-size $BOX2 --paint-order tsc --deconvolution \
--nside 512 --scheme ngp --nb-shells $NB_SHELLS --shell-spacing comoving --enable-x64 --perf --iterations 3 --seed $SEED $COSMO"

for SOLVER in kdk dkd bf; do
  for STEPS in 10 18 30 50; do
    launch 1 4 4 1 01:00:00 -- $COMMON --solver "$SOLVER" --nb-steps "$STEPS" \
      --output "$RESULTS/exp4/${SOLVER}_s${STEPS}.parquet" \
      --name "exp4_${SOLVER}_STEPS%nb_steps%_s%seed%"
  done
done
