#!/bin/bash
# Experiment 07 — Born lensing vs CosmoGrid lensed (convergence). float64.
source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 07 — Born lensing  (MODE=$MODE)"

NB_SHELLS="${NB_SHELLS:-10}"   # lightcone shells (box/2 / N ≥ min_width 50 Mpc/h)

launch 8 4 8 4 02:00:00 -- --sim-mode lensing --mesh-size 2048 2048 2048 --box-size $BOX2 \
  --solver bf --nb-steps 50 --paint-order tsc --deconvolution --nside 512 --scheme ngp \
  --nb-shells $NB_SHELLS --shell-spacing comoving --nz-shear s3 --lensing-output convergence \
  --enable-x64 --perf --iterations 3 --seed $SEED $COSMO \
  --output "$RESULTS/exp7/born_s3.parquet" --name "exp7_M%mesh_size%_s%seed%"
