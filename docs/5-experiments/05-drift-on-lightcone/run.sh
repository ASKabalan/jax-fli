#!/bin/bash
# Experiment 05 — drift on the lightcone vs none (thick shells; density-shell C_ℓ). float64.
source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 05 — drift on the lightcone  (MODE=$MODE)"

NB_SHELLS="${NB_SHELLS:-8}"    # thick shells — the point of Exp 05
COMMON="--sim-mode pm --mesh-size 1024 1024 1024 --box-size $BOX2 --solver bf --nb-steps 50 \
--paint-order tsc --deconvolution --nside 512 --scheme ngp --nb-shells $NB_SHELLS --shell-spacing comoving \
--enable-x64 --perf --iterations 3 --seed $SEED $COSMO"

for DRIFT in "" "--drift-on-lightcone"; do
  if [ -n "$DRIFT" ]; then tag="exp5_drift"; else tag="exp5_nodrift"; fi
  launch 1 4 4 1 01:00:00 -- $COMMON $DRIFT \
    --output "$RESULTS/exp5/${tag}.parquet" --name "${tag}_M%mesh_size%_s%seed%"
done
