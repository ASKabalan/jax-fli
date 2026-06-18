#!/bin/bash
# Experiment 02 — 3D mass assignment × force deconvolution (2048³, BullFrog, NGP spherical).
# {cic,tsc,pcs} × {deconv off,on} = 6 sims. float64.
source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 02 — mass assignment + force deconvolution  (MODE=$MODE)"

NB_SHELLS="${NB_SHELLS:-10}"   # lightcone shells (box/2 / N ≥ min_width 50 Mpc/h)
COMMON="--sim-mode pm --mesh-size 2048 2048 2048 --box-size $BOX2 --solver bf --nb-steps 50 \
--nside 2048 --scheme ngp --nb-shells $NB_SHELLS --shell-spacing comoving --enable-x64 --perf --iterations 3 --seed $SEED $COSMO"

for PO in cic tsc pcs; do
  for DECONV in "" "--deconvolution"; do
    if [ -n "$DECONV" ]; then tag="exp2_${PO}_deconv"; else tag="exp2_${PO}"; fi
    launch 8 4 64 1 00:45:00 -- $COMMON --paint-order "$PO" $DECONV \
      --output "$RESULTS/exp2/${tag}.parquet" --name "${tag}_M%mesh_size%_s%seed%"
  done
done
