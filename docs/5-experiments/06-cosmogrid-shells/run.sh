#!/bin/bash
# Experiment 06 — match CosmoGrid shells. float64.  DATA-PREP REQUIRED.
# Pass the CosmoGrid shell a-edges via TS_NEAR / TS_FAR (extract them from the loaded
# CosmoGrid catalog first: jfli.io.load_cosmogrid_* -> shell geometry).
source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 06 — match CosmoGrid shells  (MODE=$MODE)"

: "${TS_NEAR:?set TS_NEAR to the CosmoGrid near shell a-edges}"
: "${TS_FAR:?set TS_FAR to the CosmoGrid far shell a-edges}"

launch 8 4 8 4 02:00:00 -- --sim-mode pm --mesh-size 2048 2048 2048 --box-size $BOX2 \
  --solver bf --nb-steps 50 --paint-order tsc --deconvolution --nside 512 --scheme ngp \
  --ts-near $TS_NEAR --ts-far $TS_FAR --enable-x64 --perf --iterations 3 --seed $SEED $COSMO \
  --output "$RESULTS/exp6/cosmogrid_shells.parquet" --name "exp6_M%mesh_size%_s%seed%"
