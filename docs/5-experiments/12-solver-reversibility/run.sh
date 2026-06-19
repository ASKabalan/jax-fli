#!/bin/bash
# Experiment 12 — solver reversibility round-trip (tiny box; validates the reverse adjoint).
# forward→reverse round-trip vs nb-steps; reverse on the --density single snapshot. float64.
source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 12 — solver reversibility round-trip  (MODE=$MODE)"

for STEPS in 10 20 40 80; do
  launch 1 1 1 1 00:20:00 -- --sim-mode pm --mesh-size 128 128 128 --box-size 256.0 256.0 256.0 \
    --density --solver kdk --time-stepping a --gradient-order 0 --enable-x64 --perf --iterations 3 --nb-steps "$STEPS" \
    --grad reverse --seed $SEED $COSMO \
    --output "$RESULTS/exp12/roundtrip_s${STEPS}.parquet" --name "exp12_STEPS%nb_steps%_s%seed%"
done
