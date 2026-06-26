#!/bin/bash
# Experiment 12 — solver reversibility round-trip (tiny box; validates the reverse adjoint).
# forward→reverse round-trip vs nb-steps; reverse on the --density single snapshot. float64.
source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 12 — solver reversibility round-trip  (MODE=$MODE)"

@CLAUDE this run should be actually a benchmarking with two things in mind
Weak scaling strong scaling with 30 steps Reverse and checkpointing with multiple checkpoints like for example
The markdown product of the profiler are important (in the past they used to have the same name so they would overwrite each other)
/home/wassim/Projects/NBody/jax-fli/src/jax_fli/scripts/entry/fli_simulate.py in here I made the file change with the name

STRONG : Let's do 4 to 512 GPUs with 1024³ mesh and 2040³ mesh and sufficient halo size for each (see EXP 01)
WEAK : Let's do 4 to 512 GPUs with 256³/GPU sufficient halo size aswell

ONLY SLABS
We don't care about the round trop this was done already on Exp09


for STEPS in 10 20 40 80; do
  launch 1 1 1 1 00:20:00 -- --sim-mode pm --mesh-size 128 128 128 --box-size 256.0 256.0 256.0 \
    --density --solver kdk --time-stepping a --gradient-order 0 --enable-x64 --perf --iterations 3 --nb-steps "$STEPS" \
    --grad reverse --seed $SEED $COSMO \
    --output "$RESULTS/exp12/roundtrip_s${STEPS}.parquet" --name "exp12_STEPS%nb_steps%_s%seed%"
done
