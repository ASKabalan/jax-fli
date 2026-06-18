#!/bin/bash
# Experiment 04b — number of integration steps for convergence.
# Both solvers {kdk,bf} × nb-steps {5,6,10,20,30,50} = 12 sims at fixed resolution. float64.
# Question: how few PM steps give a converged per-shell spherical C_ℓ, for each solver?
source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 04b — step-count convergence  (MODE=$MODE)"

NB_SHELLS="${NB_SHELLS:-10}"   # lightcone shells (box/2 / N ≥ min_width 50 Mpc/h)

# Everything except --solver and --nb-steps is held constant. Several knobs are PLACEHOLDERS to be set
# from the sibling experiments that determine them — kept explicit here so the final choice is recorded:
#   --mesh-size                      resolution TBD from Exp 01 (resolution-convergence). 1024³ placeholder.
#   --paint-order tsc --deconvolution  mass-assignment scheme TBD from Exp 02 (mass-assignment).
#   --scheme ngp                     spherical-paint interpolation TBD from Exp 03 (spherical-painting).
COMMON="--sim-mode pm --mesh-size 1024 1024 1024 --box-size $BOX2 --paint-order tsc --deconvolution \
--nside 512 --scheme ngp --nb-shells $NB_SHELLS --shell-spacing comoving --enable-x64 --perf --iterations 3 --seed $SEED $COSMO"

# 1024³ @ 8 GPUs (2 nodes, --pdim 8 1): local 128³, halo int(128·0.5)=64 (EVEN), local ≤ 512³ → fits f64.
# If you raise --mesh-size (from Exp 01), scale px so local = mesh/px ≤ 512³ AND int(mesh/px·0.5) stays
# EVEN (an odd halo crashes jaxpm slice_unpad) — see 01-resolution-convergence/README.md.
#       nodes gpn px py  time
for SOLVER in kdk bf; do
  for STEPS in 5 6 10 20 30 50; do
    launch 2 4 8 1 01:00:00 -- $COMMON --solver "$SOLVER" --nb-steps "$STEPS" \
      --output "$RESULTS/exp4b/${SOLVER}_s${STEPS}.parquet" \
      --name "exp4b_${SOLVER}_STEPS%nb_steps%_s%seed%"
  done
done
