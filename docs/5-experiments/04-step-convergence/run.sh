#!/bin/bash
# Experiment 04 — step-count convergence + solver comparison.
# All three integrators {kdk,dkd,bf} × nb-steps {5,6,10,20,30,50} = 18 sims at fixed resolution. float64.
# Question: how few PM steps give a converged per-shell spherical C_ℓ, and how do the solvers compare?
# (Merges the former 04-solver-comparison + 04b-step-convergence — they were the same sweep.)
source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 04 — step convergence (solver × step count)  (MODE=$MODE)"

NB_SHELLS="${NB_SHELLS:-10}"   # lightcone shells (box/2 / N ≥ min_width 50 Mpc/h)

# Production knobs (decided by the upstream experiments — no longer placeholders):
#   --mesh-size 2048 2048 2048       Exp 01: the resolution-converged, halo-safe rung for float64.
#   --paint-order cic                Exp 02: CIC mass assignment, NO force-window --deconvolution.
#   --scheme ngp                     Exp 03: nearest-grid-point spherical painting.
#   --nside 2048                     production CosmoGrid projection (== the 2048³ mesh).
#   --shells-per-file 1              one parquet per shell into the --output *directory* (shell_NNNN.parquet);
#                                    required at nside 2048 — gathering the whole lightcone onto rank 0 OOMs
#                                    (see Exp 06). The shared launch() treats a parquet-filled dir as "done".
COMMON="--sim-mode pm --mesh-size 2048 2048 2048 --box-size $BOX2 --paint-order cic \
--nside 2048 --shells-per-file 1 --scheme ngp --nb-shells $NB_SHELLS --shell-spacing a \
--time-stepping a \
--enable-x64 --perf --iterations 3 --seed $SEED $COSMO"

# 2048³ @ 64 GPUs (16 nodes, --pdim 64 1): local 2048/64 = 32³, halo int(32·0.5)=16 (EVEN), local ≤ 512³ → fits f64.
# Halo sizing (Exp 01 max-drift rule): physical ghost zone = halo_multiplier·box/px = 0.5·2000/64 = 15.6 Mpc/h,
# vs the end-of-run 3D rms displacement σ₃D(z=0) = 10.2 Mpc/h → halo/σ = 1.53 ≥ 1.5 ✅ (the exact converged
# 2048³ rung of Exp 01). Box+cosmology fix σ_disp, so this one check covers every (solver, steps) here.
#       nodes gpn px py  time
for SOLVER in kdk dkd bf; do
  for STEPS in 10 20 30 40 50; do
    launch 16 4 64 1 01:00:00 -- $COMMON --solver "$SOLVER" --nb-steps "$STEPS" \
      --output "$RESULTS/exp4/${SOLVER}_s${STEPS}" \
      --name "exp4_${SOLVER}_STEPS%nb_steps%_s%seed%"
  done
done
