#!/bin/bash
# Experiment 10 — performance of the lightcone adjoints (memory + wall-time) on HPC.
#
# Three sweeps over the two independent axes the gradient cost depends on — integration steps
# *taken* and lightcone shells *saved* — plus the checkpoint-count knob:
#
#   A. vs STEPS   — single volumetric snapshot (--density). `reverse` is O(1) in steps (flat
#                   memory); `checkpointed` stores the step trajectory, so memory grows with steps.
#   B. vs #SHELLS — spherical lightcone, fixed steps. `reverse` is NOT O(1) here: it accumulates a
#                   per-shell save-at VJP, so memory grows with #shells (so does `checkpointed`).
#   C. vs #CHECKPOINTS — spherical lightcone, fixed steps & shells, sweeping `checkpointed_<N>`.
#                   `checkpoints` checkpoints the OUTER shell scan (NOT the inner step loop), so it
#                   trades shell-scan memory for recompute — only meaningful with >1 shell.
#
# Each run is given its OWN output subdir: the perf CSV row is keyed solely by function=pm<steps>
# + precision + mesh (grad mode is not a column; nb_shells lands only in the .md sidecar), so runs
# differing only in --grad / --nb-shells would otherwise collide in a single perf_pm.csv. The local
# figure script keys the swept variable off the subdir path.
#
# 256³, float64, 1 GPU (the trade-off shapes; multi-GPU scaling is Experiment 11). --perf writes
# wall-time + XLA memory-analysis (generated-code / argument / output / temp bytes).
source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 10 — adjoint performance: memory + time  (MODE=$MODE)"

MESH="--mesh-size 256 256 256"
PERF="--sim-mode pm --solver kdk --time-stepping a --gradient-order 0 --enable-x64 \
--perf --iterations 5 --seed $SEED $COSMO"
OUT="$RESULTS/exp10"
T=00:20:00

# --- A. memory + time vs STEPS — single volumetric snapshot (reverse flat, checkpointed grows) ----
DENS="$MESH --box-size 1000.0 1000.0 1000.0 --density $PERF"
for STEPS in 25 50 100 200 400; do
  for G in reverse checkpoint; do
    launch 1 1 1 1 $T -- $DENS --nb-steps $STEPS --grad "$G" \
      --output "$OUT/A-steps/${G}_s${STEPS}/out.parquet" --name "exp10A_${G}_s${STEPS}"
  done
done

# --- B. memory + time vs #SHELLS — spherical lightcone, fixed steps (reverse not O(1) in shells) --
# --min-width 1.0: 32 shells in the 2 Gpc/h box give ~31 Mpc/h each, below the default 50 Mpc/h floor
# (which hard-errors). Shell *width* is irrelevant to a sweep over shell *count*, so we lower the floor.
# nb_shells <= nb_steps is required, so --nb-steps 50 covers up to 32 shells.
SPH="$MESH --box-size $BOX2 --nside 512 --min-width 1.0 $PERF --nb-steps 50"
for NB in 1 2 4 8 16 32; do
  for G in reverse checkpoint; do
    launch 1 1 1 1 $T -- $SPH --nb-shells $NB --grad "$G" \
      --output "$OUT/B-shells/${G}_n${NB}/out.parquet" --name "exp10B_${G}_n${NB}"
  done
done

# --- C. memory + time vs #CHECKPOINTS — spherical lightcone, fixed 32 shells / 50 steps ----------
CK="$SPH --nb-shells 32"
for G in reverse checkpoint checkpointed_2 checkpointed_4 checkpointed_8 checkpointed_16 checkpointed_32; do
  launch 1 1 1 1 $T -- $CK --grad "$G" \
    --output "$OUT/C-ckpt/${G}/out.parquet" --name "exp10C_${G}"
done
