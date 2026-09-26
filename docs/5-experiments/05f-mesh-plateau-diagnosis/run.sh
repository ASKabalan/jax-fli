#!/bin/bash
# Experiment 05f — mesh plateau diagnosis: full factorial over mesh × solver × shell-spacing.
#
# Holds the 05e production physics fixed (50-step baseline: BullFrog, equal_vol, 3-bin Born,
# nside 2048, drift-on-lightcone, 5 Gpc/h box) and re-runs the FULL mesh ladder
# (512³–4096³, INCLUDING 2560³) at 100 steps, crossed with:
#   - Both solvers: BullFrog (bf) and DoubleKickDrift (kdk)
#   - Both shell-spacing schemes: spacing="a" (05a's voxel-like scheme) and "equal_vol" (05c/05e's)
#   - Pencil (2D) decomposition for meshes ≥2048³, slab (1D) for smaller
#
# HALO SIZING: The 05e plateau raised the question whether halo starvation is the cause.
# This experiment removes that confound: all six mesh rungs are sized with a UNIFORM
# --halo-multiplier=0.5 and decomposition (slab or balanced pencil) such that EVERY rung
# clears ≥4×σ₁D (worst-case: σ₁D(z=0)=5.89 Mpc/h, target halo ≥23.56 Mpc/h). The halo
# sizing table below shows the result — all rungs pass with margin 2.7–5.4× the target.
#
#     Mesh  | GPUs | pdim (px×py) | Local mesh | Halo (Mpc/h) | Clearance
#     ------|------|--------------|------------|--------------|----------
#     512³  |  4   | slab 4×1     |     128    |      64      | 10.9×σ₁D
#     1024³ |  8   | slab 8×1     |     128    |      64      | 10.9×σ₁D
#     2048³ | 64   | pencil 8×8   |     256    |     128      | 21.7×σ₁D
#     2560³ | 128  | pencil 16×8  | 160/320    |   80/160     | 13.6×σ₁D
#     3072³ | 256  | pencil 16×16 |     192    |      96      | 16.3×σ₁D
#     4096³ | 512  | pencil 32×16 | 128/256    |   64/128     | 10.9×σ₁D
#
# If the plateau persists despite halo comfort, its cause is NOT halo starvation — instead,
# one of: step budget, shell spacing, solver, or a genuine ceiling in the PM/Born approach
# at this box/resolution.
#
# COST ESTIMATE: 24 density configs (6 meshes × 2 solvers × 2 spacings) at 100 steps,
# pencil decomposition, 01:30:00 per run = ~1450 node-hours (~5800 GPU-hours). Born/κ stage
# (24 lensing runs, fixed decomposition) adds ~32 node-hours. Total ~1480 node-hours.
#
# This run.sh is NOT automatically submitted (MODE defaults to dryrun). Inspect the
# dryrun output with MODE=dryrun bash run.sh before committing GPU-hours.

source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 05f — mesh plateau diagnosis, 100-step factorial (MODE=$MODE)"

BOX5="5000.0 5000.0 5000.0"
SIM_MODE="${SIM_MODE:-DENSITY}"  # DENSITY → density ladder; anything else → Born lensing stage

# Shared physics: 100 steps, uniform halo, min-width 50 (the true default).
# Solver, shell-spacing, and time-stepping are loop variables (not part of COMMON).
# Time-stepping: D for BullFrog (bf), a (default) for DoubleKickDrift (kdk).
COMMON_BASE="--sim-mode pm --box-size $BOX5 --nb-steps 100 --time-stepping %TIME_STEPPING% --min-width 50.0 \
--nb-shells 20 --paint-order cic --nside 2048 --shells-per-file 1 --scheme ngp \
--shell-spacing %SPACING% --solver %SOLVER% --halo-multiplier 0.5 \
--drift-on-lightcone --enable-x64 --perf --iterations 3 --seed $SEED $COSMO"

# Mesh ladder with pencil decomposition (slab for small meshes, balanced pencil for ≥2048³).
#        mesh   nodes  gpn  px  py  (total GPUs = nodes × gpn = px × py)
RUNS=(
  "512    1      4    4   1"
  "1024   2      4    8   1"
  "2048   16     4    8   8"
  "2560   32     4   16   8"
  "3072   64     4   16  16"
  "4096  128     4   32  16"
)

SOLVERS=(bf kdk)
SPACINGS=(a equal_vol)
RUN_LIMIT="01:30:00"

if [ "$SIM_MODE" = "DENSITY" ]; then
  echo "### 24 density runs: 6 meshes × 2 solvers (bf/kdk) × 2 spacings (a/equal_vol) × 100 steps"
  for r in "${RUNS[@]}"; do
    read -r M NODES GPN PX PY <<< "$r"
    for solver in "${SOLVERS[@]}"; do
      # Time-stepping: D for bf (BullFrog), a (default) for kdk (DoubleKickDrift)
      if [ "$solver" = "bf" ]; then
        time_stepping="D"
      else
        time_stepping="a"
      fi
      for spacing in "${SPACINGS[@]}"; do
        tag="exp5f_m${M}_${solver}_${spacing}"
        COMMON=$(printf "%s" "$COMMON_BASE" | sed "s|%SOLVER%|$solver|g; s|%SPACING%|$spacing|g; s|%TIME_STEPPING%|$time_stepping|g")
        launch "$NODES" "$GPN" "$PX" "$PY" "$RUN_LIMIT" -- $COMMON \
          --mesh-size "$M" "$M" "$M" \
          --output "$RESULTS/exp5f/density/${tag}" --name "${tag}_M%mesh_size%_s%seed%"
      done
    done
  done

else
  # Born/κ lensing stage: read back the density shells and compute 3-bin lensed convergence.
  # Reuses 05e's lensing decomposition (px=16, py=1, 2 nodes, 8 gpn) — constant for all 24 configs.
  # (κ is computed on the HEALPix map, independent of the PM mesh's decomposition.)

  launch_rt() {
    local account=$1 constraint=$2 qos=$3 nodes=$4 gpn=$5 px=$6 py=$7 tlimit=$8; shift 8
    [ "$1" = "--" ] && shift
    fli-launcher --mode "$MODE" --account "$account" --constraint "$constraint" \
      --nodes "$nodes" --gpus-per-node "$gpn" --cpus-per-node "$CPUS" --qos "$qos" \
      --time-limit "$tlimit" --slurm-script "$SLURM_SCRIPT" --output-logs "$OUTPUT_LOGS" \
      --pdim "$px" "$py" -- "$@"
  }

  echo "### 24 Born/κ lensing runs (reading back density shells)"
  for r in "${RUNS[@]}"; do
    read -r M _ <<< "$r"
    for solver in "${SOLVERS[@]}"; do
      for spacing in "${SPACINGS[@]}"; do
        DATA="05-spacing-n-stepping/05f-mesh/density/exp5f_m${M}_${solver}_${spacing}/shell*.parquet"
        tag="kappa_gl_m${M}_${solver}_${spacing}"
        echo "Launching Born lensing (gauss_legendre) for ${tag}"
        launch_rt "$ACCOUNT" "$CONSTRAINT" "$QOS" 2 8 16 1 00:40:00 -- \
          fli-born-rt --repo ASKabalan/jax-fli-experiments --data-files "$DATA" \
          --nz-shear "s3[:3]" --nside 2048 --enable-x64 --normalization global --quadrature gauss_legendre \
          --perf --iterations 3 \
          --name "$tag" --output "$RESULTS/exp5f/kappa_gl/born_gl_m${M}_${solver}_${spacing}"
      done
    done
  done

fi
