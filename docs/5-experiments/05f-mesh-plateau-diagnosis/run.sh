#!/bin/bash
# Experiment 05f — mesh plateau diagnosis: mesh × solver × shell-spacing, 100 steps.
#
# Fixed 05e production physics: 5 Gpc/h box, 20 shells, nside 2048, drift on the lightcone,
# 3-bin Stage-3 Born (Gauss–Legendre), uniform --halo-multiplier 0.5.
#
# Loops:
#   mesh     512³ … 4096³ (slab below 2048³, pencil from 2048³)
#   solver   bf (BullFrog, --time-stepping D) and kdk (DoubleKickDrift, --time-stepping a)
#   spacing  a                  uniform in scale factor                       (DONE on the cluster)
#            equal_vol          plain equal volume, first shell a ~912 Mpc/h ball (DONE on the cluster)
#            eqvolc_w150_r0     equal volume, --max-width 150, --r-min 0
#            eqvolc_w150_r150   equal volume, --max-width 150, --r-min 150
#            eqvolc_w150_r300   equal volume, --max-width 150, --r-min 300
#   Born     without --resolution-cut (nside 2048) and with it (nside 512, ell_max = 1535; each shell
#            low-passed at ell_res = k_Nyq r_eff)
#
# The a / equal_vol density runs keep their original tags; launch() SKIPs them because their
# output folders already hold parquets. Only the three controlled spacings are submitted.
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
# Usage:
#   MODE=dryrun bash run.sh                    # density stage, print commands only
#   MODE=dryrun SIM_MODE=LENSING bash run.sh   # Born stage (60 without cut + 60 with cut)

# Results always live in docs/5-experiments/results (where the finished a / equal_vol runs are), whatever
# directory this script is launched from -- launch() only SKIPs a run if it finds that run's parquets there.
RESULTS="${RESULTS:-$(cd "$(dirname "$0")/.." && pwd)/results}"
source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 05f — mesh plateau diagnosis, 100-step factorial (MODE=$MODE)"

SIM_MODE="${SIM_MODE:-DENSITY}"  # DENSITY → density ladder; anything else → Born lensing stage

#        mesh   nodes  gpn  px  py  (total GPUs = nodes × gpn = px × py)
RUNS=(
  "512    1      4    4   1"
  "1024   2      4    8   1"
  "2048   16     4    8   8"
  "2560   32     4   16   8"
  "3072   64     4   16  16"
  "4096  128     4   32  16"
)

if [ "$SIM_MODE" = "DENSITY" ]; then
  for r in "${RUNS[@]}"; do
    read -r M NODES GPN PX PY <<< "$r"
    for solver in bf kdk; do
      if [ "$solver" = "bf" ]; then TS="D"; else TS="a"; fi

      COMMON="--sim-mode pm --box-size 5000.0 5000.0 5000.0 --mesh-size $M $M $M --nb-steps 100 \
--solver $solver --time-stepping $TS --nb-shells 20 --min-width 50.0 --paint-order cic --nside 2048 \
--shells-per-file 1 --scheme ngp --halo-multiplier 0.5 --drift-on-lightcone --enable-x64 \
--perf --iterations 3 --seed $SEED $COSMO"
      OUT="$RESULTS/exp5f/density"

      # done on the cluster -> SKIP
      launch "$NODES" "$GPN" "$PX" "$PY" 01:30:00 -- $COMMON --shell-spacing a \
        --output "$OUT/exp5f_m${M}_${solver}_a" --name "exp5f_m${M}_${solver}_a_M%mesh_size%_s%seed%"
      launch "$NODES" "$GPN" "$PX" "$PY" 01:30:00 -- $COMMON --shell-spacing equal_vol \
        --output "$OUT/exp5f_m${M}_${solver}_equal_vol" --name "exp5f_m${M}_${solver}_equal_vol_M%mesh_size%_s%seed%"

      # controlled equal volume
      launch "$NODES" "$GPN" "$PX" "$PY" 01:30:00 -- $COMMON --shell-spacing equal_vol --max-width 150.0 --r-min 0.0 \
        --output "$OUT/exp5f_m${M}_${solver}_eqvolc_w150_r0" --name "exp5f_m${M}_${solver}_eqvolc_w150_r0_M%mesh_size%_s%seed%"
      launch "$NODES" "$GPN" "$PX" "$PY" 01:30:00 -- $COMMON --shell-spacing equal_vol --max-width 150.0 --r-min 150.0 \
        --output "$OUT/exp5f_m${M}_${solver}_eqvolc_w150_r150" --name "exp5f_m${M}_${solver}_eqvolc_w150_r150_M%mesh_size%_s%seed%"
      launch "$NODES" "$GPN" "$PX" "$PY" 01:30:00 -- $COMMON --shell-spacing equal_vol --max-width 150.0 --r-min 300.0 \
        --output "$OUT/exp5f_m${M}_${solver}_eqvolc_w150_r300" --name "exp5f_m${M}_${solver}_eqvolc_w150_r300_M%mesh_size%_s%seed%"
    done
  done

else
  # Born/κ: read the density shells back from the HF snapshot, 2 nodes × 8 GPUs (px=16, py=1) for every config.
  launch_rt() {
    local account=$1 constraint=$2 qos=$3 nodes=$4 gpn=$5 px=$6 py=$7 tlimit=$8; shift 8
    [ "$1" = "--" ] && shift
    fli-launcher --mode "$MODE" --account "$account" --constraint "$constraint" \
      --nodes "$nodes" --gpus-per-node "$gpn" --cpus-per-node "$CPUS" --qos "$qos" \
      --time-limit "$tlimit" --slurm-script "$SLURM_SCRIPT" --output-logs "$OUTPUT_LOGS" \
      --pdim "$px" "$py" -- "$@"
  }

  for r in "${RUNS[@]}"; do
    read -r M _ <<< "$r"
    for solver in bf kdk; do
      for spacing in a equal_vol eqvolc_w150_r0 eqvolc_w150_r150 eqvolc_w150_r300; do
        tag="m${M}_${solver}_${spacing}"
        DATA="05-spacing-n-stepping/05f-mesh/density/exp5f_${tag}/shell*.parquet"
        echo "Launching 3-bin Born lensing (gauss_legendre) for exp5f_${tag}"

        # no resolution cut
        launch_rt "$ACCOUNT" "$CONSTRAINT" "$QOS" 4 4 16 1 00:40:00 -- \
          fli-born-rt --repo ASKabalan/jax-fli-experiments --data-files "$DATA" \
          --nz-shear "s3[:3]" --nside 2048 --enable-x64 --normalization global --quadrature gauss_legendre \
          --perf --iterations 3 \
          --name "kappa_gl_${tag}" --output "$RESULTS/exp5f/kappa_gl/born_gl_${tag}"

        # with resolution cut (each shell low-passed at ell_res = k_Nyq r_eff)
        launch_rt "$ACCOUNT" "$CONSTRAINT" "$QOS" 4 4 16 1 00:40:00 -- \
          fli-born-rt --repo ASKabalan/jax-fli-experiments --data-files "$DATA" \
          --nz-shear "s3[:3]" --nside 512 --enable-x64 --normalization global --quadrature gauss_legendre \
          --perf --iterations 3 --resolution-cut \
          --name "kappa_gl_rescut_${tag}" --output "$RESULTS/exp5f/kappa_gl_rescut/born_gl_rescut_${tag}"
      done
    done
  done
fi
