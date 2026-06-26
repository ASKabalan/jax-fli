#!/bin/bash
# Experiment 11 — throughput, strong & weak scaling (--perf gives time + per-device memory).
# Run in BOTH float32 and float64. Strong-scaling grids start at the smallest GPU count that fits in
# float64 (≈645³/GPU). Weak scaling fixes 512³/GPU: global = (512·PX, 512·PY, 512).
source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 11 — throughput, strong & weak scaling  (MODE=$MODE)"

PERF="--solver bf --nb-steps 50 --paint-order tsc --deconvolution --scheme ngp \
--perf --iterations 5 --seed $SEED $COSMO"
NB_SHELLS="${NB_SHELLS:-10}"   # lightcone shells (box/2 / N ≥ min_width 50 Mpc/h)

# Run every config in both precisions: f32 (no flag) and f64 (--enable-x64).
for pc in "f32:" "f64:--enable-x64"; do
  IFS=: read -r ptag pflag <<<"$pc"


  @CLAUDE only bensh pm no lensing no LPT

  # (a) per-stage cost — 1024³ on 4 GPU
  for SIM in lpt pm lensing; do
    case "$SIM" in
      lpt)     extra="" ;;
      pm)      extra="--nside 512 --nb-shells $NB_SHELLS" ;;
      lensing) extra="--nside 512 --nb-shells $NB_SHELLS --nz-shear s3" ;;
    esac
    launch 1 4 2 2 00:30:00 -- --sim-mode "$SIM" --mesh-size 1024 1024 1024 --box-size $BOX2 \
      $PERF $extra $pflag \
      --output "$RESULTS/exp11/stage_${SIM}_${ptag}.parquet" --name "exp11_${SIM}_${ptag}_s%seed%"
  done

 @CLAUDE ONLY SLABS from 4 GPUS TO 512 GPUS for both 1024 and 2048 meshes
  # (b) strong scaling — fixed grid, grow GPUs.   nodes gpn px py mesh
  for t in "1 4 2 2 1024" "2 4 2 4 1024" "4 4 4 4 1024" "8 4 4 8 1024" "16 4 8 8 1024" "32 4 8 16 1024" "128 4 16 16 1024" \
           "8 4 4 8 2048" "16 4 8 8 2048" "32 4 8 16 2048" "64 4 16 16 2048"; do
    set -- $t; nodes=$1 gpn=$2 px=$3 py=$4 M=$5
    launch "$nodes" "$gpn" "$px" "$py" 00:30:00 -- --sim-mode pm --mesh-size $M $M $M \
      --box-size $BOX2 $PERF --nside 1024 --nb-shells $NB_SHELLS $pflag \
      --output "$RESULTS/exp11/strong_M${M}_n${nodes}g${gpn}_${ptag}.parquet" \
      --name "exp11_strong_M${M}_N${nodes}x${gpn}_${ptag}_s%seed%"
  done

  @CLAUDE ONLY SLABS from 4 GPUS TO 512 GPUS with fixed 256³/GPU
  # (c) weak scaling — fixed 512³/GPU.   nodes gpn px py gx gy gz
  for t in "1 4 2 2 1024 1024 512" "2 4 2 4 1024 2048 512" "4 4 4 4 2048 2048 512" \
           "8 4 4 8 2048 4096 512" "16 4 8 8 4096 4096 512" "32 4 8 16 4096 8192 512" \
           "64 4 16 16 8192 8192 512"; do
    set -- $t; nodes=$1 gpn=$2 px=$3 py=$4 gx=$5 gy=$6 gz=$7
    launch "$nodes" "$gpn" "$px" "$py" 00:30:00 -- --sim-mode pm --mesh-size $gx $gy $gz \
      --box-size $BOX2 $PERF --nside 1024 --nb-shells $NB_SHELLS $pflag \
      --output "$RESULTS/exp11/weak_${px}x${py}_${ptag}.parquet" \
      --name "exp11_weak_${px}x${py}_${ptag}_s%seed%"
  done
done
