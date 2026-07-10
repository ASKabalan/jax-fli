#!/bin/bash
# Experiment 11 — PM throughput: strong & weak scaling on SLAB (N,1) decompositions, BullFrog only.
# Run in BOTH float32 and float64 (`--perf` gives wall-time + per-device memory). Only the PM stage is
# benchmarked here (no separate LPT-only / lensing-only timing) — PM includes its lightcone painting.
#
# Slab (N,1): px = #GPUs, py = 1, local mesh = (M/px, M, M). The ghost-zone halo int((M/px)*0.5) must be
# EVEN (an odd halo crashes jaxpm slice_unpad) — at halo_multiplier 0.5 that means (M/px) % 4 == 0, so a
# 1024³ slab tops out at 256 GPUs (512 → local 2, halo 1). Each (precision, mesh) ladder also starts at the
# smallest GPU count whose local volume fits the per-GPU ceiling (~300³ in float64, ~2·300³ in float32);
# below-ceiling rungs are skipped and logged, so "4 → 512 GPUs" is the envelope, not every rung.
source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 11 — PM strong & weak scaling, slab (N,1)  (MODE=$MODE)"

# Fixed forward-model knobs: final-version forward model — PM + lightcone painting with CIC (no force-window
# deconvolution), RBF 0.8-pixel spherical smoothing, drift on the lightcone, 20 shells. bf, 50 steps. nside
# tracks the smallest global-mesh dimension, set per launch below (nside = M for the cubic strong-scaling
# grids, 256 for the weak slab).
# --shells-per-file 1: stream the 20-shell lightcone as one parquet per shell (shell_NNNN.parquet) into the
# --output *directory* — at nside 2048 the full-lightcone gather onto rank 0 would OOM.
PERF="--sim-mode pm --solver bf --nb-steps 50 --paint-order cic --scheme rbf_neighbor --kernel-width-pixels 0.8 \
--nb-shells 20 --shells-per-file 1 --drift-on-lightcone --perf --iterations 5 --seed $SEED $COSMO"

BOX5="5000.0 5000.0 5000.0"   # 5 Gpc/h — final-version accuracy box

F64_CEIL=27000000    # 300³ cells/GPU — float64 ceiling for the heavier final-version model (512³ no longer fits)
F32_CEIL=54000000    # 2·300³ — float32 holds twice the cells per GPU

GPUS="4 8 16 32 64 128 256 512"

for pc in "f32:" "f64:--enable-x64"; do
  IFS=: read -r ptag pflag <<<"$pc"
  ceil=$F32_CEIL; [ "$ptag" = "f64" ] && ceil=$F64_CEIL

  # (a) strong scaling — fixed grid, grow GPUs (slab (N,1))
  for M in 1024 2048; do
    for g in $GPUS; do
      px=$g; py=1; nodes=$(( g / 4 ))
      (( M % px != 0 )) && continue                       # X must shard evenly across the slab
      local_x=$(( M / px ))
      if (( local_x % 4 != 0 )); then
        echo "### SKIP strong M${M} g${g} ${ptag}: halo int(${local_x}*0.5) is odd"; continue
      fi
      cells=$(( local_x * M * M ))
      if (( cells > ceil )); then
        echo "### SKIP strong M${M} g${g} ${ptag}: local ${local_x}x${M}x${M}=${cells} > ${ptag} ceiling ${ceil}"; continue
      fi
      launch "$nodes" 4 "$px" "$py" 00:30:00 -- $PERF $pflag --nside $M --mesh-size $M $M $M --box-size $BOX5 \
        --output "$RESULTS/exp11/strong_M${M}_g${g}_${ptag}" \
        --name "exp11_strong_M${M}_g${g}_${ptag}_s%seed%"
    done
  done

  # (b) weak scaling — fixed 256³/GPU (slab): global = (256·px, 256, 256), local 256³, halo 128 (always even)
  for g in $GPUS; do
    [ "$g" = 512 ] && continue   # weak g512 never landed (OOM at 512 GPUs) — strong M2048 g512 still runs above
    px=$g; py=1; nodes=$(( g / 4 ))
    gx=$(( 256 * px ))
    launch "$nodes" 4 "$px" "$py" 00:30:00 -- $PERF $pflag --nside 256 --mesh-size $gx 256 256 --box-size $BOX5 \
      --output "$RESULTS/exp11/weak_g${g}_${ptag}" \
      --name "exp11_weak_g${g}_${ptag}_s%seed%"
  done
done
