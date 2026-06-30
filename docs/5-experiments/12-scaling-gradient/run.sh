#!/bin/bash
# Experiment 12 — gradient scaling: strong & weak scaling of the IC-gradient on SLAB (N,1), float64.
# The reversible kdk (DoubleKickDrift) backsolve and equinox checkpointed scans, benchmarked with --perf
# (wall-time + per-device memory). Correctness of the gradient is settled in Exp 09; this is pure cost.
# The differentiated output is the IC-shaped gradient (the full mesh); fli-simulate saves it as usual, so
# the launcher skips a rung whose --output already exists on rerun.
#
# Slab (N,1): px = #GPUs, py = 1, local mesh (M/px, M, M). Halo int((M/px)*0.5) must be EVEN (Exp 01/11):
# 1024³ tops out at 256 GPUs. The gradient roughly DOUBLES the forward working set, so the ladder starts a
# rung higher than Exp 11 (ceiling halved to ~406³/GPU); checkpointed runs may need one rung higher still.
# Strong scaling is 1024³ only (2048³ gradient runs are out of scope here).
source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 12 — gradient strong & weak scaling, slab (N,1)  (MODE=$MODE)"

# Forward model differentiated: PM + lightcone painting (nside 1024, 10 shells), 30 steps, reversible kdk.
GRAD_COMMON="--sim-mode pm --solver kdk --nb-steps 30 --paint-order tsc --deconvolution --scheme ngp \
--nside 1024 --nb-shells 10 --enable-x64 --perf --iterations 5 --seed $SEED $COSMO"

GRAD_CEIL=67108864   # ~406³ cells/GPU — half the Exp 11 float64 ceiling (the IC gradient ~doubles the working set)
GPUS="4 8 16 32 64 128 256 512"
# <grad-value>:<tag> — the tag goes into --name (single _-free token so a func_name parser can split on _).
GRADS="reverse:rev checkpointed_4:ckpt4 checkpointed_8:ckpt8"   # O(1)-memory backsolve + two shell-scan checkpoint counts

for gradspec in $GRADS; do
  IFS=: read -r grad gtag <<<"$gradspec"

  # (a) strong scaling — fixed grid, grow GPUs (slab (N,1)). 1024³ only.
  for M in 1024; do
    for g in $GPUS; do
      px=$g; py=1; nodes=$(( g / 4 ))
      (( M % px != 0 )) && continue
      local_x=$(( M / px ))
      if (( local_x % 4 != 0 )); then
        echo "### SKIP strong M${M} g${g} ${grad}: halo int(${local_x}*0.5) is odd"; continue
      fi
      cells=$(( local_x * M * M ))
      if (( cells > GRAD_CEIL )); then
        echo "### SKIP strong M${M} g${g} ${grad}: local ${local_x}x${M}x${M}=${cells} > gradient ceiling ${GRAD_CEIL}"; continue
      fi
      launch "$nodes" 4 "$px" "$py" 00:30:00 -- $GRAD_COMMON --grad "$grad" --mesh-size $M $M $M --box-size $BOX2 \
        --output "$RESULTS/exp12/strong_M${M}_g${g}_${gtag}.parquet" \
        --name "exp12_strong_M${M}_g${g}_${gtag}_s%seed%"
    done
  done

  # (b) weak scaling — fixed 256³/GPU (slab): global = (256·px, 256, 256), local 256³, halo 128
  for g in $GPUS; do
    px=$g; py=1; nodes=$(( g / 4 ))
    gx=$(( 256 * px ))
    launch "$nodes" 4 "$px" "$py" 00:30:00 -- $GRAD_COMMON --grad "$grad" --mesh-size $gx 256 256 --box-size $BOX2 \
      --output "$RESULTS/exp12/weak_g${g}_${gtag}.parquet" \
      --name "exp12_weak_g${g}_${gtag}_s%seed%"
  done
done
