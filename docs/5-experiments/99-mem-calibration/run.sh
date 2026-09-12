#!/bin/bash
# Experiment 99 — GPU Memory Calibration Suite
# Measures per-device peak VRAM and execution throughput over 100 gradient iterations
# across a mesh resolution ladder on 8 GPUs (or multi-node setups).
#
# Sized to systematically find what mesh/halo configurations saturate GPU memory
# without triggering out-of-memory (OOM) errors during reverse-mode gradient backpropagation.
source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 99 — GPU Memory Calibration (100 gradient iterations) (MODE=$MODE)"

BOX5="5000.0 5000.0 5000.0"
SIM_MODE="${SIM_MODE:-lpt}"  # "pm" for PM density lightcone; "lensing" for 3-bin Born kappa
GRAD_SPEC="${GRAD_SPEC:-checkpointed_25}"  # gradient checkpointing strategy
SOLVER="${SOLVER:-bf}"  # "bf" (BullFrog) or "kdk" (KickDriftKick)
NB_STEPS="${NB_STEPS:-50}"
TIME_STEPPING="${TIME_STEPPING:-D}"
NSIDE="${NSIDE:-512}"
NB_SHELLS="${NB_SHELLS:-12}"
ITERATIONS="${ITERATIONS:-100}"
GPN="${GPUS_PER_NODE:-4}"  # 4 on IDRIS/Jean Zay H100 nodes, 8 on single 8-GPU box

# Shared physics & gradient benchmark configuration
COMMON="--sim-mode $SIM_MODE --box-size $BOX5 --solver $SOLVER --nb-steps $NB_STEPS --time-stepping $TIME_STEPPING \
--min-width 60.0 --paint-order cic --nside $NSIDE --shells-per-file 1 --nb-shells $NB_SHELLS \
--scheme bilinear --shell-spacing equal_vol --drift-on-lightcone \
--enable-x64 --perf --iterations $ITERATIONS --seed $SEED $COSMO --grad $GRAD_SPEC"

# Calibration ladder on 8 GPUs: mesh, px(GPUs), nodes, halo_multiplier
# Evaluates local mesh sizes from 512³ up to 1024³
RUNS=(
  "512    8     2    0.10"   # halo = 6 cells  (58.6 Mpc/h = 5.7x sigma_3D)
  "584    8     2    0.08"   # halo = 6 cells  (51.4 Mpc/h = 5.0x sigma_3D)
  "1024   16    4    0.20"   # halo = 12 cells (58.6 Mpc/h = 5.7x sigma_3D)
  "2560   32    8    0.325"  # halo = 26 cells (50.8 Mpc/h = 5.0x sigma_3D)
  "2560   64    16   0.65"   # halo = 26 cells (50.8 Mpc/h = 5.0x sigma_3D)
)

RUN_LIMIT="00:40:00"

for r in "${RUNS[@]}"; do
  read -r M PX NODES HM <<< "$r"
  tag="mem_calib_m${M}_g${PX}_${GRAD_SPEC}"
  echo ">>> Launching Memory Calibration: Mesh=${M}³, GPUs=${PX}, Nodes=${NODES} (GPN=${GPN}), HaloMul=${HM}, Grad=${GRAD_SPEC}"
  launch "$NODES" "$GPN" "$PX" 1 "$RUN_LIMIT" -- $COMMON --mesh-size "$M" "$M" "$M" --halo-multiplier "$HM" \
    --output "$RESULTS/exp99_mem_calibration/density/${tag}.parquet" \
    --name "${tag}_M%mesh_size%_s%seed%"
done
