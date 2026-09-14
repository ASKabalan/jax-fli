#!/bin/bash
# Launch 15-LPT-Multihost-MassMapping.py on Jean Zay (H100) at three resolutions.
#
# Scale cut follows the paint-nside rule: ell_max = 1.5 * paint_nside, and the
# cosine taper width is 12.5% of the cut:
#   paint_nside 512 -> ell_max 768, ell_taper 96   (all three runs here)
#
# Usage:
#   MODE=dryrun bash run.sh    # print the sbatch commands, submit nothing
#   bash run.sh                # submit the three jobs (MODE=sbatch default)
#
# Knobs (override via env): NODES TIME MAP2ALM MAP_MAX_ITER N_SNAPSHOTS SEED
#   SLURM_SCRIPT TRACES PYTHON EMAIL ACCOUNT QOS CPUS_PER_TASK
set -euo pipefail

cd "$(dirname "$0")"

MODE="${MODE:-sbatch}"
SCRIPT="15-LPT-Multihost-MassMapping.py"

# --- cluster (template from the working submission command) ---
ACCOUNT="${ACCOUNT:-tkc@h100}"
CONSTRAINT="${CONSTRAINT:-h100}"
QOS="${QOS:-qos_gpu_h100-t3}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NTASKS_PER_NODE="${NTASKS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-6}"
NODES="${NODES:-32}"
TIME="${TIME:-4:00:00}"
EMAIL="${EMAIL:-}"
SLURM_SCRIPT="${SLURM_SCRIPT:-slurm_script.sh}"   # wrapper: $SLURM_SCRIPT TRACES python ARGS
TRACES="${TRACES:-traces}"
PYTHON="${PYTHON:-python}"

# --- physics knobs ---
MAP2ALM="${MAP2ALM:-jax_cuda}"          # jax_cuda OOMs at cuFFT plan creation (see diagnose_s2fft_cuda.py)
MAP_MAX_ITER="${MAP_MAX_ITER:-400}"
N_SNAPSHOTS="${N_SNAPSHOTS:-40}"
SEED="${SEED:-0}"

# paint_nside 512 everywhere -> scale cut 768, taper 12.5% of the cut = 96.
PAINT_NSIDE=512
ELL_MAX=$(python3 -c "print(int($PAINT_NSIDE * 1.5))")          # 768
ELL_TAPER=$(python3 -c "print(int($ELL_MAX * 0.125))")          # 96

# <name>:<mesh>:<nside>
RUNS=(
  "m2048_n2048:2048:2048"
  "m2048_n1024:2048:1024"
  "m1536_n1024:1536:1024"
)

for spec in "${RUNS[@]}"; do
  IFS=: read -r name mesh nside <<<"$spec"
  out_dir="results_15/${name}"
  job_name="15map_${name}"

  cmd=(
    sbatch
    --account="$ACCOUNT"
    -C "$CONSTRAINT"
    --gres="gpu:${GPUS_PER_NODE}"
    --ntasks-per-node="$NTASKS_PER_NODE"
    --cpus-per-task="$CPUS_PER_TASK"
    --hint=nomultithread
    --qos="$QOS"
    --nodes="$NODES"
    --mail-type=BEGIN
  )
  [ -n "$EMAIL" ] && cmd+=(--mail-user="$EMAIL")
  cmd+=(
    --time="$TIME"
    --job-name="$job_name"
    "$SLURM_SCRIPT" "$TRACES" "$PYTHON" "$SCRIPT"
    --mesh "$mesh"
    --nside "$nside"
    --paint-nside "$PAINT_NSIDE"
    --ell-max "$ELL_MAX"
    --ell-taper "$ELL_TAPER"
    --map2alm-method "$MAP2ALM"
    --map-max-iter "$MAP_MAX_ITER"
    --n-snapshots "$N_SNAPSHOTS"
    --seed "$SEED"
    --out-dir "$out_dir"
    --gpus-per-node "$GPUS_PER_NODE"
  )

  echo "### ${name}: mesh=${mesh} nside=${nside} paint_nside=${PAINT_NSIDE} ell_max=${ELL_MAX} ell_taper=${ELL_TAPER}"
  if [ "$MODE" = dryrun ]; then
    echo "  ${cmd[*]}"
  else
    "${cmd[@]}"
  fi
done
