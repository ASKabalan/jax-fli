#!/bin/bash
# Shared launch helpers for the docs/5-experiments/*/run.sh scripts.
#
# Each experiment's run.sh does:  source "$(dirname "$0")/../_launch_common.sh"
# then calls:  launch <nodes> <gpus_per_node> <pdim_x> <pdim_y> <time> -- fli-simulate <args...>
#
# Usage (from inside an experiment folder):
#   MODE=dryrun bash run.sh     # print the resolved fli-launcher commands, submit nothing
#   MODE=sbatch bash run.sh     # submit to SLURM (default)
#   MODE=local  bash run.sh     # run locally via mpirun (use tiny meshes)
#
# Cluster knobs (override via env): ACCOUNT CONSTRAINT QOS CPUS SLURM_SCRIPT OUTPUT_LOGS RESULTS SEED
set -euo pipefail

MODE="${MODE:-sbatch}"
ACCOUNT="${ACCOUNT:-tkc@h100}"
CONSTRAINT="${CONSTRAINT:-h100}"
QOS="${QOS:-qos_gpu_h100-t3}"
# 96 cpus/node = 24 cpus/task on a 4-GPU H100 node (the IDRIS gpu_p6 default). Host RAM scales with
# cores on Jean Zay, so this is what gives rank 0 enough RAM to gather/save the nside-2048 lightcone.
CPUS="${CPUS:-96}"
SLURM_SCRIPT="${SLURM_SCRIPT:-slurm_script.sh}"
OUTPUT_LOGS="${OUTPUT_LOGS:-SLURM_LOGS}"
RESULTS="${RESULTS:-results}"
SEED="${SEED:-0}"

# Cosmo grid run000 Cosmology
# Cosmological parameters:
#     h:        0.73
#     Omega_b:  0.045
#     Omega_c:  0.25378877
#     Omega_k:  0.0
#     w0:       -1.1665
#     wa:       0.0
#     n:        0.97
#     sigma8:   0.9
#     Omega_nu: 0.0012112348

# CosmoGrid fiducial (== fli-simulate defaults; kept explicit for the record).
COSMO="--h 0.6774 --Omega-b 0.0486 --Omega-c 0.2589 --sigma8 0.8159 --n-s 0.9667 \
--Omega-k 0.0 --w0 -1.0 --wa 0.0 --Omega-nu 0.0"
# Cosmo grid run000 Cosmology
COSMOGRID_COSMO="--h 0.73 --Omega-b 0.045 --Omega-c 0.25378877 --sigma8 0.9 --n-s 0.97 \
--Omega-k 0.0 --w0 -1.1665 --wa 0.0 --Omega-nu 0.0012112348"

BOX2="2000.0 2000.0 2000.0"   # 2 Gpc/h accuracy box (Exp 1–7)

# launch <nodes> <gpus_per_node> <pdim_x> <pdim_y> <time> -- <fli-simulate args...>
# pdim_x * pdim_y must equal nodes * gpus_per_node (validated by fli-launcher).
launch() {
  local nodes=$1 gpn=$2 px=$3 py=$4 tlimit=$5; shift 5
  [ "$1" = "--" ] && shift
  # Skip if the run's output already exists (idempotent reruns): find --output/-o in the args
  # and bail out before submitting if that file is present and non-empty.
  local out="" prev=""
  for a in "$@"; do
    case "$prev" in --output|-o) out="$a"; break ;; esac
    prev="$a"
  done
  # A run is "done" if its --output is a non-empty file, OR (batched per-shell save) a directory
  # holding at least one parquet. `-s` is unreliable on directories, so branch on `-d` explicitly.
  if [ -n "$out" ]; then
    if [ -d "$out" ]; then
      if ls "$out"/*.parquet >/dev/null 2>&1; then
        echo "### SKIP (output dir has parquets): $out"
        return 0
      fi
    elif [ -s "$out" ]; then
      echo "### SKIP (output exists): $out"
      return 0
    fi
  fi
  fli-launcher --mode "$MODE" --account "$ACCOUNT" --constraint "$CONSTRAINT" \
    --nodes "$nodes" --gpus-per-node "$gpn" --cpus-per-node "$CPUS" --qos "$QOS" \
    --time-limit "$tlimit" --slurm-script "$SLURM_SCRIPT" --output-logs "$OUTPUT_LOGS" \
    --pdim "$px" "$py" -- fli-simulate "$@"
}
