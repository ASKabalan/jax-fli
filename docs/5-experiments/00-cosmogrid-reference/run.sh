#!/bin/bash
# Experiment 00 — CosmoGrid reference: compute the Born and ray-traced κ from the published density.
#
# Drives the lensing CLI entry points through fli-launcher (like the other experiments), streaming the
# 56 nside-2048 density shells straight from HuggingFace (no snapshot):
#   fli-born-rt    -> Born convergence       (GPU / JAX-distributed; sharded over the device mesh)
#   fli-dorian-rt  -> dorian ray-traced κ    (single-process numpy; MPI parallelism deferred)
# Each writes a κ parquet into its --output directory; publish afterwards with `python publish_local.py --yes`.
#
#   MODE=dryrun bash run.sh   # print the resolved fli-launcher / sbatch commands, submit nothing
#   MODE=sbatch bash run.sh   # submit to SLURM (default)
set -euo pipefail

source "$(dirname "$0")/../_launch_common.sh"   # MODE, ACCOUNT, CONSTRAINT, QOS, CPUS, SLURM_SCRIPT, OUTPUT_LOGS, RESULTS

echo "### Exp 00 — CosmoGrid reference lensing  (MODE=$MODE)"

# Offline cluster (Jean Zay has no internet on compute nodes)? Pre-cache the density on a LOGIN node
# (`HF_HOME=$WORK/hf_cache python download.py`), then on compute nodes:
#   export HF_HOME=$WORK/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

# Streamed HF density source (one row per shell; the glob string is kept as a single arg, not shell-globbed).
SRC=(--repo ASKabalan/jax-fli-experiments --data-files "00-cosmogrid/density/cosmogrid_density_nside2048_shell*.parquet")

# CPU knobs for the (single-process) dorian jobs — override per cluster.
CPU_ACCOUNT="${CPU_ACCOUNT:-rzt@cpu}"
CPU_QOS="${CPU_QOS:-qos_cpu-t3}"

# launch_rt <account> <constraint> <qos> <nodes> <gpus_per_node> <pdim_x> <pdim_y> <time> -- <cmd...>
# Calls fli-launcher directly (the shared launch() helper is hardcoded to fli-simulate). fli-launcher
# validates gpus_per_node*nodes == pdim_x*pdim_y and appends --nodes/--gpus-per-node/--pdim to <cmd>.
launch_rt() {
  local account=$1 constraint=$2 qos=$3 nodes=$4 gpn=$5 px=$6 py=$7 tlimit=$8; shift 8
  [ "$1" = "--" ] && shift
  fli-launcher --mode "$MODE" --account "$account" --constraint "$constraint" \
    --nodes "$nodes" --gpus-per-node "$gpn" --cpus-per-node "$CPUS" --qos "$qos" \
    --time-limit "$tlimit" --slurm-script "$SLURM_SCRIPT" --output-logs "$OUTPUT_LOGS" \
    --pdim "$px" "$py" -- "$@"
}

# --- Born κ (GPU / JAX-distributed): 8 h100 nodes × 4 = 32 devices, npix sharded (pdim "32 1") --------
# Source n(z): Stage-3 (s3) and DES Y3 (des_y3). nside 2048 native; float64; global normalization.
launch_rt "$ACCOUNT" "$CONSTRAINT" "$QOS" 8 4 32 1 01:00:00 -- \
  fli-born-rt "${SRC[@]}" --nz-shear s3     --nside 2048 --enable-x64 --normalization global \
  --output "$RESULTS/exp0/born_s3"
launch_rt "$ACCOUNT" "$CONSTRAINT" "$QOS" 8 4 32 1 01:00:00 -- \
  fli-born-rt "${SRC[@]}" --nz-shear des_y3 --nside 2048 --enable-x64 --normalization global \
  --output "$RESULTS/exp0/born_des"

# --- Ray-traced κ (single-process dorian): CPU, 1 task (--pdim 1 1, ignored by dorian) ----------------
# Holds the FULL nside-2048 lightcone in host RAM (~14 GB) -> a fat-RAM node; --nside downsamples if needed.
# --with-born also emits the Born byproduct from the same dorian pass (a cross-check, not published).
launch_rt "$CPU_ACCOUNT" cpu "$CPU_QOS" 1 1 1 1 10:00:00 -- \
  fli-dorian-rt "${SRC[@]}" --nz-shear s3     --nside 2048 --rt-interp bilinear --with-born \
  --output "$RESULTS/exp0/raytrace_s3"
launch_rt "$CPU_ACCOUNT" cpu "$CPU_QOS" 1 1 1 1 10:00:00 -- \
  fli-dorian-rt "${SRC[@]}" --nz-shear des_y3 --nside 2048 --rt-interp bilinear --with-born \
  --output "$RESULTS/exp0/raytrace_des"

# Then publish from a node with internet + HF_TOKEN:
#   RESULTS="$RESULTS" python "$(dirname "$0")/publish_local.py" --yes
