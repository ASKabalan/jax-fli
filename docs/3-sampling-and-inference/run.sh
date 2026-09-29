#!/bin/bash
# Launch 15-LPT-Multihost-MassMapping.py on Jean Zay (H100, 4 GPUs per node).
# Each job sets its own nside / paint_nside / ell_max / ell_taper. An empty field lets the script derive it from
# the mesh (notebook 14's rule): ell_max = k_Nyq * chi at the lensing peak, taper 12.5% of it,
# paint_nside = smallest 2^k >= ell_max / 2, nside = 2 * paint_nside. At the 2048 scale the jobs pin
# nside 2048, paint_nside 1024, ell_max 1024, taper 128.
#
# GPU counts: 12 * nside^2 is 3 * 2^22 (paint nside 1024) or 3 * 2^24 (nside 2048), so only device counts of
# the form 2^a or 3 * 2^a divide the pixels -> 128, 192, 256 (192 replaces 150/200, which do not divide them).
#
# Halo: 4 sigma_disp of Exp 01 (sigma_disp ~ 10 Mpc/h, 3-D rms at z = 0) = 40 Mpc/h. The box at max_z 1 is
# 4595 Mpc/h -> 2.24 Mpc/h cells at mesh 2048 (17.8 cells), 2.18 at mesh 2112 (18.4 cells) -> 20 cells
# (44.9 and 43.5 Mpc/h, both >= 4 sigma).
# Pencil grids 16 x N keep every block wider than the 20-cell halo (P_X a multiple of 4 GPUs/node for the
# hybrid mesh). The 2 source bins do not divide N here, so the lensing replicates them over y (born warns):
#   128 GPUs: 16 x  8, mesh 2048 (local 128 x 256)
#   192 GPUs: 16 x 12, mesh 2112 (local 132 x 176; 2112 = 48 * 44, first multiple of lcm(16, 12) above 2048)
#   256 GPUs: 16 x 16, mesh 2048 (local 128 x 128)
#
# Usage:
#   MODE=dryrun bash run.sh            # print the sbatch commands, submit nothing
#   WHAT=pilot bash run.sh             # the small pilot: a 1-node memory probe and a 1-node Euclid run at mesh 128
#   bash run.sh                        # submit the memory probes (WHAT=probes default)
#   WHAT=runs bash run.sh              # submit the production run(s)
#   WHAT=all bash run.sh               # probes and production runs
#
# Stored 3-D IC: <paint_density>^3 in <precision> (per job). paint_density must divide by both pdims:
# 1024 on 16 x 8 / 16 x 16, 1056 on 16 x 12, 1008 on 16 x 24. Empty -> full mesh / the script's float32.
#
# Knobs (override via env): WHAT TIME PROBE_TIME HALO MAP2ALM MAP_MAX_ITER N_SNAPSHOTS SEED
#   SLURM_SCRIPT TRACES PYTHON EMAIL ACCOUNT QOS CPUS_PER_TASK
set -euo pipefail

cd "$(dirname "$0")"

MODE="${MODE:-sbatch}"
WHAT="${WHAT:-probes}"
SCRIPT="15-LPT-Multihost-MassMapping.py"

# --- cluster (template from the working submission command) ---
ACCOUNT="${ACCOUNT:-rzt@v100}"
CONSTRAINT="${CONSTRAINT:-v100-32g}"
QOS="${QOS:-qos_gpu-t3}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NTASKS_PER_NODE="${NTASKS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-10}"
TIME="${TIME:-2:00:00}"
PROBE_TIME="${PROBE_TIME:-1:00:00}"
EMAIL="${EMAIL:-}"
SLURM_SCRIPT="${SLURM_SCRIPT:-slurm_script.sh}"   # wrapper: $SLURM_SCRIPT TRACES python ARGS
TRACES="${TRACES:-traces}"
PYTHON="${PYTHON:-python}"

# --- physics knobs ---
MAP2ALM="${MAP2ALM:-jax_cuda}"
MAP_MAX_ITER="${MAP_MAX_ITER:-400}"
N_SNAPSHOTS="${N_SNAPSHOTS:-40}"
SEED="${SEED:-0}"
HALO="${HALO:-20}"
PILOT_TIME="${PILOT_TIME:-0:30:00}"

# <name>|<mesh>|<px>|<py>|<survey>|<time>|<nside>|<paint_nside>|<ell_max>|<ell_taper>|<paint_density>|<precision>|<extra flag>
# (empty nside / paint_nside / ell_max / ell_taper -> derived from the mesh by the script)
PILOT=(
  #"pilot_probe_g4_m128|512|4|4|euclid|${PILOT_TIME}|128|128|192|32||float32|--memory-only"
  "PILOT_MESH512_EUCLID|300|4|1|euclid|${PILOT_TIME}|128|128|192|32||float32|"
  #"PILOT_MESH512_DES|300|4|1|des|${PILOT_TIME}|128|128|192|32||float32|"
)
PROBES=(
  "probe_g128_m2048|2048|16|8|des|${PROBE_TIME}|1024|512|700|64|1024|float32|--memory-only"
  "probe_g192_m2112|2112|16|12|des|${PROBE_TIME}|1024|512|700|64|1056|float32|--memory-only"
  "probe_g256_m2048|2048|16|16|des|${PROBE_TIME}|1024|512|700|64|1024|float32|--memory-only"
)
PROD=(
  "MESH1600_DESY3|1600|16|16|des|${TIME}|1024|512|700|64|1024|float32|"
  "MESH1600_EUCLID|1600|16|16|euclid|${TIME}|1024|512|700|64|1024|float32|"
  # 16 x 24 (384 GPUs, mesh 1632, density 1008) is off: unequal mesh axes make the SPMD partitioner gather the
  # k-vectors of every FFT (12288 "Involuntary full rematerialization" warnings; 0 on 16 x 16)
  #"MESH1632_DESY3|1632|16|24|des|${TIME}|1024|512|700|64|1008|float32|"
  #"MESH1632_EUCLID|1632|16|24|euclid|${TIME}|1024|512|700|64|1008|float32|"
)

submit() {
  local name mesh px py survey time nside paint_nside ell_max ell_taper paint_density precision extra
  IFS="|" read -r name mesh px py survey time nside paint_nside ell_max ell_taper paint_density precision extra <<<"$1"
  local n_gpu=$((px * py))
  local nodes=$((n_gpu / GPUS_PER_NODE))
  local out_dir="results_15/${name}"
  mkdir -p "$out_dir"   # job output

  local cmd=(
    sbatch
    --account="$ACCOUNT"
    -C "$CONSTRAINT"
    --gres="gpu:${GPUS_PER_NODE}"
    --ntasks-per-node="$NTASKS_PER_NODE"
    --cpus-per-task="$CPUS_PER_TASK"
    --hint=nomultithread
    --qos="$QOS"
    --nodes="$nodes"
    --mail-type=BEGIN
  )
  [ -n "$EMAIL" ] && cmd+=(--mail-user="$EMAIL")
  cmd+=(
    --time="$time"
    --job-name="15map_${name}"
    "$SLURM_SCRIPT" "$TRACES" "$PYTHON" "$SCRIPT"
    --mesh "$mesh"
    --pdims "$px" "$py"
    --halo-cells "$HALO"
    --survey "$survey"
    --map2alm-method "$MAP2ALM"
    --map-max-iter "$MAP_MAX_ITER"
    --n-snapshots "$N_SNAPSHOTS"
    --seed "$SEED"
    --out-dir "$out_dir"
    --gpus-per-node "$GPUS_PER_NODE"
  )
  [ -n "$paint_density" ] && cmd+=(--paint-density "$paint_density")
  [ -n "$precision" ] && cmd+=(--density-precision "$precision")
  [ -n "$nside" ] && cmd+=(--nside "$nside")
  [ -n "$paint_nside" ] && cmd+=(--paint-nside "$paint_nside")
  [ -n "$ell_max" ] && cmd+=(--ell-max "$ell_max")
  [ -n "$ell_taper" ] && cmd+=(--ell-taper "$ell_taper")
  [ -n "$extra" ] && cmd+=("$extra")

  echo "### ${name}: ${n_gpu} GPUs (${nodes} nodes, pdims ${px}x${py}) mesh=${mesh} halo=${HALO}" \
    "nside=${nside:-auto} paint_nside=${paint_nside:-auto} ell_max=${ell_max:-auto} ell_taper=${ell_taper:-auto}" \
    "survey=${survey} density=${paint_density:-$mesh}^3 ${precision:-float32} ${extra}"
  if [ "$MODE" = dryrun ]; then
    echo "  ${cmd[*]}"
  else
    "${cmd[@]}"
  fi
}

case "$WHAT" in
  pilot) SPECS=("${PILOT[@]}") ;;
  probes) SPECS=("${PROBES[@]}") ;;
  runs) SPECS=("${PROD[@]}") ;;
  all) SPECS=("${PROBES[@]}" "${PROD[@]}") ;;
  *) echo "WHAT must be pilot, probes, runs or all (got '$WHAT')" >&2; exit 1 ;;
esac

for spec in ${SPECS[@]+"${SPECS[@]}"}; do
  submit "$spec"
done
