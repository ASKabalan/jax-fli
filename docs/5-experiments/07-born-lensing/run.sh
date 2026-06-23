#!/bin/bash
# Experiment 07 — Born lensing on the CosmoGrid density shells (exp 06), straight from HuggingFace.
# For each of the 8 published nside-2048 density lightcones, run the Born approximation for BOTH the
# Stage-3 and DES Y3 source distributions, sliced to the lightcone's tomographic depth
# (2-bin -> first 2 source bins, 3-bin -> first 3). float64.
source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 07 — Born lensing from HuggingFace  (MODE=$MODE)"

REPO="${REPO:-ASKabalan/jax-fli-experiments}"
DENSITY_ROOT="06-cosmogrid-shells/density"

# Born reads the stored density lightcone and integrates once — geometry-agnostic (no mesh/box/observer).
# nside-2048 x ~46 float64 shells ~= 18 GB; --pdim "8 1" slabs it across 2 nodes x 4 H100.
BORN_COMMON="--min-z 0.01 --n-integrate 32 --lensing-output convergence \
  --normalization global --enable-x64"

# --max-z is tied to each lightcone's far edge (06-cosmogrid-shells/geometry.sh): the matter stops at
# the last shell, so integrating the source n(z) past it would just truncate the high-z tail with zero
# lensing and bias the comparison. 2-bin shells reach a=0.5564 (z=0.80), 3-bin a=0.4929 (z=1.03);
# round up to the exp06-documented depths 0.82 / 1.06 to fully include the last shell.
#
# folder  nbins  maxz   (nbins drives the source slice "[:nbins]")
RUNS="
cosmogrid_2bin_fullsky_slab    2  0.82
cosmogrid_2bin_fullsky_pencil  2  0.82
cosmogrid_2bin_quadrant_slab   2  0.82
cosmogrid_2bin_quadrant_pencil 2  0.82
cosmogrid_3bin_fullsky_slab    3  1.06
cosmogrid_3bin_fullsky_pencil  3  1.06
cosmogrid_3bin_quadrant_slab   3  1.06
cosmogrid_3bin_quadrant_pencil 3  1.06
"

born_launch() {   # born_launch <folder> <nbins> <maxz> <source:s3|des_y3>
  local folder=$1 nbins=$2 maxz=$3 src=$4
  local out="$RESULTS/exp7/$folder/$src"
  # idempotent rerun: skip if this run already produced a parquet (matches the launch helper).
  if ls "$out"/*.parquet >/dev/null 2>&1; then
    echo "### SKIP (output exists): $out"; return 0
  fi
  fli-launcher --mode "$MODE" --account "$ACCOUNT" --constraint "$CONSTRAINT" \
    --nodes 2 --gpus-per-node 4 --cpus-per-node "$CPUS" --qos "$QOS" \
    --time-limit 00:30:00 --slurm-script "$SLURM_SCRIPT" --output-logs "$OUTPUT_LOGS" \
    --pdim 8 1 -- fli-born-rt $BORN_COMMON --max-z "$maxz" \
    --repo "$REPO" --data-files "$DENSITY_ROOT/$folder/*parquet" \
    --nz-shear "${src}[:${nbins}]" \
    --output "$out"
}

while read -r folder nbins maxz; do
  [ -z "$folder" ] && continue
  born_launch "$folder" "$nbins" "$maxz" s3
  born_launch "$folder" "$nbins" "$maxz" des_y3
done <<< "$RUNS"
