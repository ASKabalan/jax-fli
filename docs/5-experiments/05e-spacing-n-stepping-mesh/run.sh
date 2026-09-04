#!/bin/bash
# Experiment 05e — mesh ladder at the production 50-step point (equal-volume, 3-bin, drift).
# Holds the 05c drift-anchor physics fixed (bf, D-stepping, 50 steps, equal_vol, min-60, 20 shells,
# nside 2048, drift-on-lightcone) and moves the mesh 512³ → 4096³, asking where the 3-bin Born κ
# stops improving with resolution. All runs are SLABS (no pencils): the ghost zone is sized by the
# Exp-01 rule — pad = hm·box/px rounded up to the next even cell count, target ≥ 1.5 σ_disp — see
# the README table. The 2560³ point is NOT re-run: it is exactly 05c's exp5c_drift_20.
# float64, --perf --iterations 3 on both pipeline phases.
source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 05e — mesh ladder, equal-volume 3-bin (5 Gpc/h, 50 steps)  (MODE=$MODE)"

BOX5="5000.0 5000.0 5000.0"
SIM_MODE="${SIM_MODE:-DENSITY}"  # DENSITY → the mesh ladder; anything else → the 3-bin Born pass

# Shared physics = 05c's drift anchor; --mesh-size and --halo-multiplier are per-run (below).
COMMON="--sim-mode pm --box-size $BOX5 --solver bf --nb-steps 50 --time-stepping D --min-width 60.0 \
--paint-order cic --nside 2048 --shells-per-file 1 --scheme ngp --shell-spacing equal_vol \
--drift-on-lightcone --enable-x64 --perf --iterations 3 --seed $SEED $COSMO"

#        mesh  px(GPUs)  nodes  hm
RUNS=(
  "512    4     1    0.5"
  "1024   8     2    0.5"
  "2048   64    16   0.5"
  "3072   256   64   1.0"
  "4096   512   128  1.5"
)

# Every run is capped at 40 min: the 05c anchor (2560³, 50 steps) records ≈49 s per iteration + ≈1 min
# JIT on 128 GPUs, and per-step time grows only with the padded cells/GPU — far below the cap even at
# 4096³. GPU counts follow the Exp-01 sizing (smallest px keeping the unpadded local mesh ≤ 512³ on a
# float64 H100); halo pads in Mpc/h: 625 / 312 / 39 / 19.5 (anchor) / 19.5 / 14.6. The 4096³ line
# (512 GPUs = 128 nodes, the QOS cap) is memory-tight, not time-tight — MODE=dryrun it first; if it
# OOMs at step 1, the documented fallback is --halo-multiplier 1.75 (see README caveat).
RUN_LIMIT="00:40:00"

if [ "$SIM_MODE" = "DENSITY" ]; then
  echo "### 2560³ is not re-run — exp5c_drift_20 is the shared anchor (published under 05c-equal-volume)"
  for r in "${RUNS[@]}"; do
    read -r M PX NODES HM <<< "$r"
    tag="exp5e_m${M}"
    launch "$NODES" 4 "$PX" 1 "$RUN_LIMIT" -- $COMMON --mesh-size "$M" "$M" "$M" --halo-multiplier "$HM" \
      --output "$RESULTS/exp5e/density/${tag}" --name "${tag}_M%mesh_size%_s%seed%"
  done
else
  # 3-bin Born lensing on the published density shells (read back from HuggingFace; push the density
  # ladder above to the dataset first, same lifecycle as 05a/05b/05c). gauss_legendre quadrature, as
  # 05c. The 2560³ Born maps + spectra already exist (05c's kappa_gauss_legendre + spectra) and are
  # NOT recomputed.
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
    DATA="05-spacing-n-stepping/05e-mesh/density/exp5e_m${M}/shell*.parquet"
    echo "Launching 3-bin Born lensing (gauss_legendre) for exp5e_m${M}"
    launch_rt "$ACCOUNT" "$CONSTRAINT" "$QOS" 2 4 8 1 00:40:00 -- \
      fli-born-rt --repo ASKabalan/jax-fli-experiments --data-files "$DATA" \
      --nz-shear "s3[:3]" --nside 2048 --enable-x64 --normalization global --quadrature gauss_legendre \
      --perf --iterations 3 \
      --name "kappa_gl_m${M}" --output "$RESULTS/exp5e/kappa_gl/born_gl_m${M}"
  done
fi
