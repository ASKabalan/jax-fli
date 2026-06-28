#!/bin/bash
# Experiment 05a — spacing & stepping: drift on the lightcone vs none (thick shells; density-shell C_ℓ).
# Single point source, 2 Gpc/h, 2560³, float64. The deeper 3-bin counterparts are 05b/05c/05d.
source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 05a — drift on the lightcone  (MODE=$MODE)"

NB_SHELLS="${NB_SHELLS:-8}"    # thick shells — the point of Exp 05a
SIM_MODE="${SIM_MODE:-DENSITY}"  # produce the density shells
# Production knobs (see Exp 01/02/03/06): 2048³ mesh, CIC paint with NO force --deconvolution, ngp
# spherical painting at nside 2048, one parquet per shell (--shells-per-file 1; required at nside 2048,
# the full-lightcone gather onto rank 0 OOMs — see Exp 06). The shared launch() treats a parquet-filled
# --output directory as "done".
COMMON="--sim-mode pm --mesh-size 2048 2048 2048 --box-size $BOX2 --solver bf --nb-steps 50 --min-width 5.0 \
--paint-order cic --nside 2048 --shells-per-file 1 --scheme ngp --shell-spacing a --time-stepping D --halo-multiplier 0.5 \
--enable-x64 --perf --iterations 3 --seed $SEED $COSMO"

# 2048³ @ 64 GPUs (16 nodes, --pdim 64 1): local 32³, halo int(32·0.5)=16 (EVEN), local ≤ 512³ → fits f64.
# Halo sizing (Exp 01 max-drift rule): ghost zone = halo_multiplier·box/px = 0.5·2000/64 = 15.6 Mpc/h vs
# the end-of-run 3D rms displacement σ₃D(z=0) = 10.2 Mpc/h → halo/σ = 1.53 ≥ 1.5 ✅ (the converged Exp 01
# rung). Drift-on-lightcone repaints existing particles, so the displacement scale — and this check — is
# unchanged.

NB_SHELLS=(5 8 10 12 16 20 25 30 40)


if [ "$SIM_MODE" = "DENSITY" ]; then
  for NB_S in "${NB_SHELLS[@]}"; do
    for DRIFT in "" "--drift-on-lightcone"; do
      if [ -n "$DRIFT" ]; then tag="exp5a_drift_${NB_S}"; else tag="exp5a_nodrift_${NB_S}"; fi
      launch 16 4 64 1 00:40:00 -- $COMMON $DRIFT --nb-shells $NB_S \
        --output "$RESULTS/exp5a/${tag}" --name "${tag}_M%mesh_size%_s%seed%"
    done
  done
else
  launch_rt() {
    local account=$1 constraint=$2 qos=$3 nodes=$4 gpn=$5 px=$6 py=$7 tlimit=$8; shift 8
    [ "$1" = "--" ] && shift
    fli-launcher --mode "$MODE" --account "$account" --constraint "$constraint" \
      --nodes "$nodes" --gpus-per-node "$gpn" --cpus-per-node "$CPUS" --qos "$qos" \
      --time-limit "$tlimit" --slurm-script "$SLURM_SCRIPT" --output-logs "$OUTPUT_LOGS" \
      --pdim "$px" "$py" -- "$@"
  }

  for NB_S in "${NB_SHELLS[@]}"; do
    echo "Launching drift-on-lightcone vs none for NB_S=$NB_S"
    launch_rt "$ACCOUNT" "$CONSTRAINT" "$QOS" 2 4 8 1 00:40:00 -- \
      fli-born-rt --repo ASKabalan/jax-fli-experiments --data-files "05-spacing-n-stepping/05a-drift/density/exp5a_drift_${NB_S}/shell*.parquet" \
      --nz-shear 0.35 --nside 2048 --enable-x64 --normalization global  --name "kappa_drift_shells_$NB_S" \
      --output "$RESULTS/exp5a/born_drift_${NB_S}"
    launch_rt "$ACCOUNT" "$CONSTRAINT" "$QOS" 1 1 1 1 00:40:00 -- \
      fli-born-rt --repo ASKabalan/jax-fli-experiments --data-files "05-spacing-n-stepping/05a-drift/density/exp5a_nodrift_${NB_S}/shell*.parquet" \
      --nz-shear 0.35 --nside 2048 --enable-x64 --normalization global --name "kappa_nodrift_shells_$NB_S" \
      --output "$RESULTS/exp5a/born_nodrift_${NB_S}"
  done
fi

# The deeper 3-bin tomographic counterpart (5 Gpc/h box, 2560³, 3 source bins) is its own experiment:
# see ../05b-spacing-n-stepping-3bin/.
