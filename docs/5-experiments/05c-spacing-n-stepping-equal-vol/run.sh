  #!/bin/bash
# Experiment 05c — spacing & stepping: equal-volume shells, 3-bin tomography. ⚠️ WIP
# Sibling of 05b (same 5 Gpc/h, 2560³, 3-bin Born, BullFrog, 50 steps, float64) but with
# EQUAL-VOLUME shell spacing instead of scale-factor spacing — isolating the near-shell shot-noise
# lever. Same drift / no-drift × shell-count sweep.
source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 05c — spacing & stepping: equal-volume, 3-bin (5 Gpc/h, 2560³)  (MODE=$MODE)"

BOX5="5000.0 5000.0 5000.0"     # 5 Gpc/h — deep enough for 3 tomographic source bins
SIM_MODE="${SIM_MODE:-DENSITY}"  # DENSITY → run the density-shell sweep; anything else → the Born lensing

# 2560³ @ 128 GPUs (32 nodes, --pdim 128 1): local 20³-slab, halo int(20·0.5)=10 (EVEN), local
# 20·2560·2560 = 1.31e8 cells ≈ 512³ → fits float64. Ghost zone 0.5·5000/128 = 19.5 Mpc/h clears the
# end-of-run rms displacement. ngp spherical paint at nside 2048, one parquet per shell (gather OOMs).
# Only change vs 05b: --shell-spacing equal_vol (the shot-noise lever). --min-width 60 floors the thin
# outer equal-volume shells (hybrid: equal-vol inner, >=60 Mpc/h outer); fits the whole sweep (r_max=2500,
# so 40*60<2500), a no-op for nb-shells<=12 and floors the razor-thin far shells for nb-shells>=16.
COMMON="--sim-mode pm --mesh-size 2560 2560 2560 --box-size $BOX5 --solver bf --nb-steps 50 --min-width 60.0 \
--paint-order cic --nside 2048 --shells-per-file 1 --scheme ngp --shell-spacing equal_vol --time-stepping D --halo-multiplier 0.5 \
--enable-x64 --perf --iterations 3 --seed $SEED $COSMO"

NB_SHELLS=(5 8 10 12 16 20 25 30 40)

if [ "$SIM_MODE" = "DENSITY" ]; then
  for NB_S in "${NB_SHELLS[@]}"; do
    for DRIFT in "" "--drift-on-lightcone"; do
      if [ -n "$DRIFT" ]; then tag="exp5c_drift_${NB_S}"; else tag="exp5c_nodrift_${NB_S}"; fi
      launch 32 4 128 1 01:00:00 -- $COMMON $DRIFT --nb-shells $NB_S \
        --output "$RESULTS/exp5c/${tag}" --name "${tag}_M%mesh_size%_s%seed%"
    done
  done
else
  # 3-bin Born lensing on the published density shells (read back from HuggingFace; push the density
  # sweep above to the dataset first, same lifecycle as 05a/05b). s3[:3] = the three lowest-z Stage-3
  # source bins, matched to the 5 Gpc/h depth.
  launch_rt() {
    local account=$1 constraint=$2 qos=$3 nodes=$4 gpn=$5 px=$6 py=$7 tlimit=$8; shift 8
    [ "$1" = "--" ] && shift
    fli-launcher --mode "$MODE" --account "$account" --constraint "$constraint" \
      --nodes "$nodes" --gpus-per-node "$gpn" --cpus-per-node "$CPUS" --qos "$qos" \
      --time-limit "$tlimit" --slurm-script "$SLURM_SCRIPT" --output-logs "$OUTPUT_LOGS" \
      --pdim "$px" "$py" -- "$@"
  }

  # Two quadratures per run: 'midpoint' (the historic center-evaluated kernel) and 'gauss_legendre'
  # (exact per-shell kernel integral — the equal-volume fat inner ball makes midpoint overshoot the
  # low-z bins; see docs/WORK_IN_PROGRESS/22-born-quadrature-equal-vol.ipynb). Published layout:
  # kappa_midpoint/ + spectra_midpoint/ and kappa_gl/ + spectra_gl/.
  for NB_S in "${NB_SHELLS[@]}"; do
    for KIND in drift nodrift; do
      echo "Launching 3-bin Born lensing ($KIND, midpoint + gauss_legendre) for NB_S=$NB_S"
      DATA="05-spacing-n-stepping/05c-equal-volume/density/exp5c_${KIND}_${NB_S}/shell*.parquet"
      launch_rt "$ACCOUNT" "$CONSTRAINT" "$QOS" 2 4 8 1 01:00:00 -- \
        fli-born-rt --repo ASKabalan/jax-fli-experiments --data-files "$DATA" \
        --nz-shear "s3[:3]" --nside 2048 --enable-x64 --normalization global --quadrature midpoint \
        --name "kappa_midpoint_${KIND}_3bin_$NB_S" --output "$RESULTS/exp5c/born_midpoint_${KIND}_${NB_S}"
      launch_rt "$ACCOUNT" "$CONSTRAINT" "$QOS" 2 4 8 1 01:00:00 -- \
        fli-born-rt --repo ASKabalan/jax-fli-experiments --data-files "$DATA" \
        --nz-shear "s3[:3]" --nside 2048 --enable-x64 --normalization global --quadrature gauss_legendre \
        --name "kappa_gl_${KIND}_3bin_$NB_S" --output "$RESULTS/exp5c/born_gl_${KIND}_${NB_S}"
    done
  done
fi
