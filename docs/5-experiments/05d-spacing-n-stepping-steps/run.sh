#!/bin/bash
# Experiment 05d — step & stepping convergence at the production geometry (equal-volume, 3-bin).
# Sibling of 05c (same 5 Gpc/h box, 2560³, 20-shell, drift-on-lightcone, equal_vol, GL Born): the ONLY
# moving parts are --nb-steps {20,30,40,50} and --time-stepping {a,bfd uses D}. Exp 04 established the
# step budget on the 2 Gpc/h accuracy box with per-shell density C_ℓ; this re-asks it at the
# production geometry with the 3-bin Born κ as the endpoint. The bfd-50 point is NOT re-run — it is
# exactly 05c's exp5c_drift_20 (density AND its published spectra_gl_drift_20.parquet), the shared
# 50-step anchor of 05d and 05e. float64, --perf --iterations 3 on both pipeline phases.
source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 05d — steps × stepping, equal-volume 3-bin (5 Gpc/h, 2560³)  (MODE=$MODE)"

BOX5="5000.0 5000.0 5000.0"
SIM_MODE="${SIM_MODE:-DENSITY}"  # DENSITY → the step sweep; anything else → the 3-bin Born pass

# Same COMMON as 05c's drift runs: 2560³ @ 128 GPUs (32 nodes, --pdim 128 1) gives a local 20³-slab
# with an EVEN halo int(20·0.5)=10 cells = 19.5 Mpc/h (clears the end-of-run rms displacement) and
# 20·2560·2560 ≈ 512³ local cells → fits float64. Only --nb-steps / --time-stepping are free.
COMMON="--sim-mode pm --mesh-size 2560 2560 2560 --box-size $BOX5 --solver bf --min-width 60.0 \
--paint-order cic --nside 2048 --shells-per-file 1 --scheme ngp --shell-spacing equal_vol \
--drift-on-lightcone --halo-multiplier 0.5 --enable-x64 --perf --iterations 3 --seed $SEED $COSMO"

# nb-steps >= nb-shells is a hard floor (Exp 04): each shell target consumes at least one clipped
# step, so below 20 steps the runs fail to integrate. The ladder starts at 20.
NB_STEPS=(20 30 40 50)

# Every run is capped at 40 min: the 05c perf anchor (the same 2560³/50-step configuration) records
# ≈49 s per simulation iteration + ≈1 min JIT on 128 GPUs, so all step counts sit far below the cap.
NB_STEPS_LIMIT="00:40:00"

if [ "$SIM_MODE" = "DENSITY" ]; then
  for TS in a D; do
    for NS in "${NB_STEPS[@]}"; do
      if [ "$TS" = "D" ] && [ "$NS" = "50" ]; then
        echo "### SKIP bfd_50 (identical to 05c's exp5c_drift_20 — reused as the shared anchor)"
        continue
      fi
      tag="exp5d_bf${TS}_${NS}"
      launch 32 4 128 1 "$NB_STEPS_LIMIT" -- $COMMON --time-stepping "$TS" --nb-steps "$NS" \
        --output "$RESULTS/exp5d/density/${tag}" --name "${tag}_M%mesh_size%_s%seed%"
    done
  done
else
  # 3-bin Born lensing on the published density shells (read back from HuggingFace; push the density
  # sweep above to the dataset first, same lifecycle as 05a/05b/05c). Quadrature is gauss_legendre —
  # 05c showed the midpoint weight breaks on equal-volume's fat inner shell. The bfd_50 Born spectra
  # also already exist (05c's spectra_gl_drift_20.parquet) and are NOT recomputed.
  launch_rt() {
    local account=$1 constraint=$2 qos=$3 nodes=$4 gpn=$5 px=$6 py=$7 tlimit=$8; shift 8
    [ "$1" = "--" ] && shift
    fli-launcher --mode "$MODE" --account "$account" --constraint "$constraint" \
      --nodes "$nodes" --gpus-per-node "$gpn" --cpus-per-node "$CPUS" --qos "$qos" \
      --time-limit "$tlimit" --slurm-script "$SLURM_SCRIPT" --output-logs "$OUTPUT_LOGS" \
      --pdim "$px" "$py" -- "$@"
  }

  for TS in a D; do
    for NS in "${NB_STEPS[@]}"; do
      if [ "$TS" = "D" ] && [ "$NS" = "50" ]; then
        echo "### SKIP born bfd_50 (already published as 05c's spectra_gl_drift_20.parquet)"
        continue
      fi
      tag="exp5d_bf${TS}_${NS}"
      DATA="05-spacing-n-stepping/05d-steps/density/exp5d_bf${TS}_${NS}/shell*.parquet"
      echo "Launching 3-bin Born lensing (gauss_legendre) for $tag"
      launch_rt "$ACCOUNT" "$CONSTRAINT" "$QOS" 2 4 8 1 00:40:00 -- \
        fli-born-rt --repo ASKabalan/jax-fli-experiments --data-files "$DATA" \
        --nz-shear "s3[:3]" --nside 2048 --enable-x64 --normalization global --quadrature gauss_legendre \
        --perf --iterations 3 \
        --name "kappa_gl_bf${TS}_3bin_${NS}" --output "$RESULTS/exp5d/kappa_gl/born_gl_bf${TS}_${NS}"
    done
  done
fi
