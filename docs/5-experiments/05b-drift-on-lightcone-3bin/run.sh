#!/bin/bash
# Experiment 05b — drift on the lightcone vs none, on a box deep enough for tomographic lensing.
# Sibling of Exp 05 (2 Gpc/h, 2048³, single point source): here a 5 Gpc/h box at 2560³ fits THREE source
# bins, so the drift improvement on thick density shells can be propagated into 3-bin Born convergence.
# float64. Same drift/no-drift × shell-count sweep, 50 BullFrog steps.
source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 05b — drift on the lightcone, 3-bin (5 Gpc/h, 2560³)  (MODE=$MODE)"

BOX5="5000.0 5000.0 5000.0"     # 5 Gpc/h — deep enough for 3 tomographic source bins (Exp 05 was 2 Gpc/h)

# 2560³ @ 128 GPUs (32 nodes, --pdim 128 1): local 20³-slab, halo int(20·0.5)=10 (EVEN), local
# 20·2560·2560 = 1.31e8 cells ≈ 512³ → fits float64. Ghost zone 0.5·5000/128 = 19.5 Mpc/h clears the
# end-of-run rms displacement. ngp spherical paint at nside 2048, one parquet per shell (gather OOMs otherwise).
COMMON="--sim-mode pm --mesh-size 2560 2560 2560 --box-size $BOX5 --solver bf --nb-steps 50 --min-width 5.0 \
--paint-order cic --nside 2048 --shells-per-file 1 --scheme ngp --shell-spacing a --time-stepping D --halo-multiplier 0.5 \
--enable-x64 --perf --iterations 3 --seed $SEED $COSMO"

NB_SHELLS=(5 8 10 12 16 20 25 30 40)

for NB_S in "${NB_SHELLS[@]}"; do
  for DRIFT in "" "--drift-on-lightcone"; do
    if [ -n "$DRIFT" ]; then tag="exp5b_drift_${NB_S}"; else tag="exp5b_nodrift_${NB_S}"; fi
    launch 32 4 128 1 01:00:00 -- $COMMON $DRIFT --nb-shells $NB_S \
      --output "$RESULTS/exp5b/${tag}" --name "${tag}_M%mesh_size%_s%seed%"
  done
done

# 3-bin Born lensing on the density shells (vs the single point source of Exp 05). fli-born-rt reads the
# published shells back from HuggingFace, so the density sweep above must be pushed to the dataset first
# (same lifecycle as Exp 05). s3[:3] = the three lowest-z Stage-3 source bins, which fit the 5 Gpc/h depth.
launch_rt() {
  local account=$1 constraint=$2 qos=$3 nodes=$4 gpn=$5 px=$6 py=$7 tlimit=$8; shift 8
  [ "$1" = "--" ] && shift
  fli-launcher --mode "$MODE" --account "$account" --constraint "$constraint" \
    --nodes "$nodes" --gpus-per-node "$gpn" --cpus-per-node "$CPUS" --qos "$qos" \
    --time-limit "$tlimit" --slurm-script "$SLURM_SCRIPT" --output-logs "$OUTPUT_LOGS" \
    --pdim "$px" "$py" -- "$@"
}

for NB_S in "${NB_SHELLS[@]}"; do
  echo "Launching 3-bin Born lensing (drift vs none) for NB_S=$NB_S"
  launch_rt "$ACCOUNT" "$CONSTRAINT" "$QOS" 2 4 8 1 01:00:00 -- \
    fli-born-rt --repo ASKabalan/jax-fli-experiments --data-files "05b-drift-3bin/density/exp5b_drift_${NB_S}/shell*.parquet" \
    --nz-shear "s3[:3]" --nside 2048 --enable-x64 --normalization global --name "kappa_drift_3bin_$NB_S" \
    --output "$RESULTS/exp5b/born_drift_${NB_S}"
  launch_rt "$ACCOUNT" "$CONSTRAINT" "$QOS" 2 4 8 1 01:00:00 -- \
    fli-born-rt --repo ASKabalan/jax-fli-experiments --data-files "05b-drift-3bin/density/exp5b_nodrift_${NB_S}/shell*.parquet" \
    --nz-shear "s3[:3]" --nside 2048 --enable-x64 --normalization global --name "kappa_nodrift_3bin_$NB_S" \
    --output "$RESULTS/exp5b/born_nodrift_${NB_S}"
done
