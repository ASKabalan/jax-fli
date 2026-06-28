#!/bin/bash
# Experiment 05d — spacing & stepping: D³ time stepping, 3-bin tomography. ⚠️ WIP
# Same 5 Gpc/h, 2560³, 3-bin Born as 05b/05c, but with uniform-D³ time stepping (--time-stepping D3),
# run over BOTH equal-volume AND scale-factor shell spacing — 2× the runs of 05b/05c. drift / no-drift
# × shell-count sweep, 50 BullFrog steps, float64.
source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 05d — spacing & stepping: D³ stepping, 3-bin (5 Gpc/h, 2560³)  (MODE=$MODE)"

BOX5="5000.0 5000.0 5000.0"     # 5 Gpc/h — deep enough for 3 tomographic source bins

# Sizing identical to 05b/05c (128 GPU slab, even halo int(20·0.5)=10, ≤512³ local, float64). The only
# change is the time-stepping schedule: --time-stepping D3 (uniform in D³, more steps at late times).
COMMON="--sim-mode pm --mesh-size 2560 2560 2560 --box-size $BOX5 --solver bf --nb-steps 50 --min-width 5.0 \
--paint-order cic --nside 2048 --shells-per-file 1 --scheme ngp --time-stepping D3 --halo-multiplier 0.5 \
--enable-x64 --perf --iterations 3 --seed $SEED $COSMO"

NB_SHELLS=(5 8 10 12 16 20 25 30 40)

# 2× the runs: D³ stepping over both equal-volume and scale-factor shell spacing.
for SPACE in equal_vol a; do
  if [ "$SPACE" = "equal_vol" ]; then sp=eqvol; else sp=a; fi
  for NB_S in "${NB_SHELLS[@]}"; do
    for DRIFT in "" "--drift-on-lightcone"; do
      if [ -n "$DRIFT" ]; then dt=drift; else dt=nodrift; fi
      tag="exp5d_${sp}_${dt}_${NB_S}"
      launch 32 4 128 1 01:00:00 -- $COMMON --shell-spacing $SPACE $DRIFT --nb-shells $NB_S \
        --output "$RESULTS/exp5d/${tag}" --name "${tag}_M%mesh_size%_s%seed%"
    done
  done
done

# 3-bin Born lensing on the published D³ density shells (both spacings; read back from HuggingFace).
launch_rt() {
  local account=$1 constraint=$2 qos=$3 nodes=$4 gpn=$5 px=$6 py=$7 tlimit=$8; shift 8
  [ "$1" = "--" ] && shift
  fli-launcher --mode "$MODE" --account "$account" --constraint "$constraint" \
    --nodes "$nodes" --gpus-per-node "$gpn" --cpus-per-node "$CPUS" --qos "$qos" \
    --time-limit "$tlimit" --slurm-script "$SLURM_SCRIPT" --output-logs "$OUTPUT_LOGS" \
    --pdim "$px" "$py" -- "$@"
}

for SPACE in equal_vol a; do
  if [ "$SPACE" = "equal_vol" ]; then sp=eqvol; else sp=a; fi
  for NB_S in "${NB_SHELLS[@]}"; do
    for dt in drift nodrift; do
      echo "Launching 3-bin Born (D³, $SPACE, $dt) for NB_S=$NB_S"
      launch_rt "$ACCOUNT" "$CONSTRAINT" "$QOS" 2 4 8 1 01:00:00 -- \
        fli-born-rt --repo ASKabalan/jax-fli-experiments \
        --data-files "05-spacing-n-stepping/05d-3bin-d3/density/exp5d_${sp}_${dt}_${NB_S}/shell*.parquet" \
        --nz-shear "s3[:3]" --nside 2048 --enable-x64 --normalization global \
        --name "kappa_d3_${sp}_${dt}_3bin_$NB_S" --output "$RESULTS/exp5d/born_${sp}_${dt}_${NB_S}"
    done
  done
done
