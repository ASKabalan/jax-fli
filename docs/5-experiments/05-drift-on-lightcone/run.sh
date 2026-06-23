#!/bin/bash
# Experiment 05 — drift on the lightcone vs none (thick shells; density-shell C_ℓ). float64.
source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 05 — drift on the lightcone  (MODE=$MODE)"

NB_SHELLS="${NB_SHELLS:-8}"    # thick shells — the point of Exp 05
# Production knobs (see Exp 01/02/03/06): 2048³ mesh, CIC paint with NO force --deconvolution, ngp
# spherical painting at nside 2048, one parquet per shell (--shells-per-file 1; required at nside 2048,
# the full-lightcone gather onto rank 0 OOMs — see Exp 06). The shared launch() treats a parquet-filled
# --output directory as "done".
COMMON="--sim-mode pm --mesh-size 2048 2048 2048 --box-size $BOX2 --solver bf --nb-steps 50 \
--paint-order cic --nside 2048 --shells-per-file 1 --scheme ngp --shell-spacing a --time-stepping D --halo-multiplier 0.5 \
--enable-x64 --perf --iterations 3 --seed $SEED $COSMO"

# 2048³ @ 64 GPUs (16 nodes, --pdim 64 1): local 32³, halo int(32·0.5)=16 (EVEN), local ≤ 512³ → fits f64.
# Halo sizing (Exp 01 max-drift rule): ghost zone = halo_multiplier·box/px = 0.5·2000/64 = 15.6 Mpc/h vs
# the end-of-run 3D rms displacement σ₃D(z=0) = 10.2 Mpc/h → halo/σ = 1.53 ≥ 1.5 ✅ (the converged Exp 01
# rung). Drift-on-lightcone repaints existing particles, so the displacement scale — and this check — is
# unchanged.
for DRIFT in "" "--drift-on-lightcone"; do
  if [ -n "$DRIFT" ]; then tag="exp5_drift"; else tag="exp5_nodrift"; fi
  launch 16 4 64 1 00:40:00 -- $COMMON $DRIFT --nb-shells $NB_SHELLS \
    --output "$RESULTS/exp5/${tag}" --name "${tag}_M%mesh_size%_s%seed%"
done

# Double the shells
launch 16 4 64 1 00:40:00 -- $COMMON --nb-shells 16 \
    --output "$RESULTS/exp5/${tag}" --name "${tag}_M%mesh_size%_s%seed%"
