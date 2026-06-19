#!/bin/bash
# Experiment 03 — spherical painting scheme + pixel-window (depends on Exp 2 winner: TSC + deconv).
# scheme ∈ {ngp, bilinear, rbf@0.8px, rbf@1.5px} × {native 1024 vs paint@2048→1024} = 8 sims. float64.
# (HEALPix pixel-window deconvolution is applied AFTER, in fli-spectra / the study notebook.)
source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 03 — spherical painting + pixel-window  (MODE=$MODE)"

NB_SHELLS="${NB_SHELLS:-10}"   # lightcone shells (box/2 / N ≥ min_width 50 Mpc/h)
COMMON="--sim-mode pm --mesh-size 2048 2048 2048 --box-size $BOX2 --solver bf --nb-steps 50 \
--paint-order tsc --deconvolution --nb-shells $NB_SHELLS --shell-spacing comoving --enable-x64 --perf --iterations 3 --seed $SEED $COSMO"

emit3() { # $1=scheme  $2=kernel-width-pixels ("" if none)  $3=nside  $4=paint-flag  $5=tag
  local kw=""; [ -n "$2" ] && kw="--kernel-width-pixels $2"
  launch 8 4 8 4 02:00:00 -- $COMMON --nside "$3" $4 --scheme "$1" $kw \
    --output "$RESULTS/exp3/$5.parquet" --name "$5_M%mesh_size%_s%seed%"
}

for cfg in "ngp::" "bilinear::" "rbf_neighbor:0.8:rbf08" "rbf_neighbor:1.5:rbf15"; do
  IFS=: read -r scheme kw tag <<<"$cfg"; tag="${tag:-$scheme}"
  emit3 "$scheme" "$kw" 1024 ""                   "exp3_${tag}_native1024"
  emit3 "$scheme" "$kw" 2048 ""                   "exp3_${tag}_paint2048"
done
  