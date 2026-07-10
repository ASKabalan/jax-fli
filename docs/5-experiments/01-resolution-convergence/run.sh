#!/bin/bash
# Experiment 01 — resolution convergence.
# Per-shell spherical C_ℓ vs particle count, 512³ → 3072³ at fixed box/steps. float64 throughout.
source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 01 — resolution convergence  (MODE=$MODE)"

NB_SHELLS="${NB_SHELLS:-10}"   # lightcone shells (box/2 / N ≥ min_width 50 Mpc/h)
COMMON="--sim-mode pm --box-size $BOX2 --solver bf --nb-steps 50 --paint-order cic \
--nside 512 --scheme ngp --nb-shells $NB_SHELLS --shell-spacing comoving --enable-x64 --perf --iterations 3 --seed $SEED $COSMO"
NAME='exp1_M%mesh_size%_B%box_size%_STEPS%nb_steps%_s%seed%'

# GPU count per rung is the smallest px (py=1) with px | mesh and local = mesh/px a multiple of 4, so:
#   - local (per-device, unpadded) mesh <= 512³  (px >= mesh³/512³) -> fits a float64 H100;
#   - default halo = int((mesh/px)·0.5) = local/2 is EVEN  -> avoids the jaxpm slice_unpad crash on an
#     odd halo, and is >= local/4 (a decent halo). Padding is x-only (py=1 drops the y-halo via
#     get_halo_size), so the painted slab is (2·local_x, mesh, mesh).
#       nodes gpn  px  py  time      mesh                  # local  halo (=local/2)
launch   1  4    4  1  00:45:00 -- $COMMON --mesh-size 512  512  512  \
  --output "$RESULTS/exp1/m512.parquet"  --name "$NAME"    # 128   64
launch   2  4    8  1  00:45:00 -- $COMMON --mesh-size 1024 1024 1024 \
  --output "$RESULTS/exp1/m1024.parquet" --name "$NAME"    # 128   64
launch  16  4   64  1  00:45:00 -- $COMMON --mesh-size 2048 2048 2048 \
  --output "$RESULTS/exp1/m2048.parquet" --name "$NAME"    #  32   16
launch  32  4  128  1  00:45:00 -- $COMMON --mesh-size 2560 2560 2560 \
  --output "$RESULTS/exp1/m2560.parquet" --name "$NAME"    #  20   10   (was 256/64: odd halo 5)
launch  64  4  256  1  00:45:00 -- $COMMON --mesh-size 3072 3072 3072 \
  --output "$RESULTS/exp1/m3072.parquet" --name "$NAME"    #  12    6
launch  32  4  32  4  00:45:00 -- $COMMON --mesh-size 2560 2560 2560 \
  --output "$RESULTS/exp1/m2560_pencils.parquet" --name "$NAME"    #  80   40  (pencil px=32)
launch  64  4  64  4  00:45:00 -- $COMMON --mesh-size 3072 3072 3072 \
  --output "$RESULTS/exp1/m3072_pencils.parquet" --name "$NAME"    #  48   24  (pencil px=64)
