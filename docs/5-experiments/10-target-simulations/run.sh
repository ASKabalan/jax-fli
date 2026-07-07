#!/bin/bash
# Experiment 10 — target simulations for inference (Exp 13 & 14).
# Four density-shell lightcone runs on the big-quadrant 3-bin box: an LPT->PM comparison trio at
# ~1024^3 (64 GPU) plus the production PM target sim at ~2560^3 (256 GPU). float64; one parquet/shell.
#
# Physics recipe (provenance):
#   quadrant sky       (Exp 06): --observer-position 0.1 0.5 0.9, box 3000x5000x3000 (1.2r,2r,1.2r; r=2500)
#   30 steps, bf + D   (Exp 04): --nb-steps 30 --solver bf --time-stepping D            (the "bfd" pairing)
#   RBF spherical paint(Exp 03): --nside 2048 --scheme rbf_neighbor --kernel-width-pixels 0.8
#   CIC, no deconv     (Exp 02): --paint-order cic  (no --deconvolution)
#   2560^3 resolution  (Exp 01): production mesh ~2560^3-equiv on this box
#   equal-vol + drift  (Exp 05c): --shell-spacing equal_vol --nb-shells 20 --min-width 60.0 --drift-on-lightcone
#
# 1-D SLAB decomposition (pdim "P 1"): only axis 0 (X) is persistently sharded, so only the X-halo is
# padded (py=1 drops the Y-halo). The distributed FFT all-to-all transposes X<->Y, so BOTH Nx AND Ny
# must be divisible by P. Isotropic mesh for the anisotropic box (aspect 1.2:2:1.2 = 3:5:3), Nx=Nz, Ny=(5/3)Nx.
# HALO: the nbody state is the displacement from the initial grid (particles are NOT re-homed), so the X-halo
# must cover a particle's TOTAL X-displacement (~6 Mpc/h 1-D rms at z=0); mass past it is silently mis-painted
# (jnp.mod wrap). Exp 06 sized this ~16 Mpc/h. Only the EVENness of halo=int(Nx/P*hm) is required (odd ->
# slice_unpad crash); the halo MAY exceed the local slab width, so hm can go above 1.0 as needed.
#   ~1024^3 @ 64 GPU  (16 nodes x 4, pdim 64 1): mesh 960 x1600 x960  (960=64*15, 1600=64*25; EXACT 3:5:3)
#       -> 1.47e9 ~= 1138^3 (+11% linear / +37% particles vs 1024^3, 0% aniso). local_x 15, cell 3.125;
#          hm 0.4 -> halo 6 = 18.75 Mpc/h (~3 sigma). NB default hm 0.5 -> odd halo 7; 0.4 keeps it even.
#   ~2560^3 @ 256 GPU (64 nodes x 4, pdim 256 1): mesh 2304x3840x2304 (2304=256*9, 3840=256*15)
#       -> 2.04e10 ~= 2732^3 (+21% particles, exact 3:5:3 iso). local_x 9, cell 1.302; hm 1.6 -> halo 14 = 18.2 Mpc/h (~3.0 sigma).
#
# Usage:
#   MODE=dryrun bash run.sh   # print the four resolved fli-launcher commands, submit nothing
#   MODE=local  bash run.sh   # run locally via mpirun (only with tiny MESH_* overrides)
#   bash run.sh               # submit to SLURM (MODE=sbatch default)
# Wall-times are first-guess estimates -- tune after the first run.
source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 10 -- target simulations: big-quadrant 3-bin density lightcones  (MODE=$MODE)"

BOX_3BIN_QUAD="3000.0 5000.0 3000.0"   # (1.2r, 2r, 1.2r), r = 2500 h^-1 Mpc  (Exp 06 3-bin quadrant)
OBS_QUAD="0.1 0.5 0.9"                 # off-corner observer -> big-quadrant footprint (Exp 06)
MESH_1024="960 1600 960"               # 64-GPU  slab: ÷64 in x&y, EXACT 3:5:3 iso, ~1138^3 (cells 3.125 Mpc/h)
MESH_2560="2304 3840 2304"             # 256-GPU slab: ÷256 in x&y, ~2732^3 (exact 3:5:3)

# Shared painting + shell geometry (LPT and PM). CIC + NO deconvolution; RBF spherical paint at 0.8 px;
# 20 equal-volume shells (>=60 Mpc/h floor); nside 2048; one parquet/shell; float64; CosmoGrid run000 cosmo.
# --halo-multiplier is set per run below (not here): the 64-way mesh has local_x=15, whose default-0.5
# halo would be odd (7 -> slice_unpad crash), so the 64-way runs use 0.4 (halo 6 = 18.75 Mpc/h, even) and
# the 256-way production run uses 1.6 (halo 14 = 18.2 Mpc/h).
COMMON="--paint-order cic --nside 2048 --scheme rbf_neighbor --kernel-width-pixels 0.8 \
--nb-shells 20 --shell-spacing a --min-width 10.0 \
--observer-position $OBS_QUAD --box-size $BOX_3BIN_QUAD \
--shells-per-file 1 --enable-x64 --perf --iterations 3 --seed $SEED $COSMOGRID_COSMO"

# N-body-only knobs (PM runs): BullFrog, 30 steps, growth-factor stepping, drift on the lightcone.
# (--solver / --nb-steps / --time-stepping / --drift-on-lightcone are no-ops under --sim-mode lpt.)
PM="--solver bf --nb-steps 30 --time-stepping D --drift-on-lightcone"

# --- ~1024^3 LPT->PM comparison trio @ 64 GPU (slab pdim 64 1) ----------------------------------
launch 16 4 64 1 00:20:00 -- $COMMON --halo-multiplier 0.4 --sim-mode lpt --lpt-order 1 --mesh-size $MESH_1024 \
  --output "$RESULTS/exp10/1lpt_1024" --name "exp10_1lpt_1024_s%seed%"

launch 16 4 64 1 00:20:00 -- $COMMON --halo-multiplier 0.4 --sim-mode lpt --lpt-order 2 --mesh-size $MESH_1024 \
  --output "$RESULTS/exp10/2lpt_1024" --name "exp10_2lpt_1024_s%seed%"

launch 16 4 64 1 00:25:00 -- $COMMON $PM --halo-multiplier 0.4 --sim-mode pm --lpt-order 2 --mesh-size $MESH_1024 \
  --output "$RESULTS/exp10/pm_1024" --name "exp10_pm_1024_s%seed%"

# --- production target sim: PM lightcone @ ~2560^3, 256 GPU (slab pdim 256 1) -------------------
# --halo-multiplier 1.6 -> halo int(9*1.6)=14 cells (~18.2 Mpc/h, ~3 sigma), covering the z=0 X-displacement
# and matching Exp 06's ~16 Mpc/h. The halo exceeds the local slab (padded local_x 9 -> 37); only evenness
# is required. Padded mesh ~37x3840x2304 float64 ~2.6 GB/field -> fits H100.
launch 64 4 256 1 00:35:00 -- $COMMON $PM --halo-multiplier 1.6 --sim-mode pm --lpt-order 2 --mesh-size $MESH_2560 \
  --output "$RESULTS/exp10/pm_2560" --name "exp10_pm_2560_s%seed%"
