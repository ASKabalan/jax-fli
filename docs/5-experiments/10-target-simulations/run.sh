#!/bin/bash
# Experiment 10 — target simulations for inference (Exp 13 & 14).
# NOTE: the full-sky 2560^3 production run (#4) starts from CosmoGrid cosmo_000001/run_0's recreated
# white noise (--ic-repo/--ic-data-files), not from --seed. See the block above that launch.
# Eight density-shell lightcone runs: an LPT->PM comparison trio at 1024^3 (64 GPU) plus the production PM
# target sim at 2560^3 (256 GPU), each run twice -- once FULL-SKY (5 Gpc/h cubic box) and once over a
# QUADRANT (partial-sky patch at the same 2500 Mpc/h depth, finer resolution). float64; one parquet/shell.
#
# Physics recipe (provenance):
#   full sky           (Exp 06): --observer-position 0.5 0.5 0.5 (box centre), box 5000x5000x5000 (r=box/2=2500)
#   30 steps, bf + D   (Exp 04): --nb-steps 30 --solver bf --time-stepping D            (the "bfd" pairing)
#   RBF spherical paint(Exp 03): --nside 2048 --scheme rbf_neighbor --kernel-width-pixels 0.8
#   CIC, no deconv     (Exp 02): --paint-order cic  (no --deconvolution)
#   2560^3 resolution  (Exp 01): production mesh 2560^3 (5 Gpc/h box)
#   20 shells + drift  (Exp 05b): --shell-spacing a --nb-shells 20 --min-width 10.0 --drift-on-lightcone
#
# 1-D SLAB decomposition (pdim "P 1"): only axis 0 (X) is persistently sharded, so only the X-halo is
# padded (py=1 drops the Y-halo). The distributed FFT all-to-all transposes X<->Y, so BOTH Nx AND Ny must
# be divisible by P. The box and mesh are CUBIC (full sky), so cells are isotropic by construction.
# HALO: the nbody state is the displacement from the initial grid (particles are NOT re-homed), so the X-halo
# must cover a particle's TOTAL X-displacement (~6 Mpc/h 1-D rms at z=0); mass past it is silently mis-painted
# (jnp.mod wrap). Exp 06 sized this ~16 Mpc/h. Only the EVENness of halo=int(Nx/P*hm) is required (odd ->
# slice_unpad crash); the halo MAY exceed the local slab width, so hm can go above 1.0 as needed.
#   1024^3 @ 64 GPU  (16 nodes x 4, pdim 64 1): cell 5000/1024 = 4.883 Mpc/h. local_x 16;
#       default hm 0.5 -> halo 8 = 39 Mpc/h (even, ample). 1024%64=0 (x & y for the all2all).
#   2560^3 @ 256 GPU (64 nodes x 4, pdim 256 1): cell 5000/2560 = 1.953 Mpc/h. local_x 10;
#       hm 1.0 -> halo 10 = 19.5 Mpc/h (even; default 0.5 -> odd 5). 2560%256=0. (128 GPU also works at
#       default 0.5: local 20, halo 10 = 19.5 Mpc/h -- same halo, half the GPUs / ~2x wall.)
#
# Usage:
#   MODE=dryrun bash run.sh   # print the eight resolved fli-launcher commands, submit nothing
#   MODE=local  bash run.sh   # run locally via mpirun (only with tiny MESH_* overrides)
#   bash run.sh               # submit to SLURM (MODE=sbatch default)
# Wall-times are first-guess estimates -- tune after the first run.
source "$(dirname "$0")/../_launch_common.sh"

echo "### Exp 10 -- target simulations: full-sky + quadrant 5 Gpc/h density lightcones  (MODE=$MODE)"

BOX_FULL="5000.0 5000.0 5000.0"   # 5 Gpc/h cubic; observer at centre -> full sphere to r=box/2=2500 (z~1.06)
OBS_FULL="0.5 0.5 0.5"            # box centre -> full sky
MESH_1024="1024 1024 1024"        # 64-GPU  slab: 1024%64=0; cubic -> isotropic; exactly 1024^3
MESH_2560="2560 2560 2560"        # 256-GPU slab: 2560%256=0; cubic -> isotropic; exactly 2560^3

# --- Quadrant: an off-centre observer sees a partial sky, so a smaller box reaches the SAME depth.
# obs (0.1,0.5,0.9) in a (3000,5000,3000) box -> factor 1+2*min(p,1-p) = (1.2,2.0,1.2) -> r = box/factor
# = 2500 Mpc/h on EVERY axis (z~1.06), identical to the full-sky depth. We hold the total cell count at the
# full-sky value (same per-GPU memory on the same GPU count), so the quadrant is a FINER-resolution partial-
# sky patch: dx 3.125 / 1.302 Mpc/h vs full-sky 4.883 / 1.953.
# Mesh constraints: (1) isotropic, box[i]/mesh[i] equal; (2) mesh[0] & mesh[1] divisible by pdim[0] (the
# all2all transposes X<->Y); (3) cells >= full-sky and minimally so. The 3:5:3 box x %64/%256 quantises the
# cell count in k^3 steps, so the closest option >= full-sky overshoots: +37% (1024) / +21% (2560).
BOX_QUADRANT="3000.0 5000.0 3000.0"  # off-centre observer -> r=2500 on every axis (z~1.06, = full sky)
OBS_QUADRANT="0.1 0.5 0.9"           # off-centre -> big quadrant footprint
MESH_1024_QUADRANT="960 1600 960"    # dx 3.125; 1.47e9 cells = 1.37x 1024^3; local_x 15 -> hm 0.8 (halo 12, even)
MESH_2560_QUADRANT="2304 3840 2304"  # dx 1.302; 2.04e10 cells = 1.21x 2560^3; local_x 9 -> hm 1.35 (halo 12, even)

# Shared painting + shell geometry (LPT and PM). CIC + NO deconvolution; RBF spherical paint at 0.8 px;
# 20 shells (scale-factor spacing, >=10 Mpc/h floor); nside 2048; one parquet/shell; float64; CosmoGrid run000.
# --halo-multiplier is set per run below: the 64-way (local_x 16) is even at the default 0.5, but the
# 256-way (local_x 10) would be odd (5) at 0.5, so it uses 1.0 (halo 10 = 19.5 Mpc/h, even).
COMMON="--paint-order cic --nside 2048 --scheme rbf_neighbor --kernel-width-pixels 0.8 \
--nb-shells 20 --shell-spacing equal_vol --min-width 50.0 \
--shells-per-file 1 --enable-x64 --perf --iterations 3 --seed $SEED $COSMOGRID_COSMO"

# N-body-only knobs (PM runs): BullFrog, 30 steps, growth-factor stepping, drift on the lightcone.
# (--solver / --nb-steps / --time-stepping / --drift-on-lightcone are no-ops under --sim-mode lpt.)
PM="--solver bf --nb-steps 30 --time-stepping D --drift-on-lightcone"

# --- 1024^3 LPT->PM comparison trio @ 64 GPU (slab pdim 64 1; default halo 0.5 -> 8 cells = 39 Mpc/h) ---
#launch 16 4 64 1 00:20:00 -- $COMMON --sim-mode lpt --lpt-order 1 --mesh-size $MESH_1024 --observer-position $OBS_FULL --box-size $BOX_FULL \
#  --output "$RESULTS/exp10/1lpt_fs_1024" --name "exp10_1lpt_1024_s%seed%"

#launch 16 4 64 1 00:20:00 -- $COMMON --sim-mode lpt --lpt-order 2 --mesh-size $MESH_1024 --observer-position $OBS_FULL --box-size $BOX_FULL \
#  --output "$RESULTS/exp10/2lpt_fs_1024" --name "exp10_2lpt_1024_s%seed%"

#launch 16 4 64 1 00:25:00 -- $COMMON $PM --sim-mode pm --lpt-order 2 --mesh-size $MESH_1024 --observer-position $OBS_FULL --box-size $BOX_FULL \
#  --output "$RESULTS/exp10/pm_fs_1024" --name "exp10_pm_1024_s%seed%"

# --- production target sim: PM lightcone @ 2560^3, 256 GPU (slab pdim 256 1) --------------------
# --halo-multiplier 1.0 -> halo int(10*1.0)=10 cells (19.5 Mpc/h, ~3 sigma), covering the z=0 X-displacement.
# Default 0.5 would give an odd halo (5). halo == local here (padded local_x 10 -> 30); only evenness is
# required. Padded mesh ~30x2560x2560 float64 ~1.6 GB/field -> fits H100.
#
# INITIAL CONDITIONS: not a --seed draw. --ic-input/--ic-repo hands fli-simulate the recreated WHITE
# noise of CosmoGrid cosmo_000001/run_0 (iSeed 111115, nGrid 832), the same run whose density shells
# Exp 0 published, so the target sim is traceable to a real external simulation with a known latent
# instead of to an arbitrary integer. jfli.resample_white_field spectrally upsamples 832^3 -> 2560^3:
# every source mode is copied at the SAME integer wavevector (transfer and coherence against the
# source are 1 by construction) and the modes above the source Nyquist index are drawn from --seed.
#   CAVEAT, and the README says it too: this reproduces CosmoGrid's REALIZATION, not its field. Mode
#   index n sits at 2*pi*n/5000 here and at 2*pi*n/900 there, so nothing physical is shared -- getting
#   coherence vs CosmoGrid's own field would need a box of 900 Mpc/h or a multiple of it.
# Jean Zay compute nodes are offline: push the truth parquet and pre-warm the HF cache on a LOGIN node
# first (snapshot_download(..., local_files_only=True) raises on a cold cache), as for Exp 07 / Exp 00.
CG_IC="--ic-repo ASKabalan/jax-fli-experiments --ic-data-files 14-inference-cosmogrid/truth/input_cg.parquet"
launch 64 4 256 1 00:35:00 -- $COMMON $PM $CG_IC --halo-multiplier 1.0 --sim-mode pm --lpt-order 2 --mesh-size $MESH_2560 --observer-position $OBS_FULL --box-size $BOX_FULL \
  --output "$RESULTS/exp10/pm_fs_2560_cgic" --name "exp10_pm_2560_cgic_s%seed%"

# === QUADRANT (partial-sky) mirror of the four runs above; same recipe, quadrant box/observer/mesh =========
# --- quadrant 1024^3 LPT->PM trio @ 64 GPU (local_x 15 is odd; --halo-multiplier 0.8 -> halo int(15*0.8)=12,
#     even, = 37.5 Mpc/h, ~= the full-sky 39 Mpc/h ghost zone; default 0.5 -> odd halo 7 -> slice_unpad crash) ---
#launch 16 4 64 1 00:20:00 -- $COMMON --halo-multiplier 0.8 --sim-mode lpt --lpt-order 1 --mesh-size $MESH_1024_QUADRANT --observer-position $OBS_QUADRANT --box-size $BOX_QUADRANT \
  #--output "$RESULTS/exp10/1lpt_quad_1024" --name "exp10_1lpt_quad_1024_s%seed%"

#launch 16 4 64 1 00:20:00 -- $COMMON --halo-multiplier 0.8 --sim-mode lpt --lpt-order 2 --mesh-size $MESH_1024_QUADRANT --observer-position $OBS_QUADRANT --box-size $BOX_QUADRANT \
  #--output "$RESULTS/exp10/2lpt_quad_1024" --name "exp10_2lpt_quad_1024_s%seed%"

#launch 16 4 64 1 00:30:00 -- $COMMON $PM --halo-multiplier 0.8 --sim-mode pm --lpt-order 2 --mesh-size $MESH_1024_QUADRANT --observer-position $OBS_QUADRANT --box-size $BOX_QUADRANT \
  #--output "$RESULTS/exp10/pm_quad_1024" --name "exp10_pm_quad_1024_s%seed%"

# --- quadrant production PM @ 2560^3, 256 GPU (local_x 9 is odd; --halo-multiplier 1.35 -> halo int(9*1.35)=12,
#     even, = 15.6 Mpc/h, ~1.5 sigma; default 0.5 -> halo 4 = 5.2 Mpc/h too thin, hm 1.0 -> odd halo 9 -> crash).
#     Odd local_x forces a large relative pad (9 -> 33 = 3.7x vs full-sky's 10 -> 30 = 3.0x): peak ~2.3 GB/field
#     (~1.5x the full-sky 1.6); still fits H100. Confirm from --perf memory before this 256-GPU submit. ---
#launch 64 4 256 1 00:45:00 -- $COMMON $PM --halo-multiplier 1.35 --sim-mode pm --lpt-order 2 --mesh-size $MESH_2560_QUADRANT --observer-position $OBS_QUADRANT --box-size $BOX_QUADRANT \
  #--output "$RESULTS/exp10/pm_quad_2560" --name "exp10_pm_quad_2560_s%seed%"
