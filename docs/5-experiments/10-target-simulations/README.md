# Experiment 10 — Target simulations

## Goal

Generate the **target simulation** that the density-field inference experiments (13 and 14) condition on, and — alongside it — a cheaper **LPT → PM progression** that motivates why the production run is a full particle-mesh (PM) simulation rather than perturbation theory. Every physics choice is inherited verbatim from an earlier converged experiment; this experiment simply assembles them on the production geometry and runs them.

All four runs paint **density-shell lightcones** on the *big-quadrant* 3-bin box (an off-corner observer that sees one sky quadrant to a comoving depth `r = 2500 h⁻¹Mpc`, i.e. source bins out to `z ≈ 1.06`), at HEALPix `nside 2048`, float64, one parquet per shell.

| # | Run | Pipeline | Mesh | pdim | GPUs (nodes×4) | ~particles |
|---|---|---|---|---|---|---|
| 1 | 1LPT lightcone | `lpt`, order 1 | 960 × 1600 × 960 | 64 × 1 | 64 (16) | 1.47 × 10⁹ ≈ 1138³ |
| 2 | 2LPT lightcone | `lpt`, order 2 | 960 × 1600 × 960 | 64 × 1 | 64 (16) | ≈ 1138³ |
| 3 | PM lightcone | `pm`, 2LPT ICs | 960 × 1600 × 960 | 64 × 1 | 64 (16) | ≈ 1138³ |
| 4 | **PM production** | `pm`, 2LPT ICs | 2304 × 3840 × 2304 | 256 × 1 | 256 (64) | 2732³ (+21 %) |

Runs 1–3 are a matched 1LPT / 2LPT / PM triptych at a cheaper ~1138³ resolution — the same seed, box, and lightcone geometry, differing only in dynamics — so the comparison isolates the effect of the gravity model. Run 4 is the actual target simulation for Experiments 13 and 14, at the production ~2560³ resolution justified by the resolution-convergence study (Experiment 01).

**Box and mesh sizing.** `BOX_3BIN_QUAD = 3000 × 5000 × 3000 h⁻¹Mpc = (1.2r, 2r, 1.2r)` with `r = 2500`; the observer at box fraction `(0.1, 0.5, 0.9)` produces the aspect factors `1 + 2·min(f, 1−f) = (1.2, 2.0, 1.2) = 3 : 5 : 3`. Both meshes follow that aspect (`Nx : Ny : Nz = 3 : 5 : 3`, `Nx = Nz`) and are distributed as **1-D slabs** (`--pdim P 1`), which shard axis 0 (X) and — because the distributed-FFT all-to-all transposes X↔Y — require **both `Nx` and `Ny` divisible by `P`**, with an even X-halo (`py = 1` drops the Y-halo). At 64 GPU, `960 × 1600 × 960` (960 = 64·15, 1600 = 64·25) is exactly isotropic (cells `3.125 h⁻¹Mpc`) and totals ≈ 1138³ (+11 % linear / +37 % particles vs 1024³; local_x = 15). At 256 GPU, `2304 × 3840 × 2304` (2304 = 256·9, 3840 = 256·15) is exactly isotropic (cells `1.302 h⁻¹Mpc`) and totals ≈ 2732³ (+21 % of 2560³; local_x = 9). Both are exactly `3 : 5 : 3`.

**Halo.** The nbody carries each particle's displacement from its initial grid cell (particles are not re-homed), so the X-halo must cover the particle's total X-displacement (≈ 6 h⁻¹Mpc 1-D rms at z = 0); mass drifting past it is silently mis-painted. Only the evenness of `halo = int(Nx/P · hm)` is required (an odd halo crashes `slice_unpad`); the halo may exceed the local slab width, so `--halo-multiplier` can go above 1 as needed. The 64-GPU runs use `--halo-multiplier 0.4` → halo 6 cells ≈ 18.75 h⁻¹Mpc (≈ 3 σ); the default 0.5 would give an odd halo of 7 here (local_x = 15). The 256-way production run sets `--halo-multiplier 1.6` → halo 14 cells ≈ 18.2 h⁻¹Mpc (≈ 3.0 σ), matching Experiment 06's ≈ 16 h⁻¹Mpc standard (the pad exceeds the 9-cell local slab, `local_x 9 → 37` padded, ≈ 2.6 GB/field at float64 — fits an H100).

## Method

The pipeline is Gaussian initial conditions → LPT (runs 1–2) or LPT + BullFrog PM (runs 3–4) → lightcone painting into 20 equal-volume shells → RBF spherical painting onto the `nside 2048` HEALPix sphere. The production recipe assembled here:

| Choice | Flags | From |
|---|---|---|
| Big-quadrant sky | `--observer-position 0.1 0.5 0.9`, `--box-size 3000 5000 3000` | Exp 06 |
| 30 steps, BullFrog + growth-factor stepping | `--nb-steps 30 --solver bf --time-stepping D` | Exp 04 (`bfd`) |
| RBF spherical painting, 0.8-pixel subpixel width | `--nside 2048 --scheme rbf_neighbor --kernel-width-pixels 0.8` | Exp 03 |
| CIC mass assignment, no force-window deconvolution | `--paint-order cic` | Exp 02 |
| ~2560³ production resolution | `--mesh-size 2304 3840 2304` | Exp 01 |
| Equal-volume spacing, 20 shells, drift on the lightcone | `--shell-spacing equal_vol --nb-shells 20 --min-width 60.0 --drift-on-lightcone` | Exp 05c |

The cosmology is **CosmoGrid run000** (`$COSMOGRID_COSMO` from `_launch_common.sh`: h = 0.73, Ω_c = 0.2538, σ₈ = 0.9, w₀ = −1.1665, with massive neutrinos), matching the CosmoGrid reference simulations (as in Experiment 06) that the inference is calibrated against. Drift on the lightcone is a property of the N-body integration, so it applies only to the PM runs (3 and 4); the LPT runs share the box, seed, and 20-shell equal-volume geometry but have no drift step. Each run is launched with `--perf --iterations 3`, so it writes both the per-shell density parquets and a `perf_<mode>.csv` timing row.

## Results

<!-- TODO: figures after the cluster runs are published to ASKabalan/jax-fli-experiments and studied in build.py. -->

_Pending the cluster runs — this experiment's figures (the 1LPT / 2LPT / PM shell and C_ℓ comparison, and the production lightcone) are produced by a `build.py` from the published parquet, per the standard lifecycle._

## How to run

```bash
MODE=dryrun bash run.sh   # print the four resolved fli-launcher commands, submit nothing
bash run.sh               # submit to SLURM (MODE=sbatch default)
```

`run.sh` sources `../_launch_common.sh` and submits four `fli-launcher → fli-simulate` jobs (idempotent — a completed output directory is skipped on rerun). Start with the three 64-GPU runs; submit the 256-GPU production run once the 64-GPU PM looks right.
