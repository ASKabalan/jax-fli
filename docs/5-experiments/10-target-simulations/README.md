# Experiment 10 — Target simulations

## Goal

Generate the **target simulation** that the density-field inference experiments (13 and 14) condition on, and — alongside it — a cheaper **LPT → PM progression** that motivates why the production run is a full particle-mesh (PM) simulation rather than perturbation theory. Every physics choice is inherited verbatim from an earlier converged experiment; this experiment simply assembles them on the production geometry and runs them.

Every run paints a **density-shell lightcone** reaching the same comoving depth `r = 2500 h⁻¹Mpc` (source bins out to `z ≈ 1.06`), at HEALPix `nside 2048`, float64, one parquet per shell. The four recipes are each run over **two geometries** — eight runs in all. **Full-sky** uses a cubic `5000³ h⁻¹Mpc` box with the observer at its centre, seeing the whole sphere. **Quadrant** uses an off-centre observer `(0.1, 0.5, 0.9)` in a smaller `3000 × 5000 × 3000 h⁻¹Mpc` box that still reaches `r = 2500` on every axis, seeing a partial-sky patch; holding the cell budget fixed at the full-sky value then buys the quadrant a **finer resolution** on that patch (see *Quadrant geometry* below).

| # | Run | Geometry | Pipeline | Mesh | pdim | GPUs (nodes×4) | particles |
|---|---|---|---|---|---|---|---|
| 1 | 1LPT lightcone | full-sky | `lpt`, order 1 | 1024 × 1024 × 1024 | 64 × 1 | 64 (16) | 1024³ (1.07 × 10⁹) |
| 2 | 2LPT lightcone | full-sky | `lpt`, order 2 | 1024 × 1024 × 1024 | 64 × 1 | 64 (16) | 1024³ |
| 3 | PM lightcone | full-sky | `pm`, 2LPT ICs | 1024 × 1024 × 1024 | 64 × 1 | 64 (16) | 1024³ |
| 4 | **PM production** | full-sky | `pm`, 2LPT ICs | 2560 × 2560 × 2560 | 256 × 1 | 256 (64) | 2560³ (1.68 × 10¹⁰) |
| 5 | 1LPT lightcone | quadrant | `lpt`, order 1 | 960 × 1600 × 960 | 64 × 1 | 64 (16) | 1.47 × 10⁹ |
| 6 | 2LPT lightcone | quadrant | `lpt`, order 2 | 960 × 1600 × 960 | 64 × 1 | 64 (16) | 1.47 × 10⁹ |
| 7 | PM lightcone | quadrant | `pm`, 2LPT ICs | 960 × 1600 × 960 | 64 × 1 | 64 (16) | 1.47 × 10⁹ |
| 8 | **PM production** | quadrant | `pm`, 2LPT ICs | 2304 × 3840 × 2304 | 256 × 1 | 256 (64) | 2.04 × 10¹⁰ |

Runs 1–3 (and their quadrant mirrors 5–7) are a matched 1LPT / 2LPT / PM triptych at a cheaper 1024³-class resolution — the same seed, box, and lightcone geometry, differing only in dynamics — so the comparison isolates the effect of the gravity model. Runs 4 and 8 are the actual target simulations for Experiments 13 and 14, at the production 2560³-class resolution justified by the resolution-convergence study (Experiment 01).

**Box and mesh sizing.** The box is cubic — `5000 × 5000 × 5000 h⁻¹Mpc` (5 Gpc/h) — with the observer at the centre `(0.5, 0.5, 0.5)`, so it is full-sky and its cells are isotropic by construction. Both meshes are cubic and distributed as **1-D slabs** (`--pdim P 1`), which shard axis 0 (X) and — because the distributed-FFT all-to-all transposes X↔Y — require `Nx` (and, after the transpose, `Ny`) divisible by `P`; only the X-axis carries a halo (`py = 1` drops the Y-halo). At 64 GPU, `1024³` (1024 = 64·16, cells `4.883 h⁻¹Mpc`). At 256 GPU, `2560³` (2560 = 256·10, cells `1.953 h⁻¹Mpc`).

**Halo.** The nbody carries each particle's displacement from its initial grid cell (particles are not re-homed), so the X-halo must cover the particle's total X-displacement (≈ 6 h⁻¹Mpc 1-D rms at z = 0); mass drifting past it is silently mis-painted. Only the evenness of `halo = int(Nx/P · hm)` is required (an odd halo crashes `slice_unpad`); the halo may exceed the local slab width, so `--halo-multiplier` can go above 1 as needed. The 64-GPU runs use the default `0.5` → halo 8 cells ≈ 39 h⁻¹Mpc (local_x = 16, even, ample). The 256-way production run sets `--halo-multiplier 1.0` → halo 10 cells ≈ 19.5 h⁻¹Mpc (≈ 3 σ; the default 0.5 would give an odd halo of 5, since local_x = 10). Here the halo equals the local slab (`local_x 10 → 30` padded, ≈ 1.6 GB/field at float64 — fits an H100). A `128 × 1` slab is an equivalent alternative (local 20, default 0.5 → halo 10 = 19.5 h⁻¹Mpc, half the GPUs).

**Quadrant geometry.** Moving the observer off-centre lets a smaller box reach the same depth: with `factor = 1 + 2·min(p, 1−p)` per axis and `r = box / factor`, the observer `(0.1, 0.5, 0.9)` in the `(3000, 5000, 3000)` box gives `factor = (1.2, 2.0, 1.2)` → `r = 2500 h⁻¹Mpc` on **every** axis (`z ≈ 1.06`), identical to full-sky, over a partial-sky footprint. The quadrant mesh is chosen to hold the **total cell count** (hence per-GPU memory on the same GPU count) at the full-sky value, subject to (1) isotropic cells `box[i]/mesh[i]` equal, and (2) `mesh[0]` and `mesh[1]` divisible by `pdim[0]` (the all-to-all transpose). Because the `3 : 5 : 3` box ratio times the `%64 / %256` divisibility quantises the cell count in `k³` steps, no isotropic mesh lands near the full-sky count exactly: the smallest option **at or above** it overshoots by **+37 %** (`960 × 1600 × 960`, `1.47 × 10⁹` cells) at 1024³-class and **+21 %** (`2304 × 3840 × 2304`, `2.04 × 10¹⁰`) at 2560³-class. Same cells in a smaller box means **finer resolution** — `dx = 3.125 / 1.302 h⁻¹Mpc` versus full-sky's `4.883 / 1.953` — so the quadrants are higher-resolution partial-sky patches, not matched-resolution sub-volumes. The halo needs care because both quadrant meshes have an **odd** `local_x` (15 and 9), for which the default `hm = 0.5` gives an odd halo (crash): the 1024-class quadrants use `--halo-multiplier 0.8` → halo 12 ≈ 37.5 h⁻¹Mpc (matching the full-sky ghost zone), and the 2560-class quadrant uses `--halo-multiplier 1.35` → halo 12 ≈ 15.6 h⁻¹Mpc (≈ 1.5 σ). The odd `local_x = 9` there forces a larger relative pad (9 → 33 padded, versus full-sky's 10 → 30), pushing peak temporary memory to ≈ 1.5× the full-sky production run's ≈ 1.6 GB/field — still within an H100, but worth confirming from the `--perf` memory analysis before the 256-GPU submit.

## Method

The pipeline is Gaussian initial conditions → LPT (runs 1–2) or LPT + BullFrog PM (runs 3–4) → lightcone painting into 20 shells (scale-factor spacing) → RBF spherical painting onto the `nside 2048` HEALPix sphere. The production recipe assembled here:

| Choice | Flags | From |
|---|---|---|
| Full-sky box | `--observer-position 0.5 0.5 0.5`, `--box-size 5000 5000 5000` | Exp 06 |
| 30 steps, BullFrog + growth-factor stepping | `--nb-steps 30 --solver bf --time-stepping D` | Exp 04 (`bfd`) |
| RBF spherical painting, 0.8-pixel subpixel width | `--nside 2048 --scheme rbf_neighbor --kernel-width-pixels 0.8` | Exp 03 |
| CIC mass assignment, no force-window deconvolution | `--paint-order cic` | Exp 02 |
| 2560³ production resolution | `--mesh-size 2560 2560 2560` | Exp 01 |
| Scale-factor shell spacing, 20 shells, drift on the lightcone | `--shell-spacing a --nb-shells 20 --min-width 10.0 --drift-on-lightcone` | Exp 05b |

The cosmology is **CosmoGrid run000** (`$COSMOGRID_COSMO` from `_launch_common.sh`: h = 0.73, Ω_c = 0.2538, σ₈ = 0.9, w₀ = −1.1665, with massive neutrinos), matching the CosmoGrid reference simulations (as in Experiment 06) that the inference is calibrated against. Drift on the lightcone is a property of the N-body integration, so it applies only to the PM runs (3 and 4); the LPT runs share the box, seed, and 20-shell geometry but have no drift step. Each run is launched with `--perf --iterations 3`, so it writes both the per-shell density parquets and a `perf_<mode>.csv` timing row.

## Results

<!-- TODO: figures after the cluster runs are published to ASKabalan/jax-fli-experiments and studied in build.py. -->

_Pending the cluster runs — this experiment's figures (the 1LPT / 2LPT / PM shell and C_ℓ comparison, and the production lightcone) are produced by a `build.py` from the published parquet, per the standard lifecycle._

## How to run

```bash
MODE=dryrun bash run.sh   # print the eight resolved fli-launcher commands, submit nothing
bash run.sh               # submit to SLURM (MODE=sbatch default)
```

`run.sh` sources `../_launch_common.sh` and submits eight `fli-launcher → fli-simulate` jobs (idempotent — a completed output directory is skipped on rerun). Start with the 64-GPU runs (full-sky and quadrant); submit each 256-GPU production run once its 64-GPU PM looks right, and confirm the quadrant production run's peak memory from its `--perf` output before the 256-GPU submit.
