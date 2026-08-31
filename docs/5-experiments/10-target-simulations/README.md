# Experiment 10 — Target simulations

## Goal

Generate the **target simulation** that the density-field inference experiments (13 and 14) condition on, and — alongside it — a cheaper **LPT → PM progression** that motivates why the production run is a full particle-mesh (PM) simulation rather than perturbation theory. Every physics choice is inherited verbatim from an earlier converged experiment; this experiment simply assembles them on the production geometry and runs them.

Every run paints a **density-shell lightcone** reaching the same comoving depth `r = 2500 h⁻¹Mpc` (source bins out to `z ≈ 1.06`), at HEALPix `nside 2048`, float64, one parquet per shell. The four recipes are each run over **two geometries** — eight runs in all. **Full-sky** uses a cubic `5000³ h⁻¹Mpc` box with the observer at its centre, seeing the whole sphere. **Quadrant** uses an off-centre observer `(0.1, 0.5, 0.9)` in a smaller `3000 × 5000 × 3000 h⁻¹Mpc` box that still reaches `r = 2500` on every axis, seeing a partial-sky patch; holding the cell budget fixed at the full-sky value then buys the quadrant a **finer resolution** on that patch (see *Quadrant geometry* below).

| # | Run | Geometry | Pipeline | Initial conditions | Mesh | pdim | GPUs (nodes×4) | particles |
|---|---|---|---|---|---|---|---|---|
| 1 | 1LPT lightcone | full-sky | `lpt`, order 1 | `--seed 0` | 1024 × 1024 × 1024 | 64 × 1 | 64 (16) | 1024³ (1.07 × 10⁹) |
| 2 | 2LPT lightcone | full-sky | `lpt`, order 2 | `--seed 0` | 1024 × 1024 × 1024 | 64 × 1 | 64 (16) | 1024³ |
| 3 | PM lightcone | full-sky | `pm`, 2LPT ICs | `--seed 0` | 1024 × 1024 × 1024 | 64 × 1 | 64 (16) | 1024³ |
| 4 | **PM production** | full-sky | `pm`, 2LPT ICs | **CosmoGrid `cosmo_000001/run_0`** | 2560 × 2560 × 2560 | 256 × 1 | 256 (64) | 2560³ (1.68 × 10¹⁰) |
| 5 | 1LPT lightcone | quadrant | `lpt`, order 1 | `--seed 0` | 960 × 1600 × 960 | 64 × 1 | 64 (16) | 1.47 × 10⁹ |
| 6 | 2LPT lightcone | quadrant | `lpt`, order 2 | `--seed 0` | 960 × 1600 × 960 | 64 × 1 | 64 (16) | 1.47 × 10⁹ |
| 7 | PM lightcone | quadrant | `pm`, 2LPT ICs | `--seed 0` | 960 × 1600 × 960 | 64 × 1 | 64 (16) | 1.47 × 10⁹ |
| 8 | **PM production** | quadrant | `pm`, 2LPT ICs | `--seed 0` | 2304 × 3840 × 2304 | 256 × 1 | 256 (64) | 2.04 × 10¹⁰ |

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
| Initial conditions of CosmoGrid `cosmo_000001/run_0` (run 4 only) | `--ic-repo`, `--ic-data-files` | Exp 14 truth artifact |

### Initial conditions of the production run

Run 4 does not draw its initial conditions from `--seed`. It is handed the **white-noise field of CosmoGrid `cosmo_000001/run_0`** — the same run whose density shells Experiment 0 published as `00-cosmogrid/density` — so the target simulation Experiments 13 and 14 condition on is traceable to a real external simulation with a known latent rather than to an arbitrary integer. CosmoGridV1 never stored its initial conditions (PKDGRAV3 generates them internally from the scalar `iSeed`); the `ic_resample` toolkit recreates the primordial white noise as a standalone NumPy generator and verifies it against PKDGRAV3 at the production size — **0 / 692,224 desynced pencils and 0 / 288,657,407 modes with `r(k) < 0.999999` at `nGrid = 832`**. `make_truth_ic.py` in [Experiment 14](../14-inference-cosmogrid-shear/) wraps that field into the single-row catalog this run reads (`iSeed 111115`, `nGrid 832`, `bFixedAmpIC 0`, `dBoxSize 900`).

The stored field is **white**, before coloring. `fli-simulate --ic-input` (or `--ic-repo` + `--ic-data-files`) spectrally upsamples it from 832³ to the run's 2560³ with `jax_fli.resample_white_field`: every source mode is copied into the target grid at the **same integer wavevector**, so the transfer function and coherence against the source are exactly 1 over the shared block — by construction, not by measurement — and the modes above the source Nyquist index are drawn fresh from `--seed` (zero-padding alone would leave the field with no small-scale power and a variance of `(832/2560)³ = 0.034`).

**What this does and does not claim.** It reproduces CosmoGrid's *realization* and uses its cosmology; it does **not** reproduce CosmoGrid's density field, and no observable is shared. Fourier modes are `k_n = 2πn/L`, so mode index `n` sits at `2π n / 5000` here and at `2π n / 900` in CosmoGrid — coherence against CosmoGrid's own field would require a box of 900 h⁻¹Mpc or an integer multiple of it, which this geometry is not. Independently of the box, `T(k) = 1` against CosmoGrid is also out of reach because `interpolate_initial_conditions` colors with jax_cosmo's Eisenstein–Hu `P(k)` while CosmoGrid ran CLASS with `A_s = 2.02 × 10⁻⁹` and three 0.02 eV neutrinos. What the run gains is **provenance and reproducibility**, not a CosmoGrid comparison.

> **Unverified: axis orientation.** The white noise is bit-verified against PKDGRAV3, but nothing checks that its `[x, y, z]` indexing enters jax-fli's mesh the right way round — and no isotropic statistic can catch a transpose, since coloring and `P(k)` are both isotropic. The upsample itself is transpose-free (verified by exact per-mode equality on asymmetric wavevectors such as `n = (1,2,3)` versus `(3,1,2)`); the `npz → mesh` ingest upstream of it is not. The only real check is cross-correlating a forward run against `00-cosmogrid/density` at matched redshift, which is Experiment 14's job.

The cosmology is **CosmoGrid run000** (`$COSMOGRID_COSMO` from `_launch_common.sh`: h = 0.73, Ω_c = 0.2538, σ₈ = 0.9, w₀ = −1.1665, with massive neutrinos), matching the CosmoGrid reference simulations (as in Experiment 06) that the inference is calibrated against. Drift on the lightcone is a property of the N-body integration, so it applies only to the PM runs (3 and 4); the LPT runs share the box, seed, and 20-shell geometry but have no drift step. Each run is launched with `--perf --iterations 3`, so it writes both the per-shell density parquets and a `perf_<mode>.csv` timing row.

## Results

### The initial condition survives resampling exactly

Before any simulation runs, the one thing worth checking is that handing the pipeline an external initial condition does not quietly change it. The figure resamples the real `832³` CosmoGrid white noise onto a `1024³` grid **in the same `900 h⁻¹Mpc` box**, so integer wavevector `n` maps to the same physical `k` on both grids and a cross-spectrum is well defined. Both fields are then coloured with the same linear `P(k)`.

![832³ CosmoGrid IC vs its 1024³ spectral upsample: P(k), transfer and coherence](assets/fig01-ic-upsample-832-1024.svg)

**The upsample is lossless on every mode the two grids share.** The top panel shows both coloured fields tracking the linear `P(k)` (measured/theory median `0.9991` over the well-sampled decades), with the `1024³` grid simply reaching a higher Nyquist. The middle panel is the point: over the shared modes the transfer function `T(k)` and the coherence `r(k)` sit at **1 across the whole band**. In this figure `|T − 1| ≡ 0` exactly — the measurement compares the two k-arrays directly, and the target's shared block is a bitwise copy of the source's, so the two spectra are sums over identical numbers — while `|1 − r| ≤ 9.7 × 10⁻⁸` reflects only the different `float32` arithmetic used for `|a|²` and `Re(a₁a₂*)`, as the bottom panel confirms.

That is the guarantee `jax_fli.resample_white_field` is built to give — source modes copied at the same integer wavevector rather than interpolated — but note the shipped function is measured a little differently. It round-trips through an inverse FFT to return a real-space field, so re-transforming its output recovers the source's modes to `≈ 1 × 10⁻⁶` rather than bitwise: the `float32` FFT round-off floor, not an approximation in the copy. `STAGE=check` in `prep_ic_spectra.py` reports both numbers side by side.

**What the upsample cannot do is invent information.** The green curve is the fraction of the `1024³` grid's modes in each shell that are inherited from CosmoGrid. It is exactly 1 out to the `832³` Nyquist (`k = 2.90 h Mpc⁻¹`), then falls through the region where the shared block is a cube inside a larger sphere — `0.46` at the `1024³` Nyquist, and 0 by the cube corner. Above the source Nyquist the field is fresh noise drawn from `--seed`; filling it is not optional, since zero-padding alone would leave the field with no small-scale power and a variance of `(832/1024)³ = 0.54`.

### It runs the other way too, for validation

`resample_white_field` is not upsample-only: the shared block is set by the **smaller** of the two grids, so the same code coarsens a field as exactly as it refines one. That is the direction you want for validation — a cheap `512³` run off the *same* CosmoGrid realization, rather than a different seed.

![832³ CosmoGrid IC vs its 512³ spectral downsample: P(k), transfer and coherence](assets/fig02-ic-downsample-832-512.svg)

**Downsampling is truncation, and it is equally exact.** Read the same way as the figure above, with the two Nyquist guides swapped: the `512³` field now stops at `k = 1.79 h Mpc⁻¹` while the source runs on to `2.90`. Over the shared modes `T(k)` and `r(k)` are again `1` — `|T − 1| ≡ 0`, `|1 − r| ≤ 9.7 × 10⁻⁸`. The coarse field inherits **99.4 %** of its modes from CosmoGrid (against 53.4 % for the `1024³` upsample, which has to invent everything above the source Nyquist); the small dip near its own Nyquist is the one structural asymmetry — a real field needs those modes self-conjugate and the source's are generic complex, so they come from the fresh draw instead.

What this is **not** is a real-space block average. Averaging `2³` cells preserves variance (`0.9997`) and would leave `P(k)` looking healthy, but it aliases high-`k` power back down and decorrelates the modes it appears to keep: measured against the source, its shared-block departure is `3.89` on `O(1)` modes — total decorrelation — versus `10⁻¹⁵` for the spectral path. That failure is invisible to a variance or `P(k)` check, which is why the truncation is done in Fourier space.

Two caveats these figures do **not** address. It compares the resampled field with its own source, which is what the resampler is responsible for — it says nothing about agreement with CosmoGrid's *evolved* density, and the box here is CosmoGrid's `900 h⁻¹Mpc`, not the `5 h⁻¹Gpc` of run 4 (see *Initial conditions of the production run* above). And it cannot detect an axis transpose in the `npz → mesh` ingest, because coherence and `P(k)` are both isotropic; that check belongs to Experiment 14.

<!-- TODO: lightcone figures after the cluster runs are published to ASKabalan/jax-fli-experiments. -->

_The remaining figures — the 1LPT / 2LPT / PM shell and `C_ℓ` comparison, and the production lightcone — are pending the cluster runs and will be produced by `build.py` from the published parquet, per the standard lifecycle._

## How to run

```bash
MODE=dryrun bash run.sh   # print the eight resolved fli-launcher commands, submit nothing
bash run.sh               # submit to SLURM (MODE=sbatch default)
```

`run.sh` sources `../_launch_common.sh` and submits eight `fli-launcher → fli-simulate` jobs (idempotent — a completed output directory is skipped on rerun). Start with the 64-GPU runs (full-sky and quadrant); submit each 256-GPU production run once its 64-GPU PM looks right, and confirm the quadrant production run's peak memory from its `--perf` output before the 256-GPU submit.

Run 4 additionally needs its initial-condition catalog reachable. Jean Zay compute nodes have no internet, so build and publish it first, then pre-warm the cache on a **login** node — `snapshot_download(..., local_files_only=True)` raises on a cold cache, the same constraint as Experiments 0 and 7:

```bash
python ../14-inference-cosmogrid-shear/make_truth_ic.py       # -> truth/input_cg.parquet (832³ white)
hf upload ASKabalan/jax-fli-experiments <local> 14-inference-cosmogrid/truth/input_cg.parquet --repo-type dataset
hf download ASKabalan/jax-fli-experiments --repo-type dataset --revision main \
  --include '14-inference-cosmogrid/truth/input_cg.parquet'   # on a LOGIN node
```

Off-cluster, `--ic-input /path/to/input_cg.parquet` replaces the `--ic-repo`/`--ic-data-files` pair. The truth parquet is already published, so the `hf upload` line above is only needed if it is ever regenerated.

The initial-condition figure is local and needs no cluster time, but the `832³` white field is 2.3 GB so the measurement is split from the plotting (as in [Experiment 06](../06-cosmogrid-shells/README.md)):

```bash
# 1. Spectra (CPU, ~15 GB peak, a few minutes each) -> ic_spectra_832_{1024,512}.csv (committed, 27 kB each)
STAGE=check python prep_ic_spectra.py    # assert the resampler's contract at a small size
STAGE=dump  python prep_ic_spectra.py    # parquet -> white_832.npy (own process; the load peaks ~12 GB)
            python prep_ic_spectra.py    # -> ic_spectra_832_1024.csv  (upsample)
     NT=512 python prep_ic_spectra.py    # -> ic_spectra_832_512.csv   (downsample)

# 2. Figure (instant; reads only the committed .csv)
JAX_PLATFORMS=cpu python build.py        # -> assets/fig01-*.svg and fig02-*.svg
```

The spectra are stored as **CSV, not `.npz`** — `*.npz` and `*.npy` are gitignored repo-wide, and this artifact has to be committed so the figure can be redrawn without the 2.3 GB initial condition. Its header carries the box, the two mesh sizes and the cosmology, from which `build.py` rebuilds the linear-theory curve.

`prep_ic_spectra.py` works in half-complex (`rfftn`) Fourier space and never inverse-transforms, because calling `jax_fli.resample_white_field` at `832³ → 1024³` materialises the full complex grids and peaks around 46 GB. `STAGE=check` asserts at a small size that both paths satisfy the same contract, so the shortcut cannot silently drift from the shipped function.
