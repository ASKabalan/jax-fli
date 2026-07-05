# Experiment 05c — Spacing & stepping: equal-volume shells

**Goal.** Experiment [05a](../05a-spacing-n-stepping-drift/README.md) established the *drift on the lightcone*: moving each particle to the scale factor at which it actually crosses the lightcone removes the frozen-epoch error of a thick shell, so a coarse *drifted* lightcone matches a much finer undrifted one. 05a used **scale-factor** shell spacing, which makes the near-observer shells thin — thin shells hold few particles, so their per-shell `C_ℓ` is shot-noise dominated. This experiment swaps the spacing to **equal volume** (`--shell-spacing equal_vol`): every shell then encloses the same comoving volume, hence the same particle count, so no near shell is starved. The geometric price is that the innermost shell becomes a **fat ball** (here `[0, 1160]` Mpc/h) while the outer shells get thin and are floored to `--min-width 60`. That concentrates the frozen-epoch error into the fat inner shell — precisely where the drift buys the most. The question 05c answers is whether equal-volume spacing *plus* drift gives clean per-region `C_ℓ` across the whole lightcone.

| sweep | values |
|-------|--------|
| `--nb-shells` (no drift) | 5, 8, 10, 12, 16, 20, 25, 30, **40** |
| `--nb-shells` (with drift) | 5, 8, 10, 12, 16, 20, 25, 30, **40** |
| drift | (none), `--drift-on-lightcone` |
| `--shell-spacing` | `equal_vol` (the one change vs 05a) |

*Fixed across the sweep:* `--sim-mode pm`, **2560³**, box **`5000³` Mpc/h**, BullFrog (`bf`), `--nb-steps 50`, `--time-stepping D`, `--paint-order cic` (no force-window `--deconvolution`), `--scheme ngp`, `--nside 2048`, `--shells-per-file 1`, **`--min-width 60.0`** (the hybrid floor: equal-volume inner shells, ≥60 Mpc/h outer shells), `--seed 0`, **float64**. **128 GPU** (32 nodes × 4, `--pdim 128 1`). Runs are published to the [`ASKabalan/jax-fli-experiments`](https://huggingface.co/datasets/ASKabalan/jax-fli-experiments) dataset under `05-spacing-n-stepping/05c-equal-volume/` (raw per-shell density maps + precomputed per-shell spectra).

> **Halo (Exp 01 rule).** 2560³ on 128 GPUs (slab) gives local `20³` and an **even** halo `int(20·0.5) = 10`; local `20·2560·2560 = 1.31e8 ≈ 512³` fits float64. The physical ghost zone `0.5·5000/128 = 19.5` Mpc/h clears the end-of-run rms displacement.

## Method

Both figures compare a coarse 10-shell lightcone to a much finer one at matched radial extent.

**fig01** is a small local illustration (256³ in the real 5 Gpc/h box): one particle cloud, coloured three ways by the redshift each particle is *assigned*. The banding uses the runs' real equal-volume shell edges (reconstructed from the stored `comoving_centers ± density_width/2`).

**fig02** measures per-region density `C_ℓ`. Because equal-volume shells do not nest cleanly (the 40-shell run is floored to 60 Mpc/h and does not tile the un-floored 10-shell run), each column is a **radial region** — a group of consecutive 10-shell shells whose extent lines up with whole 40-shell edges (to within ~30 Mpc/h). The 10-shell measurement and the 40-shell reference are each summed from **whole** shells over that region (in particle counts, so per-pixel volume cancels), then converted to overdensity and transformed with `healpy`. The 40-shell no-drift run is the finest continuous-lightcone reference available (the reference is a no-drift sum by construction).

## Results

### Redshift assignment under equal-volume shells

![Redshift assignment](assets/fig01-redshift-assignment.svg)

A thin wedge of the particle cloud, coloured by assigned redshift. **Left (10 shells, no drift):** equal volume makes the innermost shell an enormous ball out to ~1160 Mpc/h — every particle in it is frozen to a single redshift, a huge discontinuity — while the outer shells thin out into narrow bands. **Middle (10 shells, with drift):** the drift replaces the freeze with the smooth true `z(r)`, so the same 10 shells now carry a continuous redshift gradient. **Right (30 shells, no drift):** more shells shrink the fat inner ball and refine the bands, approaching the smooth gradient — but reaching it by brute force needs many shells, whereas the drift already recovers it with only 10.

### Per-region density power spectra

![Density C_ell](assets/fig02-density-shells.svg)

Density `C_ℓ` for the near (fat inner ball), mid, and far regions: the 10-shell no-drift (red) and with-drift (blue) runs against the 40-shell continuous-lightcone reference (black); the lower panels show the ratio to the reference. In the **near** region the no-drift run is biased **+21%** high — the fat inner ball frozen at one epoch — and the drift pulls it back to **+2%**. In the **mid** and **far** regions the 10-shell shells are already thin, so the frozen-epoch error is sub-percent and drift and no-drift both sit on top of the reference. Equal-volume spacing thus localises the entire convergence problem to the inner ball, and the drift-on-lightcone resolves it there with a coarse 10-shell lightcone.

### Per-shell density census against theory

The region view above sums shells together; this census instead compares **every individual shell** of every run to the analytic Limber number-counts prediction (`compute_theory_cl_for_density`, comoving-volume weighted, times the HEALPix pixel window `w_ℓ²`). Each subplot is one shell: a log-log `C_ℓ` panel (no-drift red, with-drift blue, theory dashed) over a measured/theory ratio strip. Because the drift and no-drift runs share the seed and the particles, their Poisson shot noise is the *same* realisation and cancels between them — so the physical signal is the **red↔blue gap**, not the distance from 1. The shared rise of both curves above the theory at high `ℓ` is that shot noise (the theory carries none) and is meaningful only below the marked PM-Nyquist `ℓ_max ≈ πχ/dx`.

The whole story sits in the **fat inner shell**. At 5, 8 and 10 shells the innermost equal-volume shell is a ball spanning out past ~1000 Mpc/h, and its no-drift `C_ℓ` runs ~20-25% above theory — frozen at a single epoch across a huge radial span — while the drifted shell lands on the theory. Every thinner outer shell already tracks theory with or without the drift, so red and blue coincide there. As the shell count grows the inner ball shrinks and the gap closes: by 40 shells drift and no-drift agree on every shell and both follow theory down to the resolution cutoff, where the finite mesh suppresses the small-scale power common to all runs.

![Per-shell density census vs Limber theory — 5 / 8 / 10 shells](assets/fig03-density-census-small.svg)

![Per-shell density census vs Limber theory — 16 shells](assets/fig04-density-census-16.svg)

![Per-shell density census vs Limber theory — 20 shells](assets/fig05-density-census-20.svg)

![Per-shell density census vs Limber theory — 25 shells](assets/fig06-density-census-25.svg)

![Per-shell density census vs Limber theory — 30 shells](assets/fig07-density-census-30.svg)

![Per-shell density census vs Limber theory — 40 shells](assets/fig08-density-census-40.svg)

Born lensing (the single-bin convergence figure, the analogue of [05a](../05a-spacing-n-stepping-drift/README.md)'s fig09) is deferred until the equal-volume convergence maps are published for 05c.

## How to run

```bash
MODE=dryrun bash run.sh   # print the resolved commands (submit nothing)
bash run.sh               # submit the density sweep to SLURM
```

The density sweep writes one directory of per-shell parquet (`shell_NNNN.parquet`) per (drift, shell-count) to `results/exp5c/`; once pushed to HuggingFace, the figure script renders the SVGs locally without a GPU:

```bash
JAX_PLATFORMS=cpu uv run --no-sync python build.py
```
