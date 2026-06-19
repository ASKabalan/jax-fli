# Experiment 06 — Match the CosmoGrid density shells (DES Y3 depth)

## Goal

Simulate, with jax-fli's PM engine, the **same radial density shells as CosmoGrid at nside 2048**
(the reference of [Experiment 00](../00-cosmogrid-reference/)) — i.e. reproduce CosmoGrid's shell
`z`-edges exactly — so the per-shell density `C_ℓ` and shell-to-shell cross-correlation can be
compared directly, isolating **geometry** (shell placement) from resolution and painting effects.

How *deep* in redshift we must simulate is set by the **DES Y3** weak-lensing source bins (not the
Stage-3 forecast): the box only needs to contain the structure that lenses those sources. We size a
**2×2 grid of runs** — two source depths × two observer placements:

- **2-bin set** — DES Y3 bins 1+2 (shallower, `z ≲ 0.82`).
- **3-bin set** — DES Y3 bins 1+2+3 (deeper, `z ≲ 1.06`). Bin 4 is dropped (its sources sit too far,
  forcing a deeper, coarser box — see below).
- **full sky** — observer at the box centre `(0.5, 0.5, 0.5)` → an isotropic `2r` cube.
- **big quadrant** — the [Experiment 08](../08-masked-shear/) corner geometry (one centred axis, two
  corner axes; the DES footprint sits entirely inside the visible cone). It is reoriented so the
  centred axis comes **first** — observer `(0.5, 0.1, 0.9)` — which lets the quadrant pack the **same
  2560³ cell budget** into its smaller volume at **finer, isotropic** resolution (see §4).

## Method

Everything below is computed by [`prep_geometry.py`](prep_geometry.py) (CPU only) and written into
`geometry.sh`, which [`run.sh`](run.sh) sources — the four launches carry no hand-typed numbers.

**1. Source depth from DES Y3 `n(z)`.** Each tomographic bin's `n(z)` (`jax_fli.data.get_des_y3_nz_shear`)
shares one `z`-grid (0.005–2.995), so its nominal `zmax` can't distinguish bins. We instead take each
bin's **effective end** — the last redshift where `n(z) ≥ 10 % of its peak`. The DES Y3 bins carry a
thin (~0.5–1 % of peak) high-`z` noise floor, so a 1–2 % cut clips to the grid edge and even inverts
the bin order; **10 % of peak** is robust, monotonic, and lands bin 3 at `z ≈ 1.06` — matching the
"bin 3 reaches `z ≈ 1`" read off the curve. The box scales directly with this choice, so the table
also lists the 5 %/20 % bracket; **the 10 % cut deliberately trims the upper tail** (at `z = 1.0`,
bin 3 is still ~34 % of peak; it falls to ~2 % only by `z ≈ 1.2`), which is the right trade for keeping
the box affordable.

| DES Y3 bin | `z_mean` | end (10 % peak) | (5 %) | (20 %) | |
|:--:|:--:|:--:|:--:|:--:|:--|
| 1 | 0.33 | 0.62 | 0.70 | 0.56 | |
| 2 | 0.52 | **0.82** | 0.99 | 0.75 | → 2-bin depth |
| 3 | 0.74 | **1.06** | 1.14 | 1.01 | → 3-bin depth |
| 4 | 0.93 | 1.45 | 1.51 | 1.30 | excluded (box → 6171³, dx → 2.4) |

**2. Box from redshift + observer.** `jax_fli.compute_box_size_from_redshift(cosmo, z_max, observer)`
returns `(L_x, L_y, L_z) = factor · r(z_max)` with `factor_i = 1 + 2·min(f_i, 1−f_i)` and
`r = ` radial comoving distance (CosmoGrid fiducial cosmology, to match its distances). The full-sky
observer `(0.5,0.5,0.5)` gives an isotropic `2r` cube; the quadrant `(0.5,0.1,0.9)` keeps the centred
axis at `2.0r` and clips the other two to `1.2r` (one-sided) → a `(2.0r, 1.2r, 1.2r)` box (the centred
axis first, so it is the one sharded and painted — see §4).

**3. CosmoGrid shell edges.** The published nside-2048 density (HuggingFace
`ASKabalan/jax-fli-experiments`, configs `00-cosmogrid-density-00..03`) is loaded back, **down-sampled
to nside 4** purely to free memory (we only need the per-shell metadata), and each shell's comoving
edges → scale-factor edges `a_near > a_far`. For each set we keep the shells fully inside the box
(far edge `≤ r(z_max)`) and emit them as `--ts-near` / `--ts-far`. The 3-bin selection (46 shells)
nests the 2-bin one (40 shells); both run from `z = 0`.

**4. Mesh & GPU layout (float64, 128 GPUs each).** The full-sky runs use a fixed **2560³** mesh (the
[`m2560`](../../000_RUNS/results/exp1/) template) on **128 GPUs** (`--pdim 128 1`, 32 nodes); the box
grows with depth, so `dx = L/2560` is 1.61 (2-bin) / 1.94 (3-bin). The quadrant packs the **same
2560³ cell budget** (≈1.31×10⁸ cells/GPU, the float64 ceiling) into its smaller `(2.0r, 1.2r, 1.2r)`
volume at **isotropic** `dx`: the `2.0 : 1.2 : 1.2` ratio at that budget gives **3584 × 2160 × 2160**
(centred axis first). Because the quadrant only spans `1.2r` on two axes, the same cells buy a **finer**
`dx ≈ 1.15` (2-bin) / `1.39` (3-bin) — closer to CosmoGrid's own. A `(devices, 1)` mesh shards the
spherical npix over the first (M) axis, so the centred axis is placed first (`--observer-position
0.5 0.1 0.9`); the nside-2048 lightcone is then **sharded** over 128 (~72 MB/device for ~46 shells) and
adds negligibly to the at-ceiling float64 mesh. *(A later masking analysis must use this same observer
to recover the footprint.)*

## Results

The figure below records the whole geometry decision: the DES Y3 source `n(z)` and lensing efficiency
`q(z)`, the two depth cuts, and which CosmoGrid shells each set simulates.

![Experiment 06 geometry](assets/exp06-geometry.svg)

- **Top —** DES Y3 `n(z)` per bin (dotted = each bin's 10 %-of-peak end); the dashed lines are the
  2-bin (`z = 0.82`) and 3-bin (`z = 1.06`) depths. The top axis is comoving distance `χ`.
- **Middle —** weak-lensing efficiency `q(z)`: support ends at the source plane, confirming the box
  must reach the chosen depth.
- **Bottom —** the CosmoGrid shell tessellation in `z`: 40 shells inside the 2-bin box (blue),
  6 more added for the 3-bin box (orange), the rest excluded (grey).

**The four runs:**

| Run | depth | observer | box [`h⁻¹`Mpc] | mesh | `dx` [`h⁻¹`Mpc] | shells | GPUs |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 2-bin · full sky | `z ≤ 0.82` | (0.5, 0.5, 0.5) | 4114³ | 2560³ | 1.61 | 40 | 128 |
| 2-bin · quadrant | `z ≤ 0.82` | (0.5, 0.1, 0.9) | 4114 × 2469 × 2469 | 3584 × 2160 × 2160 | 1.15 | 40 | 128 |
| 3-bin · full sky | `z ≤ 1.06` | (0.5, 0.5, 0.5) | 4972³ | 2560³ | 1.94 | 46 | 128 |
| 3-bin · quadrant | `z ≤ 1.06` | (0.5, 0.1, 0.9) | 4972 × 2983 × 2983 | 3584 × 2160 × 2160 | 1.39 | 46 | 128 |

**Resolution caveat — where the comparison is geometry-limited.** This experiment isolates geometry, so
the resolution budget matters. A force-mesh cell `dx` at comoving distance `d` subtends a Nyquist
multipole `ℓ_max ≈ π·d/dx`, ranging per shell from the innermost to the outermost: **full sky
`ℓ ≈ 31–37` → `≈ 3850–3880`**, and the finer **quadrant `ℓ ≈ 43–52` → `≈ 5400`** — the quadrant nearly
reaches nside-2048's `ℓ ≈ 6000`, the full-sky runs sit below it. So the nside-2048 maps over-resolve the
full-sky mesh (most severely the near shells), and any `C_ℓ` disagreement with CosmoGrid above each
shell's `ℓ_max` is resolution/painting, not geometry — restrict the comparison to `ℓ ≲ ℓ_max(shell)`.
Note CosmoGrid itself is a 900 `h⁻¹`Mpc / 832³ run (particle spacing ≈ 1.08 `h⁻¹`Mpc) **tiled** to fill
the lightcone; we instead use a single ~4–5 `h⁻¹`Gpc box with **no tiling**. The quadrant's `dx ≈ 1.15`
is now comparable to CosmoGrid's spacing, so the clean comparison trades only the full-sky runs' coarser
small-scale resolution for our faithful (untiled) large-scale modes.

## How to run

```bash
# 1. Geometry prep (CPU): writes geometry.sh + assets/exp06-geometry.svg
python prep_geometry.py

# 2. Inspect the four resolved fli-launcher commands without submitting
MODE=dryrun bash run.sh

# 3. Submit to SLURM (writes results/exp6/cosmogrid_{2bin,3bin}_{fullsky,quadrant}.parquet)
bash run.sh
```

The wall-times in `run.sh` are first-guess estimates — tune after the first run. The catalogs are
then published to HuggingFace and studied locally against the CosmoGrid reference, per the
[experiments lifecycle](../CLAUDE.md).

> **Gate before cluster hours.** `MODE=dryrun` only *resolves* the four commands — it does not run the
> simulator. The `m2560` template validated a *cubic* mesh, nside 512, centred observer; these runs add
> three untested axes at once — a **non-cubic** `3584×2160×2160` mesh, **nside-2048** spherical painting,
> and an **off-centre** observer. Smoke-test the plumbing locally first (a small *cubic* run, then a
> small *non-cubic* run, at tiny mesh/nside with a handful of `ts` edges) before committing 128-GPU
> hours; only non-cubic painting *under multi-host sharding* can't be reproduced locally.
