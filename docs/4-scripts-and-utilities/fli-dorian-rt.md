# fli-dorian-rt

Post-process an existing spherical density lightcone with full multi-plane **ray-tracing** via
[dorian](https://pypi.org/project/dorian-raytrace-mpi/), capturing post-Born corrections that the
Born approximation misses. Requires the `raytrace` extra (`pip install jax-fli[raytrace]`) and is
parallelised with **MPI**.

## Usage

```bash
mpirun -n 8 fli-dorian-rt --input lightcone.parquet --output kappa/ \
       --rt-interp bilinear --with-born
```

## Arguments

Same source interface as [`fli-born-rt`](fli-born-rt.md) — local `--input` **or** HuggingFace
`--repo` + `--data-files`. Output goes to a **directory** (`--output`, default `.`) as
`RAYTRACE_<base>.parquet` (plus `RAYTRACE_<base>_born.parquet` when `--with-born` is set).

| Flag | Default | Meaning |
|------|---------|---------|
| `--input FILE_OR_GLOB` / `--repo` + `--data-files` | — | local (XOR HF) density-lightcone source |
| `--output`, `-o DIR` | `.` | output **directory** (filename auto-generated) |
| `--name` | — | label stored on the output field |
| `--rt-interp {bilinear,ngp,nufft}` | `bilinear` | deflection-field interpolation |
| `--no-parallel-transport` | off | disable parallel transport of the distortion matrix |
| `--with-born` | off | also emit the Born convergence byproduct (`RAYTRACE_<base>_born.parquet`) |
| `--nside` | native | ud_grade-downsample the density lightcone before ray-tracing |
| `--normalization {global,per_plane}` | `global` | density→δ overdensity normalization |
| `--nz-shear Z…` | `s3` | source redshifts, or `s3` for Stage-3 |
| `--min-z` / `--max-z` / `--n-integrate` | `0.01` / **`3.0`** / `32` | n(z) integration grid (note the deeper `--max-z` default vs Born's 1.5) |

`--pdim` / `--nodes` / `--gpus-per-node` are accepted but ignored (dorian is numpy + MPI, not the
JAX sharding path). `--quadrature` is likewise accepted (via the shared lensing group) but **not
forwarded** into the ray-tracer — it only affects the Born path.

Unlike Born, ray-tracing is **not** JAX-differentiable. For the differentiable first-order map use
[`fli-born-rt`](fli-born-rt.md); the two are compared in the
[Lensing notebook](../2-advanced-usage/09-Lensing.ipynb).
