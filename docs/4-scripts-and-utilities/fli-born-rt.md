# fli-born-rt

Post-process an existing spherical density lightcone with the **Born approximation**, writing a `SphericalKappaField` convergence catalog. This is the batch equivalent of `jfli.born` (see the [Lensing notebook](../2-advanced-usage/09-Lensing.ipynb)). Fully JAX, runs distributed like the rest of the pipeline.

## Usage

```bash
# local density lightcone → convergence, written into the current directory
fli-born-rt --input lightcone.parquet --nz-shear s3 --quadrature gauss_legendre

# stream the lightcone from a HuggingFace dataset instead
fli-born-rt --repo ASKabalan/jax-fli-experiments \
            --data-files '05-.../density/exp5c_drift_16/*.parquet' --output kappa/
```

## Arguments

The density source is the shared source interface — **either** a local `--input` file/glob **or** a HuggingFace `--repo` + `--data-files` (mutually exclusive). The output is written to a **directory** (`--output`, default `.`) with an auto-generated name `BORN_<name or M…_B…_N…>.parquet`.

| Flag | Default | Meaning |
|------|---------|---------|
| `--input FILE_OR_GLOB` | — | local density-lightcone parquet path/glob (XOR `--repo`) |
| `--repo REPO_ID` + `--data-files GLOB` | — | HuggingFace dataset source (parquet glob inside the repo) |
| `--output`, `-o DIR` | `.` | output **directory** (filename auto-generated) |
| `--name` | — | label stored on the output field |
| `--nside` | native | ud_grade-downsample the density lightcone before lensing |
| `--normalization {global,per_plane}` | `global` | density→δ overdensity normalization (`global` = one mean over all shells) |
| `--nz-shear Z…` | `s3` | source redshifts, or `s3` for the Stage-3 tomographic n(z) |
| `--min-z` / `--max-z` / `--n-integrate` | `0.01` / `1.5` / `32` | n(z) integration grid |
| `--quadrature {midpoint,gauss_legendre}` | `midpoint` | per-shell Born weight quadrature — `midpoint` = kernel at shell centers (historic); `gauss_legendre` = exact per-shell integral |
| `--enable-x64` | off | double precision |
| `--pdim PX PY` / `--nodes` / `--gpus-per-node` | `1 1` / `1` / — | distributed Born process mesh |

`--quadrature` matters most for coarse or equal-volume shells, where wide near shells make the midpoint rectangle over-weight the true kernel — see `jfli.plot_born_windows` and [Exp 05c](../5-experiments/05c-spacing-n-stepping-equal-vol/README.md). For the post-Born ray-traced version use [`fli-dorian-rt`](fli-dorian-rt.md).
