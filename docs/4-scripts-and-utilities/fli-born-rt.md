# fli-born-rt

Post-process an existing spherical density lightcone with the **Born approximation**, writing a
`SphericalKappaField` convergence catalog. This is the batch equivalent of `jfli.born` (see the
[Lensing notebook](../2-advanced-usage/09-Lensing.ipynb)). Fully JAX, runs distributed like the
rest of the pipeline.

## Usage

```bash
# local density lightcone → convergence, written into the current directory (default: simpson quadrature)
fli-born-rt --input lightcone.parquet --nz-shear s3

# stream the lightcone from a HuggingFace dataset instead, with Gauss-Legendre quadrature + timing
fli-born-rt --repo ASKabalan/jax-fli-experiments \
            --data-files '05-.../density/exp5c_drift_16/*.parquet' --output kappa/ \
            --quadrature gauss_legendre --perf --iterations 3
```

## Arguments

The density source is the shared source interface — **either** a local `--input` file/glob **or**
a HuggingFace `--repo` + `--data-files` (mutually exclusive). The output is written to a
**directory** (`--output`, default `.`) with an auto-generated name `BORN_<name or M…_B…_N…>.parquet`.

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
| `--quadrature {midpoint,simpson,gauss_legendre}` | `simpson` | Born quadrature, applied to the per-shell weights **and** the n(z) integral — `simpson` = composite Simpson over each shell + Simpson n(z) grid; `gauss_legendre` = GL-16 over each shell + GL n(z) nodes; `midpoint` = the historic legacy route (kernel at shell centers + Simpson n(z)) |
| `--perf` | off | benchmark: compile + `--iterations` timed runs of the jitted Born call, JaxTimer CSV written to `<output-parent>/perf_born.csv` (the kappa parquet is still saved) |
| `--iterations`, `-i N` | `5` | number of timed iterations for `--perf` |
| `--enable-x64` | off | double precision |
| `--pdim PX PY` / `--nodes` / `--gpus-per-node` | `1 1` / `1` / — | distributed Born process mesh |

`--quadrature` matters most for coarse or equal-volume shells, where wide near shells make the
midpoint rectangle over-weight the true kernel — see `jfli.plot_born_windows` and
[Exp 05c](../5-experiments/05c-spacing-n-stepping-equal-vol/README.md). The multi-node rules
(`simpson`, `gauss_legendre`) are numerically interchangeable (both sit at the `a_of_chi`
interpolation floor); only the single-node `midpoint` is biased on thick shells. For the post-Born
ray-traced version use [`fli-dorian-rt`](fli-dorian-rt.md).
