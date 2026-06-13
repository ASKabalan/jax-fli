# fli-dorian-rt

Post-process an existing spherical lightcone with full multi-plane **ray-tracing** via
[dorian](https://pypi.org/project/dorian-raytrace-mpi/), capturing post-Born corrections that the
Born approximation misses. Requires the `raytrace` extra (`pip install jax-fli[raytrace]`) and is
parallelised with **MPI**.

## Usage

```bash
mpirun -n 8 fli-dorian-rt --input lightcone.parquet --output kappa_rt.parquet \
       --rt-interp bilinear
```

## Arguments

| Flag | Meaning |
|------|---------|
| `--input` | lightcone catalog |
| `--output` | destination convergence catalog |
| `--rt-interp {bilinear,ngp,nufft}` | deflection-field interpolation |
| `--no-parallel-transport` | disable parallel transport of the distortion matrix |

Unlike Born, ray-tracing is **not** JAX-differentiable. For the differentiable first-order map use
[`fli-born-rt`](fli-born-rt.md); the two are compared in the
[Lensing notebook](../2-advanced-usage/09-Lensing.ipynb).
