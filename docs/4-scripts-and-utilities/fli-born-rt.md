# fli-born-rt

Post-process an existing spherical lightcone Parquet catalog with the **Born approximation**,
writing a `SphericalKappaField` convergence catalog. This is the batch equivalent of `jfli.born`
(see the [Lensing notebook](../2-advanced-usage/07-Lensing.ipynb)).

## Usage

```bash
fli-born-rt --input lightcone.parquet --output kappa_born.parquet
```

## Arguments

| Flag | Meaning |
|------|---------|
| `--input` | lightcone catalog (a `SphericalDensity` with `status=LIGHTCONE`) |
| `--output` | destination convergence catalog |
| `--enable-x64` | run in double precision |

Source redshift distributions and the integration range come from the shared **lensing** group
(`--min-z --max-z --n-integrate`). Born is fully JAX and runs distributed like the rest of the
pipeline; for the post-Born ray-traced version use [`fli-dorian-rt`](fli-dorian-rt.md).
