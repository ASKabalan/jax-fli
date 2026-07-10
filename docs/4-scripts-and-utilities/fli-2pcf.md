# fli-2pcf

Power-spectrum-level (two-point) MCMC inference. The map is compressed to its angular power spectrum once, then the cosmology is sampled against a Gaussian (Knox) likelihood — far cheaper than [`fli-infer`](fli-infer.md) because no forward simulation runs inside the MCMC loop.

## Usage

```bash
fli-2pcf --observable kappa.parquet --path chains/ \
         --lmax 512 --f-sky 1.0 --sigma-e 0.26 \
         --num-warmup 100 --num-samples 500 --sampler NUTS
```

## Arguments

| Flag | Meaning |
|------|---------|
| `--observable` | κ-map (or spectra) catalog to condition on |
| `--path` | output directory for the chains |
| `--nside` / `--flatsky-npix` / `--field-size` | geometry of the observable |
| `--lmax` | maximum multipole for the data vector |
| `--f-sky` | observed sky fraction (covariance scaling) |
| `--sigma-e` | shape-noise dispersion |
| `--nonlinear-fn {halofit,linear}` | theory power spectrum model |
| `--num-warmup` / `--num-samples` / `--batch-count` / `--chain-index` | sampler iterations / chain shard |
| `--sampler {NUTS,MCLMC}` | inference kernel |

Plus the shared **lensing** (`--nz-shear --min-z --max-z --n-integrate --quadrature`) and **prior** groups — there is no cosmology group here; the sampled cosmology comes from the priors over a Planck18 fiducial.
