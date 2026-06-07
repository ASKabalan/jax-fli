# fli-extract

Stream MCMC sample catalogs (from [`fli-infer`](fli-infer.md) / [`fli-2pcf`](fli-2pcf.md)) and
compute **per-chain summary statistics** without loading everything into memory. Uses Welford's
online algorithm, so it scales to large chains of high-resolution fields.

## Usage

```bash
fli-extract --path chains/ --output stats/ --field-statistic --power-statistic \
            --truth truth.parquet
```

## Arguments

| Flag | Meaning |
|------|---------|
| `--path` | directory of MCMC sample catalogs (or `--repo-id` for a HuggingFace dataset) |
| `--output` | destination for the extracted statistics |
| `--field-statistic` | accumulate mean/variance of the sampled fields |
| `--power-statistic` | accumulate mean/variance of their power spectra |
| `--truth` | optional ground-truth catalog for residuals |
| `--cosmo-keys` | which cosmological parameters to summarise |
| `--ddof` | delta-degrees-of-freedom for the variance |

The result is a `CatalogExtract`, itself a Parquet catalog you can reload and plot.
