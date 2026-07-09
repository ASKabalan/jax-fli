# fli-extract

Stream MCMC sample catalogs (from [`fli-infer`](fli-infer.md) / [`fli-2pcf`](fli-2pcf.md)) and
compute **per-chain summary statistics** without loading everything into memory. Uses Welford's
online algorithm, so it scales to large chains of high-resolution fields.

## Usage

```bash
# Local: point at a root holding chain_N/samples/*.parquet (auto-detected),
# or pass one parquet glob per chain.
fli-extract --input chains/ --output stats/ --name run1 --cosmo-keys Omega_c sigma8 \
            --field-statistic --power-statistic --truth truth.parquet

# HuggingFace: one --data-files glob per chain inside the repo.
fli-extract --repo user/repo \
            --data-files 'run1/chain_0/**/*.parquet' 'run1/chain_1/**/*.parquet' \
            --output stats/ --name run1 --cosmo-keys Omega_c sigma8
```

## Arguments

The source is the shared `add_source_args` interface — local `--input` **XOR** HuggingFace `--repo` +
`--data-files`, same as [`fli-born-rt`](fli-born-rt.md) / [`fli-dorian-rt`](fli-dorian-rt.md). Here it
is **multi-pattern**: **each pattern is one MCMC chain**, streamed and accumulated independently so the
per-chain statistics stay separate (a single pooled glob would mix chains and compute a *different*,
wrong statistic). A single local `--input` root directory auto-expands its `chain_N/samples/*.parquet`
subdirectories.

| Flag | Meaning |
|------|---------|
| `--input` | local parquet glob(s)/dir(s) — one pattern per chain (a root dir auto-detects `chain_N/`) |
| `--repo` + `--data-files` | HuggingFace dataset repo + one glob per chain inside it |
| `--name` | name label stored on the resulting `CatalogExtract` |
| `--output` | destination for the extracted statistics |
| `--field-statistic` | accumulate mean/variance of the sampled fields |
| `--power-statistic` | accumulate mean/variance of their power spectra (requires `--truth`) |
| `--truth` | optional ground-truth catalog for transfer/coherence spectra |
| `--cosmo-keys` | which cosmological parameters to summarise |
| `--ddof` | delta-degrees-of-freedom for the variance |

The result is a `CatalogExtract`, itself a Parquet catalog you can reload and plot.
