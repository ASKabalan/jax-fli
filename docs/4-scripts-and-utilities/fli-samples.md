# fli-samples

Draw **prior-predictive** samples from a `jax_fli` probabilistic model — i.e. forward-simulate observables (κ maps, spectra, fields) by sampling the prior over cosmology and initial conditions. Useful for building training sets and sanity-checking a forward model before inference.

## Usage

```bash
fli-samples --model full --num-samples 100 --path samples/ --batch-id 0
```

## Arguments

| Flag | Meaning |
|------|---------|
| `--model {full,mock}` | which probabilistic model to sample (default `full`) |
| `--num-samples` | samples to draw |
| `--path` | output directory (Parquet catalogs) |
| `--batch-id` | shard id for embarrassingly-parallel batches |
| `--initial-condition` | fix / sample the initial-condition field |
| `--sigma-e` | shape-noise dispersion added to mock observables |

Plus the shared **cosmology**, **simulation**, **integration**, **lensing** (`--nz-shear --quadrature …`), **forward model** (`--lensing-output {convergence,shear,reduced_shear}`, `--mask`, `--sigma-unobserved`, `--map2alm-method`) and **prior** (`--sample {cosmo,ic}`, `--prior-omega-c …`) groups. The inference tutorials that consume these samples live under [Sampling & inference](../3-sampling-and-inference/README.md) (coming soon).
