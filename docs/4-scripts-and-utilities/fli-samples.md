# fli-samples

Draw **prior-predictive** samples from a `jax_fli` probabilistic model — i.e. forward-simulate
observables (κ maps, spectra, fields) by sampling the prior over cosmology and initial
conditions. Useful for building training sets and sanity-checking a forward model before
inference.

## Usage

```bash
fli-samples --model full_field --num-samples 100 --path samples/ --batch-id 0
```

## Arguments

| Flag | Meaning |
|------|---------|
| `--model` | which probabilistic model to sample |
| `--num-samples` | samples to draw |
| `--path` | output directory (Parquet catalogs) |
| `--batch-id` | shard id for embarrassingly-parallel batches |
| `--initial-condition` | fix / sample the initial-condition field |
| `--sigma-e` | shape-noise dispersion added to mock observables |

Plus the shared **cosmology**, **simulation**, **integration** and **prior** groups. The
inference tutorials that consume these samples live under
[Sampling & inference](../3-sampling-and-inference/) (coming soon).
