# fli-infer

Run **full-field** MCMC inference: jointly sample the initial conditions and cosmological
parameters of the forward model, conditioned on an observed convergence (κ) map. This is the
most expensive — and most information-rich — inference mode.

## Usage

```bash
fli-infer --observable kappa.parquet --path chains/ \
          --num-warmup 200 --num-samples 500 --sampler NUTS --checkpoints 10
```

## Arguments

| Flag | Meaning |
|------|---------|
| `--observable` | observed κ-map catalog to condition on |
| `--path` | output directory for the MCMC chains |
| `--num-warmup` / `--num-samples` | sampler iterations |
| `--sampler {NUTS,HMC,MCLMC}` | inference kernel |
| `--checkpoints` | gradient checkpoints (memory vs compute for the reversible solver) |
| `--sigma-e` | shape-noise dispersion of the likelihood |

Plus the shared **cosmology**, **simulation**, **integration**, **lensing** and **prior** groups.
The reversible `DoubleKickDrift` solver makes the gradient through the whole N-body run memory
efficient. Post-process the chains with [`fli-extract`](fli-extract.md); the (work-in-progress)
walk-through lives under [Sampling & inference](../3-sampling-and-inference/).
