# Sampling & Inference

Probabilistic inference with `jax-fli` — building a NumPyro forward model, custom MCMC distributions, reparameterisation for bounded parameters, and full-field (MCMC) posteriors over cosmology and the initial-condition field.

- [Probabilistic Modeling](12-Probabilistic-Modeling.ipynb) — the forward-model builder, the `Configurations` dataclass, custom MCMC distributions, and the NumPyro/BlackJAX wrappers.
- [Rosenbrock](13-Rosen.ipynb) — an MCMC sampler sanity check (NUTS / MCLMC / MAMS, compared at a matched gradient budget) on a known target before touching cosmology.
- [LPT Density MCLMC](14-LPTDensityMCLMC.ipynb) — full-posterior field-level inference of `(Ω_c, σ₈)` over the initial-condition field on a 1LPT spherical galaxy-overdensity mock (seed 0), sampled with MCLMC.
- [LPT mass mapping](14-LPT-MassMapping.ipynb) — MAP reconstruction of the initial-condition field from two DES Y3 convergence maps (2LPT spherical lightcone, 416³ on one A100), at the DES Y3 source density (Part I) and at 30 galaxies/arcmin² (Part II, Stage IV); the IC is compared through its lensing-kernel sky projection, with κ coherence against the Wiener limit and starlet statistics. The frames feed [the animation](animation/).
- [Multi-host LPT mass mapping](15-LPT-Multihost-MassMapping.py) — the script counterpart of 14, running the same MAP reconstruction over a multi-host device mesh (one process per GPU via `jax.distributed`), writing the same parquet layout so the notebook's plotting sections read its output unchanged.
- [Configuration options](configurations-options.md) — the `Configurations` fields that drive the forward model.

The command-line entry points wrap the same pipeline for batch / HPC runs, documented under [Scripts & utilities](../4-scripts-and-utilities/README.md):

- [`fli-samples`](../4-scripts-and-utilities/fli-samples.md) — prior-predictive sampling
- [`fli-infer`](../4-scripts-and-utilities/fli-infer.md) — full-field MCMC (NUTS / MCLMC)
- [`fli-2pcf`](../4-scripts-and-utilities/fli-2pcf.md) — power-spectrum-level MCMC
- [`fli-extract`](../4-scripts-and-utilities/fli-extract.md) — per-chain statistics
