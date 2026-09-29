# Sampling & Inference

Probabilistic inference with `jax-fli` — building a NumPyro forward model, custom MCMC distributions, reparameterisation for bounded parameters, and full-field (MCMC) posteriors over cosmology and the initial-condition field.

- [Probabilistic Modeling](12-Probabilistic-Modeling.ipynb) — the forward-model builder, the `Configurations` dataclass, custom MCMC distributions, and the NumPyro/BlackJAX wrappers.
- [Rosenbrock](13-Rosen.ipynb) — an MCMC sampler sanity check (NUTS / MCLMC / MAMS, compared at a matched gradient budget) on a known target before touching cosmology.
- [LPT mass mapping](14-LPT-MassMapping.ipynb) — MAP reconstruction of the initial-condition field from two DES Y3 convergence maps (2LPT spherical lightcone, 416³ on one A100), at the DES Y3 source density (Part I) and at 30 galaxies/arcmin² (Part II, Stage IV); the IC is compared through its lensing-kernel sky projection, with κ coherence against the Wiener limit and starlet statistics. The frames feed [the animation](animation/).
- [Multi-host LPT mass mapping](15-LPT-Multihost-MassMapping.py) — the script counterpart of 14, running the same MAP reconstruction over a multi-host device mesh (one process per GPU via `jax.distributed`), writing the same parquet layout so the notebook's plotting sections read its output unchanged.
- [LPT field-level inference of the initial conditions](16-LPT-FieldLevel-IC.ipynb) — the Euclid model of notebook 14 (2LPT spherical lightcone, two κ maps) sampled with MCLMC over the joint posterior of `(Ω_c, σ₈)` and the initial-condition field at 128³ on one A100 : the IC in 3D and in lensing-kernel projection (truth, mean, standard deviation, residual), its transfer function and coherence, the posterior-predictive κ, its coherence and transfer function, and the starlet ℓ₁ norm.
- [LPT field-level inference of the cosmology](17-LPT-FieldLevel-Cosmology.ipynb) — NUTS on `(Ω_c, σ₈)` with the white-noise IC of notebook 16 fixed, either at its posterior mean or at the truth: the conditional cosmology posterior against the joint chain, cross-checked on a grid of the log-posterior.
- [Configuration options](configurations-options.md) — the `Configurations` fields that drive the forward model.

The command-line entry points wrap the same pipeline for batch / HPC runs, documented under [Scripts & utilities](../4-scripts-and-utilities/README.md):

- [`fli-samples`](../4-scripts-and-utilities/fli-samples.md) — prior-predictive sampling
- [`fli-infer`](../4-scripts-and-utilities/fli-infer.md) — full-field MCMC (NUTS / MCLMC)
- [`fli-2pcf`](../4-scripts-and-utilities/fli-2pcf.md) — power-spectrum-level MCMC
- [`fli-extract`](../4-scripts-and-utilities/fli-extract.md) — per-chain statistics
