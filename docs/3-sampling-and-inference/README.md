# Sampling & Inference

Probabilistic inference with `jax-fli` — building a NumPyro forward model, custom MCMC distributions, reparameterisation for bounded parameters, and full-field (MCMC) posteriors over cosmology and the initial-condition field.

- [Probabilistic Modeling](12-Probabilistic-Modeling.ipynb) — the forward-model builder, the `Configurations` dataclass, custom MCMC distributions, and the NumPyro/BlackJAX wrappers.
- [Rosenbrock](13-Rosen.ipynb) — an MCMC sampler sanity check (NUTS / MCLMC / MAMS, compared at a matched gradient budget) on a known target before touching cosmology.
- [LPT MAP estimation](14-LPT-MAP-Estimation.ipynb) — dev notebook: joint MAP minimization of `(Ω_c, σ₈)` and the initial-condition field with cadre, plus a cheap Laplace uncertainty from the conditioned potential.
- [Configuration options](configurations-options.md) — the `Configurations` fields that drive the forward model.

The command-line entry points wrap the same pipeline for batch / HPC runs, documented under [Scripts & utilities](../4-scripts-and-utilities/README.md):

- [`fli-samples`](../4-scripts-and-utilities/fli-samples.md) — prior-predictive sampling
- [`fli-infer`](../4-scripts-and-utilities/fli-infer.md) — full-field MCMC (NUTS / MCLMC)
- [`fli-2pcf`](../4-scripts-and-utilities/fli-2pcf.md) — power-spectrum-level MCMC
- [`fli-extract`](../4-scripts-and-utilities/fli-extract.md) — per-chain statistics
