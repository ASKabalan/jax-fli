# Sampling & Inference

Probabilistic inference with `jax-fli` — building a NumPyro forward model, custom MCMC
distributions, reparameterisation for bounded parameters, and full-field vs power-spectrum-level
posteriors.

- [Probabilistic Modeling](12-Probabilistic-Modeling.ipynb) — the forward-model builder, the
  `Configurations` dataclass, and NumPyro/BlackJAX wrappers.
- [Rosenbrock](13-Rosen.ipynb) — an MCMC sanity check on a known target before touching cosmology.
- [LPT Lensing Inference](14-LPTLensingInference.ipynb) — a small end-to-end Bayesian posterior
  over cosmology + initial conditions from an LPT lensing map.
- [Full-Field Inference](15-FullFieldInference.ipynb) — full-field posterior with the PM forward
  model.
- [Configuration options](configurations-options.md) — the `Configurations` fields that drive the
  forward model.

The command-line entry points wrap the same pipeline for batch / HPC runs, documented under
[Scripts & utilities](../4-scripts-and-utilities/README.md):

- [`fli-samples`](../4-scripts-and-utilities/fli-samples.md) — prior-predictive sampling
- [`fli-infer`](../4-scripts-and-utilities/fli-infer.md) — full-field MCMC
- [`fli-2pcf`](../4-scripts-and-utilities/fli-2pcf.md) — power-spectrum-level MCMC
- [`fli-extract`](../4-scripts-and-utilities/fli-extract.md) — per-chain statistics
