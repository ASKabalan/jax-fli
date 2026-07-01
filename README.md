# jax-fli

[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue?logo=readthedocs)](https://jax-fli.readthedocs.io/en/latest/)
[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-jax--fli--experiments-yellow)](https://huggingface.co/datasets/ASKabalan/jax-fli-experiments)
[![Results Explorer](https://img.shields.io/badge/%F0%9F%A4%97%20Results-Explorer-yellow?)](https://askabalan-jax-fli-results.hf.space/)

**Differentiable cosmological forward modeling on JAX**

<p align="center">
  <img src="assets/PIPELINE.png" alt="jax-fli differentiable forward model: priors on cosmology and initial conditions evolved through LPT / PM N-body to a light-cone shell, lensed to convergence and shear, and compared to the observable" width="100%">
</p>

## Overview

`jax-fli` is a JAX toolkit for end-to-end differentiable cosmological simulations. It chains initial conditions, Lagrangian Perturbation Theory, Particle-Mesh N-body integration, lightcone painting (3D, flat-sky, HEALPix), gravitational lensing — convergence **and** shear, via the Born approximation or ray-tracing — and angular power spectrum estimation into a single differentiable pipeline. The library supports multi-GPU distribution via JAX sharding, reversible solvers for memory-efficient backpropagation, and probabilistic inference with BlackJAX/NumPyro.


<p align="center">
  <img src="assets/depgraph.png" alt="jax-fli package ecosystem: jaxdecomp and jax_cosmo underpin jaxpm, which underpins jax-fli; jax-fli drives sampling (blackjax, numpyro), cataloging (datasets, arrow), summary statistics (pysc), and ray-tracing (dorian), plus the jax_healpy / s2fft spherical-harmonics branch. Nodes are colored by contribution: authored / lead, contributed, used." width="40%">
</p>

## Key Features

- **N-body solvers** -- Pluggable symplectic integrators: reversible `DoubleKickDrift` (KKD), `DriftKickDrift`, and `BullFrog`
- **Painting targets** -- 3D density, flat-sky 2D projection, and HEALPix spherical maps with CIC / bilinear / NGP / RBF schemes
- **Interpolation kernels** -- `DriftInterp`, `OnionTiler`, and `TelephotoInterp` for on-the-fly lightcone construction beyond the box boundary
- **Correction kernels** -- PGD (position-based) and Sharpening (velocity-based, reversible) for sub-grid halo correction
- **Weak lensing** -- Born approximation (fully JIT-able) and ray-tracing (via Dorian) for convergence *and* Kaiser–Squires shear maps
- **Multi-GPU** -- Distributed simulations via JAX sharding with automatic halo exchange
- **Immutable Field PyTrees** -- `DensityField`, `ParticleField`, `FlatDensity`/`SphericalDensity`, and `FlatKappaField`/`SphericalKappaField`/`SphericalShearField` carry arrays + metadata through the pipeline
- **Summary statistics** -- 3D $P(k)$, angular $C_\ell$ (convergence and spin-2 shear EE/EB/BB), transfer functions, coherence, and Halofit theory, plus PDF, peak counts, and starlet coefficients
- **Probabilistic inference** -- Deterministic forward-model builder + NumPyro wrappers + BlackJAX batched/distributed sampling
- **I/O** -- Orbax checkpointing, Parquet serialization, HuggingFace Dataset integration, CosmoGrid/GowerStreet loaders



## Installation

This project is managed with [uv](https://docs.astral.sh/uv/) and a committed
`uv.lock` for reproducibility. Dependency groups are **not** installed by default —
request them explicitly:

```bash
uv sync                              # runtime package only (base dependencies)
uv sync --group dev                  # + dev tooling: ruff, pyright, prek, toml-sort
uv sync --group tests                # + test suite: pytest, ALL feature extras, and the
                                     #   reference/oracle backends (fastpm, pmesh, glass, ...)
uv sync --group dev --group tests    # everything for development AND testing
```

Individual feature extras can also be picked, e.g. `uv sync --extra raytrace`,
`--extra catalog`, `--extra sampling`; `--extra cuda` installs a CUDA build of JAX
(the base install is CPU).

Run anything inside the environment with `uv run` (e.g. `uv run fli-simulate ...`,
`uv run pytest`), or activate `.venv` directly.

> **Note:** This package depends on custom forks of `jaxpm` and `jax-cosmo`. They are
> declared by their PyPI names in `pyproject.toml` and pinned to the right git
> branches via `[tool.uv.sources]`, which **only `uv` reads** — installing with plain
> `pip` would fetch the upstream PyPI releases instead of the forks, so use `uv`.

Recommended installation : `uv sync --extra all`

If you have a CUDA-capable GPU, you can also install JAX with GPU support:

```bash
uv sync --extra all --extra cuda
```

__note__: You can also compile the spherical harmonics cuda kernel in s2fft just by making sure that you have a working CUDA compiler before running the above command.

## Quick Start

```python
import jax
import jax_cosmo as jc
import jax_fli as jfli

key = jax.random.PRNGKey(42)
cosmo = jc.Planck18()

# Size the box from a target redshift so the light-cone fully covers the source bins.
box_size = jfli.compute_box_size_from_redshift(cosmo, 1.5, observer_position=(0.5, 0.5, 0.5))
box_size = tuple(float(b) for b in box_size)

# 1. Gaussian initial conditions (nside enables the HEALPix / spherical projection).
initial_field = jfli.gaussian_initial_conditions(
    key, mesh_size=(256, 256, 256), box_size=box_size, cosmo=cosmo, nside=256,
)

# 2. LPT displacement + momentum at the simulation start epoch.
#    (ts must match the solver's t0 below; nbody validates this.)
dx, p = jfli.lpt(cosmo, initial_field, ts=0.001, order=1)

# 3. PM N-body with a spherical (HEALPix) light-cone output.
#    Time control (t0, t1, n_steps) lives on the solver, not on nbody().
solver = jfli.DoubleKickDrift(
    interp_kernel=jfli.NoInterp(painting=jfli.PaintingOptions(target="spherical")),
    t0=0.001, t1=1.0, n_steps=16,
)
lightcone = jfli.nbody(cosmo, dx, p, nb_shells=8, solver=solver)

# 4. DES-like tomographic n(z): 2 source bins around z ~ 0.8 (DES Y3 bins 3 & 4).
nz_sources = jfli.io.get_des_y3_nz_shear()[2:4]

# 5. Born convergence -> Kaiser-Squires shear -> angular power spectrum.
kappa = jfli.born(cosmo, lightcone, nz_shear=nz_sources)   # SphericalKappaField, (2, npix)
shear = kappa.get_shear(method="jax")                      # SphericalShearField, (2, 2, npix)
cl = kappa.angular_cl(method="healpy")                     # C_ell^kk per bin
```

## Documentation

Full tutorials and reference live in **[`docs/notebooks.md`](docs/notebooks.md)** — executable
notebooks plus markdown guides, grouped into Introduction & basics, Advanced usage, Sampling &
inference, Command-line tools, and Experiments. The notebooks are committed with **small-scale CPU
outputs** (production parameters left in the code); rerun them on a GPU / cluster to reproduce the
full-resolution figures. The rendered docs are hosted on
[Read the Docs](https://jax-fli.readthedocs.io/).

## Module Map

| Module | Purpose |
|--------|---------|
| `fields/` | Immutable PyTree containers (`DensityField`, `ParticleField`, `FlatDensity`/`SphericalDensity`, `FlatKappaField`/`SphericalKappaField`/`SphericalShearField`) |
| `initial.py` | Gaussian initial conditions and interpolation to mesh |
| `pm/` | Particle-mesh engine: `lpt()`, `nbody()`, symplectic solvers, interpolation + PGD/Sharpening kernels, integration loop |
| `lensing/` | Born approximation and Dorian ray-tracing (convergence and shear) |
| `summary_statistics/` | P(k), angular C_ell, transfer, coherence, Halofit theory, PDF, peak counts, starlet (importable as `jfli.power` for back-compat) |
| `probabilistic_models/` | Deterministic forward-model builder, NumPyro wrappers, `Configurations` dataclass (exposed as `jfli.ppl`) |
| `infer/` | Batched / distributed MCMC sampling infrastructure |
| `io/` | Orbax checkpointing, Parquet catalogs, HuggingFace / CosmoGrid / GowerStreet loaders |
| `utils.py` | Lightcone geometry helpers: comoving distances, box-size ↔ redshift, scale factors |
| `scripts/` | CLI entry points and the shared argument parser |

## Development

```bash
# Set up the full dev + test environment (locked; groups are explicit)
uv sync --group dev --group tests

# Run tests (requires the `tests` group)
uv run pytest

# Lint / format / run hooks (prek runs the local ruff + toml-sort hooks)
uv run prek run --all-files
uv run pyright            # type-check on demand (manual hook, not a CI gate)

# Build a wheel + sdist
uv build
```

## License

MIT
