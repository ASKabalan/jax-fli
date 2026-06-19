# jax-fli

**Differentiable cosmological forward modeling on JAX**

<p align="center">
  <img src="assets/PIPELINE.png" alt="HEALPix spherical lightcone shells from an N-body simulation" width="700">
</p>

## Overview

`jax-fli` is a JAX toolkit for end-to-end differentiable cosmological simulations. It chains initial conditions, Lagrangian Perturbation Theory, Particle-Mesh N-body integration, lightcone painting (3D, flat-sky, HEALPix), gravitational lensing (Born and ray-tracing), and angular power spectrum estimation into a single differentiable pipeline. The library supports multi-GPU distribution via JAX sharding, reversible solvers for memory-efficient backpropagation, and probabilistic inference with BlackJAX/NumPyro.

```
ICs ──> LPT ──> PM N-body ──> Lightcone Painting ──> Lensing ──> Power Spectra
                   │                  │
             symplectic KDK/KKD    3D / flat-sky / HEALPix
             + PGD correction      + interpolation kernels
```

## Key Features

- **N-body solvers** -- Reversible symplectic KKD (`DoubleKickDrift`) integrator
- **Painting targets** -- 3D density, flat-sky 2D projection, and HEALPix spherical maps with CIC / bilinear / NGP / RBF schemes
- **Interpolation kernels** -- `DriftInterp`, `OnionTiler`, and `TelephotoInterp` for on-the-fly lightcone construction beyond the box boundary
- **Correction kernels** -- PGD (position-based) and Sharpening (velocity-based, reversible) for sub-grid halo correction
- **Gravitational lensing** -- Born approximation (fully JIT-able) and ray-tracing (via Dorian) convergence maps
- **Multi-GPU** -- Distributed simulations via JAX sharding with automatic halo exchange
- **Immutable Field PyTrees** -- `DensityField`, `ParticleField`, `FlatDensity`, `SphericalDensity` carrying arrays + metadata through the pipeline
- **Power spectra** -- 3D P(k), angular C_ell, transfer functions, and theory predictions with Halofit
- **Probabilistic inference** -- Deterministic forward model builder + NumPyro wrappers + BlackJAX batched sampling
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

__note__: You can also compile the spherical harmonics cuda kernel in s2fft just my making sure that you have a working CUDA compiler before running the above command.

## Quick Start

```python
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import jax_fli as jfli

key = jax.random.PRNGKey(42)
cosmo = jc.Planck18()

# 1. Gaussian initial conditions
initial_field = jfli.gaussian_initial_conditions(
    key, mesh_size=(256, 256, 256), box_size=(1000.0, 1000.0, 1000.0),
    cosmo=cosmo, nside=256,
)

# 2. LPT displacement + momentum
dx, p = jfli.lpt(cosmo, initial_field, ts=0.1, order=1)

# 3. PM N-body with spherical lightcone output
solver = jfli.DoubleKickDrift(
    interp_kernel=jfli.NoInterp(painting=jfli.PaintingOptions(target="spherical")),
)
lightcone = jfli.nbody(cosmo, dx, p, t1=1.0, dt0=0.05, nb_shells=4, solver=solver)

# 4. Born lensing convergence
nz = [jfli.tophat_z(0.0, 0.5, gals_per_arcmin2=1.0)]
kappa = jfli.born(cosmo, lightcone, nz_shear=nz)

# 5. Angular power spectrum
cl = kappa.angular_cl(method="healpy")
```

<p align="center">
  <img src="assets/kappa_born.png" alt="Born approximation convergence kappa maps" width="700">
</p>

## Documentation

Full tutorials live in [`docs/`](docs/notebooks.md) — executable notebooks plus markdown guides.
The notebooks are committed with **small-scale CPU outputs** (production parameters left in the
code); rerun them on a GPU/cluster to reproduce the full-resolution figures.

**Introduction & basics**

| # | Notebook | Description |
|---|----------|-------------|
| 01 | [Library Fundamentals](docs/1-introduction-and-basics/01-basics.ipynb) | Field types, painting (3D / flat / HEALPix), power spectra, catalogs |
| 02 | [LPT Lightcone](docs/1-introduction-and-basics/02-LPT-Simulation.ipynb) | Zel'dovich lightcone, theory comparison, deconvolution + shot noise |
| 03 | [PM Simulation](docs/1-introduction-and-basics/03-PM-Simulation.ipynb) | BullFrog N-body, 3D P(k) vs Halofit with CIC deconvolution + shot noise |
| 04 | [Distributed PM](docs/1-introduction-and-basics/04-Distributed-PM.ipynb) | Multi-device sharding, halo exchange, distributed painting |

**Advanced usage**

| # | Notebook | Description |
|---|----------|-------------|
| 05 | [PM Interpolation](docs/2-advanced-usage/05-PM-Interpolation.ipynb) | TelephotoInterp / OnionTiler for shells beyond the box |
| 06 | [Drift on the Lightcone](docs/2-advanced-usage/06-Drift-on-Lightcone.ipynb) | Per-particle lightcone-crossing drift; fewer/thicker density shells |
| 07 | [Advanced PM](docs/2-advanced-usage/07-Advanced-PM.ipynb) | PGD correction and shell-partitioning strategies |
| 08 | [Weak Lensing](docs/2-advanced-usage/08-Lensing.ipynb) | Born + ray-traced convergence, shear maps, saving catalogs |
| 09 | [External Catalogs](docs/2-advanced-usage/09-External-Catalog.ipynb) | Loading CosmoGrid and GowerStreet data |
| 10 | [Multi-host PM](docs/2-advanced-usage/10-multi-host-pm.md) | Launching across nodes + validation vs CosmoGrid |

**Command-line tools** — one page per script under
[Scripts & utilities](docs/4-scripts-and-utilities/): `fli-simulate`, `fli-summary-stats`,
`fli-born-rt`, `fli-dorian-rt`, `fli-samples`, `fli-infer`, `fli-2pcf`, `fli-extract`,
`fli-launcher`. Sampling/inference tutorials and experiments are in preparation.

## Module Map

| Module | Purpose |
|--------|---------|
| `fields/` | Immutable PyTree containers (`DensityField`, `ParticleField`, `FlatDensity`, `SphericalDensity`) |
| `initial.py` | Gaussian initial conditions and interpolation to mesh |
| `pm/` | Particle-mesh engine: `lpt()`, `nbody()`, symplectic solvers, PGD correction, integration loop |
| `lensing/` | Born approximation and ray-tracing convergence maps |
| `power/` | Power spectrum estimation (P(k), C_ell, transfer, coherence), theory predictions |
| `probabilistic_models/` | Deterministic forward model builder, NumPyro wrappers, `Configurations` dataclass |
| `sampling/` | BlackJAX batched sampling, `DistributedNormal`, chain plotting |
| `io/` | Checkpoint persistence, HuggingFace catalog, CosmoGrid/GowerStreet loaders |
| `utils.py` | Lightcone geometry helpers, comoving distances, scale factors |
| `parameters.py` | Predefined cosmologies (e.g. `Planck18`) |

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
