# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development

```bash
# Install (editable, with all extras)
pip install -e ".[all]"

# Install dev tools only
pip install -e ".[dev]"

# Run tests
pytest                              # uses testpaths=["tests"], runs with --cov
pytest tests/test_against_fpm.py   # single test file
pytest tests/test_against_fpm.py::test_lpt -v  # single test

# Lint, format & type-check
ruff check .              # lint (pycodestyle, pyflakes, pyupgrade, isort)
ruff format .             # format (double quotes, 120 char line length)
pyright                   # type-check (configured in pyproject.toml)

# Pre-commit (runs ruff-format + ruff lint + pyright)
pre-commit run --all-files
```

Note: tests enable float64 via `jax.config.update("jax_enable_x64", True)` in `conftest.py`. The CI sets `JAX_PLATFORM_NAME=cpu` and initialises 8 host devices for sharding tests.

## Architecture

### Pipeline flow

```
initial conditions → LPT → N-body (PM) → lightcone painting → lensing (Born/raytrace) → power spectra
```

The probabilistic layer wraps this pipeline into a differentiable forward model sampled with BlackJAX/NumPyro.

### Module map

| Module | Purpose |
|---|---|
| `fields/` | Immutable PyTree containers (`DensityField`, `ParticleField`, lightcone maps). All inherit `AbstractField → AbstractPytree → eqx.Module`. |
| `_src/base/` | Core base classes (`AbstractPytree`, `AbstractField`), enums (`FieldStatus`, units), `tri_map`. |
| `initial.py` | Gaussian initial conditions and interpolation to mesh. |
| `pm/` | Particle-mesh engine: `lpt()`, `nbody()`, symplectic solvers, PGD correction, interpolation/tiling, integration loop. |
| `power/` | Power spectrum estimation (`power`, `transfer`, `coherence`, `angular_cl_*`), theory predictions. |
| `lensing/` | Born approximation and ray-tracing convergence maps. |
| `probabilistic_models/` | Deterministic forward model builder (`make_full_field_model`), NumPyro probabilistic wrappers, `Configurations` dataclass. |
| `infer/` | BlackJAX batched sampling, distributed priors (`DistributedNormal`), chain/posterior analysis. |
| `io/` | Checkpoint persistence (`save_sharded`/`load_sharded`), HuggingFace catalog, CosmoGrid/GowerStreet loaders. |
| `utils.py` | Lightcone geometry helpers (shell computation, comoving distances, scale factors). |
| `parameters.py` | Predefined cosmologies (e.g. `Planck18`). |
| `units.py` | Unit conversion logic for density fields. |

### Key abstractions

- **Field PyTrees**: `AbstractPytree` → `AbstractField` → `DensityField` / `ParticleField` / lightcone types. Fields carry both the JAX array (`.array`) and static metadata (`mesh_size`, `box_size`, `sharding`, `status`, `unit`). Immutable — use `.replace(**kwargs)` to produce modified copies. `_src/base/_core.py` contains the base classes; `fields/` re-exports them as the public API.
- **N-body solvers**: `AbstractNBodySolver` (equinox Module) with `init/step/save_at/reverse`. Three implementations:
  - `DoubleKickDrift` — reversible KKD, conformal momentum, requires `time_stepping='a'`.
  - `DriftKickDrift` — FastPM pi-integrator (DKD), D-time velocity, default `time_stepping='D'`.
  - `BullFrog` — second-order LPT DKD, D-time velocity, default `time_stepping='D'`.
- **`resolve_geometry()`**: must be called **outside** `jax.jit` on the solver to set lightcone shell geometry. Returns a new solver instance with updated `interp_kernel` and `n_steps`.
- **Integration**: `integrate()` in `pm/integrate.py` drives the time-stepping loop. `adjoint='checkpointed'` uses equinox scan checkpointing; `adjoint='reverse'` uses a custom VJP via `solver.reverse()` (requires `time_stepping='a'` and a reversible correction kernel).
- **Interpolation kernels**: `AbstractInterp` → `NoInterp`, `DriftInterp`, `OnionTiler`, `TelephotoInterp`. Handle lightcone painting during integration.
- **Correction kernels**: `AbstractCorrection` → `NoCorrection`, `PGDKernel`, `SharpeningKernel`. Position/velocity corrections at each step. Only `SharpeningKernel` is reversible (compatible with `adjoint='reverse'`).

## Coding Conventions

- **Equinox modules everywhere**: solvers, fields, interp kernels, corrections are all `eqx.Module` subclasses. Mark non-traced metadata with `eqx.field(static=True)`.
- **`__check_init__`**: use for validation in equinox modules (not `__post_init__`).
- **`replace()` not mutation**: fields and modules are immutable PyTrees. Use `self.replace(array=new_arr)` or `eqx.tree_at()` to update.
- **`jax.jit` with `static_argnames`**: top-level functions (`lpt`, `nbody`) use `@partial(jax.jit, static_argnames=[...])` for arguments that affect tracing (shapes, enums, orders).
- **Type hints**: use `jaxtyping.Array`, `Optional`, standard typing. `F722` (jaxtyping syntax) is suppressed in ruff.
- **Imports**: `from __future__ import annotations` in every file. Relative imports within the package.
- **`__all__`**: every public module defines `__all__`.
- **No `__pow__`**: deliberately omitted on `AbstractPytree` to avoid breaking `equinox.Omega`.

## CLI Scripts

The package installs six entrypoints (in `src/jax_fli/scripts/`, exposed via `bin/`):

| Command | Script | Purpose |
|---|---|---|
| `fli-simulate` | `fli_simulate.py` | Run LPT/N-body simulation |
| `fli-samples` | `fli_samples.py` | Draw samples from prior/posterior |
| `fli-infer` | `fli_infer.py` | Full-field inference |
| `fli-raytrace` | `fli_raytrace.py` | Ray-tracing lensing |
| `fli-extract` | `fli_extract.py` | Extract fields from checkpoints |
| `fli-grid` | `fli_grid.py` | Grid/parameter sweep utilities |

Shared utilities live in `scripts/_common.py`.

## Key Dependencies (custom forks)

| Package | Source | Branch |
|---|---|---|
| `jaxpm` | `DifferentiableUniverseInitiative/JaxPM` | `41-spherical-lensing` |
| `jax_cosmo` | `ASKabalan/jax_cosmo` | `ALL-MODIF` |
| `dorian-raytrace` (optional) | `ASKabalan/Dorian` | `main` |

These are git-pinned in `pyproject.toml`. Standard PyPI for everything else (`jax>=0.6`, `equinox`, `blackjax`, `numpyro`, `healpy`, `jax-healpy`, `orbax-checkpoint`).

## Notes

- `jax-healpy` is a mandatory dependency (not optional) — it's required for all spherical projection code.
- The `_src/` directory contains internal implementations (`_src/base/`, `_src/fields/`, `_src/lensing/`, `_src/power/`). Public modules in `fields/`, `lensing/`, `power/` re-export from `_src/` and are the intended API surface.
- `pm/legacy/` holds deprecated diffrax-based ODE solvers — do not use for new code.
