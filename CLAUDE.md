# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**jax-fli** is a JAX-based toolkit for end-to-end differentiable cosmological simulations and forward modeling. It chains: Gaussian initial conditions → Lagrangian Perturbation Theory (LPT) → Particle-Mesh N-body integration → lightcone painting (flat-sky or HEALPix) → gravitational lensing (Born / ray-tracing) → angular power spectra, with full support for probabilistic inference via NumPyro/BlackJAX.

## Stability: the library is done — only touch core code with serious motivation

The core package (`src/jax_fli/`) and its public API are **considered complete and stable**. Day-to-day
work happens in **`docs/5-experiments/`** (experiment `run.sh` / `build.py` / READMEs) and the **notebooks**
(`docs/**/*.ipynb`) — change those freely.

Do **not** modify code under `src/` (or `bin/`, `pyproject.toml`, packaging) unless there is a
**serious, explicit reason**: a real bug with a reproduction, a capability an experiment genuinely cannot
express through the existing CLI/API, or a change the user has directly asked for.

- A workaround for one experiment is **not** sufficient motivation — express it in the experiment, not the
  library.
- Never add flags, parameters, or fallbacks to the library speculatively or "to make a benchmark easier."
  If a library change seems necessary, **stop and ask the user first**, explaining why the experiment layer
  cannot do it (e.g. adding a `--no-save`-style convenience flag to an entry point counts as a library change).

## Read the source, not this file, before you quote an API

This repository is a moving source, and that changes how it must be described.

**A citation needs a state.** `src/jax_fli/pm/nbody.py:69` is meaningless on its own — this is a working tree on a feature branch with uncommitted work, and the branch moves (`docs` → `allow-ic-from-cg` inside a month). Pin it every time:

```bash
git rev-parse --abbrev-ref HEAD; git log -1 --format=%h; git status --porcelain | wc -l
```

**There are three different facts, not one.** What a docstring claims, what the code does, and what a test verifies are separate. Say which one you have; when they disagree, that disagreement is usually the finding.

**`grep` is the index.** There is nothing to build. Locate with `grep -rn "def nbody" src/`, then read the function and quote what you see. Never describe a signature, a default or a return type you did not open — a plausible-looking API with a `file:line` attached is the worst thing to produce, because the line number makes it look verified.

**Tracked beats untracked.** `git ls-files <path>` (empty output means not tracked) and `git check-ignore -v <path>` are the authority test. `AUDIT.md` is untracked and 0 bytes; `TODO.md` is untracked scratch; `HANDOUTS/`, `DEL/` and `docs/000_RUNS/` are gitignored. **This very file is untracked**, so the source outranks it — including everything below.

**Prefer the test to the source, and the oracle to the test.** The test suite checks against independent external oracles rather than self-consistency: `tests/nbody/test_against_fpm_*.py` → FastPM, `tests/nbody/test_against_disco_dj_*.py` → `discodj`, `tests/lensing/` → glass, `tests/power/test_decouple.py` → pymaster. "Verified against FastPM at `tests/test_against_fpm.py:73`" beats "the code looks correct", and **absence of a test is itself a finding**.

## The public / `_src` split — which side to cite

`src/jax_fli/*` is the public surface over a parallel `src/jax_fli/_src/*` tree (`base`, `fields`, `io`, `lensing`, `summary_statistics`) that holds the kernels.

Born is the worked example: the maths is `_src/lensing/_born.py` (`_born_core_impl`, `_born_spherical`, `_born_flat`, `_born_windows`), and public `lensing/born.py` exports `born` and `plot_born_windows`.

**Do not assume the public module is a thin shim.** `lensing/born.py` once was 17 lines and now carries real normalisation and field-construction logic. So read both sides before attributing a behaviour: **cite `_src` for the numerics, the public module for the signature, and check which of the two actually does the thing you were asked about.**

`src/jax_fli/__init__.py`'s `__all__` is the authoritative API list — prefer it over any prose, and note that a name listed there may be a **function** rather than a submodule. `jax_fli.power` is exactly that trap: `README.md:130` calls it "importable as `jfli.power` for back-compat" and `summary_statistics/__init__.py` says it "re-exports from here", but **there is no `power/` directory and no `power.py`** — the name resolves only to the function pulled in at `__init__.py:72`. Verify rather than trust this line:

```bash
uv run python -c "import jax_fli.power"   # ModuleNotFoundError
```

## Commands

This project is managed with **uv** (build backend: hatchling + hatch-vcs; the
`jaxpm`/`jax-cosmo` forks are wired via `[tool.uv.sources]` — plain `pip` would
fetch the upstream PyPI releases instead, which is a different program).

```bash
# Install (groups are explicit — none installed by default)
uv sync --group dev              # tooling only (ruff/pyright/prek/toml-sort)
uv sync --group tests            # test suite + all extras + oracle backends
uv sync --group dev --group tests  # both

# Run tests (needs the `tests` group)
uv run pytest
uv run pytest -m "not slow and not distributed"   # skip slow/multi-GPU tests
uv run pytest tests/lightcone/                     # single directory
uv run pytest -k "test_name"                       # single test by name

# Lint / format (prek runs the local ruff/toml-sort hooks); pyright is a manual hook
uv run prek run --all-files
uv run pyright

# Build a wheel + sdist
uv build
```

**Test markers**: `distributed` (multi-GPU), `single_device`, `slow`, `scripts` (CLI end-to-end).

## Architecture

### Module map

| Path | Role |
|------|------|
| `src/jax_fli/fields/` | Immutable Equinox PyTree containers for all data (density, particles, maps) |
| `src/jax_fli/pm/` | Particle-mesh engine: LPT, N-body solvers, lightcone interpolation, correction kernels |
| `src/jax_fli/lensing/` | Born approximation convergence + Dorian ray-tracing |
| `src/jax_fli/summary_statistics/` | P(k), C_ℓ, transfer functions, coherence, Halofit, PDF, peak counts, starlet |
| `src/jax_fli/probabilistic_models/` | Forward model builder, `Configurations` dataclass, NumPyro/BlackJAX wrappers |
| `src/jax_fli/infer/` | Batched/distributed MCMC sampling infrastructure |
| `src/jax_fli/io/` | Orbax checkpointing, Parquet catalogs, HuggingFace/CosmoGrid loaders |
| `src/jax_fli/initial.py` | Gaussian IC generation and mesh interpolation |
| `src/jax_fli/units.py` | Density unit enum and conversions |
| `src/jax_fli/utils.py` | Lightcone geometry: comoving distances, scale factors, redshift conversions |
| `src/jax_fli/_src/` | The kernels behind the public surface — `base`, `fields`, `io`, `lensing`, `summary_statistics` |
| `src/jax_fli/scripts/` | CLI entry points + shared argument parser |
| `bin/fli-*` | The installed executables (see below — there is no `launcher/` package) |

### Where this sits in the stack

| layer | owns | seam |
|---|---|---|
| **`jax-fli`** | the science: LPT, N-body solvers, the lightcone, Born/ray-traced lensing, summary statistics, inference | imports exactly `jaxpm.{distributed,growth,kernels,painting,pm,spherical,utils}` — and **never `jaxdecomp` directly**. `grep -rn jaxdecomp src/` hits two explanatory comments in `initial.py` and no import; re-run it if you doubt it. |
| **`JaxPM`** | the PM physics: force kernels, painting, spherical/lightcone painting, growth, multigrid — and the only place jaxDecomp is touched | see `JaxPM/CLAUDE.md` |
| **`jaxDecomp`** | the communication substrate: distributed 3D FFT, halo exchange, pencil/slab transposes | see `jaxDecomp/CLAUDE.md` |

**The local JaxPM is not the JaxPM this package runs.** `pyproject.toml` pins `jaxpm` to branch `notebook-and-docs-update`; the clone on disk sits on a different branch. For "what does jax-fli actually run", use `git show notebook-and-docs-update:<path>` in the JaxPM clone. The same pattern applies to the pinned forks of `jax-cosmo`, `blackjax`, `optimistix`, `s2fft` and `pycs`.

### Fields — the universal data container

Every quantity in the pipeline lives in an immutable Equinox PyTree field defined in `fields/`. Key types:

- `DensityField` — 3D volumetric density
- `FlatDensity` / `SphericalDensity` — 2D flat-sky or HEALPix maps
- `ParticleField` — positions + velocities
- `FlatKappaField` / `SphericalKappaField` — lensing convergence

All fields carry metadata: `mesh_size`, `box_size`, `redshift`, `a` (scale factor), `unit`, `field_status`. Density units are `OVERDENSITY`, `PARTICLE_PER_MPC3`, or `MSUN_H_PER_MPC3`; convert with `field.convert_units(target_unit)`.

### PM module — the simulation engine

- **`lpt.py`**: LPT displacement `dx` and momentum `p`
- **`solvers.py`**: Pluggable N-body integrators — `DoubleKickDrift` (reversible), `DriftKickDrift`, `BullFrog`
- **`nbody.py` / `integrate.py`**: Integration loop with multi-host sharding and `AdjointType` control
- **`interp.py`**: Lightcone painting interpolation — `NoInterp`, `DriftInterp`, `OnionTiler`, `TelephotoInterp`
- **`correction.py`**: Sub-grid halo corrections — `PGDKernel` (position-based), `SharpeningKernel` (velocity-based)

The integration itself is delegated to **diffrax**; paint/read/kick/drift come from JaxPM.

### The two adjoints

`nbody()` declares both, in `src/jax_fli/pm/nbody.py:69-71`:

- **`reverse`** — a reversible backsolve. Stores no trajectory (**O(1) memory in the integration steps**) and reconstructs it by inverting each step.
- **`checkpointed`** (the default) — an equinox checkpointed scan, recomputing forward segments instead. Two independent controls: `checkpoints` for the outer scan over saved shells, `step_checkpoints` for the inner integration-step loop within each shell. **`step_checkpoints` has no effect on the `reverse` adjoint** — the docstring at `:103-106` says so.

Neither changes the gradient value, proved by the **AD-vs-AD transpose test** at `tests/nbody/test_adjoints.py::test_adjoint_transpose` (forward-mode `⟨w, Jv⟩` against adjoint `⟨Jᵀw, v⟩`, agreeing to ~10⁻¹² in float64). That is a sharper proof than a finite difference.

**The novelty claim in `docs/5-experiments/09-gradient-validation/README.md:52` has been refuted and is still in the repo.** It states that pmwd and DISCO-DJ implement the adjoint for the final state only and do not propagate gradients through a multi-snapshot lightcone. `PMWD_COSMO` (arXiv:2211.09815) p. 14–15 treats the lightcone as the general case with an observation operator injecting cotangents at every step; `DISCO_DJ_II` (arXiv:2510.05206) §2.7 is not final-state-restricted either; and `MADLens` (arXiv:2012.07266) and `DLL` (A&A 679 A61) are published multi-shell lightcone gradients that predate it — `DLL` being the FlowPM→JaxPM lineage, i.e. Wassim's own collaborators. **The finished thesis got this right; this repository's docs did not**, so the next paper or talk drawn from `docs/5-experiments/` will reintroduce it. The defensible reframe is the *combination* — adjoint-O(1) plus a shell lensing lightcone plus distributed multi-GPU — which nothing published has together. The `/papers` skill owns those four papers; hand the claim to it before the text is written, not after.

### Painting targets

Controlled by `PaintingOptions(target=...)`:
- `"volumetric"` → `DensityField` (3D shell)
- `"flat"` → `FlatDensity` (2D flat-sky projection)
- `"spherical"` → `SphericalDensity` (HEALPix)
- `"particles"` → `ParticleField`

### Typical pipeline

`lpt` takes a `DensityField` and is keyword-only past it (`src/jax_fli/pm/lpt.py:33`):

```python
# 1. Initial conditions
ic = gaussian_initial_conditions(key, mesh_size, box_size, cosmo)

# 2. LPT
dx, p = lpt(cosmo, ic, ts=0.1, order=2)   # ic is a DensityField; ts/order are keyword-only

# 3. N-body → lightcone
solver = DoubleKickDrift(...)
lightcone = nbody(cosmo, dx, p, solver=solver, painting=PaintingOptions(target="spherical"))

# 4. Lensing
kappa = born(cosmo, lightcone, nz_shear=...)

# 5. Power spectrum
cl = kappa.angular_cl(method="healpy")
```

### CLI entry points

The implementations live in `src/jax_fli/scripts/entry/`, but the executables are the files in **`bin/`**, installed verbatim by `[tool.hatch.build.targets.wheel.shared-scripts]` (`pyproject.toml:121-124`) rather than as `[project.scripts]` console scripts. The comment there says why: each `bin/fli-*` sets environment variables and calls `jax.distributed.initialize()` **before** importing the package, so it must not import `jax_fli` at console-script generation time.

| Command | Purpose |
|---------|---------|
| `fli-simulate` | Run LPT/N-body/lensing simulation (subcommands: `lpt`, `nbody`, `lensing`) |
| `fli-samples` | Draw posterior samples from MCMC chains |
| `fli-infer` | Run MCMC inference (NUTS/MCLMC) |
| `fli-born-rt` / `fli-dorian-rt` | Born / Dorian ray-tracing |
| `fli-extract` / `fli-2pcf` | Catalog extraction / 2-point correlation |
| `fli-launcher` | SLURM job dispatcher wrapping the above |

Extensionless `bin/fli-*` files must **not** use `from __future__ import annotations` — the sh-polyglot shebang turns it into a `SyntaxError`.

### Key design principles

- **Immutable PyTrees**: all state is stateless containers — safe to `jit`, `vmap`, `grad`.
- **Composable pieces**: solvers, interpolation schemes, correction kernels, and painting targets are independently swappable.
- **Deferred cosmology**: `jax_cosmo.Cosmology` flows through the full pipeline, enabling gradient-based inference over cosmological parameters.
- **Distributed-first**: multi-GPU via `jax.sharding`; halo exchange handled automatically in the integration loop.

### JAX environment (set by the `bin/fli-*` scripts)

```
TF_GPU_ALLOCATOR=cuda_malloc_async
JAX_ENABLE_X64=False          # 32-bit throughout
XLA_PYTHON_CLIENT_MEM_FRACTION=0.97
```

Float32 N-body IC-gradients diverge at O(1) through chaos — **validate gradients in float64 only** (this is what Experiment 09 exists to show).

## The experiments — look them up, never recall them

`docs/5-experiments/` says of itself (`README.md:7`): *"End-to-end reproduction studies behind the jax-fli / jaxpm methods paper."* Each numbered folder holds hand-written prose (Goal → Method → Results), a run grid, a `run.sh`, a figure script and committed `assets/`. The prose is already written to paper standard — when a write-up needs a result, **point at the experiment and its figure** rather than paraphrasing.

**`docs/5-experiments/CLAUDE.md` is tracked and IS authoritative** for experiment conventions (no titles on figures, SVG committed, `build.py` structure). It outranks this file.

**No list of experiments belongs in a file like this one, so there is none here.** The author runs experiments continuously: a folder with no data this morning may have it now, and a figure may be regenerated under you.

`README.md:11` declares the convention — *"⚠️ marks experiments not yet run; ✅ marks finished ones."* **Those markers are intent and they drift from reality in both directions.** An experiment marked ⚠️ can already carry figures (`12-scaling-gradient` does); one marked ✅ can carry a caveat its own README shows and the index does not. So the marker is a hint and the disk is the answer:

```bash
cd docs/5-experiments
sed -n '1,40p' README.md                    # the tracked index
grep -n '⚠️\|✅' README.md                   # the markers
ls */assets/*.svg | cut -d/ -f1 | uniq -c   # which have figures
ls -d */data*/ 2>/dev/null                  # which have data
git log --oneline -8 -- .                   # what moved recently
```

- **⚠️ with no assets and no data** — nothing to lean on. Say so; it is worth more early than late.
- **⚠️ with assets** — partial. Read its README for which runs are missing. Do not upgrade it to "done" because a figure exists, and do not dismiss it because of the marker.
- **✅** — read the experiment README anyway, for the run grid and the caveats.

**The data may not be on disk.** Run data is mirrored to three HuggingFace datasets: `ASKabalan/jax-fli-experiments` (accuracy, 00–09; the results-explorer Space reads only this one), `ASKabalan/jax-fli-scaling` (experiments 11–12) and `ASKabalan/jax-fli-sampling` (experiment 13 and notebooks 14, 16, 17), with a results-explorer Space alongside — both linked from badges at the top of `docs/5-experiments/README.md`. `docs/000_RUNS/` is gitignored and runs to hundreds of gigabytes; `du -sh` it rather than quoting a size. Experiment `data*/` folders are mixed: some untracked entirely, others tracking a `.csv` summary while `*.npz` beside it is gitignored. **A number read from a local `.npz` is locally verified, not committed** — say so, because a reader cannot reproduce it from the repo alone.

**Regenerating a figure is allowed; recomputing is not, without asking.** The existing committed figure is the first answer. Experiment scripts split `_compute()` (writes `data_*/*.npz`) from `_plot()`, and `_compute()` skips when the npz exists, so re-running an experiment that ships data replots cheaply. A real recompute needs the GPU and inputs that mostly live in the HuggingFace dataset rather than on disk — **say so and ask first.**

## Known latent bugs, and what to do with them

**Report a bug; do not fix it** — that is what the stability policy above means in practice. Real ones exist in the pinned JaxPM (`pm.py:291` and `pm.py:322` call `fft3d` where the physics wants `ifft3d`; `pm.py:303` annotates an unimported `Cosmology`; `kernels.py:255` assigns into a JAX array). All sit on dead paths (PGD, the neural ODE). Re-derive the line numbers before quoting them — those files move. See `JaxPM/CLAUDE.md`.

## Environment gotchas

- Run tests with `JAX_PLATFORMS=cpu`; conftest's `JAX_PLATFORM_NAME` is ignored and plain `pytest` grabs the GPU and OOMs.
- Lint with the dev-group ruff (`uv run --group dev -- ruff`), not the system one. E501 is ignored; line length 120.
- `uv run --group X` can uninstall other groups. Restore with `uv sync --extra cuda --group dev --group tests`, then re-overlay any local editable installs.
- `s2fft`'s `method="jax_cuda"` SIGSEGVs single-process multi-GPU; use multi-host or `method="jax"`.
