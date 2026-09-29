# AGENTS.md

Guidance for AI agents working in `jax-fli` — differentiable cosmological forward modeling on JAX (IC → LPT → PM N-body → lightcone → Born/ray-traced lensing → angular power spectra, with NumPyro/BlackJAX inference). The untracked `CLAUDE.md` holds the long-form version of everything below; the tracked `docs/5-experiments/CLAUDE.md` is authoritative for experiment conventions.

## Stability policy: the library is done

- `src/jax_fli/` and its public API are complete and stable. Day-to-day work happens in `docs/5-experiments/` (experiment `run.sh`/`build.py`/READMEs) and `docs/**/*.ipynb` — change those freely.
- Do not modify `src/`, `bin/`, or packaging without a serious, explicit reason: a real bug with a reproduction, a capability no experiment can express through the existing API, or a direct user request. A workaround for one experiment is not motivation. If a library change seems needed, stop and ask first.
- Known bugs exist on dead paths in the pinned JaxPM (e.g. `fft3d` where the physics wants `ifft3d`): report them, do not fix them.

## Commands (uv only — plain pip installs the wrong program)

```bash
uv sync --group dev --group tests    # full dev+test env (no group installs by default)
uv run pytest                        # addopts already add coverage; needs the tests group
uv run pytest -m "not distributed and not scripts and not slow"   # CI's default pass
uv run pytest -k "test_name"         # single test; or point at tests/<dir>/
uv run prek run --all-files          # the lint gate: ruff + toml-sort
uv run pyright                       # manual only, NOT a CI gate (large pre-existing backlog)
uv build
```

- CI (`ci.yml`) runs three pytest passes: the default pass, `-m distributed` (with `XLA_FLAGS=--xla_force_host_platform_device_count=8`), then `-m scripts`. Reproduce the pass you care about, not the whole suite.
- Markers: `distributed` (multi-device), `single_device` (auto-applied), `slow`, `scripts` (CLI end-to-end).
- On a GPU workstation, export `JAX_PLATFORM_NAME=cpu` before plain pytest runs or JAX grabs the GPU; CI sets this explicitly.
- `uv run --group X` can uninstall packages from other groups; restore with a full `uv sync --group dev --group tests`.
- The `tests` group builds oracle backends (fastpm, pfft-python, pmesh, glass, discodj, pymaster) that need system MPI/FFTW/GSL/cfitsio — the same packages CI installs via apt.

## Read the source before quoting an API

- `__all__` in `src/jax_fli/__init__.py` is the authoritative API list. Trap: `jax_fli.power` is a *function* exported from `__init__.py`, not a module — no `power/` directory exists, despite the README's "importable as `jfli.power`".
- `src/jax_fli/_src/` holds the kernels behind the public modules. The public shim can carry real logic, so cite `_src` for numerics and the public module for signatures — check both before attributing behaviour.
- Locate with grep, then read the function. Never state a signature, default, or return type you did not open; a plausible `file:line` with wrong content is the worst output. When citing code, pin the tree state (`git rev-parse --abbrev-ref HEAD; git log -1 --format=%h; git status --porcelain | wc -l`) — this branch moves.
- The tests check against independent oracles: `tests/nbody/test_against_fpm_*` → FastPM, `test_against_disco_dj_*` → DISCO-DJ, `tests/lensing/` → glass, `tests/power/test_decouple.py` → pymaster. Prefer a test-verified statement to a docstring; the absence of a test is itself a finding.

## Forks and the sibling clones

- Every major dependency is a fork pinned by branch in `[tool.uv.sources]` (jaxpm on `notebook-and-docs-update`, plus jax-cosmo, blackjax, optimistix, s2fft, pycs and the oracle backends). This is why pip does not work here.
- The clones in the parent directory (`../JaxPM`, `../jaxDecomp`, `../jax_cosmo`, `../DISCO-DJ`, `../dorian`, …) often sit on different branches than the pins. For what jax-fli actually runs, read the pinned branch: `git show <branch>:<path>` in the clone.
- jax-fli imports `jaxpm.*` and never `jaxdecomp` directly; jaxDecomp is JaxPM's substrate.

## Experiments (where the work happens)

- `docs/5-experiments/` holds the paper's reproduction studies. Its tracked `CLAUDE.md` defines the conventions: README opens with the science and puts "How to run" last; figures are committed SVG without titles, legends spelled out; `build.py` stays flat with one function per figure; float64 via `jax.config.update("jax_enable_x64", True)` **before** importing `jax_fli`/`jax_cosmo` (the context manager breaks jax_cosmo's comoving-distance cache).
- The ⚠️/✅ markers in `docs/5-experiments/README.md` are intent and drift in both directions — check `assets/`, `data*/` and `git log` before trusting them.
- Run data is not in git: `docs/000_RUNS/` is gitignored and hundreds of GB; experiment data lives on the HuggingFace datasets `ASKabalan/jax-fli-experiments` (accuracy), `ASKabalan/jax-fli-scaling` (experiments 11–12) and `ASKabalan/jax-fli-sampling` (MAP and chains) (list files with `huggingface_hub.list_repo_files`).
- Replotting is allowed (`_compute()` skips when the npz exists); recomputing needs the GPU and HF inputs — ask first. A number read from a local `.npz` is locally verified, not committed — say so.

## CLI and environment quirks

- `bin/fli-*` are real scripts installed verbatim via `[tool.hatch.build.targets.wheel.shared-scripts]`, not `[project.scripts]` entry points, because each sets env vars and runs `jax.distributed.initialize()` **before** importing `jax_fli`.
- The `fli-*` environment is 32-bit JAX (`JAX_ENABLE_X64=False`) with 97% GPU preallocation. Float32 N-body IC-gradients diverge at O(1) through chaos — validate gradients in float64 only.

## Hygiene

- `token.md` in the repo root is untracked, not gitignored, and contains a live API key: never commit, cat, or quote it.
- Do not commit unless asked.
