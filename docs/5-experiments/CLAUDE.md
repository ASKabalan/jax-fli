# CLAUDE.md — `docs/5-experiments`

Conventions for the paper's reproduction experiments. Each experiment is a numbered, **self-contained**
folder (`NN-name`, kebab-case). These rules are authoritative for anything under `docs/5-experiments/`.

## Anatomy of an experiment

Every experiment folder contains:

1. **`README.md`** — explains the experiment (structure below).
2. **A way to run it** — exactly one of:
   - a **Python script** (`build.py`) — local experiments that compute *and* plot directly;
   - a **shell script** (`run.sh`) — cluster runs (SLURM via `fli-launcher`, sourcing `_launch_common.sh`);
   - **LaTeX / TikZ** (`*.tex`) — diagrams or algorithm listings.
3. **`assets/`** — committed figures (SVG) and any rendered `.tex`.

### Cluster (shell-script) experiments
`run.sh` submits jobs (`fli-launcher` → `fli-simulate`) that write **parquet** `Catalog`s. Those
parquet are **pushed to HuggingFace** (`ASKabalan/jax-fli-experiments`). A **local Python script**
then loads them back (`load_dataset` → `Catalog.from_dataset`) and renders the **SVG** figures — so the
heavy compute is on the cluster and figure-making is reproducible locally without a GPU.

### Local experiments
Some experiments run entirely locally (small mesh, CPU) — e.g. [`09-gradient-validation`](09-gradient-validation/) —
and ship Python scripts that compute *and* plot directly, with no `run.sh`.

## README structure — science first, "How to run" LAST

Lead with the science. **Never open the README with run instructions or CLI invocation details.**
The recommended order:

1. **Goal** — what the experiment establishes and why it matters.
2. **Method** — how it works (when applicable).
3. **Results** — each figure *embedded*, with a paragraph explaining what it shows.
4. **How to run** — the command(s) / script(s). This section is **LAST**.

In goal, I also want in the Goal section a table of the runs with their parameters

[`09-gradient-validation/README.md`](09-gradient-validation/README.md) is the reference for content
quality and ordering.

## Figures

- **SVG, committed** (Read the Docs builds without a GPU). You cannot read an SVG visually — rasterize
  to PNG (`rsvg-convert -f png -b white`) and `Read` it before claiming what a figure shows.
- **Use a grid** and clean styling; the figure should be pleasant to look at.
- **Legends must stand alone** — spell out every abbreviation in the legend or caption (e.g.
  "kdk (DoubleKickDrift)", "bf (BullFrog)"), never bare acronyms.
- **Compare float32 *and* float64** wherever precision matters (e.g. adjoint stability): show both, so
  the precision dependence is explicit.

## Algorithms / LaTeX

Keep the **source as a committed `.tex` file** (reproducibility + paper reuse) **and render it to an
image** embedded in the README — show the *rendered* algorithm, never a raw ```latex code block.
Render with, e.g.:

```bash
pdflatex algorithm.tex && pdftocairo -svg algorithm.pdf assets/algorithm.svg
```

(`\documentclass[varwidth,border=8pt]{standalone}` + `algpseudocode` auto-crops to the listing.)

## Inherited conventions
The global `fli-experiment` skill still governs the lifecycle: **plan cluster runs before submitting**;
HuggingFace dataset *configs* in the dataset README YAML; **`jax_enable_x64` before `import jax_fli`**
for any masked spin-2 `angular_cl` (float32 → all-NaN); never write the acronym for "drift on the
lightcone"; pytest helpers are fixtures; **don't commit unless asked**.


## Writing the build.py

It is very important to keep the structure of the `build.py` scripts consistent across experiments. The following is a recommended structure:


### Imports and setup

```python
from __future__ import annotations

import jax

# float64 globally BEFORE jax_cosmo (the enable_x64() context manager breaks jax_cosmo's
# pure_callback comoving-distance cache; the global config flag is the safe route).
jax.config.update("jax_enable_x64", True)

import sys
from pathlib import Path

import equinox as eqx
import healpy as hp
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from huggingface_hub import snapshot_download
from matplotlib.lines import Line2D

from jax_fli import compute_theory_cl_for_density
from jax_fli.io import Catalog

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _exputils import savefig, set_style  # noqa: E402
```

Setting up everything, making sure that the precision is set to float64 globally before importing `jax_fli`.


## Setting up files paths

```python
ASSETS = HERE / "assets"
REPO = "ASKabalan/jax-fli-experiments"

FILE_0 = "02-mass-assignement/density_spectra/spectra_exp2_cic.parquet"
FILE_1 = "02-mass-assignement/density_spectra/spectra_exp2_tsc.parquet"
```

when in doubt, you can list the files by doing

```py
from huggingface_hub import list_repo_files
files = list_repo_files("ASKabalan/jax-fli-experiments", repo_type="dataset")
```

etc..

All files in the repo are stored in the `ASKabalan/jax-fli-experiments` dataset on HuggingFace. The `snapshot_download` function is used to download the files locally.

```python
root = snapshot_download(REPO, repo_type="dataset", local_files_only=True)

cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{FILE_0}", split="train"))
```

__note__: Unless stated otherwise, all files have their precomputed spectra stored in the `ASKabalan/jax-fli-experiments`

For example

02-mass-assignement/density_spectra/spectra_exp2_cic.parquet

Is the spectra of

02-mass-assignement/density/exp2_cic/**parquet

Onces the snapshot is created you load like I showed above.
NO HELPER FUNCTION, NO PARAMETRIZATION LOOP
I need to look at the file and in 1 sec glance what HF files are used

This is illegal

```python
# DO NOT DO
def _load(path):
    return Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{path}", split="train"))

_spec_cats = {m: _load(f"01-resolution/spectra/spectra_m{m}.parquet") for m in MESHES}
# DO NOT DO
```

For multiple files then you load the filed and cosmo like so

```python
spec_XXX , cosmo = cat_0.field[0] , cat_0.cosmology[0]
spec_YYY = cat_1.field[0] # SAME COSMO AS ABOVE
```

Do not load the cosmo from the second file, it should be the same as the first one. If not then you have a problem with your data.
But silently make sure that they are the same without writing that in the file.
If not the same then stop and tell me.

### Computing the theory

```python
theory = compute_theory_cl_for_density(cosmo, spec_XXX, ell)
```

You can also compute higher order stat
This is an example /home/wassim/Projects/NBody/jax-fli/docs/5-experiments/06-cosmogrid-shells/build.py


### Figures

The main goal of `build.py` is not over factorized code, but an easy code to parse by humans.
So far the code so far was flat global code and very easy to follow.

Next each figure is a function on its own, with a good name that explains directly what the figure is about
If two figures are very similar then you can have a function that takes a parameter to generate the figure.
But deliberatly over factorizing the code is not what I want.

Finally we plot like so

```python
def main():
    set_style()
    plot_figure_1()
    plot_figure_2()
    plot_figure_3(PARAM, PARAM2)
    print(f"assets written to {ASSETS}")


if __name__ == "__main__":
    main()


```
