# CLAUDE.md — `docs/5-experiments`

Conventions for the paper's reproduction experiments. Each experiment is a numbered, **self-contained**
folder (`NN-name`, kebab-case). These rules are authoritative for anything under `docs/5-experiments/`.

## Anatomy of an experiment

Every experiment folder contains:

1. **`README.md`** — explains the experiment (structure below).
2. **A way to run it** — exactly one of:
   - a **Python script** (`NN-name.py`) — local experiments that compute *and* plot directly;
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
2. **Method** — how it works, plus any novelty vs prior work.
3. **Results** — each figure *embedded*, with a paragraph explaining what it shows.
4. **How to run** — the command(s) / script(s). This section is **LAST**.

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
