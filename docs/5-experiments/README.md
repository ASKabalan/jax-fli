# Experiments

End-to-end reproduction studies behind the **jax-fli / jaxpm** methods paper. Each experiment lives
in its own folder with a hand-written `README.md` (the goal + the exact run grid), a `run.sh` that
launches the runs, and — once the data exists — a runnable Python script that saves the figures
(SVG for the web, PDF for the paper). ⚠️ marks experiments **not yet run**; ✅ marks finished ones.

## Running

Each experiment's `run.sh` sources the shared [`_launch_common.sh`](_launch_common.sh) and submits
via `fli-launcher` → `fli-simulate`:

```bash
bash run.sh                 # submit to SLURM (MODE=sbatch, the default → the cluster)
MODE=local  bash run.sh     # run locally via mpirun (use tiny meshes)
MODE=dryrun bash run.sh     # print the resolved commands, submit nothing
```

**Precision:** every run is **float64** (`--enable-x64`) *except* [Experiment 11](11-scaling/README.md)
(scaling), which is run in **both** float32 and float64. **All experiments run on the cluster except
[Experiment 09](09-gradient-validation/README.md)** — the gradient correctness / stability check runs
**locally** (CPU, small mesh) and ships its own Python scripts instead of a `run.sh`.

## Simulation accuracy (0–7)

- **00 — [CosmoGrid reference](00-cosmogrid-reference/README.md)** ⚠️ — ground-truth density + κ
  (already on HuggingFace); one ray-traced reference run pending.
- **01 — [Resolution convergence](01-resolution-convergence/README.md)** ⚠️ — per-shell spherical
  `C_ℓ` converging with particle count, 512³ → 4096³.
- **02 — [Mass assignment + force deconvolution](02-mass-assignment/README.md)** ⚠️ — CIC / TSC / PCS
  × force-deconvolution on/off, impact on the spherical `C_ℓ`.
- **03 — [Spherical painting + pixel-window](03-spherical-painting/README.md)** ⚠️ — interpolation
  scheme (NGP / bilinear / RBF) and HEALPix pixel-window impact.
- **04 — [Step convergence](04-step-convergence/README.md)** ⚠️ — minimum step budget and the
  kdk / dkd / BullFrog comparison on the per-shell spherical `C_ℓ` vs step count.
- **05 — [Drift on the lightcone](05-drift-on-lightcone/README.md)** ⚠️ — drift-on-lightcone improves
  thick-shell density `C_ℓ`; Born convergence unaffected.
- **06 — [Match CosmoGrid shells](06-cosmogrid-shells/README.md)** ⚠️ — per-shell density `C_ℓ` +
  cross-correlation vs the CosmoGrid shells (needs the CosmoGrid shell edges).
- **07 — [Born lensing vs CosmoGrid](07-born-lensing/README.md)** ⚠️ — convergence `C_ℓ` +
  cross-correlation of our Born κ vs the CosmoGrid lensed maps.

## Weak lensing on a cut sky

- **08 — [Masked shear](08-masked-shear/README.md)** ✅ — Kaiser–Squires κ → γ on a cut sky (DES Y3
  footprint + observer visibility masks), mask-decoupled `EE` spectra. Loads the CosmoGrid
  convergence from Experiment 0.

  [![Masked shear footprints](08-masked-shear/assets/fig01-masks.svg)](08-masked-shear/README.md)

## Performance & gradients (9–12)

- **09 — [Gradient through the lightcone](09-gradient-validation/README.md)** ✅ *(local)* — adjoint
  `∂L/∂δ` vs finite differences (correctness, float64) + reversible-reconstruction stability: error vs
  step count (float32) and exactness vs number of saved shells. The lightcone gradient accumulation that
  pmwd / DISCO-DJ do not provide.

  [![Adjoint vs finite differences](09-gradient-validation/assets/fig01-grad-vs-fd.svg)](09-gradient-validation/README.md)

- **10 — [Adjoint performance: memory & wall-time](10-memory-checkpoints/README.md)** ⚠️ *(HPC)* — peak
  memory + wall-time of the `reverse` vs `checkpointed` adjoints across the two cost axes (integration
  steps, saved shells) plus the checkpoint-count trade.
- **11 — [Performance: throughput, strong & weak scaling](11-scaling/README.md)** ⚠️ — per-stage cost
  + strong & weak scaling (perf + memory), in float32 *and* float64.
- **12 — [Solver reversibility error](12-solver-reversibility/README.md)** ⚠️ — forward → reverse
  round-trip error vs step count.

## Field-level inference (13–15) — blocked on the inference pipeline

- **13 — [Toy field-level (full sky)](13-inference-toy/README.md)** ⚠️ — smallest end-to-end posterior.
- **14 — [Field-level vs CosmoGrid shear](14-inference-cosmogrid-shear/README.md)** ⚠️ — DES Y3 mask.
- **15 — [Power-spectrum level vs theory](15-inference-power-spectrum/README.md)** ⚠️ — decoupled
  angular `C_ℓ` vs theory.
