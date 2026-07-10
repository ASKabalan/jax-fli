# Experiment 14 — Field-level inference vs CosmoGrid shear ⚠️ BLOCKED

**Goal.** Field-level inference on simulated **shear**, compared against CosmoGrid: 2048³, BullFrog, TSC + deconvolution, RBF painting, shear observable; the box covers 3 DES Y3 n(z) bins. Geometry: observer `(0.1, 0.5, 0.9)` (the Exp 8 Case-3 cap), **DES Y3 mask**.

**Status.** ⚠️ **Blocked on the inference pipeline.** Differentiated *inside* `fli-infer` (`--grad checkpoint`).

**Pre-flight blockers:** (a) confirm our shear has the **same units** as CosmoGrid shear; (b) decide **where the scale cut lives** (density / convergence / shear). **Enable x64 before `import jax_fli`** for any masked spin-2 `angular_cl` (float32 → all-NaN).

## Run

Blocked — see [`run.sh`](run.sh). Reuses the Exp 8 Case-3 geometry + DES Y3 mask.
