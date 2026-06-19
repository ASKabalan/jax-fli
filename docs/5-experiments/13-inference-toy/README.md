# Experiment 13 — Toy field-level inference (full sky) ⚠️ BLOCKED

**Goal.** The smallest end-to-end field-level posterior over the initial conditions (and cosmology),
full sky, no mask — the minimal working inference that validates the pipeline before scaling up.

**Status.** ⚠️ **Blocked on the inference pipeline** (batched sampling + sample saving,
`CatalogExtract`, the power-spectrum model). The forward model is differentiated *inside* `fli-infer`
(`--grad checkpoint`, many evaluations), not as a standalone `fli-simulate` run — so there is no run
grid yet.

## Run

Blocked — see [`run.sh`](run.sh). Once the pipeline is fixed this launches `fli-infer`
(NUTS / HMC / MCLMC, `--grad checkpoint`).
