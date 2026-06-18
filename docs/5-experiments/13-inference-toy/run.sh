#!/bin/bash
# Experiment 13 — toy field-level inference (full sky). BLOCKED on the inference pipeline.
# Inference is differentiated INSIDE fli-infer (--grad checkpoint), not a standalone fli-simulate run.
# Blockers: batched sampling + sample saving, CatalogExtract, the power-spectrum model.
set -euo pipefail
echo "Exp 13 is BLOCKED on the inference pipeline (fli-infer). No runs yet."
echo "Once unblocked: launch with fli-infer (NUTS / HMC / MCLMC, --grad checkpoint), full sky, no mask."
exit 1
