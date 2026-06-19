#!/bin/bash
# Experiment 14 — field-level inference vs CosmoGrid shear. BLOCKED on the inference pipeline.
# Inference is differentiated INSIDE fli-infer (--grad checkpoint), not a standalone fli-simulate run.
# Geometry: observer (0.1,0.5,0.9) (Exp 8 Case-3 cap), DES Y3 mask; 2048³, bf, TSC+deconv, RBF, shear.
set -euo pipefail
echo "Exp 14 is BLOCKED on the inference pipeline (fli-infer). No runs yet."
echo "Pre-flight: shear units vs CosmoGrid; where the scale cut lives; x64 before import jax_fli."
exit 1
