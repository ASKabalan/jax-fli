#!/bin/bash

# export FLI_PAPER_PATH=/home/wassim/Projects/NBody/jax-fli/paper

cd /home/wassim/Projects/NBody/jax-fli/docs/5-experiments
for f in $(find . -name '*build*.py' | sort); do
  echo "═══════════════════════════════════════════════════════════"
  echo ">>> $f"
  ( cd "$(dirname "$f")" && JAX_PLATFORMS=cpu uv run python "$(basename "$f")" ) \
    && echo "✔ OK   $f" || echo "✗ FAIL $f"
done