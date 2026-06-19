#!/bin/bash
# Experiment 00 — CosmoGrid reference (cluster entry point).
#
# Four scripts in this folder (see README.md):
#   publish_density_2048.py  -> 00-cosmogrid-density (nside 2048, uint16)   [CPU / login node, ~16 GB RAM]
#   raytrace_kappa.py        -> kappa_raytrace.parquet  (dorian + MPI)      [big-RAM node; replicated]
#   born_kappa.py            -> kappa_born.parquet      (JAX-distributed)   [one process per GPU; sharded]
#   publish_local.py         -> uploads both kappa parquets to HuggingFace  [local, HF_TOKEN]
#
# The two kappa compute scripts only SAVE parquet; publish afterwards with `python publish_local.py --yes`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCOUNT="${ACCOUNT:?set ACCOUNT (e.g. export ACCOUNT=tkc@h100)}"

# Offline cluster (Jean Zay has no internet on compute nodes)? Pre-cache the density on a LOGIN node:
#   HF_HOME=$WORK/hf_cache python -c \
#     "from datasets import load_dataset; load_dataset('ASKabalan/jax-fli-experiments','00-cosmogrid-density',split='train')"
# then export on compute nodes so HF reads from the cache without touching the network:
#   export HF_HOME=$WORK/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

# Ray-traced kappa: dorian is replicated numpy+MPI (per-rank HOST RAM ~14 GB) — size the node for
# memory. JAX is pinned to CPU inside the script (host-side load); MPI parallelises the ray-tracing.
srun --account "$ACCOUNT" --constraint h100 --nodes 8 --gpus-per-node 4 \
    python "$HERE/raytrace_kappa.py" --nz s3 --interp bilinear --out "$HERE/kappa_raytrace.parquet"

# Born kappa: JAX-distributed, one process per GPU; the density is sharded across the mesh.
srun --account "$ACCOUNT" --constraint h100 --nodes 8 --gpus-per-node 4 \
    python "$HERE/born_kappa.py" --multihost --nz s3 --out "$HERE/kappa_born.parquet"

# Then publish both from a node with internet + HF_TOKEN:
#   python "$HERE/publish_local.py" --yes
