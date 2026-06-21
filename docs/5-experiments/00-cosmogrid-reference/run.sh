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

# Offline cluster (Jean Zay has no internet on compute nodes)? Pre-cache the density on a LOGIN node
# (download.py snapshots the per-shell parquets + README and streams them — a non-streaming
# load_dataset would overflow arrow's int32 list-offset across the 56 shells):
#   HF_HOME=$WORK/hf_cache python "$HERE/download.py"
# then export on compute nodes so HF reads from the cache without touching the network:
#   export HF_HOME=$WORK/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

# Ray-traced kappa: dorian is replicated numpy+MPI. Each rank holds the FULL lightcone, and the
# raytrace path keeps two copies (jnp + numpy) plus rank 0's float64 dorian shells -> ~22 GB/rank,
# ~45 GB on rank 0. So run FEW ranks per node: 4 tasks x 10 cpus on the 191 GB node (40 ranks OOM).
# The 11 GB lightcone is broadcast shell-by-shell (mpi4py pickle bcast caps at 2 GiB). Work splits
# over n_integrate+1 = 33 sources, so >4 ranks would speed it up if RAM allows (ceiling ~6-7/node).
sbatch --account=rzt@cpu --nodes 1 --ntasks-per-node=1 --cpus-per-task=40 --time 10:00:00 --job-name "RAY_TRACE_S3" \
    $SLURM_SCRIPT LENSING python "$HERE/raytrace_kappa.py" --nz s3 --interp bilinear --out "$HERE/kappa_raytrace_s3.parquet"

sbatch --account=rzt@cpu --nodes 1 --ntasks-per-node=4 --cpus-per-task=40 --time 10:00:00 --job-name "RAY_TRACE_DES" \
    $SLURM_SCRIPT LENSING python "$HERE/raytrace_kappa.py" --nz des --interp bilinear --out "$HERE/kappa_raytrace_des.parquet"


# born auto-detects the multi-process launch (no --multihost flag — its argparse would reject it).
sbatch --account=tkc@a100 -C a100  --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=8  --qos=qos_gpu_a100-dev --nodes=4 --mail-type=BEGIN --mail-user=${EMAIL} --time 01:00:00 --job-name "BORN_TRACE_S3" \
     $SLURM_SCRIPT LENSING python "$HERE/born_kappa.py" --nz s3 --out "$HERE/kappa_born_s3.parquet"
sbatch --account=tkc@a100 -C a100  --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=8  --qos=qos_gpu_a100-dev --nodes=4 --mail-type=BEGIN --mail-user=${EMAIL} --time 01:00:00 --job-name "BORN_TRACE_DES" \
     $SLURM_SCRIPT LENSING python "$HERE/born_kappa.py" --nz des --out "$HERE/kappa_born_des.parquet"

# Then publish both from a node with internet + HF_TOKEN:
#   python "$HERE/publish_local.py" --yes
