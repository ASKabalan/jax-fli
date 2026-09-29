# Experiment 99 — GPU Memory Calibration Suite

**Goal.** This suite calibrates the GPU memory limits of the full forward-and-backward model over **100 gradient iterations**, across a ladder of mesh resolutions on **8 GPUs** (or several nodes), to find where an 80 GB H100 (or A100) saturates. The model is a 50-step BullFrog (`bf`) with `D`-time stepping and the drift on the lightcone, painting 20 equal-volume spherical shells at nside 2048 with RBF smoothing of 0.8 pixel, in float64 (`--enable-x64`), with the checkpointed gradient of Equinox (`--grad checkpointed_25`) and `--perf` over the 100 iterations.

| Global Mesh $M$ | Slabs $P_x \times P_y$ | Local Grid per GPU | Halo Multiplier | Local Padded Mesh | Memory Regime |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$512^3$** | $8 \times 1$ | $64 \times 512 \times 512$ | $0.5$ | $(64 + 32) \times 512 \times 512$ | Baseline (~16 GB / GPU) |
| **$584^3$** | $8 \times 1$ | $73 \times 584 \times 584$ | $1.0$ | $(73 + 74) \times 584 \times 584$ | Calibration Point |
| **$640^3$** | $8 \times 1$ | $80 \times 640 \times 640$ | $1.0$ | $(80 + 80) \times 640 \times 640$ | Intermediate (~35 GB / GPU) |
| **$768^3$** | $8 \times 1$ | $96 \times 768 \times 768$ | $1.0$ | $(96 + 96) \times 768 \times 768$ | High Fill (~55 GB / GPU) |
| **$1024^3$** | $8 \times 1$ | $128 \times 1024 \times 1024$ | $1.0$ | $(128 + 128) \times 1024 \times 1024$ | Saturation Bound (~75–80 GB) |

With `--perf`, `fli-simulate` reports per iteration the peak memory resident on GPU rank 0 (`max_memory_allocated`), the wall-clock time of one gradient iteration (`step_time_ms`), and the initial JIT compilation and memory reservation time. A run is **calibrated** when its peak memory reaches $\approx 90–95\%$ of the device memory without triggering `CUDA_ERROR_OUT_OF_MEMORY`.

## How to run

```bash
MODE=dryrun bash run.sh               # inspect the commands without submitting
MODE=sbatch bash run.sh               # submit to the H100 partition
GPUS_PER_NODE=8 bash run.sh           # 8 GPUs on one node instead of 2 × 4
GRAD_SPEC=checkpointed_10 bash run.sh # another checkpoint count
GRAD_SPEC=reverse bash run.sh         # the reverse adjoint
SIM_MODE=lensing bash run.sh          # the gradient of the 3-bin Born lensing instead of the PM density
```
