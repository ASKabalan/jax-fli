# Experiment 99 — GPU Memory Calibration Suite

This experiment provides a dedicated benchmark suite to calibrate and stress-test GPU VRAM limits over **100 gradient backpropagation iterations** across a mesh resolution ladder on **8 GPUs** (or multi-node setups).

---

## 1. Physics & Benchmark Configuration

The calibration runs the full end-to-end forward-and-backward model:
- **Solver**: 50-step BullFrog (`bf`) with $D$-time stepping and drift-on-lightcone.
- **Lightcone**: 20 spherical shells ($NSIDE=2048$) with equal-volume spacing and RBF 0.8-pixel smoothing.
- **Precision**: `float64` (`--enable-x64`).
- **Gradient Strategy**: Equinox gradient checkpointing (`--grad checkpointed_25`).
- **Iterations & Profiling**: 100 iterations with `--perf` enabled.

---

## 2. The 8-GPU Resolution Ladder

The calibration ladder tests increasing local mesh and ghost zone sizes to determine the exact boundary where an 80 GB H100 (or A100) saturates:

| Global Mesh $M$ | Slabs $P_x \times P_y$ | Local Grid per GPU | Halo Multiplier | Local Padded Mesh | Memory Regime |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$512^3$** | $8 \times 1$ | $64 \times 512 \times 512$ | $0.5$ | $(64 + 32) \times 512 \times 512$ | Baseline (~16 GB / GPU) |
| **$584^3$** | $8 \times 1$ | $73 \times 584 \times 584$ | $1.0$ | $(73 + 74) \times 584 \times 584$ | Calibration Point |
| **$640^3$** | $8 \times 1$ | $80 \times 640 \times 640$ | $1.0$ | $(80 + 80) \times 640 \times 640$ | Intermediate (~35 GB / GPU) |
| **$768^3$** | $8 \times 1$ | $96 \times 768 \times 768$ | $1.0$ | $(96 + 96) \times 768 \times 768$ | High Fill (~55 GB / GPU) |
| **$1024^3$** | $8 \times 1$ | $128 \times 1024 \times 1024$ | $1.0$ | $(128 + 128) \times 1024 \times 1024$ | Saturation Bound (~75–80 GB) |

---

## 3. How to Run

### Dryrun (Inspect without submitting)
```bash
MODE=dryrun bash run.sh
```

### SLURM Submission (H100 Partition)
```bash
MODE=sbatch bash run.sh
```

### Custom Overrides
You can customize the run directly via environment variables:
```bash
# Test with 8 GPUs on a single 8-GPU node instead of 2x4 nodes:
GPUS_PER_NODE=8 bash run.sh

# Change checkpoint interval or test reverse adjoint:
GRAD_SPEC=checkpointed_10 bash run.sh
GRAD_SPEC=reverse bash run.sh

# Test 3-bin Born lensing gradient instead of PM density:
SIM_MODE=lensing bash run.sh
```

---

## 4. Reading the Performance Logs

When executed with `--perf`, `fli-simulate` outputs memory and timing statistics per iteration:
- **`max_memory_allocated`**: Peak VRAM resident on GPU rank 0.
- **`step_time_ms`**: Wall-clock duration per gradient backpropagation iteration.
- **Compilation overhead**: Initial JIT compilation and memory reservation time.

The run is considered **calibrated** when peak VRAM reaches $\approx 90–95\%$ of available device memory without triggering `CUDA_ERROR_OUT_OF_MEMORY`.
