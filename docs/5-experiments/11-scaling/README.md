# Experiment 11 — PM performance: strong & weak scaling ⚠️ WIP

**Goal.** Characterize the distributed performance of the **PM forward model** (N-body + lightcone
painting) under a **slab `(N, 1)`** decomposition: **strong scaling** (fixed grid, more GPUs → speedup +
parallel efficiency) and **weak scaling** (fixed work per GPU → flat wall-time = ideal). Every run captures
**wall-time *and* per-device memory** from `fli-simulate --perf` (XLA `memory_analysis`: argument / output /
temp-scratch bytes), in **both float32 and float64** — the precision/perf/memory trade-off is the point.
Only the PM stage is timed here (the LPT-only and lensing-only stage breakdown is dropped); correctness and
the gradient cost live in [Exp 12](../12-scaling-gradient/README.md).

**Status.** ⚠️ Not yet run.

*Fixed:* `--sim-mode pm`, BullFrog (`bf`), `--nb-steps 50`, `--paint-order tsc --deconvolution`,
`--scheme ngp`, lightcone `--nside 1024 --nb-shells 10`, `--perf --iterations 5`, box `2000³` Mpc/h,
`--seed 0`. **Slab `(N, 1)`**: `px = #GPUs`, `py = 1`, `nodes = GPUs/4`, local mesh `(M/px, M, M)`.

### Slab constraints (which rungs actually run)

Two hard limits prune the nominal "4 → 512 GPUs" envelope per `(precision, mesh)`:

1. **Even halo.** The ghost zone `halo = int((M/px)·0.5)` must be **even** — an odd halo crashes jaxpm
   `slice_unpad`. At `halo_multiplier 0.5` that means `(M/px) % 4 == 0`, so a **1024³ slab tops out at 256
   GPUs** (512 → local 2, halo 1, skipped).
2. **Per-GPU memory ceiling.** Each ladder starts at the smallest GPU count whose local volume fits:
   **≈512³ cells/GPU in float64** (645³ OOMs, see [Exp 01](../01-resolution-convergence/README.md)),
   **≈2·512³ in float32**. Below-ceiling rungs are skipped and logged by `run.sh`.

## (a) Strong scaling — fixed grid, grow GPUs (slab `N × 1`)

| grid | float32 GPUs | float64 GPUs |
|------|------|------|
| 1024³ | 4, 8, 16, 32, 64, 128, 256 | 8, 16, 32, 64, 128, 256 |
| 2048³ | 32, 64, 128, 256, 512 | 64, 128, 256, 512 |

(1024³ stops at 256 GPUs — odd halo at 512. 2048³ float64 starts at 64 GPUs — `local 32³` at the ≈512³ ceiling.)

## (b) Weak scaling — fixed 256³/GPU (slab `N × 1`)

`global --mesh-size = (256·px, 256, 256)`; local mesh `256³`, halo `128` (even at every rung), so all eight
GPU counts run in both precisions.

| GPUs | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|------|---|---|----|----|----|-----|-----|-----|
| `--mesh-size` | `1024 256 256` | `2048 256 256` | `4096 256 256` | `8192 256 256` | `16384 256 256` | `32768 256 256` | `65536 256 256` | `131072 256 256` |

The weak grids are anisotropic by construction (the slab shards only X; Y, Z are per-GPU depth) — these are
perf/memory benchmarks, not science runs. **≈22 strong + 16 weak ≈ 38 runs.**

## Run

```bash
MODE=dryrun bash run.sh    # print the resolved fli-launcher commands + the skipped rungs (submit nothing)
bash run.sh                # submit to SLURM
```

Writes `perf_pm.csv` rows (wall-time + memory) under `results/exp11/`. The shared
[`../_launch_common.sh`](../_launch_common.sh) skips any rung whose `--output` already exists.
