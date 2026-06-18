# Experiment 11 — Performance: throughput, strong & weak scaling ⚠️ WIP

**Goal.** Characterize the distributed performance of the forward model: (a) the per-stage cost
breakdown, (b) **strong scaling** (fixed grid, more GPUs → speedup + parallel efficiency), and
(c) **weak scaling** (fixed work per GPU → flat wall-time = ideal). Every run captures **wall-time
*and* per-device memory** from `fli-simulate --perf` (XLA `memory_analysis`: argument / output /
temp-scratch bytes), and is run in **both float32 and float64** — the precision/perf/memory trade-off
is the point.

**Status.** ⚠️ Not yet run.

*Fixed:* `--sim-mode pm`, BullFrog (`bf`), `--nb-steps 50`, `--paint-order tsc --deconvolution`,
`--scheme ngp`, `--perf --iterations 5`, box `2000³` Mpc/h, `--seed 0`. GPUs = nodes×4,
balanced-pencil `--pdim`. **float64 ceiling ≈ 645³/GPU** on 80 GB (production 4096³/256-GPU run);
each strong-scaling grid starts at the smallest GPU count that fits in double. Each config below runs
**×2** (float32 + float64).

## (a) Per-stage cost — 3 configs (1024³, 4 GPU `--pdim 2 2`)

| # | sim-mode |
|--:|------|
| 1 | `lpt` |
| 2 | `pm --nside 512 --nb-shells 10` |
| 3 | `lensing --nside 512 --nb-shells 10 --nz-shear s3` |

## (b) Strong scaling — fixed grid, grow GPUs

| # | grid | GPUs | nodes×4 | `--pdim` | f64 per-GPU |
|--:|------|--:|--:|------|------|
| 4 | 1024³ | 4   | 1  | `2 2`   | 645³ (ceiling) |
| 5 | 1024³ | 8   | 2  | `2 4`   | 512³ |
| 6 | 1024³ | 16  | 4  | `4 4`   | 406³ |
| 7 | 1024³ | 32  | 8  | `4 8`   | 323³ |
| 8 | 1024³ | 64  | 16 | `8 8`   | 256³ |
| 9 | 2048³ | 32  | 8  | `4 8`   | 645³ (ceiling) |
| 10 | 2048³ | 64  | 16 | `8 8`   | 512³ |
| 11 | 2048³ | 128 | 32 | `8 16`  | 406³ |
| 12 | 2048³ | 256 | 64 | `16 16` | 323³ |

## (c) Weak scaling — fixed 512³ per GPU, `global = (512·PX, 512·PY, 512)`

| # | GPUs | nodes×4 | `--pdim` | global `--mesh-size` |
|--:|--:|--:|------|------|
| 13 | 4   | 1  | `2 2`   | `1024 1024 512` |
| 14 | 8   | 2  | `2 4`   | `1024 2048 512` |
| 15 | 16  | 4  | `4 4`   | `2048 2048 512` |
| 16 | 32  | 8  | `4 8`   | `2048 4096 512` |
| 17 | 64  | 16 | `8 8`   | `4096 4096 512` |
| 18 | 128 | 32 | `8 16`  | `4096 8192 512` |
| 19 | 256 | 64 | `16 16` | `8192 8192 512` |

Scaling grids are anisotropic by construction (the 2D decomposition shards X, Y; Z = per-GPU depth) —
these are perf/memory benchmarks, not science runs. **19 configs × 2 precisions ≈ 38 runs.**

## Run

```bash
MODE=dryrun bash run.sh
bash run.sh
```

Writes `perf_<mode>.csv` rows (wall-time + memory) under `results/exp11/`.
