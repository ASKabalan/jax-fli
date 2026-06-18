# Experiment 01 — Resolution convergence ⚠️ WIP

**Goal.** Show the per-shell spherical-map angular power spectrum `C_ℓ` converging with particle
count at fixed box and step count — the baseline is the highest tractable resolution, cross-checked
against CosmoGrid. This fixes the mesh resolution needed for sub-percent lightcone `C_ℓ`.

**Status.** ⚠️ Not yet run.

## Grid

*Fixed:* `--sim-mode pm`, BullFrog (`bf`), `--nb-steps 50`, `--paint-order cic`, `--scheme ngp`,
`--nside 512`, `--nb-shells 10`, `--shell-spacing comoving`, box `2000³` Mpc/h, fiducial cosmology,
`--seed 0`, **float64** (`--enable-x64`).

| # | mesh | GPUs | nodes | `--pdim` | local | halo | local mesh |
|--:|------|--:|--:|------|--:|--:|------|
| 1 | 512³  | 4   | 1   | `4 1`   | 128 | 64 | 322³ |
| 2 | 1024³ | 8   | 2   | `8 1`   | 128 | 64 | 512³ |
| 3 | 2048³ | 64  | 16  | `64 1`  | 32  | 16 | 512³ |
| 4 | 2560³ | 128 | 32  | `128 1` | 20  | 10 | 509³ |
| 5 | 3072³ | 256 | 64  | `256 1` | 12  | 6  | 484³ |
| 6 | 3584³ | 448 | 112 | `448 1` | 8   | 4  | 468³ |
| 7 | 4096³ | 512 | 128 | `512 1` | 8   | 4  | 512³ |

> All rungs use the default `--halo-multiplier 0.5`. The GPU count per rung is the smallest `px`
> (with `px | mesh` and `mesh/px` a multiple of 4) such that the local mesh stays ≤ 512³ — the
> largest that fits a float64 H100 (matching the 2048³/64-GPU run). Making `mesh/px` a multiple of 4
> also makes the halo `int((mesh/px)·0.5) = local/2` **even**, which the distributed `slice_unpad`
> requires (an odd halo crashes it). Padding is x-only — with `py=1` the y-halo is dropped — so the
> painted slab is `(2·local_x, mesh, mesh)`. All rungs run in float64.

## Run

```bash
MODE=dryrun bash run.sh     # print the resolved fli-launcher commands (submit nothing)
bash run.sh                 # submit to SLURM (MODE=sbatch, the default)
```

Writes one parquet per rung to `results/exp1/`. Cluster knobs (`ACCOUNT`, `CONSTRAINT`, `QOS`, …) are
overridable env vars — see [`../_launch_common.sh`](../_launch_common.sh). The figure-making script is
added here once the runs complete.
