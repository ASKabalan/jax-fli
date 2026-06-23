# Experiment 04 — Step convergence (solver × step count) ⚠️ WIP

**Goal.** Two questions in one sweep. (1) **Step budget** — how few PM integration steps give a
converged per-shell spherical `C_ℓ`, for each integrator: the minimum `--nb-steps` at which the
lightcone maps stop changing, which fixes the cheapest accurate production run. (2) **Solver
comparison** — how the three N-body integrators `kdk` (DoubleKickDrift), `dkd` (DriftKickDrift) and
`bf` (BullFrog) converge relative to one another. (This experiment merges the former
`04-solver-comparison` and `04b-step-convergence`, which were the same solver × step-count sweep; the
**speed** half — wall-time per step — lives in [Experiment 11](../11-scaling/README.md).)

**Method.** Hold everything fixed (resolution, box, painting, shells, seed, float64) and sweep
`--nb-steps ∈ {5, 6, 10, 20, 30, 50}` for all three integrators. For each (solver, steps) we paint the
spherical lightcone and measure the per-shell `C_ℓ`; convergence is the `C_ℓ` ratio (and a summary
error) against the **same solver's** 50-step run. The low counts (5, 6) probe the regime where the
integrator starts to break down, and the solvers are expected to converge at different step budgets.

**Status.** ⚠️ Not yet run. The figure-making script is added here once the runs complete.

## Grid

*Fixed:* `--sim-mode pm`, **2048³** (Exp 01: resolution-converged, halo-safe for float64),
`--paint-order cic` with **no** force-window `--deconvolution` (Exp 02), `--scheme ngp` (Exp 03),
`--nside 2048` (production CosmoGrid projection), `--shells-per-file 1` (one parquet per shell),
`--nb-shells 10`, `--shell-spacing comoving`, box `2000³` Mpc/h, `--seed 0`, **float64**.
**64 GPU** (16 nodes × 4, `--pdim 64 1`).

| # | `--solver` | `--nb-steps` | | # | `--solver` | `--nb-steps` | | # | `--solver` | `--nb-steps` |
|--:|------|------|---|--:|------|------|---|--:|------|------|
| 1 | kdk | 5  | |  7 | dkd | 5  | | 13 | bf | 5  |
| 2 | kdk | 6  | |  8 | dkd | 6  | | 14 | bf | 6  |
| 3 | kdk | 10 | |  9 | dkd | 10 | | 15 | bf | 10 |
| 4 | kdk | 20 | | 10 | dkd | 20 | | 16 | bf | 20 |
| 5 | kdk | 30 | | 11 | dkd | 30 | | 17 | bf | 30 |
| 6 | kdk | 50 | | 12 | dkd | 50 | | 18 | bf | 50 |

> **GPU count & halo (Exp 01 rule).** 2048³ on 64 GPUs (`--pdim 64 1`) gives local mesh `32³ ≤ 512³`
> (fits a float64 H100) and an **even** halo `int(32·0.5) = 16` (an odd halo crashes jaxpm
> `slice_unpad`). The physical ghost zone `halo_multiplier·box/px = 0.5·2000/64 = 15.6 Mpc/h` clears
> the end-of-run 3D rms particle displacement `σ₃D(z=0) = 10.2 Mpc/h` with a `1.53×` margin (`≥ 1.5`)
> — the exact resolution-converged rung of [Exp 01](../01-resolution-convergence/README.md). Box and
> cosmology fix the displacement, so this one check holds for every (solver, steps) here.

## Run

```bash
MODE=dryrun bash run.sh     # print the resolved fli-launcher commands (submit nothing)
bash run.sh                 # submit to SLURM
```

Writes one directory of per-shell parquet (`shell_NNNN.parquet`) per (solver, steps) to
`results/exp4/<solver>_s<steps>/`, plus an appended `results/exp4/perf_pm.csv` timing row per run.
Reruns skip any rung whose output directory already holds parquet (handled by the shared
[`../_launch_common.sh`](../_launch_common.sh)).
