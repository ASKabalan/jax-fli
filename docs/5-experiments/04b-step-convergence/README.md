# Experiment 04b — Number of integration steps for convergence ⚠️ WIP

**Goal.** Find how few PM integration steps give a converged per-shell spherical `C_ℓ`, for each
solver — the minimum `--nb-steps` at which the lightcone maps stop changing. The baseline is the
highest step count (50); every lower count is measured against it. This fixes the **step budget** for
the production runs: the cheapest simulation that is still accurate.

**Method.** Hold everything fixed (resolution, box, painting, shells, seed, float64) and sweep
`--nb-steps ∈ {5, 6, 10, 20, 30, 50}` for both production integrators — `kdk` (DoubleKickDrift) and
`bf` (BullFrog). For each (solver, steps) we paint the spherical lightcone and measure the per-shell
`C_ℓ`; convergence is the `C_ℓ` ratio (and a summary error) against the **same solver's** 50-step run.
The low counts (5, 6) probe the regime where the integrator starts to break down, and the two solvers
are expected to converge at different step budgets — this is the per-step-count complement to the
solver sweep in [Experiment 04](../04-solver-comparison/README.md).

**Status.** ⚠️ Not yet run. Several fixed knobs are **placeholders** pending the experiments that set
them: the resolution from [Exp 01](../01-resolution-convergence/README.md), the mass-assignment scheme +
force deconvolution from [Exp 02](../02-mass-assignment/README.md), and the spherical-paint
interpolation from [Exp 03](../03-spherical-painting/README.md). The figure-making script is added here
once the runs complete.

## Grid

*Fixed (placeholders — see Status):* `--sim-mode pm`, **1024³** (Exp 01), `--paint-order tsc
--deconvolution` (Exp 02), `--scheme ngp` (Exp 03), `--nside 512`, `--nb-shells 10`, `--shell-spacing
comoving`, box `2000³` Mpc/h, `--seed 0`, **float64**. **8 GPU** (2 nodes × 4, `--pdim 8 1`; local 128³,
halo 64).

| # | `--solver` | `--nb-steps` | | # | `--solver` | `--nb-steps` |
|--:|------|------|---|--:|------|------|
| 1 | kdk | 5  | | 7  | bf | 5  |
| 2 | kdk | 6  | | 8  | bf | 6  |
| 3 | kdk | 10 | | 9  | bf | 10 |
| 4 | kdk | 20 | | 10 | bf | 20 |
| 5 | kdk | 30 | | 11 | bf | 30 |
| 6 | kdk | 50 | | 12 | bf | 50 |

> GPU count follows the Exp 01 rule: local mesh `mesh/px` ≤ 512³ and the halo `int(mesh/px·0.5)` even.
> 1024³ on 8 GPUs gives local 128³ and halo 64. If the resolution is raised once Exp 01 lands, scale the
> GPU count to keep both conditions.

## Run

```bash
MODE=dryrun bash run.sh     # print the resolved fli-launcher commands (submit nothing)
bash run.sh                 # submit to SLURM
```

Writes one parquet per (solver, steps) to `results/exp4b/`. Reruns skip any rung whose parquet already
exists (handled by the shared [`../_launch_common.sh`](../_launch_common.sh)).
