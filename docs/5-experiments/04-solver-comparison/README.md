# Experiment 04 — Solver comparison: speed & convergence ⚠️ WIP

**Goal.** Compare the three N-body integrators — `kdk` (DoubleKickDrift), `dkd` (DriftKickDrift),
`bf` (BullFrog) — on (a) convergence of the last spherical map vs step count. These are the
**convergence** runs; the **speed** half (wall-time/step) is the per-stage timing in
[Experiment 11](../11-scaling/README.md).

**Status.** ⚠️ Not yet run.

## Grid

*Fixed:* `--sim-mode pm`, 1024³, `--paint-order tsc --deconvolution`, `--scheme ngp`, `--nside 512`,
`--nb-shells 10`, `--shell-spacing comoving`, box `2000³` Mpc/h, `--seed 0`, **float64**. **4 GPU**
(1 node×4, `--pdim 4 1`).

| # | `--solver` | `--nb-steps` | | # | `--solver` | `--nb-steps` |
|--:|------|------|---|--:|------|------|
| 1 | kdk | 10 | | 7  | dkd | 50 |
| 2 | kdk | 18 | | 8  | bf  | 10 |
| 3 | kdk | 30 | | 9  | bf  | 18 |
| 4 | kdk | 50 | | 10 | bf  | 30 |
| 5 | dkd | 10 | | 11 | bf  | 50 |
| 6 | dkd | 18 | | 12 | dkd | 30 |

## Run

```bash
MODE=dryrun bash run.sh
bash run.sh
```

Writes one parquet per (solver, steps) to `results/exp4/`.
