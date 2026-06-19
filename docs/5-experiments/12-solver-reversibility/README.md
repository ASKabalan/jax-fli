# Experiment 12 — Solver reversibility error ⚠️ WIP

**Goal.** Measure the forward → reverse round-trip error vs step count — a direct check that the
reversible backsolve reconstructs the trajectory (the basis of the O(1)-memory `reverse` adjoint),
independent of the gradient.

**Status.** ⚠️ Not yet run.

## Grid

*Fixed:* `--sim-mode pm --density`, 128³, `kdk`, `--time-stepping a --gradient-order 0`,
`--enable-x64`, `--grad reverse`, box `256³` Mpc/h, `--seed 0`, **float64**. **1 GPU** (`--pdim 1 1`).

| # | `--nb-steps` |
|--:|------|
| 1 | 10 |
| 2 | 20 |
| 3 | 40 |
| 4 | 80 |

## Run

```bash
MODE=dryrun bash run.sh
bash run.sh
```

Writes one parquet per step count to `results/exp12/`.
