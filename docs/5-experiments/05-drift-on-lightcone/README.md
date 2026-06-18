# Experiment 05 — Drift on the lightcone ⚠️ WIP

**Goal.** Show that drifting particles to their lightcone-crossing epoch (`--drift-on-lightcone`)
improves the per-shell density `C_ℓ` for **thick** shells, while the Born convergence is unaffected
(the radial projection dominates). Run with thick shells (`--nb-shells 8`) so the effect is visible.

**Status.** ⚠️ Not yet run.

## Grid

*Fixed:* `--sim-mode pm`, 1024³, BullFrog (`bf`), `--nb-steps 50`, `--paint-order tsc
--deconvolution`, `--scheme ngp`, `--nside 512`, **`--nb-shells 8`** (thick), `--shell-spacing
comoving`, box `2000³` Mpc/h, `--seed 0`, **float64**. **4 GPU** (1 node×4, `--pdim 4 1`).

| # | drift |
|--:|------|
| 1 | (none) |
| 2 | `--drift-on-lightcone` |

## Run

```bash
MODE=dryrun bash run.sh
bash run.sh
```

Writes two parquet (`nodrift`, `drift`) to `results/exp5/`.
