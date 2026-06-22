# Experiment 05 — Drift on the lightcone ⚠️ WIP

**Goal.** Show that drifting particles to their lightcone-crossing epoch (`--drift-on-lightcone`)
improves the per-shell density `C_ℓ` for **thick** shells, while the Born convergence is unaffected
(the radial projection dominates). Run with thick shells (`--nb-shells 8`) so the effect is visible.

**Status.** ⚠️ Not yet run.

## Grid

*Fixed:* `--sim-mode pm`, **2048³** (Exp 01), BullFrog (`bf`), `--nb-steps 50`, `--paint-order cic`
with **no** force-window `--deconvolution` (Exp 02), `--scheme ngp` (Exp 03), `--nside 2048`
(production CosmoGrid projection), `--shells-per-file 1` (one parquet per shell), **`--nb-shells 8`**
(thick), `--shell-spacing comoving`, box `2000³` Mpc/h, `--seed 0`, **float64**.
**64 GPU** (16 nodes × 4, `--pdim 64 1`).

| # | drift |
|--:|------|
| 1 | (none) |
| 2 | `--drift-on-lightcone` |

> **Halo (Exp 01 rule).** 2048³ on 64 GPUs gives local `32³ ≤ 512³` and an even halo
> `int(32·0.5) = 16`; the physical ghost zone `0.5·2000/64 = 15.6 Mpc/h` clears the 3D rms
> displacement `σ₃D(z=0) = 10.2 Mpc/h` (`1.53×`, the converged
> [Exp 01](../01-resolution-convergence/README.md) rung). Drift-on-lightcone only repaints existing
> particles, so the displacement scale is unchanged.

## Run

```bash
MODE=dryrun bash run.sh
bash run.sh
```

Writes two directories of per-shell parquet (`shell_NNNN.parquet`) — `exp5_nodrift`, `exp5_drift` —
to `results/exp5/`, plus an appended `results/exp5/perf_pm.csv`.
