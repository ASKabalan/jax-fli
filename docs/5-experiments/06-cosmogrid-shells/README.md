# Experiment 06 — Match CosmoGrid shells ⚠️ WIP

**Goal.** Reproduce the CosmoGrid radial shell geometry exactly (its shell `z`-edges) and compare the
per-shell density `C_ℓ` + cross-correlation against the CosmoGrid shells, isolating geometry from
resolution/painting effects.

**Status.** ⚠️ Not yet run. **Data-prep required:** extract the CosmoGrid shell `a`-edges and pass
them via `--ts-near` / `--ts-far` (set `TS_NEAR` / `TS_FAR` before launching).

## Grid

*Fixed:* `--sim-mode pm`, 2048³, BullFrog (`bf`), `--nb-steps 50`, `--paint-order tsc
--deconvolution`, `--scheme ngp`, `--nside 512`, box `2000³` Mpc/h, `--seed 0`, **float64**. **32 GPU**
(8 nodes×4, `--pdim 8 4`).

| # | shells |
|--:|------|
| 1 | `--ts-near` / `--ts-far` = CosmoGrid shell edges |

## Run

```bash
TS_NEAR="..." TS_FAR="..." MODE=dryrun bash run.sh    # set the CosmoGrid shell edges first
TS_NEAR="..." TS_FAR="..." bash run.sh
```

Writes `results/exp6/cosmogrid_shells.parquet`.
