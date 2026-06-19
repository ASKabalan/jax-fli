# Experiment 07 — Born lensing vs CosmoGrid lensed ⚠️ WIP

**Goal.** Compare our Born-approximation convergence against the CosmoGrid lensed maps: convergence
`C_ℓ` + cross-correlation, for a DES Y3-like source distribution.

**Status.** ⚠️ Not yet run. *(Optional: one run per DES Y3 tomographic bin.)*

## Grid

*Fixed:* `--sim-mode lensing`, 2048³, BullFrog (`bf`), `--nb-steps 50`, `--paint-order tsc
--deconvolution`, `--scheme ngp`, `--nside 512`, `--nb-shells 10`, `--shell-spacing comoving`,
`--lensing-output convergence`, box `2000³` Mpc/h, `--seed 0`, **float64**. **32 GPU** (8 nodes×4,
`--pdim 8 4`).

| # | `--nz-shear` |
|--:|------|
| 1 | s3 *(optional: one run per DES Y3 bin)* |

## Run

```bash
MODE=dryrun bash run.sh
bash run.sh
```

Writes `results/exp7/born_s3.parquet`.
