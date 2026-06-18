# Experiment 02 — Mass assignment + force deconvolution ⚠️ WIP

**Goal.** Quantify how the 3D force mass-assignment order (CIC / TSC / PCS) and the optional
Fourier-space force-window `--deconvolution` shift the spherical-map `C_ℓ`; the baseline is
PCS + deconvolution (and/or Halofit).

**Status.** ⚠️ Not yet run.

## Grid

*Fixed:* `--sim-mode pm`, 2048³, BullFrog (`bf`), `--nb-steps 50`, `--scheme ngp`, `--nside 512`,
`--nb-shells 10`, `--shell-spacing comoving`, box `2000³` Mpc/h, `--seed 0`, **float64**
(`--enable-x64`). **32 GPU** (8 nodes×4, `--pdim 8 4`).

| # | `--paint-order` | `--deconvolution` |
|--:|------|------|
| 1 | cic | off |
| 2 | cic | on |
| 3 | tsc | off |
| 4 | tsc | on |
| 5 | pcs | off |
| 6 | pcs | on |

## Run

```bash
MODE=dryrun bash run.sh     # print the resolved fli-launcher commands
bash run.sh                 # submit to SLURM (default)
```

Writes one parquet per config to `results/exp2/`.
