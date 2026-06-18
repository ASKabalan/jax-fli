# Experiment 03 — Spherical painting + pixel-window ⚠️ WIP

**Goal.** Quantify the HEALPix painting interpolation scheme (NGP / bilinear / RBF kernel width) and
the effect of painting at a higher `paint-nside` then downgrading (pixel-window), on the spherical
`C_ℓ`. Depends on the Exp 2 winner (TSC + deconvolution).

**Status.** ⚠️ Not yet run.

## Grid

*Fixed:* `--sim-mode pm`, 2048³, BullFrog (`bf`), `--nb-steps 50`, `--paint-order tsc
--deconvolution`, `--nb-shells 10`, `--shell-spacing comoving`, box `2000³` Mpc/h, `--seed 0`,
**float64**. **32 GPU** (8 nodes×4, `--pdim 8 4`). The HEALPix pixel-window deconvolution is applied
**after** via `power.deconvolve_spherical` (in `fli-spectra` / the study notebook), not a flag.

| # | `--scheme` | `--kernel-width-pixels` | nside / paint |
|--:|------|------|------|
| 1 | ngp | — | `--nside 1024` (native) |
| 2 | ngp | — | `--nside 1024 --paint-nside 2048` |
| 3 | bilinear | — | `--nside 1024` |
| 4 | bilinear | — | `--nside 1024 --paint-nside 2048` |
| 5 | rbf_neighbor | 0.8 | `--nside 1024` |
| 6 | rbf_neighbor | 0.8 | `--nside 1024 --paint-nside 2048` |
| 7 | rbf_neighbor | 1.5 | `--nside 1024` |
| 8 | rbf_neighbor | 1.5 | `--nside 1024 --paint-nside 2048` |

## Run

```bash
MODE=dryrun bash run.sh
bash run.sh
```

Writes one parquet per config to `results/exp3/`.
