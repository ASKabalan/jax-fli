# Experiment 05c — Spacing & stepping: equal-volume shells, 3-bin ⚠️ WIP

**Goal.** Take the deep, 3-bin tomographic setup of [Experiment 05b](../05b-spacing-n-stepping-3bin/README.md)
and swap the shell spacing from scale-factor (`--shell-spacing a`) to **equal volume**
(`--shell-spacing equal_vol`), isolating the **near-shell shot-noise** lever. Equal-volume shells give
every shell the same comoving volume — and therefore a uniform particle count — instead of starving the
inner (near-observer) shells, which is the dominant cause of their poor per-shell `C_ℓ` convergence.
Everything else matches 05b (5 Gpc/h box, 2560³, BullFrog, 50 steps, `--time-stepping D`, 3-bin Born),
so the comparison cleanly attributes any change to the spacing.

| sweep | values |
|-------|--------|
| `--nb-shells` | 5, 8, 10, 12, 16, 20, 25, 30, 40 |
| drift | (none), `--drift-on-lightcone` |
| `--shell-spacing` | `equal_vol` (the one change vs 05b) |
| lensing | 3-bin Born (`--nz-shear s3[:3]`) on each density run |

*Fixed:* `--sim-mode pm`, **2560³**, box **`5000³` Mpc/h**, BullFrog (`bf`), `--nb-steps 50`,
`--time-stepping D`, `--paint-order cic` with **no** force-window `--deconvolution`, `--scheme ngp`,
`--nside 2048`, `--shells-per-file 1`, `--min-width 5.0` (guardrail only), `--seed 0`, **float64**.
**128 GPU** (32 nodes × 4, `--pdim 128 1`).

> **Halo (Exp 01 rule).** 2560³ on 128 GPUs (slab) gives local `20³` and an **even** halo
> `int(20·0.5) = 10`; local `20·2560·2560 = 1.31e8 ≈ 512³` fits float64. The physical ghost zone
> `0.5·5000/128 = 19.5 Mpc/h` clears the end-of-run rms displacement.

**Status.** ⚠️ Not yet run — `run.sh` is ready; figures and a `build.py` follow once the cluster runs
land and are published to the
[`ASKabalan/jax-fli-experiments`](https://huggingface.co/datasets/ASKabalan/jax-fli-experiments) dataset
under `05-spacing-n-stepping/05c-3bin-equalvol/`.

## Run

```bash
MODE=dryrun bash run.sh   # print the resolved commands (submit nothing)
bash run.sh               # submit to SLURM
```

The density sweep writes one directory of per-shell parquet (`shell_NNNN.parquet`) per
(drift, shell-count) to `results/exp5c/`; once those are pushed to HuggingFace, the `fli-born-rt` step
reads them back and writes the 3-bin convergence maps. A local figure script renders the SVGs
afterward (heavy compute on the cluster, figure-making reproducible locally without a GPU, per the
[experiment conventions](../CLAUDE.md)).
