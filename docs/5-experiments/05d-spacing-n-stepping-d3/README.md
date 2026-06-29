# Experiment 05d — Spacing & stepping: D³ time stepping, 3-bin ⚠️ WIP

**Goal.** Add the third lever — **time stepping** — to the deep 3-bin setup. Equal-volume spacing
([05c](../05c-spacing-n-stepping-equal-vol/README.md)) fixes *where the particles land*; D³ stepping
(`--time-stepping D3`) fixes *where the integration steps land*. A BullFrog step has a leading residual
`∝ D³`, so equalising the per-step error means stepping **uniformly in `τ = D³`**, which concentrates
steps at late times (the near, most non-linear shells). This experiment runs D³ stepping over **both**
equal-volume and scale-factor shell spacing — **2× the runs** of 05b/05c — so the stepping effect can
be read against each spacing baseline (05b for `a`, 05c for `equal_vol`).

| sweep | values |
|-------|--------|
| `--time-stepping` | `D3` (uniform in D³ — the one change vs 05b/05c) |
| `--shell-spacing` | `equal_vol` **and** `a` (the 2× factor) |
| `--nb-shells` | 5, 8, 10, 12, 16, 20, 25, 30, 40 |
| drift | (none), `--drift-on-lightcone` |
| lensing | 3-bin Born (`--nz-shear s3[:3]`) on each density run |

*Fixed:* `--sim-mode pm`, **2560³**, box **`5000³` Mpc/h**, BullFrog (`bf`), `--nb-steps 50`,
`--paint-order cic` with **no** force-window `--deconvolution`, `--scheme ngp`, `--nside 2048`,
`--shells-per-file 1`, `--min-width 5.0` (guardrail only), `--seed 0`, **float64**. **128 GPU**
(32 nodes × 4, `--pdim 128 1`).

> **Halo (Exp 01 rule).** Identical sizing to 05b/05c: local `20³`, even halo `int(20·0.5) = 10`,
> local `1.31e8 ≈ 512³` fits float64, ghost zone `19.5 Mpc/h` clears the end-of-run rms displacement.

> **Caveat (see notebook [08-Advanced-PM](../../2-advanced-usage/08-Advanced-PM.ipynb)).** A *raw*
> uniform-D³ schedule from `a₀ = 0.1` takes a very large first step and can starve the early, still-linear
> growth at small step budgets. At the 50-step production budget here this is not an issue, but the schedule
> should be validated per configuration — D³ is a late-time refinement, not a free swap.

**Status.** ⚠️ Not yet run — `run.sh` is ready; figures and a `build.py` follow once the cluster runs
land and are published to the
[`ASKabalan/jax-fli-experiments`](https://huggingface.co/datasets/ASKabalan/jax-fli-experiments) dataset
under `05-spacing-n-stepping/05d-3bin-d3/`.

## Run

```bash
MODE=dryrun bash run.sh   # print the resolved commands (submit nothing)
bash run.sh               # submit to SLURM
```

The density sweep writes one directory of per-shell parquet per (spacing, drift, shell-count) to
`results/exp5d/`; once those are pushed to HuggingFace, the `fli-born-rt` step reads them back and
writes the 3-bin convergence maps. A local figure script renders the SVGs afterward (heavy compute on
the cluster, figure-making reproducible locally without a GPU, per the
[experiment conventions](../CLAUDE.md)).
