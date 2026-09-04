# Experiment 05d — Step & stepping convergence at the production geometry

**Goal.** Experiment [04](../04-step-convergence/README.md) fixed the step budget — a modest 20–30 steps suffices — but on the accuracy box: 2048³, 2 Gpc/h, 10 scale-factor shells, judged on the per-shell density `C_ℓ`. The production lightcone this Chapter ships differs in every one of those axes: a **5 Gpc/h box at 2560³**, **20 equal-volume shells**, **drift-on-lightcone**, judged on the **tomographic Born convergence** of the three lowest-`z` Stage-3 source bins. This experiment re-asks the step question there: sweep `--nb-steps ∈ {20, 30, 40, 50}` with BullFrog in **both** time variables — `bfd` (growth factor `D`, the production choice) and `bfa` (scale factor `a`) — at everything else fixed, and measure how the 3-bin Born κ `C_ℓ` approaches the 50-step anchor. The 50-step `bfd` point is **not re-run**: it is exactly [05c](../05c-spacing-n-stepping-equal-vol/README.md)'s `exp5c_drift_20`, whose Gauss–Legendre Born spectra are already published — the shared anchor of this experiment and of the mesh ladder in [05e](../05e-spacing-n-stepping-mesh/README.md).

| sweep | values |
|-------|--------|
| `--nb-steps` | 20, 30, 40, 50 |
| `--time-stepping` | `D` (`bfd`), `a` (`bfa`) |

*Fixed across the sweep:* `--sim-mode pm`, **2560³**, box **`5000³` Mpc/h**, BullFrog, **equal-volume** shells (`--shell-spacing equal_vol`, `--min-width 60.0`), **20 shells**, `--drift-on-lightcone`, `--paint-order cic` (no force-window `--deconvolution`), `--scheme ngp`, `--nside 2048`, `--shells-per-file 1`, `--halo-multiplier 0.5`, `--seed 0`, **float64**. **128 GPU** (32 nodes × 4, `--pdim 128 1`; ghost zone `0.5·5000/128 = 19.5` Mpc/h — the 05c anchor sizing, local `20·2560·2560 ≈ 512³` cells fits float64). Born: 3-bin `--nz-shear s3[:3]`, `--quadrature gauss_legendre` (05c showed the legacy midpoint weight breaks on equal-volume's fat inner shell), `--normalization global`. Runs are published to the [`ASKabalan/jax-fli-experiments`](https://huggingface.co/datasets/ASKabalan/jax-fli-experiments) dataset under `05-spacing-n-stepping/05d-steps/` (per-shell density maps, plus the 3-bin Born convergence maps and their spectra).

> **Why the ladder starts at 20.** `--nb-steps` must reach `--nb-shells` (the Exp 04 finding): the integrator visits each shell snapshot with at least one clipped step, so below `nb-shells` steps the runs fail to integrate. The originally requested 5, 6 and 10-step runs are therefore dropped at 20 shells.

> **Wall-time.** Every run is bounded by a **40-minute** SLURM limit — generous by orders of magnitude: the measured 05c perf (the same 2560³/50-step configuration, `--perf --iterations 3`) records ≈ 49 s per simulation iteration plus ≈ 1 min of JIT on 128 GPUs, so even the 50-step runs sit minutes under the cap.

## Method

The same comparison logic as Exp 04, moved to the production geometry and re-pointed at lensing: every run shares the seed, mesh, box and cosmology with the 05c anchor, so the IC white noise is **identical across step counts** — the per-multipole ratio of each run's Born `C_ℓ` to the 50-step anchor's is phase-matched and free of cosmic variance at fixed `ℓ`. Convergence is the step count at which those ratios settle; the `bfa`/`bfd` pair re-tests Exp 04's null result (the two BullFrog steppings were indistinguishable at 2048³/10-shell) at 2560³/20 equal-volume shells, where the step *placement* relative to the shell targets differs between the two time variables. Because [05c](../05c-spacing-n-stepping-equal-vol/README.md) already established that the drift nulls the frozen-epoch error of the fat inner shell, the sweep runs the drift arm only.

The endpoint of every comparison is the Gauss–Legendre Born convergence `C_ℓ` per source bin (bandpower-binned), the same quantity the Chapter's CosmoGrid comparison rests on — so a step-budget saving established here transfers to the production analysis directly.

## Results

⚠️ *Not yet run.* The sweep is submitted via `run.sh`; figures land here once the density and Born data are on HuggingFace.

## How to run

```bash
MODE=dryrun bash run.sh    # print the resolved commands (submit nothing)
bash run.sh                # submit the density sweep to SLURM
SIM_MODE=BORN bash run.sh  # after pushing the density parquet to HF: the 3-bin Born pass
```

The density sweep writes one directory of per-shell parquet (`shell_NNNN.parquet`) per (stepping, step count) to `results/exp5d/density/`; once pushed to HuggingFace, `SIM_MODE=BORN` reads the published shells back and writes the 3-bin convergence maps under `results/exp5d/kappa_gl/` (`fli-born-rt --quadrature gauss_legendre --perf --iterations 3`). The κ spectra parquet are then derived with `tools/make_spectra.py` and published under `05-spacing-n-stepping/05d-steps/kappa_spectra/`.
