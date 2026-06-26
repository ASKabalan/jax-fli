# Experiment 05b — Drift on the lightcone, 3-bin tomography ⚠️ WIP

**Goal.** Carry the [Experiment 05](../05-drift-on-lightcone/README.md) result — drifting particles to their
lightcone-crossing epoch sharpens the per-shell density `C_ℓ` for **thick** shells — into **tomographic Born
convergence with three source bins**. Exp 05 ran a 2 Gpc/h box, deep enough only for a single point source at
`z = 0.35`; the radial projection there washes the drift effect out of the convergence. This experiment uses a
**5 Gpc/h box at 2560³**, deep enough to place **three** tomographic source bins, and asks whether the
density-shell improvement survives the deeper, multi-bin lensing projection.

**Method.** Same drift / no-drift comparison as Exp 05, swept over shell count
`--nb-shells ∈ {5, 8, 10, 12, 16, 20, 25, 30, 40}`, BullFrog, **50 steps**, float64, full-sky `--nside 2048`.
The density shells are then Born-integrated against a **3-bin** source distribution (`--nz-shear s3[:3]`, the
three lowest-`z` Stage-3 bins, matched to the 5 Gpc/h depth) to produce per-bin convergence `C_ℓ`. Born is
order-invariant, so the comparison isolates the drift effect on the projected signal.

**Status.** ⚠️ Not yet run — `run.sh` is ready; figures and a `build.py` follow once the cluster runs land
and are published to the [`ASKabalan/jax-fli-experiments`](https://huggingface.co/datasets/ASKabalan/jax-fli-experiments)
dataset. (Open question to confirm at analysis time: the exact `--nz-shear` 3-bin specification and whether the
inscribed full-sky depth of the 5 Gpc/h box reaches all three bins.)

## Grid

*Fixed:* `--sim-mode pm`, **2560³**, BullFrog (`bf`), `--nb-steps 50`, `--paint-order cic` with **no**
force-window `--deconvolution`, `--scheme ngp`, `--nside 2048`, `--shells-per-file 1`, `--shell-spacing a`,
box **`5000³` Mpc/h**, `--seed 0`, **float64**. **128 GPU** (32 nodes × 4, `--pdim 128 1`).

> **Halo (Exp 01 rule).** 2560³ on 128 GPUs (slab) gives local `20³` and an **even** halo `int(20·0.5) = 10`
> (an odd halo crashes jaxpm `slice_unpad`); local `20·2560·2560 = 1.31e8 ≈ 512³` fits float64. The physical
> ghost zone `0.5·5000/128 = 19.5 Mpc/h` clears the end-of-run rms particle displacement. Drift-on-lightcone
> only repaints existing particles, so the displacement scale is unchanged.

| sweep | values |
|------|------|
| `--nb-shells` | 5, 8, 10, 12, 16, 20, 25, 30, 40 |
| drift | (none), `--drift-on-lightcone` |
| lensing | 3-bin Born (`--nz-shear s3[:3]`) on each density run |

## Run

```bash
MODE=dryrun bash run.sh
bash run.sh
```

The density sweep writes one directory of per-shell parquet (`shell_NNNN.parquet`) per (drift, shell-count) to
`results/exp5b/`; once those are pushed to HuggingFace, the `fli-born-rt` step reads them back and writes the
3-bin convergence maps. A local figure script renders the SVGs afterward (heavy compute on the cluster,
figure-making reproducible locally without a GPU, per the [experiment conventions](../CLAUDE.md)).
