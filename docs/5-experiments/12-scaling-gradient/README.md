# Experiment 12 — Gradient scaling: strong & weak ⚠️ WIP

**Goal.** [Experiment 09](../09-gradient-validation/README.md) establishes that the lightcone IC-gradient is
**correct**; this experiment measures what it **costs at scale**. We strong- and weak-scale the
initial-condition gradient under a **slab `(N, 1)`** decomposition and capture **wall-time *and* per-device
memory** (`fli-simulate --perf`, XLA `memory_analysis`) for the two production adjoints, so you can read off
which adjoint to pick and how large a *differentiable* simulation fits at a given GPU count.

The two adjoints trade memory against compute differently:

- **`reverse`** — a reversible `kdk` (DoubleKickDrift) backsolve: it reconstructs the trajectory by inverting
  each step, so it is **O(1) in the step count** (stores no step trajectory). It is **not** O(1) in *shells*
  — it accumulates one painting VJP per saved shell.
- **`checkpointed_N`** — an equinox checkpointed scan over the **outer lightcone-shell scan**
  ([`pm/integrate.py`](../../../src/jax_fli/pm/integrate.py)): `N` trades stored shell-carries against
  recompute. The gradient is **bit-identical at any `N`** (Exp 09) — a pure memory↔compute trade.

**Status.** ⚠️ Not yet run.

*Fixed:* `--sim-mode pm`, `kdk`, `--nb-steps 30`, `--paint-order tsc --deconvolution`, `--scheme ngp`,
lightcone `--nside 1024 --nb-shells 10`, `--perf --iterations 5`, box `2000³` Mpc/h, **float64**, `--seed 0`.
Variants: `--grad ∈ {reverse, checkpointed_4, checkpointed_8}`. **Slab `(N, 1)`**: `px = #GPUs`, `py = 1`.
The differentiated output (the IC-shaped gradient, a full mesh) is saved as usual, so a rerun skips a rung
whose `--output` already exists. Strong scaling is **1024³ only** (2048³ gradient runs are out of scope here).

### Slab constraints (which rungs run)

Same two limits as [Exp 11](../11-scaling/README.md), plus a gradient memory penalty:

1. **Even halo** — `int((M/px)·0.5)` must be even, so **1024³ tops out at 256 GPUs** (512 → halo 1).
2. **Per-GPU ceiling, halved.** The IC gradient roughly **doubles** the forward working set (reverse holds the
   forward + backward state; checkpointed stores shell carries), so each ladder starts a rung higher than
   Exp 11 — ceiling **≈406³/GPU**. `run.sh` skips and logs below-ceiling rungs; the heaviest `checkpointed`
   runs may still need to start one rung higher.

## (a) Strong scaling — fixed grid, grow GPUs (slab `N × 1`, per `--grad` variant)

| grid | GPUs |
|------|------|
| 1024³ | 16, 32, 64, 128, 256 |

## (b) Weak scaling — fixed 256³/GPU (slab `N × 1`, per `--grad` variant)

`global --mesh-size = (256·px, 256, 256)`; local `256³`, halo `128` — all eight GPU counts run.

| GPUs | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|------|---|---|----|----|----|-----|-----|-----|

**3 grad variants × (5 strong + 8 weak rungs) ≈ 39 runs.**

## Run

```bash
MODE=dryrun bash run.sh    # print the resolved commands + skipped rungs (submit nothing)
bash run.sh                # submit to SLURM
```

Writes `perf_pm.csv` rows + per-run markdown (wall-time + memory, keyed by the unique `--name`) under
`results/exp12/`. A local figure script then loads those and renders the strong/weak-scaling SVGs.
