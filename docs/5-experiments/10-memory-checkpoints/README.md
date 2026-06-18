# Experiment 10 — Adjoint performance: memory & wall-time ⚠️ WIP

## Goal

Experiment 09 establishes that the lightcone gradient is **correct**. This experiment measures what it
**costs**. The cost of the initial-condition gradient depends on two *independent* axes — the number of
integration **steps taken** and the number of lightcone **shells saved** — and on a third knob, the
checkpoint count. Mapping peak per-device memory and wall-time across all three tells you which adjoint
to pick and how large a differentiable simulation fits on a single GPU.

The two adjoints trade memory against compute differently:

- **`reverse`** — a reversible backsolve. It reconstructs the trajectory by inverting each step, so it is
  **O(1) in the number of steps** (it stores no step trajectory). It is **not** O(1) in the number of
  *shells*: it accumulates one painting VJP per saved shell, so its working set grows with `--nb-shells`.
- **`checkpointed`** — an equinox checkpointed scan. It stores (a fraction of) the step trajectory, so
  its memory **grows with the step count**. Its `checkpoints=N` argument checkpoints the **outer
  lightcone-shell scan** — *not* the inner step loop ([`pm/integrate.py`](../../../src/jax_fli/pm/integrate.py),
  the `eqxi.scan` at the end of `_fwd_loop`) — so `N` trades shell-scan storage for recompute and is a
  no-op on a single snapshot.

Correctness is settled in [Experiment 09](../09-gradient-validation/README.md); here everything is
performance. This runs on **HPC** (exact, uncontended timing); Experiment 09 is the laptop-scale
correctness check.

## Method

`fli-simulate --perf` compiles the IC-gradient (the forward model wrapped in `jax.grad`, exactly the
`--grad` path), runs a warmup + `--iterations` timed iterations, and writes one `perf_pm.csv` row with
the wall-time statistics **and** XLA's compiled memory analysis — generated-code, argument, output and
**temp** (scratch) bytes. `temp_size` is the peak working buffer and is the memory metric reported here.

Three sweeps isolate the three axes (all at **256³, float64, 1 GPU** — these are the trade-off *shapes*;
multi-GPU strong/weak scaling is [Experiment 11](../11-scaling/README.md)):

| sweep | varies | output | fixed | what it isolates |
|------|--------|--------|-------|------------------|
| **A — vs steps** | `--nb-steps` ∈ {25, 50, 100, 200, 400} | single volumetric snapshot (`--density`) | — | `reverse` flat (O(1) in steps) vs `checkpointed` growing with steps |
| **B — vs #shells** | `--nb-shells` ∈ {1, 2, 4, 8, 16, 32} | spherical lightcone (`--nside 512`) | 50 steps | both grow with #shells — `reverse` is **not** O(1) in shells |
| **C — vs #checkpoints** | `--grad checkpointed_{2,4,8,16,32}` (+ `reverse`, default `checkpoint`) | spherical lightcone | 50 steps, 32 shells | the checkpoint count's storage-vs-recompute trade |

Each sweep runs both adjoints where meaningful. Every run writes to its **own output subdirectory**: the
perf CSV row is keyed only by `function=pm{steps}` + precision + mesh — the `--grad` mode is not a column
and `--nb-shells` lands only in the `.md` sidecar — so runs that differ only in adjoint or shell count
would otherwise overwrite each other's `perf_pm.csv`. The figure script keys the swept variable off the
subdirectory path.

## Results

⚠️ **Not yet run on HPC** — figures are added once the sweeps produce their `perf_pm.csv`. The expected
shapes (the mechanism is verified at small scale; this experiment measures the production-scale
magnitudes):

- **A — vs steps.** `reverse` peak memory is **flat** in the step count (it stores no trajectory); its
  wall-time grows roughly linearly (it evaluates the forces twice per step on the backward pass).
  `checkpointed` peak memory **rises** with steps. So on a single snapshot `reverse` is the memory winner,
  increasingly so as the chain lengthens.
- **B — vs #shells.** Both adjoints' memory **rises** with the number of saved shells — `reverse`
  accumulates a painting VJP per shell, so its O(1)-in-steps advantage does **not** extend to shells. This
  is the counterpart to A and the reason the checkpoint knob (sweep C) exists.
- **C — vs #checkpoints.** The `checkpoints=N` knob trades two competing terms on the shell scan: stored
  shell carries (grow with `N`) against recompute scratch (grows as `N` shrinks). At 256³ a single
  particle-state carry is ≈ 0.8 GB, so storage dominates and fewer checkpoints should mean **less peak
  memory at the cost of more wall-time**. Crucially the gradient is **bit-identical at any `N`** (proven in
  Experiment 09) — this is a pure memory↔compute trade, never an accuracy one.

## How to run

The sweeps run on the cluster via the SLURM launcher; figures are rendered locally afterwards.

```bash
MODE=dryrun bash run.sh    # print the 29 resolved fli-launcher / fli-simulate commands, submit nothing
bash run.sh                # submit to SLURM (default MODE=sbatch)
MODE=local  bash run.sh     # run locally via mpirun (use a tiny mesh first)
```

Each run writes a `perf_pm.csv` (wall-time + memory) under its own `results/exp10/<sweep>/<label>/`
directory. Once the runs complete, a local Python script loads those CSVs (keyed by subdirectory path)
and renders the committed SVG figures — heavy compute on the cluster, figure-making reproducible locally
without a GPU, per the [experiment conventions](../CLAUDE.md).
