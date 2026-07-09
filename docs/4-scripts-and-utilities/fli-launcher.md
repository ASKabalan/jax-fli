# fli-launcher

A thin **launcher** for the other `fli-*` scripts (or any command). You give it placement options;
everything after a literal `--` is the command it runs — so it launches any entry point at
single-GPU, multi-GPU, or multi-node scale from one interface. It does **not** sweep grids or
generate job scripts: it forwards *one* command, appending the JAX world-shape flags the entry
scripts need (`--nodes --gpus-per-node --pdim`), and can print it (`dryrun`), run it under
`mpirun` (`local`), or submit it to SLURM (`sbatch`).

## Usage

```bash
# dryrun (default): resolve + print the SLURM config and command, submit nothing
fli-launcher --nodes 2 --gpus-per-node 4 --pdim 8 1 -- fli-simulate --sim-mode pm --nside 512

# local: run under mpirun, no SLURM
fli-launcher --mode local --gpus-per-node 4 --pdim 4 1 -- fli-simulate --sim-mode pm

# sbatch: submit to SLURM (needs a wrapper script — see $SLURM_SCRIPT below)
fli-launcher --mode sbatch --nodes 2 --gpus-per-node 4 --pdim 8 1 \
    --account myacct --constraint h100 --qos qos_gpu_h100-t3 --time-limit 01:00:00 \
    --slurm-script slurm_script.sh -- fli-simulate --sim-mode pm --nside 512
```

Everything after the first `--` is the payload command. The launcher checks
`gpus-per-node × nodes == PX × PY`, appends `--nodes N --gpus-per-node G --pdim PX PY` to the
payload, and writes a reproducible `*.args.log` next to the run's `--output`.

## Options (group "SLURM / cluster")

| Flag | Default | Meaning |
|------|---------|---------|
| `--mode {local,sbatch,dryrun}` | `dryrun` | dryrun = print only; local = `mpirun`; sbatch = submit |
| `--nodes` | `1` | node count |
| `--gpus-per-node` | `4` | GPUs per node |
| `--cpus-per-node` | `16` | CPUs per node (÷ tasks-per-node → `--cpus-per-task`) |
| `--tasks-per-node` | = `--gpus-per-node` | MPI tasks per node |
| `--pdim PX PY` | `1 1` | JAX process mesh; must satisfy `PX*PY == nodes*gpus-per-node`; forwarded to the payload |
| `--constraint` | `h100` | node constraint; `cpu` = CPU-only (skips `-C`/`--gres`, counts tasks not GPUs locally) |
| `--account` | `XXX` | SLURM account |
| `--qos` | `qos_gpu_h100-t3` | SLURM QoS |
| `--time-limit` | `00:30:00` | wall time `HH:MM:SS` |
| `--slurm-script` | — | wrapper script (`$SLURM_SCRIPT`); **required for `--mode sbatch`** |
| `--output-logs` | `SLURM_LOGS` | directory for the `%x_%j.out` / `.err` logs |

There are **no subcommands** and no `--grid`. Payload values may contain `%placeholder%` tokens the
launcher substitutes from the payload + constraint before submitting —
`%constraint% %mesh_size% %box_size% %nb_steps% %omega_c% %sigma8% %seed% %lpt_order%` (multi-value
flags join with `x`, e.g. `--mesh-size 256 256 256` → `256x256x256`), e.g.
`--output results/%constraint%_M%mesh_size%_S%seed%.parquet`.

## `$SLURM_SCRIPT` — the sbatch wrapper

`--slurm-script` (equivalently the `SLURM_SCRIPT` env var the experiment `run.sh` files export) is a
**user-supplied** wrapper that `fli-launcher` hands to `sbatch`. The repo does not ship it as an
installed asset — a ready template is committed alongside this page. All SLURM directives are passed
as `sbatch` CLI flags (they override any `#SBATCH` lines in the wrapper), then:

```
sbatch [--account … -C … --gres … --time … --nodes … …] <wrapper> TRACES_DIR <resolved cmd> --nodes N --gpus-per-node G --pdim PX PY
```

So inside the wrapper `$1` is the run / traces name and `"${@:2}"` is the full command — a minimal
wrapper is just `srun "${@:2}"`. The committed template [`slurm_script.sh`](slurm_script.sh) does
this and more: it loads the right modules per `--constraint` (A100/H100/CPU/V100), activates the uv
env, and exposes `slaunch` (plain `srun`) / `plaunch` (`nsys profile`) helpers. Copy it and point the
launcher at it:

```bash
export SLURM_SCRIPT=path/to/slurm_script.sh
fli-launcher --mode sbatch … --slurm-script "$SLURM_SCRIPT" -- fli-simulate …
```

The experiment `run.sh` files (via [`_launch_common.sh`](../5-experiments/_launch_common.sh)) drive
exactly this pattern with `MODE=dryrun|sbatch|local`; the multi-host physics is in
[multi-host PM](../2-advanced-usage/11-multi-host-pm.md).
