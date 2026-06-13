# fli-launcher

A SLURM **dispatcher** that submits grids of the other scripts — sweeping cosmologies, seeds and
resolutions — so a whole campaign is one command instead of hundreds of hand-written `sbatch`
files. It wraps the entry points in `launcher/` and emits/*submits* the job scripts.

## Usage

```bash
fli-launcher simulate --grid cosmologies.yaml --nodes 4 --gpus-per-node 4
```

## Subcommands

| Subcommand | Dispatches |
|------------|------------|
| `simulate` | [`fli-simulate`](fli-simulate.md) over a parameter grid |
| `grid` | mesh/box grid set-up and scaling tests |
| `samples` | [`fli-samples`](fli-samples.md) batches |
| `infer` | [`fli-infer`](fli-infer.md) chains |
| `2pcf` / `born_rt` / `extract` / `dorian_rt` | the matching scripts |

Each subcommand forwards the relevant argument groups to the underlying script, plus SLURM
placement options (nodes, GPUs per node, time, partition). Run `fli-launcher <subcommand> --help`
for the per-command options. The multi-host launch pattern it generates is described in
[multi-host PM](../2-advanced-usage/11-multi-host-pm.md).
