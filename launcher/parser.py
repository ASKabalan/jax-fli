"""Launcher-specific argument groups and dispatch logic.

Thin wrapper around shared_args — imports all shared groups
and adds the SLURM/cluster layer on top.
"""
from __future__ import annotations

import os
import subprocess
import sys

from .shared_args import (  # noqa: F401 (re-exported for subcommand modules)
    DEFAULT_NAME_TEMPLATE,
    add_common_sim_args,
    add_cosmo_args,
    add_distributed_args,
    add_lensing_args,
    add_lightcone_args,
    add_mesh_args,
    add_output_target_args,
)


def add_slurm_args(p):
    """Cluster/SLURM arguments common to all launcher subcommands."""
    add_distributed_args(p)  # pdim/nodes with single-device defaults
    g = p.add_argument_group("SLURM / cluster")
    g.add_argument(
        "--mode", choices=["local", "sbatch", "dryrun"], default="dryrun", help="Execution mode (default: dryrun)"
    )
    g.add_argument("--account", default="XXX")
    g.add_argument("--constraint", default="h100")
    g.add_argument("--gpus-per-node", type=int, default=4)
    g.add_argument("--cpus-per-node", type=int, default=16)
    g.add_argument("--tasks-per-node", type=int, default=None, help="Defaults to --gpus-per-node when not set")
    g.add_argument("--qos", default="qos_gpu_h100-t3")
    g.add_argument("--time-limit", default="00:30:00")
    g.add_argument("--slurm-script", default=None, help="Path to SLURM job script (required when --mode=sbatch)")
    g.add_argument("--output-logs", default="SLURM_LOGS", help="Directory for SLURM log files")
    # Override distributed defaults for cluster use
    p.set_defaults(pdim=[16, 1], nodes=4)


def add_integration_args(p):
    """Launcher-only: simulation type + multi-mesh flat list."""
    g = p.add_argument_group("integration")
    g.add_argument("--simulation-type", choices=["lpt", "nbody", "lensing"], default="nbody")
    add_mesh_args(p, nargs="+")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _slurm_params(args):
    """Return (tasks_per_node, cpus_per_task, total_gpus)."""
    tpn = args.tasks_per_node if args.tasks_per_node is not None else args.gpus_per_node
    cpt = args.cpus_per_node // tpn
    total_gpus = args.gpus_per_node * args.nodes
    return tpn, cpt, total_gpus


def _print_dryrun(job_name, args, fli_cmd):
    tpn, cpt, _ = _slurm_params(args)
    print("=======================================================")
    print(f"Submitting job {job_name}")
    print("=======================================================")
    print(f"{'ACCOUNT':<16} | {args.account}")
    print(f"{'CONSTRAINT':<16} | {args.constraint}")
    print(f"{'TIME_LIMIT':<16} | {args.time_limit}")
    print(f"{'GPUS_PER_NODE':<16} | {args.gpus_per_node}")
    print(f"{'CPUS_PER_TASK':<16} | {cpt}")
    print(f"{'NODES':<16} | {args.nodes}")
    print(f"{'TASKS_PER_NODE':<16} | {tpn}")
    print(f"{'QOS':<16} | {args.qos}")
    print("*******************************************************")
    print(" ".join(fli_cmd))
    print("*******************************************************")
    print("======= end of job =======")
    print()


def dispatch(args, job_name, tag, fli_cmd, *, use_gpu=True, always_mpirun=False, env=None):
    """Dispatch *fli_cmd* according to args.mode.

    Parameters
    ----------
    use_gpu:
        When False, sbatch omits ``--gres=gpu`` and ``-C constraint``, and
        the MPI task count is ``tasks_per_node * nodes`` rather than
        ``gpus_per_node * nodes``.
    always_mpirun:
        When True, local mode always wraps with ``mpirun`` regardless of
        the GPU/task count (used by dorian-rt which is always MPI).
    env:
        Optional environment dict passed to ``subprocess.run``.
    """
    tpn, cpt, total_gpus = _slurm_params(args)

    if args.mode == "dryrun":
        _print_dryrun(job_name, args, fli_cmd)
        return

    if args.mode == "local":
        n = total_gpus if use_gpu else tpn * args.nodes
        if always_mpirun or n > 1:
            prefix = ["mpirun", "-n", str(n), "--oversubscribe"]
        else:
            prefix = []
        subprocess.run(prefix + fli_cmd, check=True, env=env)
        return

    # ---- sbatch mode ----
    if not args.slurm_script:
        print("Error: --slurm-script is required when --mode=sbatch", file=sys.stderr)
        sys.exit(1)

    constraint = args.constraint
    sbatch = ["sbatch", f"--account={args.account}"]
    if use_gpu and constraint and constraint != "cpu":
        sbatch += ["-C", constraint]
        sbatch += [f"--gres=gpu:{args.gpus_per_node}"]
    sbatch += [
        f"--time={args.time_limit}",
        f"--cpus-per-task={cpt}",
        f"--nodes={args.nodes}",
        f"--tasks-per-node={tpn}",
        f"--qos={args.qos}",
        f"--job-name={job_name}",
        f"--output={args.output_logs}/%x_%j.out",
        f"--error={args.output_logs}/%x_%j.err",
        os.path.expandvars(args.slurm_script),
        tag,
    ]
    subprocess.run(sbatch + fli_cmd, check=True, env=env)
