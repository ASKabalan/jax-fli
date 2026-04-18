"""Generic, script-agnostic launcher for jax-fli entry points.

Usage
-----

    fli-launcher [SLURM OPTS] -- <cmd> [cmd args...]

``<cmd>`` is anything the user wants to run (typically ``fli-simulate``,
``fli-infer``, ...). The launcher never inspects it beyond the first token,
which is used to derive the SLURM job name.

Modes
-----
* ``--mode dryrun`` — print the resolved SLURM configuration and forwarded
  command, then exit without running anything.
* ``--mode local`` — run ``mpirun -n <nodes*gpus_per_node> <cmd>`` (or just
  the command if only one task).
* ``--mode sbatch`` — submit ``sbatch [flags] <slurm_script> <cmd>`` so the
  wrapper script receives the full payload as its positional arguments.

The launcher validates ``gpus_per_node * nodes == prod(pdim)`` and appends
``--nodes <n> --pdim <p0> <p1>`` to the forwarded command so that entry
scripts still see the world shape they need for JAX sharding.

Template substitution
---------------------

Values in the forwarded command that contain ``%placeholder%`` tokens are
resolved before dispatch from the payload itself + the SLURM ``--constraint``.
Supported tokens:

    %constraint%  %mesh_size%  %box_size%  %nb_steps%
    %omega_c%     %sigma8%     %seed%      %lpt_order%

Typical use:

    --output results/%constraint%_M%mesh_size%_S%seed%.parquet
    --name   cosmo_c%omega_c%_S8%sigma8%

After substitution, the launcher writes ``<output_dir>/args.log`` with the
resolved command so each run is self-documenting. ``<output_dir>`` is the
parent of ``--output`` (when it looks like a file) or the value of
``--output`` / ``--path`` (when it looks like a directory).
"""
from __future__ import annotations

import argparse
import math
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from jax_fli.scripts.parser import add_slurm_args


# Flags whose first value should feed the template lookup table.
_TEMPLATE_KEYS = {
    "--mesh-size": "mesh_size",
    "--box-size": "box_size",
    "--nb-steps": "nb_steps",
    "--Omega-c": "omega_c",
    "--sigma8": "sigma8",
    "--seed": "seed",
    "--lpt-order": "lpt_order",
}

# Flags that point at an output location (used to pick the log directory).
# Ordered by preference: the first match wins.
_OUTPUT_KEYS = ("--output", "--path")


def _split_on_separator(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split argv on the first bare ``--``; everything after is the payload."""
    if "--" not in argv:
        return argv, []
    idx = argv.index("--")
    return argv[:idx], argv[idx + 1 :]


def _resolve_slurm_params(args: argparse.Namespace) -> tuple[int, int, int]:
    """Return (tasks_per_node, cpus_per_task, total_gpus)."""
    tpn = args.tasks_per_node if args.tasks_per_node is not None else args.gpus_per_node
    cpt = args.cpus_per_node // tpn
    if cpt < 1:
        sys.exit(f"Error: --cpus-per-node={args.cpus_per_node} is too low for tasks_per_node={tpn}")
    total_gpus = args.gpus_per_node * args.nodes
    return tpn, cpt, total_gpus


def _validate_topology(args: argparse.Namespace) -> None:
    """Check gpus_per_node * nodes == prod(pdim)."""
    expected = args.gpus_per_node * args.nodes
    actual = math.prod(args.pdim)
    if expected != actual:
        sys.exit(
            f"Error: --gpus-per-node ({args.gpus_per_node}) * --nodes ({args.nodes}) = {expected} "
            f"does not match prod(--pdim {args.pdim}) = {actual}"
        )


def _job_name(cmd_name: str, payload: list[str]) -> str:
    """Derive a SLURM job name: use ``--name`` if present, else FLI_<STEM>."""
    if "--name" in payload:
        idx = payload.index("--name")
        if idx + 1 < len(payload):
            return payload[idx + 1]
    stem = Path(cmd_name).name.replace("fli-", "").replace("-", "_").upper()
    return f"FLI_{stem}"


# ── Template substitution ───────────────────────────────────────────────────

def _collect_template_vars(payload: list[str], args: argparse.Namespace) -> dict[str, str]:
    """Build the ``%key% → value`` table from the payload + launcher args.

    For multi-value flags (--mesh-size, --box-size), values are joined with ``x``
    so ``--mesh-size 64 64 64`` → ``64x64x64``.
    """
    table: dict[str, str] = {"constraint": str(args.constraint)}
    i = 0
    while i < len(payload):
        tok = payload[i]
        if tok in _TEMPLATE_KEYS:
            key = _TEMPLATE_KEYS[tok]
            values: list[str] = []
            j = i + 1
            while j < len(payload) and not payload[j].startswith("--"):
                values.append(payload[j])
                j += 1
            table[key] = "x".join(values) if len(values) > 1 else (values[0] if values else "")
            i = j
        else:
            i += 1
    return table


def _substitute(value: str, table: dict[str, str]) -> str:
    if "%" not in value:
        return value
    out = value
    for key, val in table.items():
        out = out.replace(f"%{key}%", val)
    return out


def _apply_templates(payload: list[str], table: dict[str, str]) -> list[str]:
    """Return a copy of ``payload`` with every ``%key%`` token resolved."""
    return [_substitute(tok, table) for tok in payload]


# ── Log / output-dir resolution ─────────────────────────────────────────────

def _resolve_output_dir(payload: list[str]) -> Path | None:
    """Find the directory to drop ``args.log`` into.

    Looks for ``--output`` first, then ``--path``. If the value has a common
    archive-ish suffix (``.parquet``, ``.h5``, ``.nc``) we use its parent.
    Returns ``None`` if no output-like flag is present (e.g. ``fli-spectra``).
    """
    for key in _OUTPUT_KEYS:
        if key not in payload:
            continue
        idx = payload.index(key)
        if idx + 1 >= len(payload):
            continue
        raw = payload[idx + 1]
        p = Path(raw)
        file_suffixes = {".parquet", ".h5", ".hdf5", ".nc", ".json", ".zarr"}
        return p.parent if p.suffix.lower() in file_suffixes else p
    return None


def _write_log(out_dir: Path, cmd: list[str], launcher_args: argparse.Namespace) -> None:
    """Write ``<out_dir>/args.log`` with a shell-quoted, reproducible command."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "args.log"
    with log_path.open("a") as f:
        f.write(f"# {datetime.now().isoformat(timespec='seconds')} mode={launcher_args.mode}\n")
        f.write(shlex.join(["fli-launcher", *sys.argv[1:sys.argv.index('--')], "--"] + cmd) + "\n")


# ── Dispatch ────────────────────────────────────────────────────────────────

def _print_dryrun(args: argparse.Namespace, cmd: list[str]) -> None:
    tpn, cpt, total_gpus = _resolve_slurm_params(args)
    bar = "=" * 60
    print(bar)
    print(f"  fli-launcher dryrun — {_job_name(cmd[0], cmd)}")
    print(bar)
    for label, value in [
        ("mode", args.mode),
        ("account", args.account),
        ("constraint", args.constraint),
        ("nodes", args.nodes),
        ("gpus-per-node", args.gpus_per_node),
        ("cpus-per-node", args.cpus_per_node),
        ("tasks-per-node", tpn),
        ("cpus-per-task", cpt),
        ("total gpus", total_gpus),
        ("qos", args.qos),
        ("time-limit", args.time_limit),
        ("pdim", args.pdim),
        ("slurm-script", args.slurm_script),
        ("output-logs", args.output_logs),
    ]:
        print(f"  {label:<16} {value}")
    print("-" * 60)
    print("  " + " ".join(cmd))
    print(bar)


def _run_local(args: argparse.Namespace, cmd: list[str]) -> None:
    tpn, _, total_gpus = _resolve_slurm_params(args)
    use_gpu = args.constraint != "cpu"
    n = total_gpus if use_gpu else tpn * args.nodes
    prefix = ["mpirun", "-n", str(n), "--oversubscribe"] if n > 1 else []
    subprocess.run(prefix + cmd, check=True)


def _run_sbatch(args: argparse.Namespace, cmd: list[str]) -> None:
    if not args.slurm_script:
        sys.exit("Error: --slurm-script is required when --mode=sbatch")

    tpn, cpt, _ = _resolve_slurm_params(args)
    use_gpu = args.constraint != "cpu"
    job_name = _job_name(cmd[0], cmd)

    sbatch = ["sbatch", f"--account={args.account}"]
    if use_gpu:
        sbatch += ["-C", args.constraint, f"--gres=gpu:{args.gpus_per_node}"]
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
    ]
    subprocess.run(sbatch + cmd, check=True)


def main() -> None:
    argv = sys.argv[1:]
    launcher_argv, payload = _split_on_separator(argv)
    if not payload:
        sys.exit(
            "Error: missing command payload. Usage:\n"
            "  fli-launcher [SLURM OPTS] -- <cmd> [cmd args...]"
        )

    parser = argparse.ArgumentParser(
        prog="fli-launcher",
        description="Submit any command as a SLURM job, run it locally under mpirun, or print a dryrun.",
    )
    add_slurm_args(parser)
    args = parser.parse_args(launcher_argv)

    _validate_topology(args)

    # Resolve %placeholders% in the payload BEFORE appending world-shape flags,
    # so templates only see user-provided values (not the injected --nodes/--pdim).
    table = _collect_template_vars(payload, args)
    resolved_payload = _apply_templates(payload, table)

    cmd = list(resolved_payload) + [
        "--nodes", str(args.nodes),
        "--pdim", str(args.pdim[0]), str(args.pdim[1]),
    ]

    # Write a per-run log in the resolved output directory (when there is one).
    out_dir = _resolve_output_dir(resolved_payload)
    if out_dir is not None and args.mode != "dryrun":
        _write_log(out_dir, cmd, args)

    if args.mode == "dryrun":
        _print_dryrun(args, cmd)
        return
    if args.mode == "local":
        _run_local(args, cmd)
        return
    if args.mode == "sbatch":
        _run_sbatch(args, cmd)
        return
    sys.exit(f"Error: unknown --mode {args.mode!r}")


if __name__ == "__main__":
    main()
