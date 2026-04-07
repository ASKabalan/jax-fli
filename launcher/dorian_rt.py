"""dorian_rt subcommand — wraps fli-dorian-rt."""
from __future__ import annotations

import os

from .parser import (
    add_lensing_args,
    add_slurm_args,
    dispatch,
)


def add_subparser(sub):
    p = sub.add_parser(
        "dorian-rt",
        help="Submit a fli-dorian-rt ray-tracing lensing job (CPU/MPI)",
    )
    add_slurm_args(p)
    add_lensing_args(p)

    g = p.add_argument_group("dorian-rt")
    g.add_argument("--input-dir", default="results/cosmology_runs")
    g.add_argument("--output-dir", default="results/lensing/multi_shell_raytrace")
    g.add_argument(
        "--rt-interp",
        choices=["bilinear", "ngp", "nufft"],
        default="bilinear",
    )
    g.add_argument("--no-parallel-transport", action="store_true")

    p.set_defaults(
        constraint="cpu",
        cpus_per_node=24,
        tasks_per_node=4,
        nodes=1,
        qos="qos_cpu",
        time_limit="01:00:00",
        func=run,
    )


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    job_name = "fli_dorian_rt"

    print(f"Submitting fli-dorian-rt job for {args.input_dir}/*.parquet")

    fli_cmd = [
        "fli-dorian-rt",
        "--input",
        f"{args.input_dir}/*.parquet",
        "--output",
        args.output_dir,
        "--nz-shear",
        *[str(v) for v in args.nz_shear],
        "--min-z",
        str(args.min_z),
        "--max-z",
        str(args.max_z),
        "--n-integrate",
        str(args.n_integrate),
        "--rt-interp",
        args.rt_interp,
    ]
    if args.no_parallel_transport:
        fli_cmd.append("--no-parallel-transport")

    dispatch(args, job_name, "FLI_DORIAN_RT", fli_cmd, use_gpu=False, always_mpirun=True)
