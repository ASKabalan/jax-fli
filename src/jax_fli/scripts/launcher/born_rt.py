"""born_rt subcommand — wraps fli-born-rt."""
from __future__ import annotations

import os

from .parser import (
    add_lensing_args,
    add_slurm_args,
    dispatch,
)


def add_subparser(sub):
    p = sub.add_parser(
        "born-rt",
        help="Submit a fli-born-rt Born lensing post-processing job",
    )
    add_slurm_args(p)
    add_lensing_args(p)

    g = p.add_argument_group("born-rt")
    g.add_argument("--input-dir", default="results/cosmology_runs")
    g.add_argument("--output-dir", default="results/lensing/multi_shell")
    g.add_argument("--enable-x64", action="store_true")

    # born_rt_script.bash: CONSTRAINT="" (empty), PDIMS="4 1", NODES=1
    p.set_defaults(
        constraint="",
        pdim=[4, 1],
        nodes=1,
        func=run,
    )


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    px, py = args.pdim
    job_name = "fli_born_rt"

    print(f"Submitting fli-born-rt job for {args.input_dir}/*.parquet")

    fli_cmd = [
        "fli-born-rt",
        "--pdim",
        str(px),
        str(py),
        "--nodes",
        str(args.nodes),
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
    ]
    if args.enable_x64:
        fli_cmd.append("--enable-x64")

    # use_gpu=False when constraint is "cpu" or empty
    use_gpu = bool(args.constraint) and args.constraint != "cpu"
    dispatch(args, job_name, "FLI_BORN_RT", fli_cmd, use_gpu=use_gpu)
