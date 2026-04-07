"""extract subcommand — wraps fli-extract."""
from __future__ import annotations

import os
import sys

from .parser import (
    add_slurm_args,
    dispatch,
)


def add_subparser(sub):
    p = sub.add_parser(
        "extract",
        help="Submit a fli-extract job to compute chain statistics",
    )
    add_slurm_args(p)

    g = p.add_argument_group("extract")
    g.add_argument(
        "--input-dir",
        default="test_fli_samples",
        help="Local root dir (chain_N/samples layout); mutually exclusive with --repo-id",
    )
    g.add_argument("--repo-id", default=None, help="HuggingFace Hub repo ID, e.g. 'ASKabalan/jax-fli-experiments'")
    g.add_argument(
        "--config", nargs="*", default=None, help="Config names, one per chain (required when --repo-id is used)"
    )
    g.add_argument("--truth-parquet", default="test_fli_samples/chain_0/samples/samples_0.parquet")
    g.add_argument("--output-file", default="results/extracts/extract.parquet")
    g.add_argument("--set-name", default="my_extract")
    g.add_argument("--cosmo-keys", nargs="+", default=["Omega_c", "sigma8"])
    g.add_argument("--field-statistic", action="store_true", default=True)
    g.add_argument("--power-statistic", action="store_true", default=True)
    g.add_argument("--ddof", type=int, default=0)
    g.add_argument("--enable-x64", action="store_true")

    p.set_defaults(func=run)


def run(args):
    if args.repo_id is not None and not args.config:
        print("Error: --config must be set when --repo-id is used.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    px, py = args.pdim
    job_name = f"{args.constraint}_extract_{args.set_name}"

    print(f"Submitting {job_name}")

    fli_cmd = ["fli-extract"]
    if args.repo_id is not None:
        fli_cmd += ["--repo-id", args.repo_id]
        if args.config:
            fli_cmd += ["--config", *args.config]
    else:
        fli_cmd += ["--path", args.input_dir]

    fli_cmd += ["--set-name", args.set_name, "--output", args.output_file]
    fli_cmd += ["--cosmo-keys", *args.cosmo_keys]
    if args.truth_parquet:
        fli_cmd += ["--truth", args.truth_parquet]
    if args.field_statistic:
        fli_cmd.append("--field-statistic")
    if args.power_statistic:
        fli_cmd.append("--power-statistic")
    fli_cmd += ["--ddof", str(args.ddof)]
    fli_cmd += ["--pdim", str(px), str(py), "--nodes", str(args.nodes)]
    if args.enable_x64:
        fli_cmd.append("--enable-x64")

    dispatch(args, job_name, "FLI_EXTRACT", fli_cmd)
