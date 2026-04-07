"""samples subcommand — wraps fli-samples."""
from __future__ import annotations

import os

from .parser import (
    add_common_sim_args,
    add_lensing_args,
    add_lightcone_args,
    add_slurm_args,
    dispatch,
)


def add_subparser(sub):
    p = sub.add_parser(
        "samples",
        help="Submit fli-samples jobs across chains × batches",
    )
    add_slurm_args(p)
    add_common_sim_args(p)
    add_lensing_args(p)
    add_lightcone_args(p)

    g = p.add_argument_group("samples")
    g.add_argument("--output-dir", default="test_fli_samples")
    g.add_argument("--model", choices=["full", "mock"], default="mock")
    g.add_argument("--mesh-size", nargs=3, type=int, default=[64, 64, 64], metavar=("MX", "MY", "MZ"))
    g.add_argument("--box-size", nargs=3, type=float, default=[250.0, 250.0, 250.0], metavar=("BX", "BY", "BZ"))
    g.add_argument("--nside", type=int, default=64)
    g.add_argument("--nb-shells", type=int, default=8, help="Number of lightcone shells (default: 8)")
    g.add_argument("--enable-x64", action="store_true", help="Enable JAX 64-bit precision")
    g.add_argument("--num-samples", type=int, default=10)
    g.add_argument("--chains", nargs="+", type=int, default=[0, 1, 2, 3], help="Chain IDs to run")
    g.add_argument("--batches", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5], help="Batch IDs to run")

    p.set_defaults(t0=0.01, nb_steps=100, func=run)


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    px, py = args.pdim

    for chain in args.chains:
        for batch in args.batches:
            job_name = f"{args.constraint}_samples_chain{chain}_batch{batch}"
            print(f"Submitting {job_name}")

            fli_cmd = [
                "fli-samples",
                "--model",
                args.model,
                "--mesh-size",
                *[str(m) for m in args.mesh_size],
                "--box-size",
                *[str(b) for b in args.box_size],
                "--pdim",
                str(px),
                str(py),
                "--nodes",
                str(args.nodes),
                "--nside",
                str(args.nside),
                "--lpt-order",
                str(args.lpt_order),
                "--t0",
                str(args.t0),
                "--t1",
                str(args.t1),
                "--nb-steps",
                str(args.nb_steps),
                "--nb-shells",
                str(args.nb_shells),
                "--halo-multiplier",
                str(args.halo_multiplier),
                "--observer-position",
                *[str(v) for v in args.observer_position],
                "--nz-shear",
                *[str(v) for v in args.nz_shear],
                "--min-z",
                str(args.min_z),
                "--max-z",
                str(args.max_z),
                "--n-integrate",
                str(args.n_integrate),
                "--interp",
                args.interp,
                "--scheme",
                args.scheme,
            ]
            if args.paint_nside is not None:
                fli_cmd += ["--paint-nside", str(args.paint_nside)]
            if args.kernel_width_arcmin is not None:
                fli_cmd += ["--kernel-width-arcmin", str(args.kernel_width_arcmin)]
            if args.equal_vol:
                fli_cmd.append("--equal-vol")
            if args.dealiased:
                fli_cmd.append("--dealiased")
            if args.exact_growth:
                fli_cmd.append("--exact-growth")
            fli_cmd += ["--gradient-order", str(args.gradient_order)]
            if args.laplace_fd:
                fli_cmd.append("--laplace-fd")
            fli_cmd += [
                "--num-samples",
                str(args.num_samples),
                "--seed",
                str(batch),
                "--path",
                f"{args.output_dir}/chain_{chain}",
                "--batch-id",
                str(batch),
            ]
            if args.enable_x64:
                fli_cmd.append("--enable-x64")

            dispatch(args, job_name, "FLI_SAMPLES", fli_cmd)
