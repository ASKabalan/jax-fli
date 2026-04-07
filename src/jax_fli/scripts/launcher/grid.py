"""grid subcommand — wraps fli-grid."""
from __future__ import annotations

import os

from .parser import (
    add_common_sim_args,
    add_cosmo_args,
    add_integration_args,
    add_lensing_args,
    add_lightcone_args,
    add_output_target_args,
    add_slurm_args,
    dispatch,
)

# Defaults matching simulation_grid.bash
_DEFAULT_TS_NEAR = [
    "0.3938",
    "0.4052",
    "0.4165",
    "0.4276",
    "0.4387",
    "0.4497",
    "0.4606",
    "0.4714",
    "0.4822",
    "0.4929",
]
_DEFAULT_TS_FAR = [
    "0.4052",
    "0.4165",
    "0.4276",
    "0.4387",
    "0.4497",
    "0.4606",
    "0.4714",
    "0.4822",
    "0.4929",
    "0.5036",
]


def add_subparser(sub):
    p = sub.add_parser(
        "grid",
        help="Submit a single fli-grid job (full parameter-grid exploration in one process)",
    )
    add_slurm_args(p)
    add_common_sim_args(p)
    add_cosmo_args(p, sweep=True)
    add_lensing_args(p)
    add_lightcone_args(p)
    add_output_target_args(p)
    add_integration_args(p)

    g = p.add_argument_group("grid")
    g.add_argument(
        "--nb-shells", type=int, default=None, help="Number of lightcone shells (default: None = fli-grid decides)"
    )
    g.add_argument("--enable-x64", action="store_true", help="Enable JAX 64-bit precision")
    g.add_argument("--output-dir", default="results/grid_runs")

    p.set_defaults(
        time_limit="24:00:00",  # grid runs ALL combos — set generously
        drift_on_lightcone=True,
        ts_near=_DEFAULT_TS_NEAR,
        ts_far=_DEFAULT_TS_FAR,
        box_size=[500.0, 500.0, 500.0, 1000.0, 1000.0, 1000.0],
        Omega_c=["0.2"],
        sigma8=["0.8"],
        func=run,
    )


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    px, py = args.pdim
    job_name = f"fli_grid_{args.simulation_type}"

    fli_cmd = [
        "fli-grid",
        args.simulation_type,
        "--mesh-size",
        *[str(m) for m in args.mesh_size],
        "--box-size",
        *[str(b) for b in args.box_size],
        "--Omega-c",
        *[str(v) for v in args.Omega_c],
        "--sigma8",
        *[str(v) for v in args.sigma8],
        "--seed",
        *[str(v) for v in args.seed],
    ]
    if args.nb_shells is not None:
        fli_cmd += ["--nb-shells", str(args.nb_shells)]
    if args.ts:
        fli_cmd += ["--ts", *[str(t) for t in args.ts]]
    if args.ts_near:
        fli_cmd += ["--ts-near", *[str(t) for t in args.ts_near]]
    if args.ts_far:
        fli_cmd += ["--ts-far", *[str(t) for t in args.ts_far]]
    fli_cmd += [
        "--nb-steps",
        str(args.nb_steps),
        "--t0",
        str(args.t0),
        "--t1",
        str(args.t1),
        "--lpt-order",
        str(args.lpt_order),
        "--halo-multiplier",
        str(args.halo_multiplier),
        "--pdim",
        str(px),
        str(py),
        "--nodes",
        str(args.nodes),
        "--interp",
        args.interp,
        "--scheme",
        args.scheme,
    ]
    if args.nside is not None:
        fli_cmd += ["--nside", str(args.nside)]
    elif getattr(args, "density", False):
        fli_cmd.append("--density")
    elif getattr(args, "flatsky_npix", None) is not None:
        fli_cmd += ["--flatsky-npix", *[str(v) for v in args.flatsky_npix]]
        if getattr(args, "field_size", None) is not None:
            fli_cmd += ["--field-size", *[str(v) for v in args.field_size]]
    if args.paint_nside is not None:
        fli_cmd += ["--paint-nside", str(args.paint_nside)]
    if args.kernel_width_arcmin is not None:
        fli_cmd += ["--kernel-width-arcmin", str(args.kernel_width_arcmin)]
    if args.drift_on_lightcone:
        fli_cmd.append("--drift-on-lightcone")
    fli_cmd += ["--min-width", str(args.min_width)]
    fli_cmd += ["--shell-spacing", args.shell_spacing]
    fli_cmd += ["--solver", args.solver]
    if args.density_widths:
        fli_cmd += ["--density-widths", *[str(d) for d in args.density_widths]]
    if args.dealiased:
        fli_cmd.append("--dealiased")
    if args.exact_growth:
        fli_cmd.append("--exact-growth")
    fli_cmd += ["--gradient-order", str(args.gradient_order)]
    if args.laplace_fd:
        fli_cmd.append("--laplace-fd")
    if args.simulation_type == "lensing":
        fli_cmd += [
            "--nz-shear",
            *[str(v) for v in args.nz_shear],
            "--min-z",
            str(args.min_z),
            "--max-z",
            str(args.max_z),
            "--n-integrate",
            str(args.n_integrate),
        ]
    fli_cmd += [
        "--h",
        str(args.h),
        "--Omega-b",
        str(args.Omega_b),
        "--n-s",
        str(args.n_s),
        "--Omega-k",
        str(args.Omega_k),
        "--w0",
        str(args.w0),
        "--wa",
        str(args.wa),
        "--Omega-nu",
        str(args.Omega_nu),
        "--output-dir",
        args.output_dir,
    ]
    if args.enable_x64:
        fli_cmd.append("--enable-x64")

    dispatch(args, job_name, "FLI_GRID", fli_cmd)
