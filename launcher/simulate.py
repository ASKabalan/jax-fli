"""simulate subcommand — wraps fli-simulate."""
from __future__ import annotations

import os

from .parser import (
    DEFAULT_NAME_TEMPLATE,
    add_common_sim_args,
    add_cosmo_args,
    add_integration_args,
    add_lensing_args,
    add_lightcone_args,
    add_output_target_args,
    add_slurm_args,
    dispatch,
)


def add_subparser(sub):
    p = sub.add_parser(
        "simulate",
        help="Submit fli-simulate jobs over a mesh × box × cosmology × seed grid",
    )
    add_slurm_args(p)
    add_common_sim_args(p)
    add_cosmo_args(p, sweep=True)
    add_lensing_args(p)
    add_lightcone_args(p)
    add_output_target_args(p)
    add_integration_args(p)

    g = p.add_argument_group("simulate")
    g.add_argument("--nb-shells", type=int, default=10, help="Number of lightcone shells (default: 10)")
    g.add_argument("--enable-x64", action="store_true", help="Enable JAX 64-bit precision")
    g.add_argument("--output-dir", default="results/cosmology_runs")
    g.add_argument(
        "--name-template",
        default=DEFAULT_NAME_TEMPLATE,
        dest="name_template",
        help="Template for the output parquet name. Placeholders: %%constraint%%, %%mesh_size%%, "
        "%%box_size%%, %%nb_steps%%, %%omega_c%%, %%sigma8%%, %%seed%%",
    )
    g.add_argument("--perf", action="store_true", default=False, help="Benchmark: warmup + N timed iterations")
    g.add_argument("--iterations", type=int, default=3, help="Number of timed iterations for --perf")

    # drift-on-lightcone is ON by default
    p.set_defaults(drift_on_lightcone=True, func=run)


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    mesh_list = args.mesh_size
    if len(mesh_list) % 3 != 0:
        raise ValueError(f"--mesh-size must be a multiple of 3 values, got {len(mesh_list)}")
    meshes = [mesh_list[i : i + 3] for i in range(0, len(mesh_list), 3)]

    box_list = args.box_size
    if len(box_list) % 3 != 0:
        raise ValueError(f"--box-size must be a multiple of 3 values, got {len(box_list)}")
    boxes = [box_list[i : i + 3] for i in range(0, len(box_list), 3)]

    px, py = args.pdim

    print("Launching cosmology simulations...")

    for box in boxes:
        box_name = "x".join(str(b) for b in box).replace(".0", "")

        for mesh in meshes:
            mesh_name = "x".join(str(m) for m in mesh)

            for oc in args.Omega_c:
                for s8 in args.sigma8:
                    for sd in args.seed:
                        template = getattr(args, "name_template", DEFAULT_NAME_TEMPLATE)
                        job_name = (
                            template.replace("%constraint%", str(args.constraint))
                            .replace("%mesh_size%", mesh_name)
                            .replace("%box_size%", box_name)
                            .replace("%nb_steps%", str(args.nb_steps))
                            .replace("%omega_c%", str(oc))
                            .replace("%sigma8%", str(s8))
                            .replace("%seed%", str(sd))
                        )
                        out_file = f"{args.output_dir}/{job_name}.parquet"

                        fli_cmd = [
                            "fli-simulate",
                            args.simulation_type,
                            "--mesh-size",
                            *[str(m) for m in mesh],
                            "--box-size",
                            *[str(b) for b in box],
                            "--pdim",
                            str(px),
                            str(py),
                            "--nodes",
                            str(args.nodes),
                            "--halo-multiplier",
                            str(args.halo_multiplier),
                            "--observer-position",
                            *[str(v) for v in args.observer_position],
                        ]
                        # Output target (mutually exclusive)
                        if args.nside is not None:
                            fli_cmd += ["--nside", str(args.nside)]
                        elif getattr(args, "density", False):
                            fli_cmd.append("--density")
                        else:
                            if getattr(args, "flatsky_npix", None) is not None:
                                fli_cmd += ["--flatsky-npix", *[str(v) for v in args.flatsky_npix]]
                            if getattr(args, "field_size", None) is not None:
                                fli_cmd += ["--field-size", *[str(v) for v in args.field_size]]
                        if not args.ts and not args.ts_near and not args.ts_far:
                            fli_cmd += ["--nb-shells", str(args.nb_shells)]
                        if args.ts:
                            fli_cmd += ["--ts", *[str(t) for t in args.ts]]
                        if args.ts_near:
                            fli_cmd += ["--ts-near", *[str(t) for t in args.ts_near]]
                        if args.ts_far:
                            fli_cmd += ["--ts-far", *[str(t) for t in args.ts_far]]
                        fli_cmd += [
                            "--t0",
                            str(args.t0),
                            "--nb-steps",
                            str(args.nb_steps),
                            "--t1",
                            str(args.t1),
                            "--lpt-order",
                            str(args.lpt_order),
                            "--interp",
                            args.interp,
                            "--scheme",
                            args.scheme,
                        ]
                        if args.paint_nside is not None:
                            fli_cmd += ["--paint-nside", str(args.paint_nside)]
                        if args.kernel_width_arcmin is not None:
                            fli_cmd += ["--kernel-width-arcmin", str(args.kernel_width_arcmin)]
                        if args.drift_on_lightcone:
                            fli_cmd.append("--drift-on-lightcone")
                        fli_cmd += [
                            "--min-width",
                            str(args.min_width),
                            "--shell-spacing",
                            args.shell_spacing,
                            "--solver",
                            args.solver,
                        ]
                        if getattr(args, "density_widths", None):
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
                            "--Omega-c",
                            str(oc),
                            "--sigma8",
                            str(s8),
                            "--Omega-b",
                            str(args.Omega_b),
                            "--h",
                            str(args.h),
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
                            "--seed",
                            str(sd),
                            "--output",
                            out_file,
                            "--name",
                            job_name,
                        ]
                        if getattr(args, "perf", False):
                            iterations = getattr(args, "iterations", 3) or 3
                            fli_cmd += ["--perf", "--iterations", str(iterations)]
                        if args.enable_x64:
                            fli_cmd.append("--enable-x64")

                        dispatch(args, job_name, "FLI_SIMULATION", fli_cmd)
