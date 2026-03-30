"""fli-simulate: CLI entry point for the jax_fli pipeline.

Provides subcommands:
  lpt      — IC → LPT → lightcone / particles
  nbody    — IC → LPT(particles) → NBody → lightcone / particles
  lensing  — IC → LPT → NBody → Born → kappa

JAX is imported lazily (after argument parsing) so --help is instantaneous.
"""

import os
import sys
from argparse import ArgumentParser, Namespace
from functools import partial

import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax.experimental.multihost_utils import sync_global_devices

import jax_fli as jfli
from jax_fli.scripts._common import _build_sharding, _resolve_nz_shear, _try_parse_s3  # noqa: F401

# ---------------------------------------------------------------------------
# Cosmology builder
# ---------------------------------------------------------------------------


def _build_cosmo(args: Namespace):
    """Construct a jax_cosmo.Cosmology from CLI flags."""
    return jc.Cosmology(
        Omega_c=args.Omega_c,
        Omega_b=args.Omega_b,
        h=args.h,
        n_s=args.n_s,
        sigma8=args.sigma8,
        Omega_k=args.Omega_k,
        w0=args.w0,
        wa=args.wa,
        Omega_nu=args.Omega_nu,
    )


# ---------------------------------------------------------------------------
# Painting options builder
# ---------------------------------------------------------------------------


def _build_painting(args: Namespace):
    """Return a PaintingOptions and (nside, flatsky_npix) for IC generation."""
    from jax_fli.fields import PaintingOptions

    nside = getattr(args, "nside", None)
    flatsky_npix = getattr(args, "flatsky_npix", None)
    density = getattr(args, "density", False)

    if nside is not None:
        return PaintingOptions(target="spherical", scheme=args.scheme, paint_nside=args.paint_nside), nside, None
    elif flatsky_npix is not None:
        h, w = flatsky_npix
        return PaintingOptions(target="flat"), None, (h, w)
    elif density:
        return PaintingOptions(target="density"), None, None
    else:
        return PaintingOptions(target="particles"), None, None


# ---------------------------------------------------------------------------
# Snapshot/lightcone time-step resolver
# ---------------------------------------------------------------------------


def _resolve_ts(args: Namespace):
    """Return ts (jnp array or None) from --ts / --ts-near+far."""
    if args.ts is not None:
        return jnp.array(args.ts)
    if args.ts_near is not None and args.ts_far is not None:
        # Build (2, N) near/far array — convert lists from nargs="+" to JAX arrays first
        return jnp.stack([jnp.array(args.ts_near), jnp.array(args.ts_far)], axis=0)

    return None


# ---------------------------------------------------------------------------
# Solver builder
# ---------------------------------------------------------------------------


def _build_solver(args: Namespace, painting):
    """Build an AbstractNBodySolver from CLI flags."""
    inter = getattr(args, "interp", "none")
    drift_on_lightcone = getattr(args, "drift_on_lightcone", False)
    if inter == "none":
        if drift_on_lightcone:
            interp_kernel = jfli.DriftInterp(painting=painting)
        else:
            interp_kernel = jfli.NoInterp(painting=painting)
    elif inter == "onion":
        if painting.target != "spherical":
            raise ValueError("--interp onion requires --nside (spherical painting target)")
        interp_kernel = jfli.OnionTiler(painting=painting, drift_on_lightcone=drift_on_lightcone)
    elif inter == "telephoto":
        interp_kernel = jfli.TelephotoInterp(painting=painting, drift_on_lightcone=drift_on_lightcone)
    else:
        raise ValueError(f"Unknown --interp value: {inter}")

    shell_spacing = getattr(args, "shell_spacing", "comoving")
    solver_name = getattr(args, "solver", "kdk")
    common_kwargs = dict(
        interp_kernel=interp_kernel,
        gradient_order=getattr(args, "gradient_order", 1),
        laplace_fd=getattr(args, "laplace_fd", False),
        t0=args.t0,
        t1=getattr(args, "t1", 1.0),
        n_steps=getattr(args, "nb_steps", 19),
        shell_spacing=shell_spacing,
        min_width=getattr(args, "min_width", 50.0),
    )
    if solver_name == "kdk":
        return jfli.DoubleKickDrift(**common_kwargs)
    elif solver_name == "dkd":
        return jfli.DriftKickDrift(**common_kwargs)
    elif solver_name == "bf":
        return jfli.BullFrog(**common_kwargs)
    else:
        raise ValueError(f"Unknown --solver value: {solver_name}")


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------


def _save_result(result, cosmo, args: Namespace, output: str | None = None) -> None:
    """Save result to parquet (process 0 only)."""
    out_path = output if output is not None else args.output
    parent_folder = os.path.dirname(out_path)
    if parent_folder:
        os.makedirs(parent_folder, exist_ok=True)
    catalog = jfli.io.Catalog(field=result, cosmology=cosmo)
    catalog.to_parquet(out_path)
    print(f"Saved to {out_path}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parser() -> ArgumentParser:
    """Build the full argparse tree with subcommands."""

    # ------------------------------------------------------------------
    # Common parent (shared by all subcommands)
    # ------------------------------------------------------------------
    common = ArgumentParser(add_help=False)

    # Mesh / box
    common.add_argument(
        "--mesh-size",
        type=int,
        nargs=3,
        default=[64, 64, 64],
        metavar=("NX", "NY", "NZ"),
        help="Mesh resolution (default: 64 64 64)",
    )
    common.add_argument(
        "--box-size",
        type=float,
        nargs=3,
        default=[200.0, 200.0, 200.0],
        metavar=("LX", "LY", "LZ"),
        help="Box side lengths in Mpc/h (default: 200 200 200)",
    )
    common.add_argument(
        "--pdim",
        type=int,
        nargs=2,
        default=[1, 1],
        metavar=("PX", "PY"),
        help="Process mesh dimensions (default: 1 1 = single device)",
    )
    common.add_argument("--nodes", type=int, default=1, help="Number of nodes (default: 1)")
    common.add_argument(
        "--halo-fraction",
        type=int,
        default=8,
        metavar="F",
        help="Halo size as mesh // fraction for distributed painting (default: 8)",
    )
    common.add_argument(
        "--observer-position",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 0.5],
        metavar=("OX", "OY", "OZ"),
        help="Observer position in box coordinates (default: 0.5 0.5 0.5, i.e. center of the box)",
    )

    # Random seed and output
    common.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    common.add_argument(
        "--output", "-o", default="sim_output.parquet", help="Output file path (default: sim_output.parquet)"
    )

    # Performance
    common.add_argument("--perf", action="store_true", help="Benchmark: warmup + N timed iterations")
    common.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=5,
        metavar="N",
        help="Number of timed iterations for --perf (default: 5)",
    )
    common.add_argument("--trace", action="store_true", help="Run with JAX profiler trace")
    common.add_argument(
        "--trace-dir", default="/tmp/jax_trace", help="Directory for profiler trace (default: /tmp/jax_trace)"
    )
    common.add_argument("--enable-x64", action="store_true", help="Enable JAX 64-bit precision (default: False)")

    # Cosmology
    cosmo_group = common.add_argument_group("cosmology")
    cosmo_group.add_argument("--Omega-c", type=float, default=0.2589, help="Cold dark matter density (default: 0.2589)")
    cosmo_group.add_argument("--Omega-b", type=float, default=0.0486, help="Baryon density (default: 0.0486)")
    cosmo_group.add_argument("--h", type=float, default=0.6774, help="Dimensionless Hubble parameter (default: 0.6774)")
    cosmo_group.add_argument("--n-s", type=float, default=0.9667, dest="n_s", help="Spectral index (default: 0.9667)")
    cosmo_group.add_argument("--sigma8", type=float, default=0.8159, help="sigma8 (default: 0.8159)")
    cosmo_group.add_argument("--Omega-k", type=float, default=0.0, help="Curvature density (default: 0.0)")
    cosmo_group.add_argument("--w0", type=float, default=-1.0, help="Dark energy EOS w0 (default: -1.0)")
    cosmo_group.add_argument("--wa", type=float, default=0.0, help="Dark energy EOS wa (default: 0.0)")
    cosmo_group.add_argument("--Omega-nu", type=float, default=0.0, help="Neutrino density (default: 0.0)")

    # Painting target (mutually exclusive)
    paint_group = common.add_mutually_exclusive_group()
    paint_group.add_argument("--nside", type=int, default=None, help="HEALPix NSIDE for spherical painting")
    paint_group.add_argument(
        "--flatsky-npix",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Flat-sky pixel resolution (height width)",
    )
    paint_group.add_argument(
        "--field-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="2D field pixel resolution (alternative to --flatsky-npix)",
    )
    paint_group.add_argument("--density", action="store_true", default=False, help="3D density field painting")

    common.add_argument(
        "--scheme",
        choices=["ngp", "bilinear", "rbf_neighbor"],
        default="bilinear",
        help="Spherical painting interpolation scheme (default: bilinear)",
    )
    common.add_argument(
        "--paint-nside",
        type=int,
        default=None,
        dest="paint_nside",
        help="Override nside for spherical painting (default: same as --nside)",
    )

    # ------------------------------------------------------------------
    # LPT parent (shared by lpt, nbody, lensing)
    # ------------------------------------------------------------------
    lpt_parent = ArgumentParser(add_help=False)
    lpt_parent.add_argument("--t0", type=float, default=0.1, help="LPT starting scale factor (default: 0.1)")
    lpt_parent.add_argument("--lpt-order", type=int, default=2, choices=[1, 2], help="LPT order (default: 2)")
    lpt_parent.add_argument("--nb-shells", type=int, default=None, help="Number of lightcone shells")
    lpt_parent.add_argument(
        "--density-widths", type=float, nargs="+", default=None, metavar="W", help="Override shell widths (Mpc/h)"
    )
    # Mutually exclusive ts group
    ts_group = lpt_parent.add_mutually_exclusive_group()
    ts_group.add_argument(
        "--ts", type=float, nargs="+", default=None, metavar="A", help="Scale factors for snapshot/shell output"
    )
    ts_group.add_argument(
        "--ts-near",
        type=float,
        nargs="+",
        default=None,
        metavar="A_NEAR",
        help="Near scale factor edge(s) (use with --ts-far; one value per shell pair)",
    )
    lpt_parent.add_argument(
        "--ts-far",
        type=float,
        nargs="+",
        default=None,
        metavar="A_FAR",
        help="Far scale factor edge(s) (use with --ts-near; one value per shell pair)",
    )
    lpt_parent.add_argument(
        "--shell-spacing",
        choices=["comoving", "equal_vol", "a", "growth"],
        default="comoving",
        dest="shell_spacing",
        help="Shell spacing mode (default: comoving)",
    )
    lpt_parent.add_argument(
        "--min-width",
        type=float,
        default=50.0,
        dest="min_width",
        help="Minimum shell width in Mpc/h (default: 50.0)",
    )
    lpt_parent.add_argument(
        "--gradient-order",
        type=int,
        default=1,
        choices=[0, 1],
        dest="gradient_order",
        help="Force gradient order (0=exact ik, 1=finite-difference) (default: 1)",
    )
    lpt_parent.add_argument(
        "--laplace-fd",
        action="store_true",
        default=False,
        dest="laplace_fd",
        help="Use finite-difference Laplacian (default: False)",
    )
    lpt_parent.add_argument(
        "--dealiased",
        action="store_true",
        default=False,
        help="Enable dealiased mode (default: False)",
    )
    lpt_parent.add_argument(
        "--exact-growth",
        action="store_true",
        default=False,
        dest="exact_growth",
        help="Use exact growth factor computation (default: False)",
    )

    # ------------------------------------------------------------------
    # NBody parent (adds t1, nb_steps, solver, interp)
    # ------------------------------------------------------------------
    nbody_parent = ArgumentParser(add_help=False)
    nbody_parent.add_argument("--t1", type=float, default=1.0, help="NBody final scale factor (default: 1.0)")
    nbody_parent.add_argument(
        "--nb-steps",
        type=int,
        default=19,
        dest="nb_steps",
        help="Number of integration steps (default: 19)",
    )
    nbody_parent.add_argument(
        "--interp",
        choices=["none", "onion", "telephoto"],
        default="none",
        help="Interpolation kernel (default: none)",
    )
    nbody_parent.add_argument(
        "--drift-on-lightcone", action="store_true", help="Apply drift correction when painting lightcone shells"
    )
    nbody_parent.add_argument(
        "--solver",
        choices=["kdk", "dkd", "bf"],
        default="kdk",
        help="N-body integrator: kdk=DoubleKickDrift, dkd=DriftKickDrift, bf=BullFrog (default: kdk)",
    )

    # ------------------------------------------------------------------
    # Top-level parser
    # ------------------------------------------------------------------
    parser = ArgumentParser(
        prog="fli-simulate",
        description="jax_fli simulation pipeline CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # lpt subcommand
    subparsers.add_parser(
        "lpt",
        parents=[common, lpt_parent],
        help="Run IC → LPT only",
        description="Generate initial conditions and apply LPT displacements.",
    )

    # nbody subcommand
    subparsers.add_parser(
        "nbody",
        parents=[common, lpt_parent, nbody_parent],
        help="Run IC → LPT(particles) → NBody",
        description="Run full N-body integration from LPT initial conditions.",
    )

    # lensing subcommand
    lensing_p = subparsers.add_parser(
        "lensing",
        parents=[common, lpt_parent, nbody_parent],
        help="Run IC → LPT → NBody → Born lensing",
        description="Run full pipeline including weak lensing convergence maps.",
    )
    lensing_p.add_argument(
        "--nz-shear",
        nargs="+",
        required=True,
        metavar="Z",
        help="Source redshifts or 'stage3'/'s3' for 4-bin Stage 3 distributions",
    )
    lensing_p.add_argument(
        "--min-z", type=float, default=0.01, help="Minimum redshift for nz integration (default: 0.01)"
    )
    lensing_p.add_argument(
        "--max-z", type=float, default=1.5, help="Maximum redshift for nz integration (default: 1.5)"
    )
    lensing_p.add_argument(
        "--n-integrate", type=int, default=32, help="Number of integration points for nz distributions (default: 32)"
    )
    return parser


# ---------------------------------------------------------------------------
# Pre-JAX argument validation
# ---------------------------------------------------------------------------


def _validate_args(args: Namespace, parser: ArgumentParser) -> None:
    """Validate argument combinations before importing JAX."""
    # --ts-near requires --ts-far and --nb-shells
    ts_near = getattr(args, "ts_near", None)
    ts_far = getattr(args, "ts_far", None)

    if ts_near is not None:
        if ts_far is None:
            parser.error("--ts-near requires --ts-far")
        if len(ts_near) != len(ts_far):
            parser.error(
                f"--ts-near and --ts-far must have the same number of values ({len(ts_near)} vs {len(ts_far)})"
            )

    # lensing requires a projection target
    if args.subcommand == "lensing":
        nside = getattr(args, "nside", None)
        flatsky_npix = getattr(args, "flatsky_npix", None)
        if nside is None and flatsky_npix is None:
            parser.error("lensing subcommand requires --nside or --flatsky-npix")

    # --perf and --trace are mutually exclusive (perf wins)
    if getattr(args, "perf", False) and getattr(args, "trace", False):
        print("Warning: --perf and --trace both specified; --perf takes precedence, --trace ignored", file=sys.stderr)
        args.trace = False

    # --interp onion requires --nside
    interp = getattr(args, "interp", "none")
    nside = getattr(args, "nside", None)
    if interp == "onion" and nside is None:
        parser.error("--interp onion requires --nside")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


@partial(
    jax.jit,
    static_argnames=[
        "lpt_order",
        "painting",
        "nb_shells",
        "shell_spacing",
        "min_width",
        "gradient_order",
        "laplace_fd",
        "dealiased",
        "exact_growth",
    ],
)
def run_lpt(
    cosmo,
    initial_conditions,
    ts,
    *,
    lpt_order,
    painting,
    nb_shells,
    shell_spacing,
    min_width,
    density_widths=None,
    gradient_order=1,
    laplace_fd=False,
    dealiased=False,
    exact_growth=False,
):
    dx, p = jfli.lpt(
        cosmo,
        initial_conditions,
        ts=ts,
        nb_shells=nb_shells,
        density_widths=density_widths,
        order=lpt_order,
        painting=painting,
        shell_spacing=shell_spacing,
        min_width=min_width,
        gradient_order=gradient_order,
        laplace_fd=laplace_fd,
        dealiased=dealiased,
        exact_growth=exact_growth,
    )
    return dx


@partial(
    jax.jit,
    static_argnames=["lpt_order", "sim_type", "nb_shells", "gradient_order", "laplace_fd", "dealiased", "exact_growth"],
)
def run_simulations(
    cosmo,
    initial_conditions,
    solver,
    *,
    lpt_order,
    sim_type,
    nz_shear=None,
    ts=None,
    nb_shells=None,
    density_widths=None,
    gradient_order=1,
    laplace_fd=False,
    dealiased=False,
    exact_growth=False,
):
    # LPT to particles snapshot at t0, then run NBody
    dx, p = jfli.lpt(
        cosmo,
        initial_conditions,
        ts=solver.t0,
        order=lpt_order,
        painting=jfli.PaintingOptions(target="particles"),
        gradient_order=gradient_order,
        laplace_fd=laplace_fd,
        dealiased=dealiased,
        exact_growth=exact_growth,
    )

    # Run NBody
    lightcone = jfli.nbody(
        cosmo,
        dx,
        p,
        solver=solver,
        ts=ts,
        nb_shells=nb_shells,
        density_widths=density_widths,
    )
    if sim_type == "nbody":
        return lightcone

    # Run lensing
    if sim_type == "born":
        return jfli.born(cosmo, lightcone, nz_shear)
    else:
        raise ValueError(f"Unknown sim_type: {sim_type}")


def main() -> None:
    """CLI entry point registered as fli-simulate."""
    p = parser()
    args, unknown = p.parse_known_args()
    if unknown:
        print(
            f"Warning: the following arguments are not recognized by the "
            f"'{args.subcommand}' subcommand and will be ignored: {unknown}",
            file=sys.stderr,
        )
    jax.config.update("jax_enable_x64", args.enable_x64)
    _validate_args(args, p)

    # Prepare arguments

    cosmo = _build_cosmo(args)

    painting, nside, flatsky_npix = _build_painting(args)
    sharding = _build_sharding(args)

    ts = _resolve_ts(args)
    nz_shear = _resolve_nz_shear(args)
    solver = _build_solver(args, painting)

    mesh = tuple(args.mesh_size)
    halo_size = (mesh[0] // args.halo_fraction, mesh[1] // args.halo_fraction)

    key = jax.random.key(args.seed)

    initial_field = jfli.gaussian_initial_conditions(
        key,
        mesh,
        tuple(args.box_size),
        observer_position=tuple(args.observer_position),
        cosmo=cosmo,
        nside=args.nside,
        flatsky_npix=tuple(args.flatsky_npix) if args.flatsky_npix is not None else None,
        field_size=tuple(args.field_size) if args.field_size is not None else None,
        field_sharding=sharding,
        halo_size=halo_size,
    )

    sim_type = args.subcommand
    lpt_order = args.lpt_order
    if args.subcommand == "lensing":
        sim_type = "born"

    density_widths = jnp.array(args.density_widths) if args.density_widths is not None else None
    nb_shells = args.nb_shells
    shell_spacing = getattr(args, "shell_spacing", "comoving")

    if sim_type == "lpt":
        # LPT mode: pass geometry params directly to lpt()
        # For lpt: if nb_shells is set, don't pass ts (they're mutually exclusive)
        lpt_ts = ts if ts is not None else (args.t0 if nb_shells is None else None)

        run_fn = run_lpt
        run_kwargs = {
            "ts": lpt_ts,
            "lpt_order": lpt_order,
            "painting": painting,
            "nb_shells": nb_shells,
            "shell_spacing": shell_spacing,
            "min_width": getattr(args, "min_width", 50.0),
            "density_widths": density_widths,
            "gradient_order": args.gradient_order,
            "laplace_fd": args.laplace_fd,
            "dealiased": args.dealiased,
            "exact_growth": args.exact_growth,
        }
    else:
        run_fn = run_simulations
        run_kwargs = {
            "solver": solver,
            "lpt_order": lpt_order,
            "nz_shear": nz_shear,
            "sim_type": sim_type,
            "ts": ts,
            "nb_shells": nb_shells,
            "density_widths": density_widths,
            "gradient_order": args.gradient_order,
            "laplace_fd": args.laplace_fd,
            "dealiased": args.dealiased,
            "exact_growth": args.exact_growth,
        }

    if args.perf:
        try:
            from jax_hpc_profiler import JaxTimer
        except ImportError:
            print("Error: jax-hpc-profiler not found. Please install it to use --perf.", file=sys.stderr)
            sys.exit(1)

        timer = JaxTimer(save_jaxpr=False)
        print("Compiling and running first iteration...")
        result = timer.chrono_jit(run_fn, cosmo, initial_field, **run_kwargs)
        del result
        print(f"Running {args.iterations} timed iterations...")
        for i in range(args.iterations):
            result = timer.chrono_fun(run_fn, cosmo, initial_field, **run_kwargs)
            print(f"Iteration {i + 1}/{args.iterations} completed.")
            if i < args.iterations - 1:
                del result

        metadata = {
            "precision": "float64" if jax.config.jax_enable_x64 else "float32",
            "x": str(args.mesh_size[0]),
            "y": str(args.mesh_size[1]),
            "z": str(args.mesh_size[2]),
            "px": str(args.pdim[0]),
            "py": str(args.pdim[1]),
            "nodes": str(args.nodes),
        }
        extra_info = {
            "halo_fraction": str(args.halo_fraction),
            "painting_target": painting.target,
            "ts": str(args.ts) if args.ts is not None else f"near={args.ts_near}, far={args.ts_far}",
            "nb_shells": str(args.nb_shells),
            "lpt_order": str(args.lpt_order),
        }

        report_file = f"perf_{sim_type}.csv"
        nb_steps = getattr(args, "nb_steps", "")
        func_name = f"{sim_type}{nb_steps}"
        timer.report(report_file, function=func_name, extra_info=extra_info, **metadata)
        print(f"Performance report saved to {report_file}")
    else:
        result = jax.block_until_ready(run_fn(cosmo, initial_field, **run_kwargs))

    print("Simulation completed... saving results.")
    sync_global_devices("Done")
    # --- Save output ---
    _save_result(result, cosmo, args)  # pyright: ignore
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
