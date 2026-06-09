"""fli-simulate: CLI entry point for the jax_fli pipeline.

Use --sim-mode to select the pipeline:
  lpt      — IC → LPT → lightcone / particles
  pm       — IC → LPT(particles) → NBody → lightcone / particles
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
from jax_fli.scripts._common import _build_sharding, _resolve_nz_shear, _save_args_log, _try_parse_s3  # noqa: F401

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
        return (
            PaintingOptions(
                target="spherical",
                scheme=args.scheme,
                paint_nside=args.paint_nside,
                kernel_width_arcmin=getattr(args, "kernel_width_arcmin", None),
                kernel_width_pixels=getattr(args, "kernel_width_pixels", None),
            ),
            nside,
            None,
        )
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

    solver_name = getattr(args, "solver", "kdk")
    common_kwargs = dict(
        interp_kernel=interp_kernel,
        gradient_order=getattr(args, "gradient_order", 1),
        laplace_fd=getattr(args, "laplace_fd", False),
        order=getattr(args, "paint_order", "cic"),
        deconvolution=getattr(args, "deconvolution", False),
        t0=args.t0,
        t1=getattr(args, "t1", 1.0),
        n_steps=getattr(args, "nb_steps", 19),
        time_stepping=getattr(args, "time_stepping", "a"),
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
    name = getattr(args, "name", None)
    if name is not None:
        result = result.replace(name=name)
    catalog = jfli.io.Catalog(field=result, cosmology=cosmo)
    catalog.to_parquet(out_path)
    print(f"Saved to {out_path}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parser() -> ArgumentParser:
    """Build the flat argument parser with --sim-mode."""
    from jax_fli.scripts.parser import (
        add_cosmo_args,
        add_distributed_args,
        add_integration_settings_args,
        add_simulation_settings_args,
    )

    p = ArgumentParser(prog="fli-simulate", description="jax_fli simulation pipeline CLI")
    add_distributed_args(p)
    add_simulation_settings_args(p)
    add_integration_settings_args(p)
    add_cosmo_args(p)
    p.add_argument(
        "--output", "-o", default="sim_output.parquet", help="Output file path (default: sim_output.parquet)"
    )
    p.add_argument("--name", default=None, help="Label stored as AbstractField.name inside the output catalog")
    p.add_argument("--perf", action="store_true", help="Benchmark: warmup + N timed iterations")
    p.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=5,
        metavar="N",
        help="Number of timed iterations for --perf (default: 5)",
    )

    return p


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
    if args.sim_mode == "lensing":
        nside = getattr(args, "nside", None)
        flatsky_npix = getattr(args, "flatsky_npix", None)
        if nside is None and flatsky_npix is None:
            parser.error("--sim-mode lensing requires --nside or --flatsky-npix")

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
        "paint_order",
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
    lpt_order,
    painting,
    nb_shells,
    shell_spacing,
    min_width,
    density_widths=None,
    paint_order="cic",
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
        paint_order=paint_order,
        gradient_order=gradient_order,
        laplace_fd=laplace_fd,
        dealiased=dealiased,
        exact_growth=exact_growth,
    )
    return dx


@partial(
    jax.jit,
    static_argnames=[
        "lpt_order",
        "sim_type",
        "nb_shells",
        "shell_spacing",
        "min_width",
        "paint_order",
        "gradient_order",
        "laplace_fd",
        "dealiased",
        "exact_growth",
        "n_integrate",
    ],
)
def run_simulations(
    cosmo,
    initial_conditions,
    solver,
    lpt_order,
    sim_type,
    nz_shear=None,
    ts=None,
    nb_shells=None,
    density_widths=None,
    shell_spacing: str = "a",
    min_width: float = 50.0,
    paint_order="cic",
    gradient_order=1,
    laplace_fd=False,
    dealiased=False,
    exact_growth=False,
    min_z=0.01,
    max_z=1.5,
    n_integrate=32,
):
    # LPT to particles snapshot at t0, then run NBody
    dx, p = jfli.lpt(
        cosmo,
        initial_conditions,
        ts=solver.t0,
        order=lpt_order,
        painting=jfli.PaintingOptions(target="particles"),
        paint_order=paint_order,
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
        shell_spacing=shell_spacing,
        min_width=min_width,
    )
    if sim_type == "pm":
        return lightcone

    # Run lensing
    if sim_type == "born":
        return jfli.born(cosmo, lightcone, nz_shear, min_z=min_z, max_z=max_z, n_integrate=n_integrate)
    else:
        raise ValueError(f"Unknown sim_type: {sim_type}")


def main() -> None:
    """CLI entry point registered as fli-simulate."""
    p = parser()
    args = p.parse_args()
    jax.config.update("jax_enable_x64", args.enable_x64)
    _validate_args(args, p)

    # Prepare arguments

    cosmo = _build_cosmo(args)

    painting, nside, flatsky_npix = _build_painting(args)
    sharding = _build_sharding(args)

    ts = _resolve_ts(args)
    nz_shear = _resolve_nz_shear(args)
    solver = _build_solver(args, painting)

    output_dir = os.path.dirname(args.output) or "."
    _save_args_log(args, output_dir, f"fli-simulate {args.sim_mode}")

    mesh = tuple(args.mesh_size)
    px, py = args.pdim
    halo_size = (int(mesh[0] / px * args.halo_multiplier), int(mesh[1] / py * args.halo_multiplier))

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

    sim_type = args.sim_mode
    lpt_order = args.lpt_order
    if args.sim_mode == "lensing":
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
            "paint_order": args.paint_order,
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
            "shell_spacing": shell_spacing,
            "min_width": getattr(args, "min_width", 50.0),
            "paint_order": args.paint_order,
            "gradient_order": args.gradient_order,
            "laplace_fd": args.laplace_fd,
            "dealiased": args.dealiased,
            "exact_growth": args.exact_growth,
            "min_z": getattr(args, "min_z", 0.01),
            "max_z": getattr(args, "max_z", 1.5),
            "n_integrate": getattr(args, "n_integrate", 32),
        }

    if args.perf:
        try:
            from jax_hpc_profiler import JaxTimer
        except ImportError:
            print("Error: jax-hpc-profiler not found. Please install it to use --perf.", file=sys.stderr)
            sys.exit(1)

        if sim_type == "lpt":
            _static_argnums = (3, 4, 5, 6, 7, 9, 10, 11)
        else:
            _static_argnums = (3, 4, 7, 9, 10, 11, 12, 13, 16)
        timer = JaxTimer(save_jaxpr=False, static_argnums=_static_argnums)
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
            "halo_multiplier": str(args.halo_multiplier),
            "painting_target": painting.target,
            "ts": str(args.ts) if args.ts is not None else f"near={args.ts_near}, far={args.ts_far}",
            "nb_shells": str(args.nb_shells),
            "lpt_order": str(args.lpt_order),
        }

        output_dir = f"{os.path.dirname(args.output)}/" if args.output else ""
        report_file = f"{output_dir}/perf_{sim_type}.csv"
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
