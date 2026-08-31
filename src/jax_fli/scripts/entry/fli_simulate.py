"""fli-simulate: CLI entry point for the jax_fli pipeline.

Use --sim-mode to select the pipeline:
  lpt      — IC → LPT → lightcone / particles
  pm       — IC → LPT(particles) → NBody → lightcone / particles
  lensing  — IC → LPT → NBody → Born → kappa

JAX is imported lazily (after argument parsing) so --help is instantaneous.
"""

import os
import re
import sys
from argparse import ArgumentParser, Namespace
from functools import partial

import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax.experimental.multihost_utils import sync_global_devices

import jax_fli as jfli
from jax_fli.scripts._common import (  # noqa: F401
    _build_sharding,
    _resolve_nz_shear,
    _resolve_source,
    _save_args_log,
    _try_parse_s3,
)

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
# Gradient mode resolver
# ---------------------------------------------------------------------------


def _parse_grad(grad: str):
    """Map a --grad spec to ``(compute_grad, adjoint, checkpoints)``.

    none             -> forward only (no gradient)
    reverse          -> reversible backsolve adjoint (O(1) memory)
    checkpoint       -> equinox checkpointed scan, default ~log2(n_steps) checkpoints
    checkpointed_<N> -> checkpointed scan with N checkpoints
    """
    if grad == "none":
        return False, "checkpointed", None
    if grad == "reverse":
        return True, "reverse", None
    if grad == "checkpoint":
        return True, "checkpointed", None
    m = re.fullmatch(r"checkpointed_(\d+)", grad)
    if m:
        return True, "checkpointed", int(m.group(1))
    raise ValueError(f"Invalid --grad value: {grad!r}. Use none | reverse | checkpoint | checkpointed_<N>.")


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
    """Save result to parquet (process 0 only).

    With ``--shells-per-file N`` (N >= 1) on a batched, multi-shell result, ``out_path`` is treated
    as a *directory* and the lightcone is streamed N shells at a time into ``shell_{i:04d}.parquet``,
    so only N shells are all-gathered to host RAM at once. This avoids the host OOM of gathering the
    whole nside-2048 lightcone (e.g. 46 shells x ~402 MB at float64) onto every task. The per-chunk
    write reuses the field ``__getitem__`` slice (array + per-shell metadata) and the loop runs in
    lockstep on every rank, so each ``to_parquet`` stays a synchronized collective.

    Otherwise (flag unset, or a non-batched single field), the whole result is written to one parquet.
    """
    out_path = output if output is not None else args.output
    name = getattr(args, "name", None)
    if name is not None:
        result = result.replace(name=name)

    shells_per_file = int(getattr(args, "shells_per_file", 0) or 0)
    if shells_per_file >= 1 and result.is_batched():
        os.makedirs(out_path, exist_ok=True)
        n_shells = int(result.array.shape[0])
        for i in range(0, n_shells, shells_per_file):
            chunk = result[i : i + shells_per_file]
            chunk_path = os.path.join(out_path, f"shell_{i:04d}.parquet")
            jfli.io.Catalog(field=chunk, cosmology=cosmo).to_parquet(chunk_path)
        if jax.process_index() == 0:
            print(f"Saved {n_shells} shells (chunks of {shells_per_file}) to {out_path}/")
        return

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
    """Build the flat argument parser with --sim-mode."""
    from jax_fli.scripts.parser import (
        add_common_args,
        add_cosmo_args,
        add_distributed_args,
        add_integration_settings_args,
        add_lensing_args,
        add_output_target_args,
        add_simulation_settings_args,
        add_source_args,
    )

    p = ArgumentParser(prog="fli-simulate", description="jax_fli simulation pipeline CLI")
    add_common_args(p)
    add_distributed_args(p)
    add_simulation_settings_args(p)
    add_output_target_args(p)
    # Optional external initial condition: --ic-input (local) XOR --ic-repo + --ic-data-files (HF).
    # A single-row Catalog holding a WHITE DensityField -- the opposite convention to fli-infer's
    # --ic-input, which expects the COLORED delta and de-colors it. Nothing in the schema
    # distinguishes the two, so the source has to be the right one by construction.
    add_source_args(p, prefix="ic")
    # --sim-mode is required for fli-simulate and adds the 'lensing' choice (pm + Born -> kappa).
    add_integration_settings_args(p, sim_mode_default=None, sim_mode_choices=("lpt", "pm", "lensing"))
    add_lensing_args(p)
    add_cosmo_args(p)
    p.add_argument(
        "--grad",
        type=str,
        default="none",
        metavar="MODE",
        help="Differentiate the forward model w.r.t. initial conditions: 'none' (forward only), "
        "'reverse', 'checkpoint' (~log2(steps) checkpoints), or 'checkpointed_<N>' (N checkpoints). "
        "When set, the output becomes the IC-shaped gradient field (pm/lensing modes only).",
    )
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
    p.add_argument(
        "--shells-per-file",
        type=int,
        default=0,
        metavar="N",
        help="Stream a multi-shell lightcone to disk N shells per parquet file (default: 0 = one "
        "single file). When N>=1, --output is treated as a directory and one shell_NNNN.parquet is "
        "written per N-shell chunk, gathering only N shells to host RAM at a time (avoids OOM at "
        "large nside / many shells).",
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

    # --grad: valid spec, pm/lensing only, and reverse adjoint requires uniform a-stepping
    try:
        compute_grad, grad_adjoint, _ = _parse_grad(getattr(args, "grad", "none"))
    except ValueError as e:
        parser.error(str(e))
    if compute_grad:
        if args.sim_mode == "lpt":
            parser.error("--grad is not supported with --sim-mode lpt (use pm or lensing)")
        if grad_adjoint == "reverse":
            if getattr(args, "time_stepping", "a") != "a":
                parser.error("--grad reverse requires --time-stepping a (the reverse adjoint assumes uniform a-steps)")
            if getattr(args, "interp", "none") in ("onion", "telephoto"):
                parser.error(
                    "--grad reverse is not yet validated with --interp onion/telephoto (the onion / "
                    "telephoto tilers). Use --interp none — optionally with --drift-on-lightcone, which "
                    "keeps interp='none' (a DriftInterp) and leaves the reverse adjoint valid — or "
                    "--grad checkpoint / checkpointed_<N>."
                )


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
        "adjoint",
        "checkpoints",
        "compute_grad",
        "quadrature",
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
    adjoint="checkpointed",
    checkpoints=None,
    compute_grad=False,
    quadrature="midpoint",
):
    def _forward(ic):
        # LPT to particles snapshot at t0, then run NBody
        dx, p = jfli.lpt(
            cosmo,
            ic,
            ts=solver.t0,
            order=lpt_order,
            painting=jfli.PaintingOptions(target="particles"),
            paint_order=paint_order,
            gradient_order=gradient_order,
            laplace_fd=laplace_fd,
            dealiased=dealiased,
            exact_growth=exact_growth,
        )

        # Run NBody. adjoint / checkpoints select the gradient strategy used by --grad when this
        # forward model is differentiated w.r.t. the initial conditions (ignored for the pure
        # forward pass).
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
            adjoint=adjoint,
            checkpoints=checkpoints,
        )
        if sim_type == "pm":
            return lightcone

        # Run lensing (Born) -> convergence. Shear is a forward-model concern only (fli-infer);
        # fli-simulate emits density (pm) or convergence (born), never shear.
        if sim_type == "born":
            return jfli.born(
                cosmo, lightcone, nz_shear, min_z=min_z, max_z=max_z, n_integrate=n_integrate, quadrature=quadrature
            )
        raise ValueError(f"Unknown sim_type: {sim_type}")

    if not compute_grad:
        return _forward(initial_conditions)

    # --grad: differentiate a scalar machinery-benchmark loss L = 1/2 * sum(observable**2) w.r.t.
    # the initial-condition array, and return the IC-shaped gradient field. The IC array is the
    # sole differentiation target; the field's static metadata is carried over via .replace.
    def _loss(ic_array):
        observable = _forward(initial_conditions.replace(array=ic_array))
        return 0.5 * jnp.sum(jnp.square(observable.array))

    grad_array = jax.grad(_loss)(initial_conditions.array)
    return initial_conditions.replace(array=grad_array)


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
    compute_grad, grad_adjoint, grad_checkpoints = _parse_grad(args.grad)

    output_dir = os.path.dirname(args.output) or "."
    # Per-run log name from the output file (e.g. m512.parquet -> m512.args.log) so runs sharing a
    # directory don't clobber one another; append so it sits after the fli-launcher command.
    args_log_name = f"{os.path.splitext(os.path.basename(args.output))[0]}.args.log"
    _save_args_log(args, output_dir, f"fli-simulate {args.sim_mode}", filename=args_log_name, mode="a")

    mesh = tuple(args.mesh_size)
    px, py = args.pdim
    halo_size = (int(mesh[0] / px * args.halo_multiplier), int(mesh[1] / py * args.halo_multiplier))

    key = jax.random.key(args.seed)

    if args.ic_input or args.ic_repo:
        # External white field. gaussian_initial_conditions is just normal_field +
        # interpolate_initial_conditions, so this swaps one white field for another and nothing
        # downstream changes. The source is spectrally upsampled to `mesh` when it is smaller --
        # its modes are copied at the same integer wavevector and the rest drawn from --seed.
        rows = []
        for row in _resolve_source(args, prefix="ic").with_format("numpy"):
            rows.append(row)
            if len(rows) > 1:
                raise ValueError("The initial-condition source must contain exactly one row, but found more than one.")
        if not rows:
            raise ValueError("The initial-condition source contained no rows.")
        # Load REPLICATED unless the source already has the run's mesh. Applying the run's sharding
        # to a smaller source fails outright -- 832 % 256 != 0 raises IndivisibleError -- and even
        # where it divides, resample_white_field needs the source replicated for its eager fft3d.
        # (fli-infer can pass `sharding` because there the IC always has the run mesh.)
        ic_sharding = sharding if tuple(int(v) for v in rows[0]["mesh_size"]) == mesh else None
        ic_field = jfli.io.Catalog.from_dataset(rows[0], sharding=ic_sharding).field[0]
        if not isinstance(ic_field, jfli.DensityField):
            raise TypeError(f"The initial condition must be a DensityField, got {type(ic_field).__name__}.")
        if jax.process_index() == 0:
            print(f"Initial condition from {args.ic_input or args.ic_repo}: {tuple(ic_field.mesh_size)} -> {mesh}")
        # The catalog's own box_size is the SOURCE simulation's (e.g. CosmoGrid's 900 Mpc/h). It is
        # deliberately ignored: the realization is re-interpreted in THIS run's --box-size, so the
        # array -- not the DensityField -- is what gets passed on (the field overload would override
        # mesh_size/box_size with the source's).
        white = jfli.resample_white_field(ic_field.array, key, mesh, field_sharding=sharding)
        initial_field = jfli.interpolate_initial_conditions(
            white,
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
    else:
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
            "adjoint": grad_adjoint,
            "checkpoints": grad_checkpoints,
            "compute_grad": compute_grad,
            "quadrature": getattr(args, "quadrature", "simpson"),
        }

    if args.perf:
        try:
            from jax_hpc_profiler import JaxTimer
        except ImportError:
            print("Error: jax-hpc-profiler not found. Please install it to use --perf.", file=sys.stderr)
            sys.exit(1)

        # Positional indices of the run_lpt / run_simulations static_argnames listed above: the
        # timer re-jits with static_argnums, so every static_argname must appear here or it gets
        # traced (a traced bool then fails the inner jit's static-hash). Keep in sync with the sigs.
        if sim_type == "lpt":
            _static_argnums = (3, 4, 5, 6, 7, 9, 10, 11, 12, 13)
        else:
            # 22 = quadrature (appended last in run_simulations so earlier indices are stable)
            _static_argnums = (3, 4, 7, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22)
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
        name = getattr(args, "name", "")
        func_name = f"{sim_type}{nb_steps}_{name}"
        timer.report(report_file, function=func_name, extra_info=extra_info, **metadata)
        print(f"Performance report saved to {report_file}")
        # Keep the last timed result and fall through to the save below, so one --perf run yields BOTH
        # the perf CSV and the parquet output(s) (per-shell when --shells-per-file is set).
    else:
        result = jax.block_until_ready(run_fn(cosmo, initial_field, **run_kwargs))

    print("Simulation completed... saving results.")
    sync_global_devices("Done")
    if compute_grad:
        # The IC-shaped gradient inherits the initial field's dynamic metadata, some of which is
        # None (z_sources / comoving_centers / density_width). Populate zero placeholders so the
        # parquet serializer accepts it — these fields are meaningless for a gradient.
        result = result.replace(  # pyright: ignore
            z_sources=jnp.zeros((1,)),
            comoving_centers=jnp.zeros((1,)),
            density_width=jnp.zeros((1,)),
        )
    # --- Save output ---
    _save_result(result, cosmo, args)  # pyright: ignore
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
