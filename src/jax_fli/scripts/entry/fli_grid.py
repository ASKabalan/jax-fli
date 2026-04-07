"""fli-grid: single-process Cartesian grid runner for fli-simulate.

Accepts list-valued parameters for the griddable dimensions and runs all
Cartesian combinations sequentially, reusing the same JAX JIT cache across
runs that share the same static shapes.

Range notation
--------------
Scalar griddable parameters (--Omega-c, --sigma8, --seed, --nb-shells,
--nside) accept either explicit values or a compact
``start:stop:step`` range token (stop is inclusive).  Both styles can be
mixed freely in the same argument.

    --Omega-c 0.2:0.4:0.01          # 21 values: 0.20, 0.21, …, 0.40
    --seed 0:9:1                     # 10 seeds: 0, 1, …, 9
    --sigma8 0.7 0.8:0.9:0.05       # explicit 0.7 + range 0.80, 0.85, 0.90

Example
-------
    fli-grid nbody \\
        --mesh-size 64 64 64 128 128 128 \\
        --box-size 200 200 200 \\
        --Omega-c 0.2:0.4:0.1 \\
        --sigma8 0.7:0.9:0.1 \\
        --seed 0:2:1 \\
        --nb-shells 10 \\
        --nb-steps 18 \\
        --nside 16 32 --output-dir /tmp/grid_out --dry-run
"""

from __future__ import annotations

import copy
from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expand_range_values(raw: list, dtype: type) -> list:
    """Expand a mixed list of plain scalars and ``start:stop:step`` range tokens.

    Each element of *raw* is either:

    * A plain scalar (string or already-converted number) → converted to *dtype*.
    * A ``"start:stop:step"`` string → expanded via ``np.arange``; *stop* is
      inclusive (uses a small epsilon to guard against floating-point rounding).

    Examples
    --------
    >>> _expand_range_values(["0.2:0.4:0.1"], float)
    [0.2, 0.3, 0.4]
    >>> _expand_range_values(["0:2:1"], int)
    [0, 1, 2]
    >>> _expand_range_values([0.8159, "0.9:1.0:0.05"], float)
    [0.8159, 0.9, 0.95, 1.0]
    """
    result: list = []
    for v in raw:
        s = str(v)
        if ":" in s:
            parts = s.split(":")
            if len(parts) != 3:
                raise ValueError(f"Range notation must be 'start:stop:step', got {v!r}")
            start, stop, step = float(parts[0]), float(parts[1]), float(parts[2])
            arr = np.arange(start, stop + step * 1e-9, step)
            result.extend(dtype(x) for x in arr)
        else:
            result.append(dtype(s))
    return result


def _parse_groups(values: list, group_size: int) -> list[tuple]:
    """Split a flat list into tuples of ``group_size``.

    e.g. [64, 64, 64, 128, 128, 128] with group_size=3
         → [(64, 64, 64), (128, 128, 128)]
    """
    if len(values) % group_size != 0:
        raise ValueError(f"Expected a multiple of {group_size} values, got {len(values)}: {values}")
    return [tuple(values[i : i + group_size]) for i in range(0, len(values), group_size)]


def _make_stem(
    subcommand: str, mesh, box, omega_c, sigma8, seed, nb_shells, nb_steps, nside, density_widths=None
) -> str:
    """Build a descriptive filename stem for a single grid combination."""
    mesh_str = "x".join(str(m) for m in mesh)
    box_str = "x".join(str(int(b)) if b == int(b) else str(b) for b in box)
    nside_tag = f"_nside{nside}" if nside is not None else ""
    dw_tag = f"_dw{density_widths}" if density_widths is not None else ""
    return f"{subcommand}_M{mesh_str}_B{box_str}_Oc{omega_c}_S8{sigma8}_s{seed}_Ns{nb_shells}_Nst{nb_steps}{nside_tag}{dw_tag}"


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parser() -> ArgumentParser:
    """Build the fli-grid argument parser."""
    from jax_fli.scripts.parser import (
        add_common_sim_args,
        add_cosmo_args,
        add_distributed_args,
        add_lensing_args,
        add_lightcone_args,
        add_mesh_args,
    )

    p = ArgumentParser(
        prog="fli-grid",
        description="Run a Cartesian grid of fli-simulate combinations.",
    )
    subparsers = p.add_subparsers(dest="subcommand", required=True)

    # Shared grid parent (used by all subcommands)
    grid_parent = ArgumentParser(add_help=False)

    add_mesh_args(grid_parent, nargs="+")  # flat list, groups of 3
    add_distributed_args(grid_parent)
    add_common_sim_args(grid_parent)
    add_lightcone_args(grid_parent)
    add_cosmo_args(grid_parent, sweep=True)  # range-notation strings for Omega-c/sigma8/seed

    # Griddable: shells and nside
    grid_parent.add_argument(
        "--nb-shells",
        type=str,
        nargs="+",
        default=[10],
        metavar="N",
        help="Number of lightcone shells; supports start:stop:step range notation",
    )
    grid_parent.add_argument(
        "--nside",
        type=str,
        nargs="+",
        default=None,
        metavar="NS",
        help="HEALPix NSIDE value(s); supports start:stop:step range notation",
    )

    # Painting targets (mutually exclusive; nside is handled separately above)
    paint_group = grid_parent.add_mutually_exclusive_group()
    paint_group.add_argument("--flatsky-npix", type=int, nargs=2, default=None, metavar=("H", "W"))
    paint_group.add_argument("--density", action="store_true", default=False)
    grid_parent.add_argument("--field-size", type=int, nargs=2, default=None, metavar=("H", "W"))

    # Output / control
    grid_parent.add_argument("--output-dir", default=".", metavar="DIR", help="Output directory (default: .)")
    grid_parent.add_argument("--dry-run", action="store_true", help="Print combinations without running")
    grid_parent.add_argument("--enable-x64", action="store_true")

    grid_parent.set_defaults(nb_steps=19, t0=0.1)

    # Subcommands
    subparsers.add_parser("lpt", parents=[grid_parent], help="Grid over LPT runs")
    subparsers.add_parser("nbody", parents=[grid_parent], help="Grid over NBody runs")
    lensing_p = subparsers.add_parser("lensing", parents=[grid_parent], help="Grid over full lensing pipeline runs")
    add_lensing_args(lensing_p)

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point registered as fli-grid."""
    import jax
    import jax.numpy as jnp

    from jax_fli.scripts._common import _build_sharding, _resolve_nz_shear
    from jax_fli.scripts.entry.fli_simulate import (
        _build_cosmo,
        _build_painting,
        _build_solver,
        _resolve_ts,
        _save_result,
        run_lpt,
        run_simulations,
    )

    p = parser()
    args = p.parse_args()
    jax.config.update("jax_enable_x64", args.enable_x64)

    # --- Expand range notation for griddable scalar parameters ---
    args.Omega_c = _expand_range_values(args.Omega_c, float)
    args.sigma8 = _expand_range_values(args.sigma8, float)
    args.seed = _expand_range_values(args.seed, int)
    args.nb_shells = _expand_range_values(args.nb_shells, int)

    # Expand nside if provided
    if args.nside is not None:
        nside_values = _expand_range_values(args.nside, int)
    else:
        nside_values = [None]

    # Validate: cannot use both nside and flatsky-npix/field-size/density
    if nside_values != [None]:
        if args.flatsky_npix is not None or args.field_size is not None or args.density:
            p.error("--nside cannot be combined with --flatsky-npix, --field-size, or --density")

    # --- Expand griddable parameters ---
    mesh_configs = _parse_groups(args.mesh_size, 3)
    box_configs = _parse_groups(args.box_size, 3)

    # Each --density-widths value is a scalar broadcast to all shells; None = use defaults
    density_width_values = args.density_widths if args.density_widths is not None else [None]

    # Compute total from dimension sizes, then iterate lazily
    total = (
        len(mesh_configs)
        * len(box_configs)
        * len(args.Omega_c)
        * len(args.sigma8)
        * len(args.seed)
        * len(args.nb_shells)
        * len(nside_values)
        * len(density_width_values)
    )
    grid = product(
        mesh_configs,
        box_configs,
        args.Omega_c,
        args.sigma8,
        args.seed,
        args.nb_shells,
        nside_values,
        density_width_values,
    )
    print(f"Grid: {total} combination(s) — subcommand={args.subcommand}")
    if args.dry_run:
        print("Dry run — combinations:")

    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve nz_shear once (shared across all combos; only valid for lensing subcommand)
    nz_shear = _resolve_nz_shear(args) if args.subcommand == "lensing" else None

    # Sharding from fixed args (shared across all combos)
    sharding = _build_sharding(args)

    import jax_fli as jfli

    for idx, (mesh, box, omega_c, sigma8, seed, nb_shells, nside_val, density_width_val) in enumerate(grid):
        # Build combo Namespace by shallow-copying fixed args and overriding grid dims
        combo = copy.copy(args)
        combo.mesh_size = list(mesh)
        combo.box_size = list(box)
        combo.Omega_c = omega_c
        combo.sigma8 = sigma8
        combo.seed = seed
        combo.nb_shells = nb_shells
        # Set nside for this combo (used by _build_painting)
        combo.nside = nside_val

        stem = _make_stem(
            args.subcommand, mesh, box, omega_c, sigma8, seed, nb_shells, args.nb_steps, nside_val, density_width_val
        )

        print(f"[{idx + 1}/{total}] {stem}")
        if args.dry_run:
            continue

        # Build objects for this combo
        cosmo = _build_cosmo(combo)
        painting, nside, flatsky_npix = _build_painting(combo)
        ts = _resolve_ts(combo)
        solver = _build_solver(combo, painting)

        px, py = args.pdim
        halo_size = (int(mesh[0] / px * args.halo_multiplier), int(mesh[1] / py * args.halo_multiplier))

        key = jax.random.key(seed)
        initial_field = jfli.gaussian_initial_conditions(
            key,
            tuple(mesh),
            tuple(box),
            observer_position=tuple(combo.observer_position),
            cosmo=cosmo,
            nside=combo.nside,
            flatsky_npix=tuple(combo.flatsky_npix) if combo.flatsky_npix is not None else None,
            field_size=tuple(combo.field_size) if combo.field_size is not None else None,
            field_sharding=sharding,
            halo_size=halo_size,
        )

        lpt_order = combo.lpt_order
        shell_spacing = getattr(combo, "shell_spacing", "comoving")
        density_widths_val = jnp.array(density_width_val) if density_width_val is not None else None
        # ts and nb_shells are mutually exclusive in resolve_geometry
        effective_nb_shells = nb_shells if ts is None else None

        if args.subcommand == "lpt":
            # LPT mode: use run_lpt with geometry params
            lpt_ts = ts if ts is not None else (combo.t0 if effective_nb_shells is None else None)
            run_fn = run_lpt
            run_kwargs = {
                "cosmo": cosmo,
                "initial_conditions": initial_field,
                "ts": lpt_ts,
                "lpt_order": lpt_order,
                "painting": painting,
                "nb_shells": effective_nb_shells,
                "shell_spacing": shell_spacing,
                "min_width": getattr(combo, "min_width", 50.0),
                "density_widths": density_widths_val,
                "gradient_order": getattr(combo, "gradient_order", 1),
                "laplace_fd": getattr(combo, "laplace_fd", False),
                "dealiased": getattr(combo, "dealiased", False),
                "exact_growth": getattr(combo, "exact_growth", False),
            }
        else:
            sim_type = "born" if args.subcommand == "lensing" else args.subcommand
            run_fn = run_simulations
            run_kwargs = {
                "cosmo": cosmo,
                "initial_conditions": initial_field,
                "solver": solver,
                "lpt_order": lpt_order,
                "sim_type": sim_type,
                "nz_shear": nz_shear,
                "ts": ts,
                "nb_shells": effective_nb_shells,
                "density_widths": density_widths_val,
                "gradient_order": getattr(combo, "gradient_order", 1),
                "laplace_fd": getattr(combo, "laplace_fd", False),
                "dealiased": getattr(combo, "dealiased", False),
                "exact_growth": getattr(combo, "exact_growth", False),
                "min_z": getattr(combo, "min_z", 0.01),
                "max_z": getattr(combo, "max_z", 1.5),
                "n_integrate": getattr(combo, "n_integrate", 32),
            }

        result = jax.block_until_ready(run_fn(**run_kwargs))

        out_path = output_dir / f"{stem}.parquet"
        _save_result(result, cosmo, combo, output=str(out_path))
        del result

        del cosmo, initial_field, solver, painting, ts

    if not args.dry_run:
        print(f"\nAll {total} combination(s) completed.")
    else:
        print(f"\n{total} combination(s) listed (dry run — nothing executed).")


if __name__ == "__main__":
    main()
