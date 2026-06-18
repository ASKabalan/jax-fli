"""fli-summary-stats: compute summary statistics from lightcone/density parquet catalogs.

For each matched parquet file, the catalog is loaded, the field is converted to OVERDENSITY (for
density types), the requested summary statistic is computed, and the result is saved next to the
original file with a "summary_stats_" prefix.

Two-point statistics (the default) by field type:
  - FlatDensity / FlatKappaField            → angular_cl (flat-sky)
  - SphericalDensity / SphericalKappaField  → angular_cl (HEALPix; optional footprint mask)
  - DensityField                            → power (3D P(k))

Spherical maps additionally support an apodized footprint mask (``--mask``) derived from the
observer position (``infer_from_observer_position``), the DES Y3 footprint (``des_y3``), or a
HEALPix map; with a mask the spherical C_ell is the mode-decoupled (MCM) bandpower estimate.

Kappa fields skip the OVERDENSITY unit conversion (they are already in dimensionless convergence
units appropriate for Cl computation).
"""

from __future__ import annotations

import re
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

# ---- Field types that need OVERDENSITY conversion -------------------------
_DENSITY3D_TYPE = {"DensityField"}
_DENSITY_TYPES = {"FlatDensity", "SphericalDensity"}
# ---- Kappa/shear types — skip unit conversion -----------------------------
_KAPPA_TYPES = {"FlatKappaField", "SphericalKappaField"}
# ---- Spherical (HEALPix) types — support the footprint mask ---------------
_SPHERICAL_TYPES = {"SphericalDensity", "SphericalKappaField"}
# ---- All supported types --------------------------------------------------
_SUPPORTED_TYPES = _DENSITY_TYPES | _KAPPA_TYPES | _DENSITY3D_TYPE

# Output files are written next to the input with this prefix; re-runs skip their own outputs.
_OUTPUT_PREFIX = "summary_stats_"


def parser() -> ArgumentParser:
    """Build the argument parser for fli-summary-stats."""
    from jax_fli.scripts.parser import (
        add_common_args,
        add_summary_stats_common_args,
        add_summary_stats_density_args,
        add_summary_stats_flat_args,
        add_summary_stats_mask_args,
        add_summary_stats_scan_args,
        add_summary_stats_spherical_args,
    )

    p = ArgumentParser(
        prog="fli-summary-stats",
        description=(
            "Compute summary statistics from lightcone/density parquet catalogs.\n"
            "\n"
            "Scans a folder for .parquet files (optionally filtered by regex),\n"
            "computes the requested summary statistic for each field type, and\n"
            "saves the result next to the original file with a 'summary_stats_' prefix.\n"
            "Spherical maps support an apodized footprint mask (observer position / DES Y3 / file)."
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )
    add_summary_stats_scan_args(p)
    add_summary_stats_flat_args(p)
    add_summary_stats_spherical_args(p)
    add_summary_stats_density_args(p)
    add_summary_stats_mask_args(p)
    add_summary_stats_common_args(p)
    add_common_args(p)
    return p


def _find_parquet_files(folder: str, regex: str, recursive: bool, force_regen: bool) -> list[Path]:
    """Scan *folder* for parquet files whose **name** matches *regex*.

    Files already starting with ``summary_stats_`` are silently skipped so that re-running the
    tool does not process its own outputs.
    """
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {folder}")
    pattern = re.compile(regex)
    glob_fn = root.rglob if recursive else root.glob
    files = []
    for p in sorted(glob_fn("*")):
        if p.is_file() and pattern.match(p.name) and not p.name.startswith(_OUTPUT_PREFIX):
            # if the same file name exists with the output prefix, skip to avoid re-processing
            if (p.parent / f"{_OUTPUT_PREFIX}{p.name}").exists() and not force_regen:
                print(f"  SKIP (output exists): {p.name}")
            else:
                files.append(p)
    return files


def _convert_to_overdensity(field, field_type: str, normalization: str):
    """Convert a density field to OVERDENSITY units if needed.

    Kappa fields are returned as-is (they are already in convergence units
    that are the correct physical quantity for Cl computation).
    """
    if field_type not in _DENSITY_TYPES:
        return field
    from jax_fli.fields.units import DensityUnit

    try:
        return field.to(DensityUnit.OVERDENSITY, normalization=normalization)
    except Exception as e:
        print(f"    WARNING: OVERDENSITY conversion failed ({e}). Proceeding with original unit.")
        return field


def _resolve_mask_for_field(field, field_type: str, args):
    """Resolve the apodized footprint mask for a spherical field (else None).

    The observer position comes from ``--observer-position`` when given, otherwise from the
    field's stored metadata. Flat / 3D fields never carry a footprint mask.
    """
    if field_type not in _SPHERICAL_TYPES:
        return None
    from jax_fli.scripts._common import _resolve_summary_stats_mask

    observer_position = (
        tuple(args.observer_position) if args.observer_position is not None else tuple(field.observer_position)
    )
    return _resolve_summary_stats_mask(args.mask, field.nside, observer_position, args.apodization_scale_deg)


def _compute_summary_stats(field, field_type: str, args, mask=None):
    """Route to the appropriate summary-statistic method based on field type."""
    import jax.numpy as jnp
    import numpy as np

    ell_edges = jnp.array(args.ell_edges) if args.ell_edges is not None else None
    kedges = jnp.array(args.kedges) if args.kedges is not None else None

    if field_type in ("FlatDensity", "FlatKappaField"):
        return field.angular_cl(
            field_size=field.field_size,
            ell_edges=ell_edges,
            batch_size=args.batch_size,
        )
    elif field_type in ("SphericalDensity", "SphericalKappaField"):
        # TODO(summary-stats): also dispatch PDF / peak counts / starlets (spherical maps only) to
        # SphericalDensity.compute_pdf / compute_peak_counts / starlet_coefficients once the
        # jax_fli.summary_statistics module lands on this branch (it currently lives, uncommitted,
        # in the summary-stat worktree). For now only the angular C_ell — optionally restricted to
        # the apodized footprint mask and MCM-decoupled — is computed.
        return field.angular_cl(
            lmax=args.lmax,
            method=args.method,
            batch_size=args.batch_size,
            mask=mask,
        )
    elif field_type == "DensityField":
        if args.kedges is not None and args.dk is not None:
            raise ValueError("Cannot specify both kedges and dk. Please choose one.")
        compensate_order = getattr(args, "compensate_order", None)
        shotnoise = None
        sn_order = getattr(args, "shotnoise_order", None)
        if sn_order is not None:
            # mean number density for one particle per mesh cell
            nbar = float(np.prod(np.asarray(field.mesh_size)) / np.prod(np.asarray(field.box_size)))
            shotnoise = (sn_order, nbar)
        return field.power(
            kedges=kedges,
            dk=args.dk,
            kmax=args.kmax,
            multipoles=tuple(args.multipoles),
            los=tuple(args.los),
            batch_size=args.batch_size,
            compensate_order=compensate_order,
            shotnoise=shotnoise,
        )
    else:
        raise ValueError(f"Unsupported field type for summary statistics: {field_type!r}")


def main() -> None:
    """CLI entry point registered as fli-summary-stats."""
    import jax

    from jax_fli.io import Catalog

    p = parser()
    args = p.parse_args()
    jax.config.update("jax_enable_x64", args.enable_x64)

    # ------------------------------------------------------------------
    # Discover files
    # ------------------------------------------------------------------
    files = _find_parquet_files(args.folder, args.regex, args.recursive, args.force_regen)
    if not files:
        print("No matching parquet files found.")
        return

    print(f"Found {len(files)} file(s) to process.")

    errors: list[tuple[Path, str]] = []

    for path in files:
        print(f"\n[{path.parent.name}/{path.name}]")
        try:
            catalog = Catalog.from_parquet(str(path))
        except Exception as e:
            msg = f"  ERROR loading: {e}"
            print(msg)
            errors.append((path, str(e)))
            continue

        field = catalog.field[0]
        import numpy as np

        field = field.apply_fn(
            lambda x: np.asarray(x, dtype=np.float32)
        )  # Ensure array is in numpy for JAX compatibility
        cosmo = catalog.cosmology[0]
        field_type = type(field).__name__

        print(f"  type: {field_type}  unit: {field.unit}")

        if field_type not in _SUPPORTED_TYPES:
            msg = f"  SKIP: unsupported field type '{field_type}'"
            print(msg)
            continue

        # Unit conversion
        field = _convert_to_overdensity(field, field_type, args.normalization)

        # Footprint mask (spherical only; None for flat/3D and for a centered observer)
        try:
            mask = _resolve_mask_for_field(field, field_type, args)
        except Exception as e:
            msg = f"  ERROR resolving mask: {e}"
            print(msg)
            errors.append((path, str(e)))
            continue
        if mask is not None:
            print(f"  mask: applied ('{args.mask}', apodization {args.apodization_scale_deg} deg)")

        # Compute summary statistics
        print("  Computing summary statistics...")
        try:
            ps = jax.block_until_ready(_compute_summary_stats(field, field_type, args, mask=mask))
        except Exception as e:
            msg = f"  ERROR computing summary statistics: {e}"
            print(msg)
            errors.append((path, str(e)))
            continue

        print(f"  Result: {type(ps).__name__}  shape={ps.array.shape}  unit={ps.unit}")

        # Save
        out_path = path.parent / f"{_OUTPUT_PREFIX}{path.name}"
        try:
            Catalog(ps, cosmo).to_parquet(str(out_path))
            print(f"  Saved: {out_path}")
        except Exception as e:
            msg = f"  ERROR saving: {e}"
            print(msg)
            errors.append((path, str(e)))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    n_ok = len(files) - len(errors)
    print(f"Done: {n_ok}/{len(files)} files processed successfully.")
    if errors:
        print(f"Errors ({len(errors)}):")
        for p, e in errors:
            print(f"  {p.name}: {e}")


if __name__ == "__main__":
    main()
