"""fli-spectra: compute power spectra from lightcone/density parquet catalogs.

For each matched parquet file, the catalog is loaded, the field is converted to
OVERDENSITY (for density types), spectra are computed with the appropriate method,
and saved next to the original file with a "spectra_" prefix.

Supported field types:
  - FlatDensity / FlatKappaField  → angular_cl (flat-sky)
  - SphericalDensity / SphericalKappaField  → angular_cl (HEALPix)
  - DensityField  → power (3D P(k))

Kappa fields skip the OVERDENSITY unit conversion (they are already in
dimensionless convergence units appropriate for Cl computation).
"""

from __future__ import annotations

import re
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path


# ---- Field types that need OVERDENSITY conversion -------------------------
_DENSITY_TYPES = {"FlatDensity", "SphericalDensity", "DensityField"}
# ---- Kappa/shear types — skip unit conversion -----------------------------
_KAPPA_TYPES = {"FlatKappaField", "SphericalKappaField"}
# ---- All supported types --------------------------------------------------
_SUPPORTED_TYPES = _DENSITY_TYPES | _KAPPA_TYPES


def parser() -> ArgumentParser:
    """Build the argument parser for fli-spectra."""
    from jax_fli.scripts.parser import (
        add_spectra_common_args,
        add_spectra_density_args,
        add_spectra_flat_args,
        add_spectra_scan_args,
        add_spectra_spherical_args,
    )

    p = ArgumentParser(
        prog="fli-spectra",
        description=(
            "Compute power spectra from lightcone/density parquet catalogs.\n"
            "\n"
            "Scans a folder for .parquet files (optionally filtered by regex),\n"
            "computes the appropriate power spectrum for each field type, and\n"
            "saves the result next to the original file with a 'spectra_' prefix."
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )
    add_spectra_scan_args(p)
    add_spectra_flat_args(p)
    add_spectra_spherical_args(p)
    add_spectra_density_args(p)
    add_spectra_common_args(p)
    return p


def _find_parquet_files(folder: str, regex: str, recursive: bool) -> list[Path]:
    """Scan *folder* for parquet files whose **name** matches *regex*.

    Files already starting with ``spectra_`` are silently skipped so that
    re-running the tool does not process its own outputs.
    """
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {folder}")
    pattern = re.compile(regex)
    glob_fn = root.rglob if recursive else root.glob
    files = []
    for p in sorted(glob_fn("*")):
        if p.is_file() and pattern.match(p.name) and not p.name.startswith("spectra_"):
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


def _compute_spectra(field, field_type: str, args):
    """Route to the appropriate spectra method based on field type."""
    import jax.numpy as jnp

    ell_edges = jnp.array(args.ell_edges) if args.ell_edges is not None else None
    kedges = jnp.array(args.kedges) if args.kedges is not None else None

    if field_type in ("FlatDensity", "FlatKappaField"):
        return field.angular_cl(
            field_size=field.field_size,
            ell_edges=ell_edges,
            batch_size=args.batch_size,
        )
    elif field_type in ("SphericalDensity", "SphericalKappaField"):
        return field.angular_cl(
            lmax=args.lmax,
            method=args.method,
            batch_size=args.batch_size,
        )
    elif field_type == "DensityField":
        return field.power(
            kedges=kedges,
            multipoles=tuple(args.multipoles),
            los=tuple(args.los),
            batch_size=args.batch_size,
        )
    else:
        raise ValueError(f"Unsupported field type for spectra: {field_type!r}")


def main() -> None:
    """CLI entry point registered as fli-spectra."""
    import jax

    from jax_fli.io import Catalog

    p = parser()
    args = p.parse_args()
    jax.config.update("jax_enable_x64", args.enable_x64)

    # ------------------------------------------------------------------
    # Discover files
    # ------------------------------------------------------------------
    files = _find_parquet_files(args.folder, args.regex, args.recursive)
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
        field = field.apply_fn(lambda x: np.asarray(x, dtype=np.float32))  # Ensure array is in numpy for JAX compatibility
        cosmo = catalog.cosmology[0]
        field_type = type(field).__name__

        print(f"  type: {field_type}  unit: {field.unit}")

        if field_type not in _SUPPORTED_TYPES:
            msg = f"  SKIP: unsupported field type '{field_type}'"
            print(msg)
            continue

        # Unit conversion
        field = _convert_to_overdensity(field, field_type, args.normalization)

        # Compute spectra
        print("  Computing spectra...")
        try:
            import jax
            ps = jax.block_until_ready(_compute_spectra(field, field_type, args))
        except Exception as e:
            msg = f"  ERROR computing spectra: {e}"
            print(msg)
            errors.append((path, str(e)))
            continue

        print(f"  Result: {type(ps).__name__}  shape={ps.array.shape}  unit={ps.unit}")

        # Save
        out_path = path.parent / f"spectra_{path.name}"
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
    print(f"\n{'='*60}")
    n_ok = len(files) - len(errors)
    print(f"Done: {n_ok}/{len(files)} files processed successfully.")
    if errors:
        print(f"Errors ({len(errors)}):")
        for p, e in errors:
            print(f"  {p.name}: {e}")


if __name__ == "__main__":
    main()
