"""fli-born-rt: post-process existing lightcone parquet files with Born lensing.

For each input parquet file, each row (cosmology run) is processed independently
and saved to BORN_<suffix>_row<N>.parquet in the output directory.
Memory is freed after each row to stay frugal on large grids.
"""

from __future__ import annotations

import os
from argparse import ArgumentParser
from pathlib import Path


def parser() -> ArgumentParser:
    """Build the argument parser for fli-born-rt."""
    from jax_fli.scripts.parser import add_distributed_args, add_lensing_args

    p = ArgumentParser(
        prog="fli-born-rt",
        description="Post-process lightcone parquet files with Born lensing.",
    )
    p.add_argument(
        "--input",
        required=True,
        metavar="FILE_OR_GLOB",
        help="Input parquet file(s) — single path or shell glob (e.g. 'results/*.parquet')",
    )
    p.add_argument("--output", "-o", default=".", metavar="DIR", help="Output directory (default: .)")
    p.add_argument("--enable-x64", action="store_true", help="Enable JAX 64-bit precision (default: False)")

    add_lensing_args(p)
    add_distributed_args(p)

    return p


def main() -> None:
    """CLI entry point registered as fli-born-rt."""
    import jax

    from jax_fli.io import Catalog
    from jax_fli.scripts._common import _build_sharding, _resolve_nz_shear

    p = parser()
    args = p.parse_args()
    jax.config.update("jax_enable_x64", args.enable_x64)
    sharding = _build_sharding(args)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    nz_shear = _resolve_nz_shear(args)

    import datasets

    import jax_fli as jfli

    min_z = args.min_z
    max_z = args.max_z
    n_integrate = args.n_integrate

    ds = datasets.load_dataset("parquet", data_files=args.input, split="train", streaming=True).with_format("numpy")
    row_count = 0
    for i, row in enumerate(ds):
        catalog = Catalog.from_dataset(row, sharding=sharding)
        field = catalog.field[0]
        cosmo = catalog.cosmology[0]

        print(f"  row {i}: field={type(field).__name__} cosmo=Oc={float(cosmo.Omega_c):.4f}")

        born_result = jax.block_until_ready(
            jfli.born(cosmo, field, nz_shear, min_z=min_z, max_z=max_z, n_integrate=n_integrate)
        )
        out_path = (
            output_dir / f"BORN_M_{field.mesh_size[0]}_B_{int(field.box_size[0])}_N_{field.nside}_row{i:04d}.parquet"
        )
        os.makedirs(out_path.parent, exist_ok=True)
        Catalog(field=born_result, cosmology=cosmo).to_parquet(str(out_path))
        print(f"    Saved Born kappa → {out_path}")
        del born_result, field, cosmo, catalog
        row_count += 1

    print(f"  Done: {row_count} row(s)")
    print("\nAll files processed.")


if __name__ == "__main__":
    main()
