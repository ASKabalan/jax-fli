"""fli-extract: stream MCMC catalogs and compute per-chain statistics."""

from __future__ import annotations

import argparse

import jax

from jax_fli.io.catalog import Catalog
from jax_fli.io.extract import extract_catalog
from jax_fli.scripts._common import _build_sharding, _resolve_chain_sources


def parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for fli-extract."""
    from jax_fli.scripts.parser import add_common_args, add_distributed_args, add_source_args

    p = argparse.ArgumentParser(
        prog="fli-extract",
        description="Stream MCMC catalog parquet files and compute per-chain statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Source: the generic add_source_args builder in multi-pattern mode — local --input glob(s)/dir(s)
    # XOR HuggingFace --repo + --data-files glob(s). Each pattern is ONE MCMC chain (so per-chain
    # statistics stay separate); a single local --input root dir auto-expands its chain_N/ subdirs.
    add_source_args(p, multi=True)

    p.add_argument("--name", type=str, required=True, metavar="NAME", help="Name label for the CatalogExtract.")
    p.add_argument("--output", type=str, required=True, metavar="PATH", help="Output parquet file path.")
    p.add_argument(
        "--cosmo-keys",
        nargs="+",
        required=True,
        metavar="KEY",
        help="Cosmological parameter names to extract (e.g. Omega_c sigma8).",
    )
    p.add_argument("--truth", type=str, default=None, metavar="PATH", help="Parquet path of a truth Catalog.")
    p.add_argument(
        "--field-statistic", action="store_true", help="Compute per-chain mean and std of the density fields."
    )
    p.add_argument(
        "--power-statistic", action="store_true", help="Compute per-chain transfer function and coherence spectra."
    )
    p.add_argument("--ddof", type=int, default=0, help="Delta degrees of freedom for std computation (default: 0)")

    add_common_args(p)
    add_distributed_args(p)

    return p


def main() -> None:
    """CLI entry point registered as fli-extract."""
    p = parser()
    args = p.parse_args()

    jax.config.update("jax_enable_x64", args.enable_x64)

    if args.input is None and args.repo is None:
        p.error("a source is required: --input (local glob[s]/dir) or --repo + --data-files (HuggingFace).")
    if args.repo is not None and not args.data_files:
        p.error("--data-files is required when --repo is set (one glob per chain).")
    if args.power_statistic and args.truth is None:
        p.error("--truth is required when --power-statistic is set.")

    repo, patterns = _resolve_chain_sources(args)

    sharding = _build_sharding(args)

    truth = None
    if args.truth is not None:
        truth = Catalog.from_parquet(args.truth, sharding=sharding)

    ce = extract_catalog(
        cosmo_keys=args.cosmo_keys,
        set_name=args.name,
        patterns=patterns,
        repo=repo,
        truth=truth,
        field_statistic=args.field_statistic,
        power_statistic=args.power_statistic,
        ddof=args.ddof,
        sharding=sharding,
    )
    if args.field_statistic and ce.mean_field is not None and jax.process_index() == 0:
        print(f"sharding of mean fields {ce.mean_field.array.sharding}")

    ce.to_parquet(args.output)
    if jax.process_index() == 0:
        print(f"Saved CatalogExtract '{ce.name}' with {ce.n_chains} chains to {args.output}")


if __name__ == "__main__":
    main()
