"""fli-born-rt: stack a density lightcone and post-process it with Born lensing.

Reads every matched density shell — a local glob (``--input``) or a HuggingFace dataset repo
(``--repo`` + ``--data-files``) — stacks them into ONE ``(S, npix)`` ``SphericalDensity`` lightcone
(optionally ud_grade-downsampled to ``--nside``), and runs the differentiable Born approximation
**once**, writing a single ``SphericalKappaField`` parquet named from the lightcone's field name.

Replaces ``docs/5-experiments/00-cosmogrid-reference/born_kappa.py`` for the post-processing path.
Distributed Born is supported via ``--pdim`` (the lightcone is sharded ``P("x","y")``).
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path


def parser() -> ArgumentParser:
    """Build the argument parser for fli-born-rt."""
    from jax_fli.scripts.parser import (
        add_common_args,
        add_distributed_args,
        add_lensing_args,
        add_lensing_postproc_args,
        add_source_args,
    )

    p = ArgumentParser(
        prog="fli-born-rt",
        description="Stack a density lightcone and post-process it with Born lensing.",
    )
    p.add_argument("--name", default=None, help="Label stored as AbstractField.name inside the output catalog")

    add_source_args(p)
    add_lensing_postproc_args(p)
    add_common_args(p)
    add_lensing_args(p)
    add_distributed_args(p)
    return p


def main() -> None:
    """CLI entry point registered as fli-born-rt."""
    import jax

    from jax_fli.io import Catalog
    from jax_fli.scripts._common import _build_sharding, _load_lightcone, _resolve_nz_shear, _save_args_log

    args = parser().parse_args()
    jax.config.update("jax_enable_x64", args.enable_x64)
    sharding = _build_sharding(args)
    nz_shear = _resolve_nz_shear(args)
    assert nz_shear is not None  # --nz-shear defaults to ['s3']; Born needs a source distribution
    lead = jax.process_index() == 0

    import jax_fli as jfli

    lightcone, cosmo = _load_lightcone(args, sharding=sharding)
    assert cosmo is not None
    if lead:
        print(
            f"  density {type(lightcone).__name__} {tuple(lightcone.array.shape)} nside={lightcone.nside} "
            f"| n(z)={len(nz_shear)} bin(s), normalization={args.normalization}, quadrature={args.quadrature}"
        )

    kappa = jax.block_until_ready(
        jfli.born(
            cosmo,
            lightcone,
            nz_shear,
            min_z=args.min_z,
            max_z=args.max_z,
            n_integrate=args.n_integrate,
            normalization=args.normalization,
            quadrature=args.quadrature,
        )
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = lightcone.name or f"M{lightcone.mesh_size[0]}_B{int(lightcone.box_size[0])}_N{lightcone.nside}"
    out_path = out_dir / f"BORN_{base}.parquet"
    if lead:
        _save_args_log(args, str(out_dir), "fli-born-rt")
    # to_parquet is collective (process_allgather); only the lead rank writes the file.
    if args.name is not None:
        kappa = kappa.replace(name=args.name)
    Catalog(field=kappa, cosmology=cosmo).to_parquet(str(out_path))
    if lead:
        print(f"  Saved Born {kappa.name} {tuple(kappa.array.shape)} → {out_path}")


if __name__ == "__main__":
    main()
