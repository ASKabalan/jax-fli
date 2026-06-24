"""fli-dorian-rt: stack a density lightcone and ray-trace it with dorian via MPI.

Reads every matched density shell — a local glob (``--input``) or a HuggingFace dataset repo
(``--repo`` + ``--data-files``) — stacks them into ONE ``(S, npix)`` ``SphericalDensity`` lightcone
(optionally ud_grade-downsampled to ``--nside``), and ray-traces it through dorian **once**. Rank 0
writes a single ``SphericalKappaField`` parquet (the ray-traced κ), named from the lightcone's field
name; with ``--with-born`` it also writes the Born byproduct from the same dorian pass.

Replaces ``docs/5-experiments/00-cosmogrid-reference/raytrace_kappa.py`` for the post-processing path.
Each rank holds the full replicated lightcone in RAM (dorian is numpy+MPI) — use ``--nside`` to keep
it small.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path


def parser() -> ArgumentParser:
    """Build the argument parser for fli-dorian-rt."""
    from jax_fli.scripts.parser import (
        add_distributed_args,
        add_lensing_args,
        add_lensing_postproc_args,
        add_source_args,
    )

    p = ArgumentParser(
        prog="fli-dorian-rt",
        description="Stack a density lightcone and ray-trace it with dorian (MPI).",
    )
    add_source_args(p)
    add_lensing_postproc_args(p)
    p.add_argument("--name", default=None, help="Label stored as AbstractField.name inside the output catalog")
    p.add_argument(
        "--rt-interp",
        choices=["bilinear", "ngp", "nufft"],
        default="bilinear",
        help="Interpolation method for raytrace (default: bilinear)",
    )
    p.add_argument("--no-parallel-transport", action="store_true", help="Disable parallel transport in raytrace")
    p.add_argument(
        "--with-born",
        action="store_true",
        help="Also emit the Born convergence byproduct from the same dorian pass (default: ray-traced only)",
    )
    add_lensing_args(p)
    # dorian is single-process numpy+MPI (it gets its world from MPI.COMM_WORLD, not a JAX mesh), but
    # fli-launcher unconditionally appends --nodes/--gpus-per-node/--pdim to every payload. Accept and
    # ignore them here so dorian can be launched uniformly through fli-launcher like fli-born-rt.
    add_distributed_args(p)
    # Ray-tracing integrates through high-z shells, so default the n(z) ceiling to 3.0 (the reference
    # raytrace_kappa.py value), vs Born's 1.5.
    p.set_defaults(max_z=3.0)
    return p


def main() -> None:
    """CLI entry point registered as fli-dorian-rt."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    from jax_fli.io import Catalog
    from jax_fli.scripts._common import _load_lightcone, _resolve_nz_shear, _save_args_log

    args = parser().parse_args()
    nz_shear = _resolve_nz_shear(args)
    assert nz_shear is not None  # --nz-shear defaults to ['s3']; raytrace needs a source distribution

    import jax_fli as jfli

    # Every rank holds the full replicated lightcone (dorian is numpy + MPI).
    # WARNING _load_lightcone is not MPI friendly
    lightcone, cosmo = _load_lightcone(args)
    assert cosmo is not None

    kappa_rt, kappa_born = jfli.raytrace(
        cosmo,
        lightcone,
        nz_shear,
        min_z=args.min_z,
        max_z=args.max_z,
        n_integrate=args.n_integrate,
        interp=args.rt_interp,
        parallel_transport=not args.no_parallel_transport,
        born=args.with_born,
        raytrace=True,
        comm=comm,
        normalization=args.normalization,
    )

    # raytrace returns (None, None) on non-lead ranks; the lead holds the gathered maps and writes.
    if rank == 0:
        assert kappa_rt is not None
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_args_log(args, str(out_dir), "fli-dorian-rt")
        base = lightcone.name or f"M{lightcone.mesh_size[0]}_B{int(lightcone.box_size[0])}_N{lightcone.nside}"
        out_rt = out_dir / f"RAYTRACE_{base}.parquet"
        if args.name is not None:
            name_raytrace = f"{args.name} raytraced"
            kappa_rt = kappa_rt.replace(name=name_raytrace)
        Catalog(field=kappa_rt, cosmology=cosmo).to_parquet(str(out_rt))
        print(f"  Saved raytrace kappa {tuple(kappa_rt.array.shape)} → {out_rt}")
        if args.with_born and kappa_born is not None:
            if args.name is not None:
                name_born = f"{args.name} born"
                kappa_born = kappa_born.replace(name=name_born)
            out_born = out_dir / f"RAYTRACE_{base}_born.parquet"
            Catalog(field=kappa_born, cosmology=cosmo).to_parquet(str(out_born))
            print(f"  Saved Born byproduct {tuple(kappa_born.array.shape)} → {out_born}")


if __name__ == "__main__":
    main()
