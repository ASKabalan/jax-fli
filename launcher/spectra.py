"""spectra subcommand — wraps fli-spectra."""
from __future__ import annotations

from .parser import (
    add_slurm_args,
    dispatch,
)


def add_subparser(sub):
    p = sub.add_parser(
        "spectra",
        help="Submit a fli-spectra power-spectrum computation job",
    )
    add_slurm_args(p)

    g = p.add_argument_group("spectra")
    g.add_argument("--folder", required=True, help="Folder to scan for parquet files")
    g.add_argument(
        "--regex",
        default=r".*\.parquet$",
        metavar="PATTERN",
        help="Regex pattern to match parquet filenames (default: all .parquet files)",
    )
    g.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    g.add_argument("--force-regen", action="store_true", help="Force regeneration even if output exists")
    g.add_argument(
        "--normalization",
        choices=["global", "per_plane"],
        default="global",
        help="Overdensity normalization (default: global)",
    )
    # Flat-sky spectra
    g.add_argument("--ell-edges", type=float, nargs="+", default=None, metavar="E", help="Ell bin edges for flat-sky Cl")
    # Spherical spectra
    g.add_argument("--lmax", type=int, default=None, help="Max multipole lmax (default: 3*nside-1)")
    g.add_argument("--method", choices=["healpy", "jax"], default="healpy", help="SHT method (default: healpy)")
    # 3D P(k)
    g.add_argument("--kedges", type=float, nargs="+", default=None, metavar="K", help="k bin edges for P(k)")
    g.add_argument("--multipoles", type=int, nargs="+", default=[0], metavar="L", help="Multipole moments (default: 0)")
    g.add_argument(
        "--los",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 1.0],
        metavar=("LX", "LY", "LZ"),
        help="Line-of-sight direction (default: 0 0 1)",
    )
    # Common
    g.add_argument("--batch-size", type=int, default=None, help="Batch size for jax.lax.map (default: no batching)")
    g.add_argument("--enable-x64", action="store_true", help="Enable JAX 64-bit precision")

    p.set_defaults(func=run)


def run(args):
    fli_cmd = ["fli-spectra", args.folder]

    if args.regex != r".*\.parquet$":
        fli_cmd += ["--regex", args.regex]
    if args.recursive:
        fli_cmd.append("--recursive")
    if args.force_regen:
        fli_cmd.append("--force-regen")
    if args.normalization != "global":
        fli_cmd += ["--normalization", args.normalization]
    if args.ell_edges is not None:
        fli_cmd += ["--ell-edges", *[str(e) for e in args.ell_edges]]
    if args.lmax is not None:
        fli_cmd += ["--lmax", str(args.lmax)]
    if args.method != "healpy":
        fli_cmd += ["--method", args.method]
    if args.kedges is not None:
        fli_cmd += ["--kedges", *[str(k) for k in args.kedges]]
    if args.multipoles != [0]:
        fli_cmd += ["--multipoles", *[str(m) for m in args.multipoles]]
    if args.los != [0.0, 0.0, 1.0]:
        fli_cmd += ["--los", *[str(v) for v in args.los]]
    if args.batch_size is not None:
        fli_cmd += ["--batch-size", str(args.batch_size)]
    if args.enable_x64:
        fli_cmd.append("--enable-x64")

    job_name = f"fli_spectra_{args.folder.replace('/', '_')}"
    dispatch(args, job_name, "FLI_SPECTRA", fli_cmd, use_gpu=False)
