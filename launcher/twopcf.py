"""2pcf subcommand — wraps fli-2pcf."""
from __future__ import annotations

from .parser import (
    add_lensing_args,
    add_slurm_args,
    dispatch,
)


def add_subparser(sub):
    p = sub.add_parser(
        "2pcf",
        help="Submit a fli-2pcf power-spectrum MCMC inference job",
    )
    add_slurm_args(p)
    add_lensing_args(p)

    g = p.add_argument_group("2pcf")
    g.add_argument("--observable", required=True, metavar="PATH", help="Parquet Catalog with observed C_ell")
    g.add_argument(
        "--path", default="results/2pcf_inference", help="Output directory (default: results/2pcf_inference)"
    )

    geom = p.add_mutually_exclusive_group()
    geom.add_argument("--nside", type=int, default=None, help="HEALPix NSIDE")
    geom.add_argument(
        "--flatsky-npix",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Flat-sky pixel resolution (H W)",
    )
    g.add_argument(
        "--field-size",
        type=float,
        nargs=2,
        default=None,
        metavar=("H_DEG", "W_DEG"),
        help="Angular field size in degrees (required for flat-sky)",
    )

    g.add_argument("--lmax", type=int, default=2047, help="Maximum multipole lmax (default: 2047)")
    g.add_argument("--f-sky", type=float, default=1.0, help="Sky fraction (default: 1.0)")
    g.add_argument("--sigma-e", type=float, default=0.26, help="Shape-noise dispersion (default: 0.26)")
    g.add_argument(
        "--nonlinear-fn",
        choices=["halofit", "linear"],
        default="halofit",
        help="Nonlinear power spectrum (default: halofit)",
    )
    g.add_argument("--chain-index", type=int, default=0, help="Chain index in output filenames (default: 0)")
    g.add_argument("--num-warmup", type=int, default=100, help="MCMC warmup iterations (default: 100)")
    g.add_argument("--num-samples", type=int, default=500, help="Samples per batch (default: 500)")
    g.add_argument("--batch-count", type=int, default=10, help="Number of sequential batches (default: 10)")
    g.add_argument("--sampler", choices=["NUTS", "HMC", "MCLMC"], default="NUTS", help="MCMC sampler (default: NUTS)")
    g.add_argument(
        "--backend", choices=["numpyro", "blackjax"], default="blackjax", help="Sampling backend (default: blackjax)"
    )
    g.add_argument("--sample", nargs="+", default=["cosmo"], help="Parameters to sample (default: cosmo)")
    g.add_argument(
        "--prior-omega-c", type=float, nargs=2, default=[0.1, 0.5], metavar=("MIN", "MAX"), dest="prior_omega_c"
    )
    g.add_argument(
        "--prior-sigma8", type=float, nargs=2, default=[0.6, 1.0], metavar=("MIN", "MAX"), dest="prior_sigma8"
    )
    g.add_argument("--prior-h", type=float, nargs=2, default=[0.5, 0.9], metavar=("MIN", "MAX"), dest="prior_h")
    g.add_argument("--seed", type=int, default=0, help="JAX PRNGKey seed (default: 0)")
    g.add_argument("--enable-x64", action="store_true", help="Enable JAX 64-bit precision")

    p.set_defaults(func=run)


def run(args):
    out_path = f"{args.path}/chain_{args.chain_index}"

    obs_name = args.observable.replace("/", "_").replace(".parquet", "")
    job_name = f"fli_2pcf_{obs_name}_s{args.seed}"

    fli_cmd = [
        "fli-2pcf",
        "--observable",
        args.observable,
        "--path",
        out_path,
    ]

    if args.nside is not None:
        fli_cmd += ["--nside", str(args.nside)]
    elif args.flatsky_npix is not None:
        fli_cmd += ["--flatsky-npix", str(args.flatsky_npix[0]), str(args.flatsky_npix[1])]
        if args.field_size is not None:
            fli_cmd += ["--field-size", str(args.field_size[0]), str(args.field_size[1])]

    fli_cmd += [
        "--lmax",
        str(args.lmax),
        "--f-sky",
        str(args.f_sky),
        "--sigma-e",
        str(args.sigma_e),
        "--nonlinear-fn",
        args.nonlinear_fn,
        "--nz-shear",
        *[str(v) for v in args.nz_shear],
        "--min-z",
        str(args.min_z),
        "--max-z",
        str(args.max_z),
        "--n-integrate",
        str(args.n_integrate),
        "--num-warmup",
        str(args.num_warmup),
        "--num-samples",
        str(args.num_samples),
        "--batch-count",
        str(args.batch_count),
        "--sampler",
        args.sampler,
        "--backend",
        args.backend,
        "--sample",
        *args.sample,
        "--prior-omega-c",
        str(args.prior_omega_c[0]),
        str(args.prior_omega_c[1]),
        "--prior-sigma8",
        str(args.prior_sigma8[0]),
        str(args.prior_sigma8[1]),
        "--prior-h",
        str(args.prior_h[0]),
        str(args.prior_h[1]),
        "--chain-index",
        str(args.chain_index),
        "--seed",
        str(args.seed),
    ]
    if args.enable_x64:
        fli_cmd.append("--enable-x64")

    dispatch(args, job_name, "FLI_2PCF", fli_cmd)
