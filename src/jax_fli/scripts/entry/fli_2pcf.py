"""fli-2pcf: 2-point-function (power-spectrum level) MCMC inference.

Loads pre-computed angular power spectra (C_ell) from a parquet Catalog,
builds a Knox-formula Gaussian likelihood, and runs batched MCMC to infer
cosmological parameters.

This is the computationally cheap complement to fli-infer: the heavy simulation
forward model is replaced by a jax_cosmo Limber integral, making warmup and
sampling orders of magnitude faster.
"""

from __future__ import annotations

import argparse

import jax
import jax_cosmo as jc
from numpyro.handlers import condition

import jax_fli as jfli
from jax_fli.scripts._common import _resolve_nz_shear, _save_args_log


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for fli-2pcf."""
    from jax_fli.scripts.parser import (
        add_lensing_args,
        add_prior_args,
    )

    p = argparse.ArgumentParser(
        prog="fli-2pcf",
        description=(
            "Run 2-point-function (power-spectrum level) MCMC inference "
            "conditioned on observed angular power spectra."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required args
    p.add_argument(
        "--observable",
        type=str,
        required=True,
        metavar="PATH",
        help="Parquet Catalog containing a PowerSpectrum field with observed C_ell.",
    )
    p.add_argument(
        "--path",
        type=str,
        required=True,
        metavar="PATH",
        help="Output directory for MCMC checkpoints and parquet catalogs.",
    )

    # Geometry (mutually exclusive)
    geom_group = p.add_mutually_exclusive_group()
    geom_group.add_argument(
        "--nside",
        type=int,
        default=None,
        help="HEALPix NSIDE (spherical geometry).",
    )
    geom_group.add_argument(
        "--flatsky-npix",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Flat-sky pixel resolution (height width).",
    )
    p.add_argument(
        "--field-size",
        type=float,
        nargs=2,
        default=None,
        metavar=("H_DEG", "W_DEG"),
        help="Angular field size in degrees (required for flat-sky geometry).",
    )

    # Power-spectrum config
    p.add_argument(
        "--lmax",
        type=int,
        default=2047,
        help="Maximum multipole lmax for C_ell computation (default: 2047).",
    )
    p.add_argument(
        "--f-sky",
        type=float,
        default=1.0,
        dest="f_sky",
        help="Sky fraction for Gaussian covariance (default: 1.0).",
    )
    p.add_argument(
        "--sigma-e",
        type=float,
        default=0.26,
        dest="sigma_e",
        help="Shape-noise dispersion per component (default: 0.26).",
    )
    p.add_argument(
        "--nonlinear-fn",
        choices=["halofit", "linear"],
        default="halofit",
        dest="nonlinear_fn",
        help="Nonlinear power spectrum function (default: halofit).",
    )

    # Shared arg groups
    add_lensing_args(p)
    add_prior_args(p)

    # MCMC
    p.add_argument("--chain-index", type=int, default=0, dest="chain_index",
                   help="Chain index written into output filenames (default: 0)")
    p.add_argument("--num-warmup", type=int, default=100, dest="num_warmup",
                   help="MCMC warmup iterations (default: 100)")
    p.add_argument("--num-samples", type=int, default=500, dest="num_samples",
                   help="Samples per batch (default: 500)")
    p.add_argument("--batch-count", type=int, default=10, dest="batch_count",
                   help="Number of sequential batches (default: 10)")
    p.add_argument(
        "--sampler", choices=["NUTS", "HMC", "MCLMC"], default="NUTS",
        help="MCMC sampler (default: NUTS)"
    )
    p.add_argument(
        "--backend", choices=["numpyro", "blackjax"], default="blackjax",
        help="Sampling backend (default: blackjax)"
    )
    p.add_argument("--seed", type=int, default=0,
                   help="JAX PRNGKey seed (default: 0)")
    p.add_argument("--no-progress-bar", action="store_true",
                   help="Suppress tqdm progress bars")
    p.add_argument("--enable-x64", action="store_true",
                   help="Enable JAX 64-bit precision (default: False)")

    return p


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point registered as fli-2pcf."""
    import jax.numpy as jnp

    p = parser()
    args = p.parse_args()

    jax.config.update("jax_enable_x64", args.enable_x64)

    # --- validate geometry ---
    if args.nside is None and args.flatsky_npix is None:
        p.error("One of --nside or --flatsky-npix is required.")
    if args.nside is not None and args.flatsky_npix is not None:
        p.error("Only one of --nside or --flatsky-npix can be specified.")
    if args.flatsky_npix is not None and args.field_size is None:
        p.error("--field-size is required when using --flatsky-npix.")
    if args.sampler == "MCLMC" and args.backend != "blackjax":
        p.error("--sampler MCLMC requires --backend blackjax.")

    _save_args_log(args, args.path, "fli-2pcf")

    # --- load observed C_ell from parquet ---
    catalog = jfli.io.Catalog.from_parquet(args.observable)
    obs_ps = catalog.field[0]
    obs_cell = jnp.asarray(obs_ps.array)      # shape: (N_pairs, N_ell) or (N_ell,)
    obs_ells = jnp.asarray(obs_ps.wavenumber)  # ells at which Cl was measured

    # --- resolve nz_shear ---
    nz_shear = _resolve_nz_shear(args)

    # --- determine geometry ---
    if args.flatsky_npix is not None:
        geometry = "flat"
        nside = None
        flatsky_npix = tuple(args.flatsky_npix)
        field_size = tuple(args.field_size)
    else:
        geometry = "spherical"
        nside = args.nside
        flatsky_npix = None
        field_size = None

    # --- build priors from CLI args ---
    sample_set = set(args.sample)
    priors = {}
    if "cosmo" in sample_set:
        priors["Omega_c"] = jfli.infer.PreconditionnedUniform(*args.prior_omega_c)
        priors["sigma8"] = jfli.infer.PreconditionnedUniform(*args.prior_sigma8)
        priors["h"] = jfli.infer.PreconditionnedUniform(*args.prior_h)

    if not priors:
        p.error("--sample must include 'cosmo' to have anything to infer.")

    # --- resolve nonlinear function ---
    nonlinear_fn = jc.power.halofit if args.nonlinear_fn == "halofit" else jc.power.linear

    # --- build Configurations ---
    config = jfli.ppl.Configurations(
        mesh_size=(1, 1, 1),    # unused by power-spectrum model
        box_size=(1.0, 1.0, 1.0),  # unused by power-spectrum model
        nside=nside,
        flatsky_npix=flatsky_npix,
        field_size=field_size,
        geometry=geometry,
        fiducial_cosmology=jc.Planck18,
        nz_shear=nz_shear,
        priors=priors,
        sigma_e=args.sigma_e,
        ells=obs_ells,
        f_sky=args.f_sky,
    )

    # --- build probabilistic model and condition on observed C_ell ---
    prob_model = jfli.ppl.powerspec_probmodel(config, nonlinear_fn=nonlinear_fn)
    conditioned_model = condition(prob_model, data={"C_ell": obs_cell.flatten()})

    # --- run batched MCMC ---
    jfli.infer.batched_sampling(
        conditioned_model,
        path=args.path,
        rng_key=jax.random.PRNGKey(args.seed),
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        batch_count=args.batch_count,
        sampler=args.sampler,
        backend=args.backend,
        progress_bar=not args.no_progress_bar,
        save_callback=jfli.ppl.sample2catalog(config),
    )


if __name__ == "__main__":
    main()
