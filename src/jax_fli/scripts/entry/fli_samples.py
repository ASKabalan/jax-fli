"""fli-samples: generate prior-predictive samples from a probabilistic model."""

from __future__ import annotations

import argparse

import jax
import jax_cosmo as jc
from numpyro.infer import Predictive

import jax_fli as jfli
from jax_fli.scripts._common import _build_sharding, _resolve_nz_shear

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for fli-samples."""
    from jax_fli.scripts.parser import (
        add_common_sim_args,
        add_distributed_args,
        add_lensing_args,
        add_lightcone_args,
        add_mesh_args,
    )

    p = argparse.ArgumentParser(
        prog="fli-samples",
        description="Generate prior-predictive samples from a probabilistic model.",
    )

    p.add_argument(
        "--model",
        choices=["full", "mock"],
        default="full",
        help="Probabilistic model to sample from: 'full' or 'mock' (default: full)",
    )

    add_mesh_args(p, nargs=3)
    add_distributed_args(p)
    add_common_sim_args(p)
    add_lightcone_args(p)
    add_lensing_args(p)

    # Geometry (mutually exclusive)
    geom_group = p.add_mutually_exclusive_group()
    geom_group.add_argument("--nside", type=int, default=None, help="HEALPix NSIDE for spherical painting")
    geom_group.add_argument(
        "--flatsky-npix",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Flat-sky pixel resolution (height width)",
    )

    # Samples-specific
    p.add_argument("--nb-shells", type=int, default=8, help="Number of lightcone shells (default: 8)")
    p.add_argument("--sigma-e", type=float, default=0.26, help="Shape-noise dispersion (default: 0.26)")
    p.add_argument(
        "--density-plane-smoothing", type=float, default=0.0, help="Density plane smoothing scale (default: 0.0)"
    )
    p.add_argument("--num-samples", type=int, default=100, help="Number of prior-predictive samples (default: 100)")
    p.add_argument("--seed", type=int, default=0, help="JAX PRNGKey seed (default: 0)")
    p.add_argument("--path", type=str, required=True, help="Output directory")
    p.add_argument("--batch-id", type=int, default=0, help="Batch index written into output filenames (default: 0)")
    p.add_argument("--enable-x64", action="store_true", help="Enable JAX 64-bit precision (default: False)")

    p.set_defaults(t0=0.01, nb_steps=100)

    return p


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point registered as fli-samples."""
    args = parser().parse_args()

    jax.config.update("jax_enable_x64", args.enable_x64)

    # --- validate geometry ---
    if args.nside is None and args.flatsky_npix is None:
        parser().error("One of --nside or --flatsky-npix is required.")
    if args.nside is not None and args.flatsky_npix is not None:
        parser().error("Only one of --nside or --flatsky-npix can be specified.")

    # --- resolve nz_shear ---
    nz_shear = _resolve_nz_shear(args)

    priors = {
        "Omega_c": jfli.infer.PreconditionnedUniform(0.1, 0.5),
        "sigma8": jfli.infer.PreconditionnedUniform(0.6, 1.0),
    }

    # --- determine geometry ---
    if args.flatsky_npix is not None:
        geometry, nside, flatsky_npix = "flat", None, tuple(args.flatsky_npix)
    else:
        geometry, nside, flatsky_npix = "spherical", args.nside, None

    # -- determine sharding ---
    sharding = _build_sharding(args)

    # --- compute halo_size ---
    mesh = tuple(args.mesh_size)
    px, py = args.pdim
    halo_size = (int(mesh[0] / px * args.halo_multiplier), int(mesh[1] / py * args.halo_multiplier))

    # --- build Configurations ---
    config = jfli.ppl.Configurations(
        mesh_size=mesh,
        box_size=tuple(args.box_size),
        nside=nside,
        flatsky_npix=flatsky_npix,
        fiducial_cosmology=jc.Planck18,
        nz_shear=nz_shear,
        priors=priors,
        sigma_e=args.sigma_e,
        density_plane_smoothing=args.density_plane_smoothing,
        halo_size=halo_size,
        observer_position=tuple(args.observer_position),
        t0=args.t0,
        nb_steps=args.nb_steps,
        t1=args.t1,
        lpt_order=args.lpt_order,
        number_of_shells=args.nb_shells,
        geometry=geometry,
        scheme=args.scheme,
        paint_nside=args.paint_nside,
        kernel_width_arcmin=args.kernel_width_arcmin,
        field_sharding=sharding,
        lensing="born",
        drift_on_lightcone=args.drift_on_lightcone,
        shell_spacing=args.shell_spacing,
        min_width=args.min_width,
        min_redshift=args.min_z,
        max_redshift=args.max_z,
    )

    # --- select model ---
    if args.model == "full":
        model = jfli.ppl.full_field_probmodel(config)
    else:
        model = jfli.ppl.mock_probmodel(config)

    # --- sample with NumPyro Predictive ---
    rng_key = jax.random.PRNGKey(args.seed)
    pred = Predictive(model, num_samples=args.num_samples)
    samples = pred(rng_key)

    print(f"sharding {samples['initial_conditions'].array.sharding} samples with {config.field_sharding}...")
    # --- save via sample2catalog ---
    saving_fn = jfli.infer.sample2catalog(config)
    saving_fn(samples, args.path, args.batch_id)


if __name__ == "__main__":
    main()
