"""fli-infer: run full-field MCMC inference conditioned on observed kappa maps."""

from __future__ import annotations

import argparse
import sys
import warnings
from argparse import Namespace

import jax
import jax_cosmo as jc
from numpyro.handlers import condition

import jax_fli as jfli
from jax_fli.fields import FlatKappaField, SphericalKappaField
from jax_fli.scripts._common import _build_sharding, _resolve_nz_shear, _save_args_log

# ---------------------------------------------------------------------------
# Observable loading
# ---------------------------------------------------------------------------


def _load_observable(path: str, sharding):
    """Load a kappa Catalog from parquet and extract per-bin arrays + metadata."""
    catalog = jfli.io.Catalog.from_parquet(path, sharding=sharding)
    obs_field = catalog.field[0]
    obs_cosmo = catalog.cosmology[0]

    if isinstance(obs_field, SphericalKappaField):
        geometry = "spherical"
        nside = obs_field.nside
        flatsky_npix = None
        field_size = None
    elif isinstance(obs_field, FlatKappaField):
        geometry = "flat"
        nside = None
        flatsky_npix = obs_field.flatsky_npix
        field_size = obs_field.field_size
    else:
        raise ValueError(
            f"Observable must be a SphericalKappaField or FlatKappaField, got {type(obs_field).__name__}. "
            "Generate observables with fli-samples or fli-simulate lensing."
        )

    n_kappas = obs_field.array.shape[0]
    kappa_arrays = [obs_field.array[i] for i in range(n_kappas)]

    return kappa_arrays, obs_cosmo, obs_field.box_size, geometry, nside, flatsky_npix, field_size, n_kappas


# ---------------------------------------------------------------------------
# Initial condition loading
# ---------------------------------------------------------------------------


def _load_initial_condition(path: str):
    """Load an IC DensityField from a parquet Catalog."""
    catalog = jfli.io.Catalog.from_parquet(path)
    return catalog.field[0]


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for fli-infer."""
    from jax_fli.scripts.parser import (
        add_common_sim_args,
        add_distributed_args,
        add_lensing_args,
        add_lightcone_args,
        add_prior_args,
    )

    p = argparse.ArgumentParser(
        prog="fli-infer",
        description="Run full-field MCMC inference conditioned on observed kappa maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required args
    p.add_argument(
        "--observable",
        type=str,
        required=True,
        metavar="PATH",
        help="Parquet Catalog with batched kappa field and cosmology.",
    )
    p.add_argument(
        "--path",
        type=str,
        required=True,
        metavar="PATH",
        help="Output directory for MCMC checkpoints and parquet catalogs.",
    )
    p.add_argument(
        "--mesh-size", type=int, nargs=3, required=True, metavar=("NX", "NY", "NZ"), help="Inference mesh resolution."
    )
    p.add_argument(
        "--box-size", type=float, nargs=3, required=True, metavar=("LX", "LY", "LZ"), help="Box side lengths in Mpc/h."
    )

    # Shared arg groups
    add_distributed_args(p)
    add_common_sim_args(p)
    add_lightcone_args(p)
    add_lensing_args(p)
    add_prior_args(p)

    # Infer-specific
    p.add_argument("--nb-shells", type=int, default=8, metavar="INT", help="Number of lightcone shells (default: 8)")
    p.add_argument(
        "--initial-condition",
        type=str,
        default=None,
        metavar="PATH",
        help="Parquet Catalog with IC DensityField for initialization or fixing IC.",
    )
    p.add_argument(
        "--init-cosmo",
        action="store_true",
        help="Warm-start cosmological parameters from the observable's stored cosmology.",
    )
    p.add_argument("--sigma-e", type=float, default=0.26, help="Shape noise dispersion (default: 0.26)")
    p.add_argument(
        "--density-plane-smoothing", type=float, default=0.0, help="Density plane smoothing scale (default: 0.0)"
    )
    p.add_argument(
        "--adjoint",
        choices=["checkpointed", "recursive"],
        default="checkpointed",
        help="Gradient strategy for NUTS (default: checkpointed)",
    )
    p.add_argument("--checkpoints", type=int, default=10, help="Number of gradient checkpoints (default: 10)")
    p.add_argument("--num-warmup", type=int, default=500, help="MCMC warmup iterations (default: 500)")
    p.add_argument("--num-samples", type=int, default=1000, help="Samples per batch (default: 1000)")
    p.add_argument("--batch-count", type=int, default=5, help="Number of sequential batches (default: 5)")
    p.add_argument("--sampler", choices=["NUTS", "HMC", "MCLMC"], default="NUTS", help="MCMC sampler (default: NUTS)")
    p.add_argument(
        "--backend", choices=["numpyro", "blackjax"], default="numpyro", help="Sampling backend (default: numpyro)"
    )
    p.add_argument("--seed", type=int, default=0, help="JAX PRNGKey seed (default: 0)")
    p.add_argument("--no-progress-bar", action="store_true", help="Suppress tqdm progress bars")
    p.add_argument("--enable-x64", action="store_true", help="Enable JAX 64-bit precision (default: False)")

    return p


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def _validate_args(args: Namespace, p: argparse.ArgumentParser) -> None:
    """Validate argument combinations before running JAX."""
    if not args.sample:
        p.error("--sample must contain at least one of 'cosmo' or 'ic'.")

    sample_set = set(args.sample)
    if not sample_set & {"cosmo", "ic"}:
        p.error(f"--sample must contain at least one of 'cosmo' or 'ic', got: {args.sample}")

    if "ic" not in sample_set and args.initial_condition is None:
        p.error("--initial-condition is required when 'ic' is not in --sample (IC must be fixed).")

    if args.sampler == "MCLMC" and args.backend != "blackjax":
        p.error("--sampler MCLMC requires --backend blackjax.")

    if args.nb_steps < 2:
        p.error(f"--nb-steps must be >= 2, got {args.nb_steps}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point registered as fli-infer."""
    p = parser()
    args = p.parse_args()

    jax.config.update("jax_enable_x64", args.enable_x64)

    _validate_args(args, p)
    _save_args_log(args, args.path, "fli-infer")
    sharding = _build_sharding(args)

    kappa_arrays, obs_cosmo, obs_box_size, geometry, nside, flatsky_npix, field_size, n_kappas = _load_observable(
        args.observable, sharding
    )

    cli_box = tuple(args.box_size)
    if cli_box != tuple(obs_box_size):
        warnings.warn(
            f"--box-size {cli_box} differs from the observable's stored box_size {tuple(obs_box_size)}. "
            "Make sure this is intentional.",
            stacklevel=2,
        )

    ic_field = None
    if args.initial_condition is not None:
        ic_field = _load_initial_condition(args.initial_condition)

    sample_set = set(args.sample)

    condition_data = {f"kappa_{i}": kappa_arrays[i] for i in range(n_kappas)}

    if "cosmo" not in sample_set:
        condition_data["Omega_c"] = float(obs_cosmo.Omega_c)
        condition_data["sigma8"] = float(obs_cosmo.sigma8)

    if "ic" not in sample_set:
        assert ic_field is not None  # guaranteed by _validate_args
        condition_data["initial_conditions"] = ic_field.array

    init_params = None
    if "ic" in sample_set and ic_field is not None:
        init_params = {"initial_conditions": ic_field.array}
    if args.init_cosmo:
        init_params = init_params or {}
        init_params.update({"Omega_c": float(obs_cosmo.Omega_c), "sigma8": float(obs_cosmo.sigma8)})

    nz_shear = _resolve_nz_shear(args)
    if len(nz_shear) != n_kappas:
        print(
            f"Warning: observable has {n_kappas} kappa maps but nz_shear has {len(nz_shear)} bins. "
            "Inference may fail if the numbers don't match.",
            file=sys.stderr,
        )

    priors = {}
    if "cosmo" in sample_set:
        priors["Omega_c"] = jfli.infer.PreconditionnedUniform(*args.prior_omega_c)
        priors["sigma8"] = jfli.infer.PreconditionnedUniform(*args.prior_sigma8)
        priors["h"] = jfli.infer.PreconditionnedUniform(*args.prior_h)

    mesh = tuple(args.mesh_size)
    px, py = args.pdim
    halo_size = (int(mesh[0] / px * args.halo_multiplier), int(mesh[1] / py * args.halo_multiplier))

    config = jfli.ppl.Configurations(
        mesh_size=mesh,
        box_size=tuple(args.box_size),
        nside=nside,
        flatsky_npix=flatsky_npix,
        field_size=field_size,
        geometry=geometry,
        observer_position=tuple(args.observer_position),
        fiducial_cosmology=jc.Planck18,
        nz_shear=nz_shear,
        priors=priors,
        sigma_e=args.sigma_e,
        density_plane_smoothing=args.density_plane_smoothing,
        halo_size=halo_size,
        t0=args.t0,
        nb_steps=args.nb_steps,
        t1=args.t1,
        lpt_order=args.lpt_order,
        number_of_shells=args.nb_shells,
        lensing="born",
        scheme=args.scheme,
        paint_nside=args.paint_nside,
        kernel_width_arcmin=args.kernel_width_arcmin,
        adjoint=args.adjoint,
        checkpoints=args.checkpoints,
        field_sharding=sharding,
        drift_on_lightcone=args.drift_on_lightcone,
        shell_spacing=args.shell_spacing,
        min_width=args.min_width,
        min_redshift=args.min_z,
        max_redshift=args.max_z,
    )

    prob_model = jfli.ppl.full_field_probmodel(config)
    conditioned_model = condition(prob_model, data=condition_data)

    # Run batched MCMC
    jfli.infer.batched_sampling(
        conditioned_model,
        path=args.path,
        rng_key=jax.random.PRNGKey(args.seed),
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        batch_count=args.batch_count,
        sampler=args.sampler,
        backend=args.backend,
        init_params=init_params,
        progress_bar=not args.no_progress_bar,
        save_callback=jfli.ppl.sample2catalog(config),
    )


if __name__ == "__main__":
    main()
