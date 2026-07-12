"""fli-samples: generate prior-predictive samples from a probabilistic model."""

from __future__ import annotations

import argparse

import jax
import jax_cosmo as jc
from numpyro.infer import Predictive

import jax_fli as jfli
from jax_fli.scripts._common import (
    _build_sharding,
    _resolve_mask,
    _resolve_nz_shear,
    _resolve_solver_name,
    _save_args_log,
)

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for fli-samples."""
    from jax_fli.scripts.parser import (
        add_common_args,
        add_distributed_args,
        add_forward_model_args,
        add_integration_settings_args,
        add_lensing_args,
        add_output_target_args,
        add_prior_args,
        add_simulation_settings_args,
    )

    p = argparse.ArgumentParser(
        prog="fli-samples",
        description="Generate prior-predictive samples from a probabilistic model.",
    )

    add_common_args(p)
    add_distributed_args(p)
    add_simulation_settings_args(p)
    add_output_target_args(p)
    # Full-field model uses BullFrog by default (the Configurations default); set it in the parser.
    add_integration_settings_args(p, solver_default="bf")
    add_lensing_args(p)
    add_prior_args(p)
    add_forward_model_args(p)

    g = p.add_argument_group("Sampling settings")
    g.add_argument(
        "--model",
        choices=["full", "mock"],
        default="full",
        help="Probabilistic model to sample from: 'full' or 'mock' (default: full)",
    )

    # Samples-specific (--sigma-e now comes from the shared add_forward_model_args)
    g.add_argument("--num-samples", type=int, default=100, help="Number of prior-predictive samples (default: 100)")
    g.add_argument("--path", required=True, metavar="PATH", help="Output directory for samples and catalogs.")
    g.add_argument("--batch-id", type=int, default=0, help="Batch index written into output filenames (default: 0)")
    g.add_argument(
        "--initial-condition",
        type=str,
        default=None,
        metavar="PATH",
        help="Parquet Catalog with IC DensityField (required when 'ic' is not in --sample).",
    )

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

    _save_args_log(args, args.path, "fli-samples")

    # --- resolve nz_shear ---
    nz_shear = _resolve_nz_shear(args)

    # --- build priors from CLI args ---
    sample_set = set(args.sample)
    priors = {}
    if "cosmo" in sample_set:
        priors["Omega_c"] = jfli.infer.PreconditionnedUniform(*args.prior_omega_c)
        priors["sigma8"] = jfli.infer.PreconditionnedUniform(*args.prior_sigma8)
        priors["h"] = jfli.infer.PreconditionnedUniform(*args.prior_h)

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

    # Survey footprint mask (likelihood-only; a no-op for prior-predictive sampling, kept
    # consistent with fli-infer). Resolves at the model nside.
    mask = _resolve_mask(args.mask, nside)

    # --- build Configurations ---
    # Mirrors the fli-infer config, minus adjoint/checkpoints (prior-predictive Predictive sampling
    # runs no gradients, so the adjoint strategy is irrelevant here).
    config = jfli.ppl.Configurations(
        mesh_size=mesh,
        box_size=tuple(args.box_size),
        nside=nside,
        flatsky_npix=flatsky_npix,
        field_size=args.field_size,
        fiducial_cosmology=jc.Planck18,
        nz_shear=nz_shear,
        priors=priors,
        sigma_e=args.sigma_e,
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
        kernel_width_pixels=args.kernel_width_pixels,
        pixel_window_deconvolution=args.pixel_window_deconvolution,
        field_sharding=sharding,
        lensing_output=args.lensing_output,
        drift_on_lightcone=args.drift_on_lightcone,
        shell_spacing=args.shell_spacing,
        min_width=args.min_width,
        min_redshift=args.min_z,
        max_redshift=args.max_z,
        quadrature=args.quadrature,
        # N-body / force / painting knobs (previously not forwarded from the CLI)
        sim_mode=args.sim_mode,
        nbody_solver=_resolve_solver_name(args.solver),
        paint_order=args.paint_order,
        gradient_order=args.gradient_order,
        laplace_fd=args.laplace_fd,
        deconvolution=args.deconvolution,
        dealiased=args.dealiased,
        exact_growth=args.exact_growth,
        # Masking / likelihood + observer visibility mask
        mask=mask,
        sigma_unobserved=args.sigma_unobserved,
        apodization_scale_deg=args.apodization_scale_deg,
        log_lightcone=args.log_lightcone,
    )

    # --- select model ---
    if args.model == "full":
        model = jfli.ppl.full_field_probmodel(config)
    else:
        model = jfli.ppl.mock_probmodel(config)

    # --- sample with NumPyro Predictive ---
    chain_key = jax.random.PRNGKey(args.seed)
    rng_key = jax.random.fold_in(chain_key, args.batch_id)
    pred = Predictive(model, num_samples=args.num_samples)
    samples = pred(rng_key)

    print(f"sharding {samples['initial_conditions'].array.sharding} samples with {config.field_sharding}...")
    # Recolor the WHITE initial_conditions -> physical delta, one sample at a time (lax.map is
    # sequential; never vmap a field-sized batch), using each sample's stored cosmology.
    samples["initial_conditions"] = jax.lax.map(
        lambda wc: (
            jfli.interpolate_initial_conditions(
                wc[0], config.mesh_size, config.box_size, cosmo=wc[1], field_sharding=config.field_sharding
            ).array
        ),
        (samples["initial_conditions"].array, samples["cosmo"]),
    )
    # --- save via sample2catalog ---
    saving_fn = jfli.infer.sample2catalog(config)
    saving_fn(samples, args.path, args.batch_id)


if __name__ == "__main__":
    main()
