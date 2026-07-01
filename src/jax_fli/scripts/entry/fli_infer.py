"""fli-infer: run full-field MCMC inference conditioned on observed kappa maps."""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace

import jax
import jax_cosmo as jc
from numpyro.handlers import condition

import jax_fli as jfli
from jax_fli.fields import (
    DensityField,
    FlatKappaField,
    FlatShearField,
    SphericalKappaField,
    SphericalShearField,
)
from jax_fli.scripts._common import (
    _build_sharding,
    _resolve_mask,
    _resolve_nz_shear,
    _resolve_solver_name,
    _resolve_source,
    _save_args_log,
)

# ---------------------------------------------------------------------------
# Observable loading
# ---------------------------------------------------------------------------


def _single_row_catalog(ds, *, sharding=None, what="source"):
    """Materialize a streamed source expected to hold exactly ONE catalog row; return its Catalog.

    Reads at most two rows so a mistakenly-batched source fails fast instead of streaming in full.
    """
    rows = []
    for row in ds.with_format("numpy"):
        rows.append(row)
        if len(rows) > 1:
            raise ValueError(f"The {what} source must contain exactly one row, but found more than one.")
    if not rows:
        raise ValueError(f"The {what} source contained no rows.")
    catalog = jfli.io.Catalog.from_dataset(rows[0], sharding=sharding)
    if len(catalog.field) != 1:
        raise ValueError(f"The {what} row expands to {len(catalog.field)} catalog entries; expected a single field.")
    return catalog


def _load_observable(args, *, lensing_output, target_nside=None, sharding=None):
    """Load the single-row observable and return **only** the per-bin map arrays.

    Spherical only for now. The source is a local ``--input`` glob or a HuggingFace ``--repo`` +
    ``--data-files`` (streamed); it must be a SINGLE catalog row whose field type matches
    ``--lensing-output``: ``convergence`` → ``SphericalKappaField``; ``shear`` / ``reduced_shear`` →
    ``SphericalShearField``. The map is ud_graded onto the model's analysis ``target_nside`` (a no-op
    when equal) so the conditioned data lives on the model grid. Cosmology and geometry are NOT read
    from the observable — the truth cosmology used for conditioning / warm-starting comes from the
    initial-condition catalog (see :func:`_load_initial_condition`).
    """
    catalog = _single_row_catalog(_resolve_source(args), sharding=sharding, what="observable")
    obs_field = catalog.field[0]

    want_shear = lensing_output != "convergence"
    expected = (FlatShearField, SphericalShearField) if want_shear else (FlatKappaField, SphericalKappaField)
    if not isinstance(obs_field, expected):
        kind = "shear" if want_shear else "convergence"
        raise TypeError(
            f"--lensing-output {lensing_output} expects a {kind} observable "
            f"({' or '.join(c.__name__ for c in expected)}), got {type(obs_field).__name__}. "
            "Generate observables with fli-samples or fli-simulate lensing."
        )

    # ud_grade the observable onto the model's analysis nside for the likelihood (no-op if equal).
    if target_nside is not None and obs_field.nside != target_nside:
        obs_field = obs_field.ud_sample(target_nside)

    # One source bin per leading batch axis; a single-bin map round-trips UNBATCHED (so its leading
    # axis is npix / the spin-2 component, not a bin) — is_batched() accounts for the spin-2 core.
    arr = obs_field.array
    return [arr[i] for i in range(arr.shape[0])] if obs_field.is_batched() else [arr]


# ---------------------------------------------------------------------------
# Initial condition loading
# ---------------------------------------------------------------------------


def _load_initial_condition(args, *, sharding=None):
    """Load the optional single-row IC ``DensityField`` **and its cosmology**.

    The source is ``--ic-input`` (local glob) or ``--ic-repo`` + ``--ic-data-files`` (HuggingFace),
    streamed exactly like the observable. It must be a single catalog row holding a ``DensityField``.
    Returns ``(ic_field, ic_cosmo)``: the IC array conditions the model (fixed IC) or warm-starts a
    sampled IC, and ``ic_cosmo`` is the truth cosmology used to condition / warm-start the cosmological
    parameters (the observable no longer carries cosmology).
    """
    catalog = _single_row_catalog(_resolve_source(args, prefix="ic"), sharding=sharding, what="initial condition")
    ic_field = catalog.field[0]
    if not isinstance(ic_field, DensityField):
        raise TypeError(f"The initial condition must be a DensityField, got {type(ic_field).__name__}.")
    return ic_field, catalog.cosmology[0]


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for fli-infer."""
    from jax_fli.scripts.parser import (
        add_common_args,
        add_distributed_args,
        add_forward_model_args,
        add_infer_args,
        add_integration_settings_args,
        add_lensing_args,
        add_output_target_args,
        add_prior_args,
        add_simulation_settings_args,
        add_source_args,
    )

    p = argparse.ArgumentParser(
        prog="fli-infer",
        description="Run full-field MCMC inference conditioned on an observed kappa / shear map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Observable source: a single-row kappa / shear Catalog (local --input glob or HF --repo +
    # --data-files), validated against --lensing-output in _load_observable. The OPTIONAL initial
    # condition uses the parallel prefixed source (--ic-input / --ic-repo / --ic-data-files); it must be
    # a single-row DensityField, used to fix (condition) the IC or to warm-start a sampled IC.
    add_source_args(p)
    add_source_args(p, prefix="ic")
    p.add_argument(
        "--path",
        type=str,
        required=True,
        metavar="PATH",
        help="Output directory for MCMC checkpoints and parquet catalogs.",
    )

    add_common_args(p)
    add_distributed_args(p)
    add_simulation_settings_args(p)
    add_output_target_args(p)
    # Full-field model uses BullFrog by default (the Configurations default); set it in the parser.
    add_integration_settings_args(p, solver_default="bf")
    add_lensing_args(p)
    add_prior_args(p)
    # IC comes from the prefixed --ic-* source above, so drop add_infer_args's --initial-condition.
    add_infer_args(p, with_initial_condition=False)
    add_forward_model_args(p)

    return p


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def _validate_args(args: Namespace, p: argparse.ArgumentParser) -> None:
    """Validate argument combinations before running JAX."""
    # Observable source (add_source_args): --input XOR --repo + --data-files (required).
    if args.input is None and args.repo is None:
        p.error("an observable source is required: --input (local) or --repo + --data-files (HuggingFace).")
    if args.repo is not None and not args.data_files:
        p.error("--data-files is required when --repo is set.")
    # Initial-condition source (prefixed add_source_args, optional): --ic-input XOR --ic-repo + --ic-data-files.
    if args.ic_repo is not None and not args.ic_data_files:
        p.error("--ic-data-files is required when --ic-repo is set.")

    if args.mesh_size is None:
        p.error("--mesh-size is required for fli-infer.")
    if args.box_size is None:
        p.error("--box-size is required for fli-infer.")
    if not args.sample:
        p.error("--sample must contain at least one of 'cosmo' or 'ic'.")

    sample_set = set(args.sample)
    if not sample_set & {"cosmo", "ic"}:
        p.error(f"--sample must contain at least one of 'cosmo' or 'ic', got: {args.sample}")

    if "ic" not in sample_set and not (args.ic_input or args.ic_repo):
        p.error(
            "an initial condition is required when 'ic' is not in --sample (IC must be fixed): "
            "pass --ic-input or --ic-repo + --ic-data-files."
        )

    if args.sim_mode == "pm" and args.nb_steps < 2:
        p.error(f"--nb-steps must be >= 2 for --sim-mode pm, got {args.nb_steps}")


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

    # Spherical only for now. The model painting nside comes from the CLI args; the observable is
    # ud_graded onto it inside _load_observable.
    model_nside = args.nside

    # The observable provides ONLY the per-bin map arrays (no cosmology, no geometry).
    observable_arrays = _load_observable(
        args, lensing_output=args.lensing_output, target_nside=model_nside, sharding=sharding
    )
    n_observables = len(observable_arrays)

    # The truth cosmology used for conditioning / warm-starting comes from the IC catalog, not the observable.
    ic_field, ic_cosmo = (None, None)
    if args.ic_input or args.ic_repo:
        ic_field, ic_cosmo = _load_initial_condition(args, sharding=sharding)

    sample_set = set(args.sample)  # _validate_args already guarantees this contains 'cosmo' and/or 'ic'

    condition_data = {f"observable_{i}": observable_arrays[i] for i in range(n_observables)}
    # ==================================================================================================
    # Fix parameters that are NOT sampled. Cosmology (incl. h) is conditioned on the IC catalog's cosmology.
    # ==================================================================================================
    if "cosmo" not in sample_set:
        if ic_cosmo is None:
            p.error(
                "fixing cosmology (--sample without 'cosmo') needs an IC source (--ic-input / --ic-repo) "
                "to read the truth cosmology from."
            )
        condition_data["Omega_c"] = float(ic_cosmo.Omega_c)
        condition_data["sigma8"] = float(ic_cosmo.sigma8)
        condition_data["h"] = float(ic_cosmo.h)

    if "ic" not in sample_set:
        assert ic_field is not None  # guaranteed by _validate_args
        condition_data["initial_conditions"] = ic_field.array
    # ==================================================================================================

    # ==================================================================================================
    # Initializing the probabilistic model configuration
    # ==================================================================================================
    init_params = None
    if "ic" in sample_set and ic_field is not None:
        init_params = {"initial_conditions": ic_field.array}
    if args.init_cosmo:
        if ic_cosmo is None:
            p.error("--init-cosmo needs an IC source (--ic-input / --ic-repo) to warm-start cosmology from.")
        # Warm-start every sampled cosmological parameter (Omega_c, sigma8, h) from the IC's cosmology.
        init_params = init_params or {}
        init_params.update(
            {"Omega_c": float(ic_cosmo.Omega_c), "sigma8": float(ic_cosmo.sigma8), "h": float(ic_cosmo.h)}
        )
    # ==================================================================================================

    nz_shear = _resolve_nz_shear(args)
    if len(nz_shear) != n_observables:
        print(
            f"Warning: observable has {n_observables} maps but nz_shear has {len(nz_shear)} bins. "
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

    # Survey footprint mask resolves at the model nside (from the CLI args).
    mask = _resolve_mask(args.mask, model_nside)

    config = jfli.ppl.Configurations(
        # Simulation setting
        mesh_size=mesh,
        box_size=tuple(args.box_size),
        halo_size=halo_size,
        nb_steps=args.nb_steps,
        field_sharding=sharding,
        # N-body / force / painting knobs (previously not forwarded from the CLI)
        sim_mode=args.sim_mode,
        nbody_solver=_resolve_solver_name(args.solver),
        t0=args.t0,
        t1=args.t1,
        lpt_order=args.lpt_order,
        number_of_shells=args.nb_shells,
        drift_on_lightcone=args.drift_on_lightcone,
        paint_order=args.paint_order,
        gradient_order=args.gradient_order,
        laplace_fd=args.laplace_fd,
        deconvolution=args.deconvolution,
        dealiased=args.dealiased,
        exact_growth=args.exact_growth,
        shell_spacing=args.shell_spacing,
        time_stepping=args.time_stepping,
        min_width=args.min_width,
        # Lensing
        lensing_output=args.lensing_output,
        map2alm_method=args.map2alm_method,
        min_redshift=args.min_z,
        max_redshift=args.max_z,
        n_integrate=args.n_integrate,
        apodization_scale_deg=args.apodization_scale_deg,
        # Geometry / painting (spherical only for now)
        nside=model_nside,
        flatsky_npix=None,
        field_size=None,
        geometry="spherical",
        scheme=args.scheme,
        observer_position=tuple(args.observer_position),
        paint_nside=args.paint_nside,
        kernel_width_arcmin=args.kernel_width_arcmin,
        kernel_width_pixels=args.kernel_width_pixels,
        pixel_window_deconvolution=args.pixel_window_deconvolution,
        # Cosmology and nz
        fiducial_cosmology=jc.Planck18,
        nz_shear=nz_shear,
        # Masking / likelihood + observer visibility mask
        mask=mask,
        sigma_unobserved=args.sigma_unobserved,
        log_lightcone=args.log_lightcone,
        # Priors and inference settings
        priors=priors,
        sigma_e=args.sigma_e,
        adjoint=args.adjoint,
        checkpoints=args.checkpoints,
        # Sampler settings (BlackJAX NUTS/MCLMC)
        sampler=args.sampler,
        nuts_max_num_doublings=args.max_num_doublings,
        nuts_target_accept=args.target_accept,
        mclmc_desired_energy_var=args.mclmc_desired_energy_var,
        mclmc_init_step_size_scale=args.mclmc_init_step_size_scale,
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
        sampler=config.sampler,
        max_num_doublings=config.nuts_max_num_doublings,
        target_accept=config.nuts_target_accept,
        mclmc_desired_energy_var=config.mclmc_desired_energy_var,
        mclmc_num_tune=config.mclmc_num_tune,
        mclmc_init_step_size_scale=config.mclmc_init_step_size_scale,
        mclmc_diagonal_preconditioning=config.mclmc_diagonal_preconditioning,
        init_params=init_params,
        progress_bar=not args.no_progress_bar,
        save_callback=jfli.ppl.sample2catalog(config),
    )


if __name__ == "__main__":
    main()
