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


def _load_observable(args, *, lensing_output, sharding=None):
    """Load the single-row observable via ``add_source_args`` and split it into per-bin map arrays.

    The source is a local ``--input`` glob or a HuggingFace ``--repo`` + ``--data-files`` (streamed).
    It must be a SINGLE catalog row whose field type matches ``--lensing-output``: ``convergence`` →
    ``Flat``/``SphericalKappaField``; ``shear`` / ``reduced_shear`` → ``Flat``/``SphericalShearField``
    (there is no density observable — ``lensing_output`` has no ``density`` option). Returns the per-bin
    map arrays (convergence ``(npix,)`` / ``(ny,nx)``; shear ``(2,npix)`` / ``(2,ny,nx)``) plus the
    geometry metadata that ``main`` forwards to ``Configurations``.
    """
    catalog = _single_row_catalog(_resolve_source(args), sharding=sharding, what="observable")
    obs_field = catalog.field[0]
    obs_cosmo = catalog.cosmology[0]

    want_shear = lensing_output != "convergence"
    expected = (FlatShearField, SphericalShearField) if want_shear else (FlatKappaField, SphericalKappaField)
    if not isinstance(obs_field, expected):
        kind = "shear" if want_shear else "convergence"
        raise TypeError(
            f"--lensing-output {lensing_output} expects a {kind} observable "
            f"({' or '.join(c.__name__ for c in expected)}), got {type(obs_field).__name__}. "
            "Generate observables with fli-samples or fli-simulate lensing."
        )

    if isinstance(obs_field, (SphericalKappaField, SphericalShearField)):
        geometry, nside, flatsky_npix, field_size = "spherical", obs_field.nside, None, None
    else:
        geometry, nside, flatsky_npix, field_size = "flat", None, obs_field.flatsky_npix, obs_field.field_size

    # One source bin per leading batch axis; a single-bin map round-trips UNBATCHED (so its leading
    # axis is npix / the spin-2 component, not a bin) — is_batched() accounts for the spin-2 core.
    arr = obs_field.array
    kappa_arrays = [arr[i] for i in range(arr.shape[0])] if obs_field.is_batched() else [arr]
    n_kappas = len(kappa_arrays)

    return kappa_arrays, obs_cosmo, obs_field.box_size, geometry, nside, flatsky_npix, field_size, n_kappas


# ---------------------------------------------------------------------------
# Initial condition loading
# ---------------------------------------------------------------------------


def _load_initial_condition(args, *, sharding=None):
    """Load the optional single-row IC ``DensityField`` via the prefixed ``add_source_args('ic')`` source.

    The source is ``--ic-input`` (local glob) or ``--ic-repo`` + ``--ic-data-files`` (HuggingFace),
    streamed exactly like the observable. It must be a single catalog row holding a ``DensityField``;
    the IC array conditions the model (when the IC is fixed) or warm-starts a sampled IC.
    """
    catalog = _single_row_catalog(_resolve_source(args, prefix="ic"), sharding=sharding, what="initial condition")
    ic_field = catalog.field[0]
    if not isinstance(ic_field, DensityField):
        raise TypeError(f"The initial condition must be a DensityField, got {type(ic_field).__name__}.")
    return ic_field


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
        args, lensing_output=args.lensing_output, sharding=sharding
    )

    cli_box = tuple(args.box_size)
    if cli_box != tuple(obs_box_size):
        warnings.warn(
            f"--box-size {cli_box} differs from the observable's stored box_size {tuple(obs_box_size)}. "
            "Make sure this is intentional.",
            stacklevel=2,
        )

    ic_field = None
    if args.ic_input or args.ic_repo:
        ic_field = _load_initial_condition(args)

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

    # Survey footprint mask resolves at the model nside, which comes from the observable.
    mask = _resolve_mask(args.mask, nside)

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
        halo_size=halo_size,
        t0=args.t0,
        nb_steps=args.nb_steps,
        t1=args.t1,
        lpt_order=args.lpt_order,
        number_of_shells=args.nb_shells,
        lensing="born",
        lensing_output=args.lensing_output,
        scheme=args.scheme,
        paint_nside=args.paint_nside,
        kernel_width_arcmin=args.kernel_width_arcmin,
        kernel_width_pixels=args.kernel_width_pixels,
        pixel_window_deconvolution=args.pixel_window_deconvolution,
        adjoint=args.adjoint,
        checkpoints=args.checkpoints,
        field_sharding=sharding,
        drift_on_lightcone=args.drift_on_lightcone,
        shell_spacing=args.shell_spacing,
        min_width=args.min_width,
        min_redshift=args.min_z,
        max_redshift=args.max_z,
        # N-body / force / painting knobs (previously not forwarded from the CLI)
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
