"""fli-infer: run full-field MCMC inference conditioned on observed kappa maps."""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace

import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax.scipy.special import ndtri
from jaxpm.distributed import fft3d, ifft3d
from jaxpm.kernels import fftk
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


def _load_observable(
    args, *, lensing_output, target_nside=None, sharding=None, ell_max=None, ell_taper_width=8, method="jax"
):
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
    per_bin = [obs_field[i] for i in range(obs_field.array.shape[0])] if obs_field.is_batched() else [obs_field]
    # Apply the SAME map-level scale cut as the model likelihood (so data and model match) when requested.
    if ell_max is not None:
        per_bin = [f.scale_cut(ell_max, ell_taper_width, method=method) for f in per_bin]
    return [f.array for f in per_bin]


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
        add_gradient_args,
        add_infer_args,
        add_integration_settings_args,
        add_lensing_args,
        add_output_target_args,
        add_prior_args,
        add_scale_cut_args,
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
    add_scale_cut_args(p)  # optional pixel-likelihood ell-tapered scale cut
    add_gradient_args(p)  # adjoint / checkpoints (moved out of add_infer_args)

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
        args,
        lensing_output=args.lensing_output,
        target_nside=model_nside,
        sharding=sharding,
        ell_max=args.ell_max,
        ell_taper_width=args.ell_taper_width,
        method=args.map2alm_method,
    )
    n_observables = len(observable_arrays)

    # The truth cosmology used for conditioning / warm-starting comes from the IC catalog, not the observable.
    ic_field, ic_cosmo = (None, None)
    if args.ic_input or args.ic_repo:
        ic_field, ic_cosmo = _load_initial_condition(args, sharding=sharding)

    sample_set = set(args.sample)  # _validate_args already guarantees this contains 'cosmo' and/or 'ic'

    # The catalog stores the COLORED physical delta, but the model samples the WHITE field and colors
    # it inline (with the sampled cosmology). De-color the loaded IC to white via the inverse
    # power-spectrum transform (using the IC's own cosmology) so both fixing and warm-starting condition
    # the white `initial_conditions` site the model actually traverses -- not a doubly-colored field.
    white_ic = None
    if ic_field is not None:
        # Exact inverse of the model's inline coloring (interpolate_initial_conditions): divide the
        # colored field by sqrt(P(k)) on the SAME 128-point grid and mesh/box kmesh (no k=0 special
        # case, which is what the transform-based inverse gets wrong), so the white field round-trips.
        mesh_t, box_t = tuple(args.mesh_size), tuple(args.box_size)
        k = jnp.logspace(-4, 1, 128)
        pk = jc.power.linear_matter_power(ic_cosmo, k)
        colored = fft3d(ic_field.array)
        kmesh = sum((kk / box_t[i] * mesh_t[i]) ** 2 for i, kk in enumerate(fftk(colored))) ** 0.5
        pkmesh = jnp.interp(kmesh, k, pk) * (mesh_t[0] * mesh_t[1] * mesh_t[2]) / (box_t[0] * box_t[1] * box_t[2])
        white_ic = ifft3d(colored / jnp.sqrt(pkmesh)).real

    condition_data = {f"observable_{i}": observable_arrays[i] for i in range(n_observables)}
    # ==================================================================================================
    # Fix parameters that are NOT sampled.
    # ==================================================================================================
    # Fixing cosmology: with 'cosmo' unsampled the model builds `fiducial_cosmology()` (empty priors,
    # no cosmo sample sites), so PIN the fiducial to the IC catalog's cosmology. Conditioning the name
    # `Omega_c` is a no-op here -- TransformReparam turns it deterministic and the latent site is
    # `Omega_c_base` -- so it must NOT be added to condition_data.
    if "cosmo" not in sample_set:
        if ic_cosmo is None:
            p.error(
                "fixing cosmology (--sample without 'cosmo') needs an IC source (--ic-input / --ic-repo) "
                "to read the truth cosmology from."
            )
        fiducial_cosmology = lambda **_: ic_cosmo
    else:
        fiducial_cosmology = jc.Planck18

    if "ic" not in sample_set:
        assert white_ic is not None  # guaranteed by _validate_args
        condition_data["initial_conditions"] = white_ic
    # ==================================================================================================

    # ==================================================================================================
    # Initializing the probabilistic model configuration
    # ==================================================================================================
    init_params = None
    if "ic" in sample_set and white_ic is not None:
        init_params = {"initial_conditions": white_ic}
    if args.init_cosmo:
        if ic_cosmo is None:
            p.error("--init-cosmo needs an IC source (--ic-input / --ic-repo) to warm-start cosmology from.")
        if "cosmo" in sample_set:
            # Warm-start the WHITE reparam bases `<name>_base` (the latent sites NUTS moves), computing
            # each from the IC cosmology via the inverse PreconditionnedUniform bijector
            # `base = ndtri((phys - low) / (high - low))` (Probit o Affine). Seeding the physical names
            # would be a no-op under reparam.
            init_params = init_params or {}
            init_params.update(
                {
                    f"{name}_base": float(ndtri((phys - lo) / (hi - lo)))
                    for name, (lo, hi), phys in (
                        ("Omega_c", args.prior_omega_c, float(ic_cosmo.Omega_c)),
                        ("sigma8", args.prior_sigma8, float(ic_cosmo.sigma8)),
                        ("h", args.prior_h, float(ic_cosmo.h)),
                    )
                }
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
        # Lensing + optional pixel-likelihood scale cut (ell_max None -> no cut)
        lensing_output=args.lensing_output,
        ell_max=args.ell_max,
        ell_taper_width=args.ell_taper_width,
        map2alm_method=args.map2alm_method,
        min_redshift=args.min_z,
        max_redshift=args.max_z,
        n_integrate=args.n_integrate,
        quadrature=args.quadrature,
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
        # Cosmology and nz (fiducial is Planck18 when sampling cosmology, else pinned to the IC cosmology)
        fiducial_cosmology=fiducial_cosmology,
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
        mclmc_init_step_size=args.mclmc_init_step_size,
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
        mclmc_init_step_size=config.mclmc_init_step_size,
        mclmc_diagonal_preconditioning=config.mclmc_diagonal_preconditioning,
        init_params=init_params,
        progress_bar=not args.no_progress_bar,
        save_callback=jfli.ppl.sample2catalog(config),
        # Recolor the WHITE initial_conditions -> physical delta per sample (inside the sampling scan,
        # one field at a time) with each sample's cosmology; sample2catalog then just saves it.
        post_process=jfli.infer.colour_ic(config),
    )


if __name__ == "__main__":
    main()
