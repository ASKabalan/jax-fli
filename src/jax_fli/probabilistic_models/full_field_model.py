"""Probabilistic wrappers that build on the deterministic forward model."""

from __future__ import annotations

import jax.numpy as jnp
import jax_cosmo as jc
import numpyro
from numpyro.handlers import reparam
from numpyro.infer.reparam import TransformReparam

from ..fields import DensityField
from ..infer import DistributedNormal
from ..initial import interpolate_initial_conditions
from .config import Configurations
from .forward_model import make_full_field_model

__all__ = ["make_full_field_model", "full_field_probmodel", "mock_probmodel"]


def _dispersion(config: Configurations) -> float:
    """Per-galaxy noise dispersion by observable (methods note, section 4): ``1`` for density
    (Poisson), ``sigma_e / sqrt(2)`` for convergence, ``sigma_e`` for shear."""
    return {
        "density": 1.0,
        "convergence": config.sigma_e / jnp.sqrt(2.0),
        "shear": config.sigma_e,
        "reduced_shear": config.sigma_e,
    }[config.lensing_output]


def _flat_pixel_area_arcmin2(field: DensityField) -> float:
    if field.flatsky_npix is None or field.field_size is None:
        raise ValueError("Flat maps require both flatsky_npix and field_size metadata")
    size_y, size_x = field.field_size
    ny, nx = field.flatsky_npix
    return (size_y * 60.0 / ny) * (size_x * 60.0 / nx)


def _spherical_pixel_area_arcmin2(field: DensityField) -> float:
    if field.nside is None:
        raise ValueError("Spherical maps require nside metadata")
    arcmin_per_rad = (180.0 / jnp.pi) * 60.0
    return (4.0 * jnp.pi / (12.0 * field.nside**2)) * arcmin_per_rad**2


def _nz_to_distributions(nz_shear):
    """Ensure every nz entry is a redshift_distribution; wrap floats in delta_nz."""
    if isinstance(nz_shear, (list | tuple)):
        return [
            nz if isinstance(nz, jc.redshift.redshift_distribution) else jc.redshift.delta_nz(nz) for nz in nz_shear
        ]
    # JAX float array (from _resolve_nz_shear scalar path)
    return [jc.redshift.delta_nz(z) for z in nz_shear]


def make_likelihood(config: Configurations):
    """Build the pixel-space likelihood closure ``likelihood(observable)`` (dispatch once at build time).

    A per-pixel Gaussian on every observable map, registering conditionable ``observable_i`` sites.
    The survey ``config.mask`` (if any) enters as an inflated per-pixel sigma (``config.sigma_unobserved``)
    outside the footprint. If ``config.ell_max`` is set, each observable map is first band-limited to
    ``ell_max`` (``map2alm`` -> cosine taper -> ``alm2map``, via ``scale_cut``) before the Gaussian --
    the field-level scale cut. The scale cut is spherical-only (spin-0 density/convergence via
    ``SphericalDensity.scale_cut`` and spin-2 shear via ``SphericalShearField.scale_cut``); flat
    geometry with ``ell_max`` set raises.
    """
    nz_list = _nz_to_distributions(config.nz_shear)
    geometry = config.geometry

    if geometry == "spherical":
        pixel_area = _spherical_pixel_area_arcmin2
    elif geometry == "flat":
        pixel_area = _flat_pixel_area_arcmin2
    else:
        raise ValueError("geometry must be 'flat' or 'spherical'")

    dispersion = _dispersion(config)
    is_spherical_shear = geometry == "spherical" and config.lensing_output in ("shear", "reduced_shear")
    # Optional survey footprint mask (e.g. DES Y3): inflate the per-pixel sigma on unobserved
    # pixels (mask == 0) so they don't constrain the fit.
    mask = None if config.mask is None else jnp.asarray(config.mask)

    # Optional field-level scale cut: band-limit each observable map to ell_max before the Gaussian.
    do_scale_cut = config.ell_max is not None
    if do_scale_cut:
        if geometry != "spherical":
            raise NotImplementedError("config.ell_max scale cut is spherical-only (no flat scale_cut)")
        if not 0 < config.ell_taper_width <= config.ell_max:
            raise ValueError(
                f"ell_taper_width must be in (0, ell_max]; got ell_taper_width={config.ell_taper_width}, "
                f"ell_max={config.ell_max}"
            )

    def pixel_likelihood(observable):
        pixel_area_arcmin2 = pixel_area(observable)
        observed = []
        for idx, (observable_map, nz) in enumerate(zip(observable, nz_list)):
            loc = (
                observable_map.scale_cut(config.ell_max, config.ell_taper_width, method=config.map2alm_method)
                if do_scale_cut
                else observable_map
            )
            sigma_obs = dispersion / jnp.sqrt(nz.gals_per_arcmin2 * pixel_area_arcmin2)
            if mask is None:
                scale = sigma_obs
            else:
                # spherical shear is (2, npix) per bin: insert the spin-2 component axis
                m = mask[None, :] if is_spherical_shear else mask
                scale = jnp.where(m > 0, sigma_obs, config.sigma_unobserved)
            observed.append(numpyro.sample(f"observable_{idx}", DistributedNormal(loc=loc, scale=scale)))
        return observed

    return pixel_likelihood


def full_field_probmodel(config: Configurations):
    """Return a NumPyro model for joint inference of cosmology and initial-condition fields.

    The initial conditions are sampled WHITE (the ``initial_conditions`` site is a unit
    ``DistributedNormal`` the sampler traverses directly) and colored inline with the
    cosmology-dependent power spectrum; the cosmology priors come from ``config.priors``.
    The pixel-space likelihood registers conditionable ``observable_i`` sites (optionally band-limited
    to ``config.ell_max``); the observable is selected by ``config.lensing_output`` and
    ``config.geometry``.
    """
    forward_model = make_full_field_model(config=config)
    likelihood = make_likelihood(config)

    def model():
        # Sample the cosmology in WHITE space: TransformReparam on the (PreconditionnedUniform) priors
        # so NUTS moves the `<name>_base` Gaussians. Double-storing the small cosmo scalars is fine.
        with reparam(config={name: TransformReparam() for name in config.priors}):
            cosmo = config.fiducial_cosmology(**{k: numpyro.sample(k, prior) for k, prior in config.priors.items()})
        numpyro.deterministic("cosmo", cosmo)

        # Sample WHITE initial conditions, then color inline with the power spectrum for the forward
        # model. `interpolate_initial_conditions` returns a ready DensityField; the trace keeps only the
        # white `initial_conditions` (recolored to the physical delta at save time in sample2catalog).
        white = numpyro.sample(
            "initial_conditions",
            DistributedNormal(
                loc=jnp.zeros(config.mesh_size),
                scale=jnp.ones(config.mesh_size),
                mesh_size=config.mesh_size,
                box_size=config.box_size,
                observer_position=config.observer_position,
                halo_size=config.halo_size,
                nside=config.nside,
                flatsky_npix=config.flatsky_npix,
                field_size=config.field_size,
                field_sharding=config.field_sharding,
                field_type="density",
            ),
        )
        white_array = white.array if isinstance(white, DensityField) else white
        initial_conditions = interpolate_initial_conditions(
            white_array,
            config.mesh_size,
            config.box_size,
            cosmo=cosmo,
            observer_position=config.observer_position,
            halo_size=config.halo_size,
            nside=config.nside,
            flatsky_npix=config.flatsky_npix,
            field_size=config.field_size,
            field_sharding=config.field_sharding,
        )
        numpyro.deterministic("initial_conditions_meta_data", initial_conditions.to_metadata())

        observable, lightcone = forward_model(cosmo, initial_conditions)

        if config.log_lightcone:
            numpyro.deterministic("lightcone", lightcone)

        if observable.shape[0] != len(config.nz_shear):
            raise ValueError("Number of observable maps does not match nz_shear entries")

        numpyro.deterministic("observable_meta_data", observable.to_metadata())
        return likelihood(observable)

    return model


def mock_probmodel(config: Configurations):
    """Return a NumPyro model that samples only cosmology and initial conditions.

    No forward simulation is run. Registered deterministic sites match those
    expected by :func:`~jax_fli.probabilistic_models.sample2catalog`,
    so the callback can be used directly to save IC catalogs.
    """

    def model():
        with reparam(config={name: TransformReparam() for name in config.priors}):
            cosmo = config.fiducial_cosmology(**{k: numpyro.sample(k, prior) for k, prior in config.priors.items()})
        numpyro.deterministic("cosmo", cosmo)

        white = numpyro.sample(
            "initial_conditions",
            DistributedNormal(
                loc=jnp.zeros(config.mesh_size),
                scale=jnp.ones(config.mesh_size),
                mesh_size=config.mesh_size,
                box_size=config.box_size,
                observer_position=config.observer_position,
                halo_size=config.halo_size,
                nside=config.nside,
                flatsky_npix=config.flatsky_npix,
                field_size=config.field_size,
                field_sharding=config.field_sharding,
                field_type="density",
            ),
        )
        white_array = white.array if isinstance(white, DensityField) else white
        initial_conditions = interpolate_initial_conditions(
            white_array,
            config.mesh_size,
            config.box_size,
            cosmo=cosmo,
            observer_position=config.observer_position,
            halo_size=config.halo_size,
            nside=config.nside,
            flatsky_npix=config.flatsky_npix,
            field_size=config.field_size,
            field_sharding=config.field_sharding,
        )
        numpyro.deterministic("initial_conditions_meta_data", initial_conditions.to_metadata())
        return initial_conditions

    return model
