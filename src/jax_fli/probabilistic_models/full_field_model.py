"""Probabilistic wrappers that build on the deterministic forward model."""

from __future__ import annotations

import jax.numpy as jnp
import jax_cosmo as jc
import numpyro
from numpyro.handlers import reparam
from numpyro.infer.reparam import TransformReparam

from ..fields import DensityField, FlatDensity, FlatShearField, SphericalDensity, SphericalShearField
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


def build_harmonic_whiteners(config: Configurations, nz_list, dispersion, observed_field):
    """Return ``residual_fn(observable) -> rho`` for the harmonic (ell-tapered) scale-cut likelihood,
    so the factor is ``-0.5 * sum(rho ** 2)``.

    Packs the residual MAP ``observable - observed_field`` through the shared, precomputed ell-taper
    (built once from ``observed_field`` via ``harmonic_pack_precompute`` and reused, so the SHT/taper
    convention cannot drift between data and model), then whitens by ``1/sqrt(N_ell)`` with the flat
    white noise ``N_ell = dispersion**2 / nbar``. Packing the residual (not ``pack(model) -
    pack(data)``) avoids cancellation of two large packed vectors near the solution; the two are equal
    because the pack is linear.
    """
    l_cut, width = config.ell_max, config.ell_taper_width
    arcmin_per_rad = (180.0 / jnp.pi) * 60.0
    inv_sqrt_nl = (
        jnp.asarray([jnp.sqrt(nz.gals_per_arcmin2 * arcmin_per_rad**2) for nz in nz_list])[:, None] / dispersion
    )
    pack = observed_field.harmonic_pack_precompute(l_cut, width, method=config.map2alm_method)

    def residual_fn(observable):
        residual = observable.replace(array=observable.array - observed_field.array)
        return residual.harmonic_pack(precompute=pack) * inv_sqrt_nl

    return residual_fn


def make_likelihood(config: Configurations, observed_maps):
    """Build the likelihood closure ``likelihood(observable)`` (dispatch once, at build time).

    Implements the whitened residual ``rho = W * N^{-1/2} * (d - A[s])`` in the space selected
    by ``config.likelihood_space``:

    - ``"pixel"``: a per-pixel Gaussian on every observable map, registering conditionable
      ``observable_i`` sites.
    - ``"harmonic"``: the ell-tapered Gaussian on the whitened harmonic residual, entering as a
      ``numpyro.factor`` bound to ``observed_maps`` (no conditionable sites).
    """
    nz_list = _nz_to_distributions(config.nz_shear)
    geometry = config.geometry

    if config.likelihood_space == "harmonic":
        if config.ell_max is None:
            raise ValueError("likelihood_space='harmonic' requires config.ell_max")
        if config.mask is not None:
            raise NotImplementedError(
                "likelihood_space='harmonic' does not support a survey mask (a masked-sky scale cut "
                "needs the pseudo-Cl machinery); use the pixel likelihood or a full-sky setup"
            )
        if observed_maps is None:
            raise ValueError("likelihood_space='harmonic' requires the observed_maps argument")

        # Wrap the observed maps in the field class matching (geometry, spin) so the data goes through
        # the SAME harmonic_pack as the model observable (the pack itself is data-independent).
        obs = jnp.asarray(observed_maps)
        if obs.shape[0] != len(nz_list):  # accept a single unbatched map ((npix,)/(2,npix)/(ny,nx)/(2,ny,nx))
            obs = obs[None]
        if obs.shape[0] != len(nz_list):
            raise ValueError("observed_maps must have one map per nz_shear entry")
        spin2 = config.lensing_output in ("shear", "reduced_shear")
        if geometry == "spherical":
            observed_cls = SphericalShearField if spin2 else SphericalDensity
        else:
            observed_cls = FlatShearField if spin2 else FlatDensity
        observed_field = observed_cls(
            array=obs,
            mesh_size=config.mesh_size,
            box_size=config.box_size,
            nside=config.nside,
            flatsky_npix=config.flatsky_npix,
            field_size=config.field_size,
        )
        residual_fn = build_harmonic_whiteners(config, nz_list, _dispersion(config), observed_field)

        def harmonic_likelihood(observable):
            numpyro.factor("harmonic_loglik", -0.5 * jnp.sum(residual_fn(observable) ** 2))
            return observable

        return harmonic_likelihood

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

    def pixel_likelihood(observable):
        pixel_area_arcmin2 = pixel_area(observable)
        observed = []
        for idx, (observable_map, nz) in enumerate(zip(observable, nz_list)):
            sigma_obs = dispersion / jnp.sqrt(nz.gals_per_arcmin2 * pixel_area_arcmin2)
            if mask is None:
                scale = sigma_obs
            else:
                # spherical shear is (2, npix) per bin: insert the spin-2 component axis
                m = mask[None, :] if is_spherical_shear else mask
                scale = jnp.where(m > 0, sigma_obs, config.sigma_unobserved)
            observed.append(numpyro.sample(f"observable_{idx}", DistributedNormal(loc=observable_map, scale=scale)))
        return observed

    return pixel_likelihood


def full_field_probmodel(config: Configurations, observed_maps=None):
    """Return a NumPyro model for joint inference of cosmology and initial-condition fields.

    The initial conditions use a :class:`~numpyro.infer.reparam.TransformReparam` base: NUTS traverses
    the WHITE ``initial_conditions_base`` while the power-spectrum transform yields the colored
    ``initial_conditions`` the forward model uses; the cosmology priors come from ``config.priors``.
    The observable and likelihood are selected by
    ``config.likelihood_space`` ("pixel" per-pixel Normal, or "harmonic" ell-tapered scale cut),
    ``config.lensing_output`` and ``config.geometry``.

    Parameters
    ----------
    config : Configurations
        Model configuration.
    observed_maps : array_like, optional
        Observed per-bin maps ``(n_bins, npix)``. Required when
        ``config.likelihood_space == "harmonic"``: the ell-tapered likelihood enters as a
        ``numpyro.factor`` on the whitened residual, so there are no conditionable
        ``observable_*`` sites in that mode (do NOT wrap the model in ``condition``). Ignored by
        the pixel likelihood, which keeps the conditionable sites.
    """
    if config.likelihood_space not in ("pixel", "harmonic"):
        raise ValueError(f"Unknown likelihood_space: {config.likelihood_space!r}")

    forward_model = make_full_field_model(config=config)
    likelihood = make_likelihood(config, observed_maps)

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
