"""Galaxy number-counts (clustering) projection of a density lightcone.

The clustering analogue of :func:`jax_fli.lensing.born`: it projects the density shells
onto per-bin galaxy-overdensity maps with a redshift-selection kernel ``n(z)`` and no
lensing weight. The observable is the projected galaxy overdensity (``OVERDENSITY`` units,
NOT the ``COUNTS`` unit — "number counts" names the clustering probe, not the output units).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax.scipy.ndimage import map_coordinates

from .._src.lensing._metadata import _attach_source_metadata
from .._src.lensing._normalize_nz import _normalize_sources
from ..fields import DensityUnit, FieldStatus, FlatDensity, SphericalDensity

__all__ = ["number_counts"]


def _selection_weights(source_kind, sources, z_shells):
    """Per-bin shell weights ``w`` (K, n_shells), rows summing to 1, for the radial sum.

    - ``"distribution"``: ``w_{k,i} = n_k(z_i) * dz_i`` (the source n(z) sampled at the shell
      redshifts, trapezoidal ``dz``), normalized.
    - ``"redshift"`` (delta sources): linear-interpolation weights so the sum returns the
      density at the source plane ``z_s`` (interpolated between the two bracketing shells).
    """
    n = z_shells.size
    if source_kind == "distribution":
        dz = jnp.abs(jnp.gradient(z_shells)) if n > 1 else jnp.ones_like(z_shells)
        w = jnp.stack([nz(z_shells) for nz in sources], axis=0) * dz[None, :]
        return w / jnp.sum(w, axis=1, keepdims=True)

    if n == 1:
        return jnp.ones((len(sources), 1))
    order = jnp.argsort(z_shells)
    z_sorted = z_shells[order]

    def one(z_s):
        idx = jnp.clip(jnp.searchsorted(z_sorted, z_s) - 1, 0, n - 2)
        span = z_sorted[idx + 1] - z_sorted[idx]
        t = jnp.clip(jnp.where(span > 0, (z_s - z_sorted[idx]) / span, 0.0), 0.0, 1.0)
        w_sorted = jnp.zeros(n).at[idx].add(1.0 - t).at[idx + 1].add(t)
        return jnp.zeros(n).at[order].set(w_sorted)

    return jnp.stack([one(jnp.asarray(z_s)) for z_s in sources], axis=0)


def _remap_shells_to_grid(shells, r_center, lightcone):
    """Remap each flat density shell onto the observer angular grid (as Born does for flat).

    Each shell lives at its own comoving distance, so it subtends a different angular scale;
    ``map_coordinates`` resamples it onto the fixed observer grid before the radial sum.
    """
    ny, nx = lightcone.flatsky_npix
    field_size = lightcone.field_size
    fx, fy = field_size if isinstance(field_size, (tuple, list)) else (field_size, field_size)
    pixel_size = jnp.asarray([lightcone.box_size[1] / ny, lightcone.box_size[0] / nx])
    xgrid, ygrid = jnp.meshgrid(jnp.linspace(0, fx, nx, endpoint=False), jnp.linspace(0, fy, ny, endpoint=False))
    coords = jnp.stack([xgrid, ygrid], axis=0) * (jnp.pi / 180)  # angular grid in radians

    def interpolate_plane(shell, chi):
        physical_coords = coords * chi / pixel_size[:, None, None]
        return map_coordinates(shell, physical_coords - 0.5, order=1, mode="wrap")

    return jax.vmap(interpolate_plane)(shells, r_center)


def number_counts(
    cosmo,
    lightcone,
    nz_shear,
    min_z=0.01,
    max_z=1.5,
    n_integrate=32,
    bias=1.0,
    normalization="per_plane",
):
    """Project a density lightcone onto per-bin galaxy-overdensity maps.

    ``delta_g^k(n) = bias * sum_i w_{k,i} delta_i(n)`` with ``w_{k,i} = n_k(z_i) * dz_i``
    normalized so ``sum_i w_{k,i} = 1`` per bin — the shell overdensities ``delta_i``
    weighted by the source redshift distribution ``n(z)``. No lensing kernel. Returns a
    :class:`SphericalDensity` / :class:`FlatDensity` with ``status=PROJECTED_DENSITY`` and
    per-bin source metadata; shape ``(K, npix)`` / ``(K, ny, nx)``.

    ``min_z`` / ``max_z`` / ``n_integrate`` only set the per-bin source metadata (the
    projection itself is shell-based). The observable is spin-0, so the pixel and harmonic
    likelihoods reach it directly.

    Note (unvalidated): pairing this with the density noise model (dispersion 1 ->
    ``N_l = 1 / nbar``) assumes ``delta_g``'s Poisson shot noise is exactly ``1 / nbar``,
    which depends on the ``n(z)`` normalization and bias convention — validate that
    shot-noise consistency before trusting density in inference.
    """
    if nz_shear is None:
        raise ValueError("nz_shear must be provided for number counts")
    if lightcone.status != FieldStatus.LIGHTCONE:
        raise ValueError(f"Expected lightcone with status=LIGHTCONE, got {lightcone.status}")

    overdensity = lightcone.to(DensityUnit.OVERDENSITY, normalization=normalization)
    scale_factors = jnp.atleast_1d(lightcone.scale_factors)
    z_shells = jc.utils.a2z(scale_factors)

    source_kind, sources = _normalize_sources(nz_shear)
    weights = bias * _selection_weights(source_kind, sources, z_shells)  # (K, n_shells)

    is_spherical = isinstance(lightcone, SphericalDensity)
    if is_spherical:
        maps = jnp.einsum("ki,ip->kp", weights, overdensity.array)  # (K, npix)
    else:
        r_center = jc.background.radial_comoving_distance(cosmo, scale_factors)
        remapped = _remap_shells_to_grid(overdensity.array, r_center, lightcone)  # (n_shells, ny, nx)
        maps = jnp.einsum("ki,iyx->kyx", weights, remapped)  # (K, ny, nx)

    base = lightcone.replace(status=FieldStatus.PROJECTED_DENSITY)
    field_cls = SphericalDensity if is_spherical else FlatDensity
    result = field_cls.FromDensityMetadata(
        array=maps,
        field=base,
        status=FieldStatus.PROJECTED_DENSITY,
        unit=DensityUnit.OVERDENSITY,
        z_sources=nz_shear,
    )
    # replace shell-level metadata with per-source-bin metadata (as Born does)
    result = _attach_source_metadata(result, cosmo, source_kind, sources, min_z, max_z, n_integrate)
    if is_spherical:
        result = result.apply_sharding()
    return result
