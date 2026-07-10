from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import jax_cosmo.constants as constants
import numpy as np
from jax.scipy.ndimage import map_coordinates
from jaxtyping import ArrayLike

from ...fields import FlatDensity, SphericalDensity
from ..base._enums import DensityUnit
from ._normalize_nz import _normalize_sources

__all__ = ["_born_core_impl", "_born_spherical", "_born_flat", "_born_windows"]

_GL_ORDER = 16  # GL-8 vs GL-16 differ only at the a_of_chi interpolation floor (~1.6e-6); no knob.
_SIMPSON_ORDER = 16  # subintervals per shell (17 nodes, node-matched to GL-16); the floor is reached by ~9 nodes.


def _simps_weights(N: int):
    """Normalized composite-Simpson pattern ``[1, 4, 2, 4, ..., 4, 1] / 3`` for N subintervals (N+1 points).

    A folded numpy constant shared by the outer n(z) rule and the per-shell ``"simpson"`` windows;
    the caller multiplies by its own (possibly traced, per-shell) step ``dx = (b - a) / N`` to get the
    ``jax_cosmo.scipy.integrate.simps`` weights ``dx/3 * [1, 4, 2, ..., 4, 1]``.
    """
    w_np = np.ones(N + 1)
    w_np[1:-1:2] = 4.0  # odd indices → weight 4
    w_np[2:-2:2] = 2.0  # interior even indices → weight 2
    return jnp.asarray(w_np) / 3.0  # shape (N+1,)


def _born_windows(cosmo, r, a, d_r, chi_s, quadrature):
    """Per-shell Born weights ``W`` of shape ``(S, Z)`` for shells ``r``/``a``/``d_r`` and sources ``chi_s``.

    ``"midpoint"`` is the historic estimate ``d_r * (r/a) * clip(1 - r/chi_s, 0, 1000)`` with everything
    evaluated at the shell centers — exact only for shells thin compared to the kernel curvature.
    ``"simpson"`` and ``"gauss_legendre"`` integrate the kernel ``chi * (1 + z(chi)) * (1 - chi/chi_s)``
    over each shell's TRUNCATED support ``[r - d_r/2, min(r + d_r/2, chi_s)]`` (never across the clip
    kink), using ``1 + z = 1/a_of_chi`` (same distance-table cache, jit-safe) — composite Simpson with
    17 equispaced nodes vs fixed GL-16 nodes; both sit at the a_of_chi interpolation floor.
    """
    if quadrature == "midpoint":
        g = jnp.clip(1.0 - r[:, None] / chi_s[None, :], 0, 1000)  # (S, Z)
        return (d_r * r / a)[:, None] * g
    if quadrature == "simpson":
        S, Z = r.shape[0], chi_s.shape[0]
        lo, hi = r - d_r / 2, r + d_r / 2
        w = _simps_weights(_SIMPSON_ORDER)  # normalized pattern; the per-shell step span/Q scales it below
        upper = jnp.minimum(hi[:, None], chi_s[None, :])  # (S, Z)
        span = jnp.clip(upper - lo[:, None], 0.0, None)  # 0 where chi_s <= lo → weight exactly 0
        chi_q = lo[:, None, None] + span[..., None] * jnp.linspace(0.0, 1.0, _SIMPSON_ORDER + 1)  # (S, Z, Q+1)
        inv_a = 1.0 / jc.background.a_of_chi(cosmo, chi_q.reshape(-1)).reshape(S, Z, _SIMPSON_ORDER + 1)
        integrand = chi_q * inv_a * (1.0 - chi_q / chi_s[None, :, None])  # 0 at a truncated upper endpoint
        return (span / _SIMPSON_ORDER) * jnp.einsum("q,szq->sz", w, integrand)
    if quadrature == "gauss_legendre":
        S, Z = r.shape[0], chi_s.shape[0]
        lo, hi = r - d_r / 2, r + d_r / 2
        x_np, w_np = np.polynomial.legendre.leggauss(_GL_ORDER)  # folded numpy constants
        x, w = jnp.asarray(x_np), jnp.asarray(w_np)
        upper = jnp.minimum(hi[:, None], chi_s[None, :])  # (S, Z)
        span = jnp.clip(upper - lo[:, None], 0.0, None)  # 0 where chi_s <= lo
        chi_q = lo[:, None, None] + 0.5 * span[..., None] * (x + 1.0)  # (S, Z, Q)
        inv_a = 1.0 / jc.background.a_of_chi(cosmo, chi_q.reshape(-1)).reshape(S, Z, _GL_ORDER)
        integrand = chi_q * inv_a * (1.0 - chi_q / chi_s[None, :, None])
        return 0.5 * span * jnp.einsum("q,szq->sz", w, integrand)
    raise ValueError(f"Unknown quadrature: {quadrature!r} (expected 'midpoint', 'simpson' or 'gauss_legendre')")


def _born_core_impl(
    cosmo,
    delta_planes,
    r,
    a,
    z_source,
    d_r,
    pixel_size=None,
    coords=None,
    field_size=None,
    quadrature="simpson",
):
    constant_factor = 3 / 2 * cosmo.Omega_m * (constants.H0 / constants.c) ** 2
    chi_s = jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(z_source))
    n_planes = len(r)

    is_spherical = delta_planes.ndim == 2

    if not is_spherical:
        if pixel_size is None:
            raise AssertionError("pixel_size is required for flat geometry")
        pixel_size = jnp.asarray(pixel_size)

        if coords is None:
            if field_size is None:
                raise AssertionError("field_size is required when coords not provided")
            ny, nx = delta_planes.shape[-2:]
            if isinstance(field_size, (tuple | list | ArrayLike)):
                fx, fy = field_size
            else:
                fx = fy = field_size
            xgrid, ygrid = jnp.meshgrid(
                jnp.linspace(0, fx, nx, endpoint=False),
                jnp.linspace(0, fy, ny, endpoint=False),
            )
            coords = jnp.stack([xgrid, ygrid], axis=0) * (jnp.pi / 180)

    # (S, Z) per-shell weights — the only place the shell quadrature enters. The shell map is
    # constant along chi within its own shell, so refining the weight integral covers spherical
    # and flat alike with no extra interpolation.
    W = constant_factor * _born_windows(cosmo, r, a, d_r, chi_s.reshape(-1), quadrature)

    if not is_spherical:

        def interpolate_plane(delta_plane, chi_plane):
            assert pixel_size is not None and coords is not None, "pixel_size and coords are required for interpolation"
            physical_coords = coords * chi_plane / pixel_size[:, None, None]
            return map_coordinates(delta_plane, physical_coords - 0.5, order=1, mode="wrap")

        # map_coordinates is linear, so the per-shell scalar weights commute with the remap.
        delta_planes = jax.vmap(interpolate_plane)(delta_planes, r)

    final_contributions = W.reshape(n_planes, -1, *((1,) * (delta_planes.ndim - 1))) * delta_planes[:, None, ...]
    convergence = jnp.sum(final_contributions, axis=0)

    if jnp.ndim(z_source) == 0:
        convergence = jnp.squeeze(convergence)

    return convergence


def _born_spherical(
    cosmo,
    lightcone: SphericalDensity,
    r_center,
    scale_factors,
    nz_shear,
    density_plane_width,
    min_z,
    max_z,
    n_integrate,
    normalization="global",
    quadrature="simpson",
):
    source_kind, sources = _normalize_sources(nz_shear)
    lightcone_od = lightcone.to(DensityUnit.OVERDENSITY, normalization=normalization)

    if source_kind == "distribution":
        # Evaluate _born_core_impl once for all source-grid points (shape: n_z, npix), then collapse
        # with per-distribution n(z) weights. The n(z) rule follows the quadrature: gauss_legendre uses
        # GL nodes/weights on [min_z, max_z]; simpson and the legacy midpoint route keep the historic
        # equispaced Simpson grid.
        if quadrature == "gauss_legendre":
            x_np, w_np = np.polynomial.legendre.leggauss(n_integrate)  # folded numpy constants
            half = 0.5 * (max_z - min_z)
            z_grid = min_z + half * (jnp.asarray(x_np) + 1.0)  # (n_z,)
            weights = half * jnp.asarray(w_np)  # (n_z,)
        else:
            z_grid = jnp.linspace(min_z, max_z, n_integrate + 1)  # (n_z,)
            weights = _simps_weights(n_integrate) * ((max_z - min_z) / n_integrate)  # (n_z,)
        kappa_grid = _born_core_impl(
            cosmo,
            lightcone_od.array,
            r_center,
            scale_factors,
            z_grid,
            density_plane_width,
            quadrature=quadrature,
        )  # (n_z, npix)
        nz_weights = jnp.stack([nz(z_grid) for nz in sources], axis=0)  # (K, n_z)
        kappa_maps = jnp.einsum("kz,z,zp->kp", nz_weights, weights, kappa_grid)  # (K, npix)
    else:
        kappa_maps = _born_core_impl(
            cosmo, lightcone_od.array, r_center, scale_factors, sources, density_plane_width, quadrature=quadrature
        )
    return kappa_maps


def _born_flat(
    cosmo,
    lightcone: FlatDensity,
    r_center,
    scale_factors,
    nz_shear,
    density_plane_width,
    min_z,
    max_z,
    n_integrate,
    normalization="global",
    quadrature="simpson",
):
    if lightcone.field_size is None:
        raise ValueError("field_size is required on lightcone for flat-sky convergence")

    source_kind, sources = _normalize_sources(nz_shear)
    lightcone_od = lightcone.to(DensityUnit.OVERDENSITY, normalization=normalization)

    field_size = lightcone.field_size
    if isinstance(field_size, (tuple | list | ArrayLike)):
        field_size_tuple = tuple(field_size)
    else:
        field_size_tuple = (field_size, field_size)

    if lightcone.flatsky_npix is None:
        raise ValueError("flatsky_npix must be set on lightcone for flat geometry")
    ny, nx = lightcone.flatsky_npix
    pixel_size = (
        lightcone.box_size[1] / ny,
        lightcone.box_size[0] / nx,
    )

    if source_kind == "distribution":
        # Evaluate _born_core_impl once for all source-grid points (shape: n_z, ny, nx), then collapse
        # with per-distribution n(z) weights. The n(z) rule follows the quadrature: gauss_legendre uses
        # GL nodes/weights on [min_z, max_z]; simpson and the legacy midpoint route keep the historic
        # equispaced Simpson grid.
        if quadrature == "gauss_legendre":
            x_np, w_np = np.polynomial.legendre.leggauss(n_integrate)  # folded numpy constants
            half = 0.5 * (max_z - min_z)
            z_grid = min_z + half * (jnp.asarray(x_np) + 1.0)  # (n_z,)
            weights = half * jnp.asarray(w_np)  # (n_z,)
        else:
            z_grid = jnp.linspace(min_z, max_z, n_integrate + 1)  # (n_z,)
            weights = _simps_weights(n_integrate) * ((max_z - min_z) / n_integrate)  # (n_z,)
        kappa_grid = _born_core_impl(
            cosmo,
            lightcone_od.array,
            r_center,
            scale_factors,
            z_grid,
            density_plane_width,
            pixel_size=pixel_size,
            field_size=field_size_tuple,
            quadrature=quadrature,
        )  # (n_z, ny, nx)
        nz_weights = jnp.stack([nz(z_grid) for nz in sources], axis=0)  # (K, n_z)
        kappa_maps = jnp.einsum("kz,z,zyx->kyx", nz_weights, weights, kappa_grid)  # (K, ny, nx)
    else:
        kappa_maps = _born_core_impl(
            cosmo,
            lightcone_od.array,
            r_center,
            scale_factors,
            sources,
            density_plane_width,
            pixel_size=pixel_size,
            field_size=field_size_tuple,
            quadrature=quadrature,
        )

    return kappa_maps
