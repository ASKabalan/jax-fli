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


def _simps_weights(a: float, b: float, N: int):
    """Simpson's rule quadrature weights for N subintervals (N+1 points).

    Reproduces the exact weights used by ``jax_cosmo.scipy.integrate.simps``:
    ``dx/3 * [1, 4, 2, 4, ..., 4, 1]``.
    """
    dx = (b - a) / N
    w = jnp.ones(N + 1)
    w = w.at[1:-1:2].set(4.0)  # odd indices → weight 4
    w = w.at[2:-2:2].set(2.0)  # interior even indices → weight 2
    return w * (dx / 3.0)  # shape (N+1,)


def _born_windows(cosmo, r, a, d_r, chi_s, quadrature):
    """Per-shell Born weights ``W`` of shape ``(S, Z)`` for shells ``r``/``a``/``d_r`` and sources ``chi_s``.

    ``"midpoint"`` is the historic estimate ``d_r * (r/a) * clip(1 - r/chi_s, 0, 1000)`` with everything
    evaluated at the shell centers — exact only for shells thin compared to the kernel curvature.
    ``"gauss_legendre"`` integrates the kernel ``chi * (1 + z(chi)) * (1 - chi/chi_s)`` exactly over each
    shell's TRUNCATED support ``[r - d_r/2, min(r + d_r/2, chi_s)]`` with fixed GL-16 nodes (never across
    the clip kink), using ``1 + z = 1/a_of_chi`` (same distance-table cache, jit-safe).
    """
    if quadrature == "midpoint":
        g = jnp.clip(1.0 - r[:, None] / chi_s[None, :], 0, 1000)  # (S, Z)
        return (d_r * r / a)[:, None] * g
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
    raise ValueError(f"Unknown quadrature: {quadrature!r} (expected 'midpoint' or 'gauss_legendre')")


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
    quadrature="midpoint",
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
    quadrature="midpoint",
):
    source_kind, sources = _normalize_sources(nz_shear)
    lightcone_od = lightcone.to(DensityUnit.OVERDENSITY, normalization=normalization)

    if source_kind == "distribution":
        # Evaluate _born_core_impl once for all quadrature points (shape: n_z, npix),
        # then apply per-distribution n(z) weights via a manual Simpson's rule.
        z_grid = jnp.linspace(min_z, max_z, n_integrate + 1)  # (n_z,)
        kappa_grid = _born_core_impl(
            cosmo,
            lightcone_od.array,
            r_center,
            scale_factors,
            z_grid,
            density_plane_width,
            quadrature=quadrature,
        )  # (n_z, npix)
        weights = _simps_weights(min_z, max_z, n_integrate)  # (n_z,)
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
    quadrature="midpoint",
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
        # Evaluate _born_core_impl once for all quadrature points (shape: n_z, ny, nx),
        # then apply per-distribution n(z) weights via a manual Simpson's rule.
        z_grid = jnp.linspace(min_z, max_z, n_integrate + 1)  # (n_z,)
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
        weights = _simps_weights(min_z, max_z, n_integrate)  # (n_z,)
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
