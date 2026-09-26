"""Lensing-kernel-weighted projection of a 3D field onto the sphere (the kernel behind DensityField.sky_projection)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import jax_healpy as jhp
from jax.scipy.ndimage import map_coordinates
from jax_cosmo import constants


def lensing_efficiency(cosmo, nz_shear, chi, n_z=512):
    """Lensing efficiency per source bin on the comoving-distance grid ``chi`` (Mpc/h), shape ``(n_bins, n_chi)``.

    ``q_b(chi) = 3/2 Omega_m (H0/c)^2 chi / a(chi) * int dz n_b(z) (1 - chi / chi(z))``, the Born
    weight of a thin shell at ``chi`` for the normalised source distribution ``n_b``.
    """
    z_max = max(float(getattr(nz, "zmax", 3.0)) for nz in nz_shear)
    z = jnp.linspace(1e-3, z_max, n_z)
    chi_z = jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(z))
    a = jc.background.a_of_chi(cosmo, chi)
    geometry = jnp.clip(1.0 - chi[:, None] / chi_z[None, :], 0.0, None)
    prefactor = 1.5 * cosmo.Omega_m * (constants.H0 / constants.c) ** 2 * chi / a
    q = []
    for nz in nz_shear:
        n = nz(z)
        n = n / jnp.trapezoid(n, z)
        q.append(prefactor * jnp.trapezoid(n[None, :] * geometry, z, axis=1))
    return jnp.stack(q)


def sky_projection(array, mesh_size, box_size, observer_position, weights, chi, nside):
    """``sum_j weights[:, j] * f(x_obs + chi_j n) * dchi`` for every HEALPix pixel direction ``n`` (RING order).

    ``array`` is the 3D field on the mesh (voxel ``i`` at grid coordinate ``i``), ``weights`` the per-bin
    radial weights on the uniform grid ``chi`` (Mpc/h). Trilinear sampling; one radius at a time in a scan,
    so the memory is a few maps, not ``n_chi`` of them.
    """
    npix = jhp.nside2npix(nside)
    dtype = jnp.promote_types(array.dtype, weights.dtype)  # e.g. a float32 field with float64 weights
    array, weights, chi = array.astype(dtype), weights.astype(dtype), chi.astype(dtype)
    vec = jnp.asarray(jhp.pix2vec(nside, jnp.arange(npix)), dtype=dtype)
    vec = vec if vec.shape[0] == 3 else vec.T  # (3, npix) unit vectors
    mesh = jnp.asarray(mesh_size, dtype=dtype)
    cell = jnp.asarray(box_size, dtype=dtype) / mesh
    obs = jnp.asarray(observer_position, dtype=dtype) * mesh  # observer in grid units
    dchi = chi[1] - chi[0]

    def add_radius(acc, radius_and_weight):
        radius, w = radius_and_weight
        coords = obs[:, None] + radius * vec / cell[:, None]
        values = map_coordinates(array, [coords[0], coords[1], coords[2]], order=1, mode="wrap")
        return acc + w[:, None] * values[None, :], None

    acc0 = jnp.zeros((weights.shape[0], npix), dtype=dtype)
    projected, _ = jax.lax.scan(add_radius, acc0, (chi, weights.T))
    return projected * dchi
