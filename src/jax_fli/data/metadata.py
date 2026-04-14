from __future__ import annotations

import jax.numpy as jnp

__all__ = ["_max_z_source"]


def _max_z_source(nz, min_z: float, max_z: float, N: int = 32) -> jnp.ndarray:
    """Compute mean source redshift ⟨z⟩ = ∫ z n(z) dz / ∫ n(z) dz.

    Uses composite Simpson's rule over [min_z, max_z] with N intervals (N+1 points, N must be even).

    Parameters
    ----------
    nz : callable
        Redshift distribution callable n(z).
    min_z : float
        Lower integration bound.
    max_z : float
        Upper integration bound.
    N : int, optional
        Number of intervals (must be even). Default 32.

    Returns
    -------
    jnp.ndarray
        Scalar effective redshift.
    """
    z_quad = jnp.linspace(min_z, max_z, N + 1)
    dz = (max_z - min_z) / N
    nz_v = nz(z_quad)
    num = dz / 3 * jnp.sum((nz_v * z_quad)[0:-1:2] + 4 * (nz_v * z_quad)[1::2] + (nz_v * z_quad)[2::2])
    den = dz / 3 * jnp.sum(nz_v[0:-1:2] + 4 * nz_v[1::2] + nz_v[2::2])
    return num / den
