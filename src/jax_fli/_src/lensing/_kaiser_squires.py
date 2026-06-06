"""Kaiser-Squires spin-0 <-> spin-2 transforms on the sphere (pure ``jax_healpy``).

Array-level, jittable helpers behind ``SphericalKappaField.get_shear`` /
``SphericalShearField.get_convergence``. Convergence kappa is a spin-0 (scalar) field; shear
``(gamma1, gamma2)`` is its spin-2 partner. We use the BornRaytrace prefactor
``E_l = sqrt((l+2)(l-1) / (l(l+1))) * kappa_l`` (and its inverse), zeroed at ``l = 0, 1``, applied
with ``jhp.almxfl`` so the functions are ``jit``-safe (no ``hp.Alm.getlm``). Shear produced this way
is pure E-mode by construction.

Validated full-sky round-trip in ``tests/lensing/purification_study/ks_forward_study.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_healpy as jhp
import numpy as np

__all__ = ["kappa2shear", "shear2kappa"]


def _ks_factors(lmax: int):
    """(kappa->E, E->kappa) per-l prefactors, zeroed at l=0,1."""
    ell = np.arange(lmax + 1)
    fk2g = np.zeros(lmax + 1)
    fk2g[2:] = np.sqrt((ell[2:] + 2.0) * (ell[2:] - 1.0) / (ell[2:] * (ell[2:] + 1.0)))
    fg2k = np.where(fk2g > 0, 1.0 / np.where(fk2g == 0, 1.0, fk2g), 0.0)
    return jnp.asarray(fk2g), jnp.asarray(fg2k)


def _k2g_one(kappa, nside: int, lmax: int, method: str):
    fk2g, _ = _ks_factors(lmax)
    alm = jhp.map2alm(kappa, lmax=lmax, pol=False, healpy_ordering=True, method=method)
    E = jhp.almxfl(alm, fk2g, healpy_ordering=True)
    g1, g2 = jhp.alm2map([E, jnp.zeros_like(E)], nside, lmax=lmax, pol=True, healpy_ordering=True, method=method)
    return jnp.stack([g1, g2], axis=0)  # (2, npix)


def _g2k_one(gamma, nside: int, lmax: int, method: str):
    _, fg2k = _ks_factors(lmax)
    E, _B = jhp.map2alm_spin([gamma[0], gamma[1]], spin=2, lmax=lmax, healpy_ordering=True, method=method)
    return jhp.alm2map(jhp.almxfl(E, fg2k, healpy_ordering=True), nside, lmax=lmax, pol=False, healpy_ordering=True, method=method)


def kappa2shear(kappa, *, lmax: int | None = None, method: str = "jax"):
    """Convergence -> shear via Kaiser-Squires. ``(..., npix)`` -> ``(..., 2, npix)`` (pure E).

    Leading dimensions are treated as batch and mapped with ``vmap``. ``lmax``/``method`` are
    static (defaults: ``lmax = 3*nside-1``, ``method='jax'``); jittable.
    """
    kappa = jnp.asarray(kappa)
    npix = kappa.shape[-1]
    nside = jhp.npix2nside(npix)
    if lmax is None:
        lmax = 3 * nside - 1
    lead = kappa.shape[:-1]
    flat = kappa.reshape((-1, npix))
    out = jax.vmap(lambda k: _k2g_one(k, nside, lmax, method))(flat)  # (B, 2, npix)
    return out.reshape((*lead, 2, npix))


def shear2kappa(gamma, *, lmax: int | None = None, method: str = "jax"):
    """Shear -> convergence via inverse Kaiser-Squires. ``(..., 2, npix)`` -> ``(..., npix)``.

    Uses the E-mode only. Leading dimensions are batch (``vmap``). ``lmax``/``method`` static.
    """
    gamma = jnp.asarray(gamma)
    if gamma.shape[-2] != 2:
        raise ValueError(f"shear array must have a trailing (2, npix) spin-2 axis, got shape {gamma.shape}.")
    npix = gamma.shape[-1]
    nside = jhp.npix2nside(npix)
    if lmax is None:
        lmax = 3 * nside - 1
    lead = gamma.shape[:-2]
    flat = gamma.reshape((-1, 2, npix))
    out = jax.vmap(lambda g: _g2k_one(g, nside, lmax, method))(flat)  # (B, npix)
    return out.reshape((*lead, npix))
