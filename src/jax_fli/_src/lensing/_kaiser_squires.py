"""Kaiser-Squires spin-0 <-> spin-2 transforms (flat-sky and on the sphere).

Array-level, jittable helpers behind the field methods ``FlatKappaField.get_shear`` /
``FlatShearField.get_convergence`` and ``SphericalKappaField.get_shear`` /
``SphericalShearField.get_convergence``. These functions are internal: the public surface is the
field methods, which are what the tests exercise.

Convergence kappa is a spin-0 (scalar) field; shear ``(gamma1, gamma2)`` is its spin-2 partner.

Spherical (HEALPix)
    Uses the BornRaytrace prefactor ``E_l = sqrt((l+2)(l-1) / (l(l+1))) * kappa_l`` (and its
    inverse), zeroed at ``l = 0, 1``, applied with ``jhp.almxfl`` so the functions are ``jit``-safe
    (no ``hp.Alm.getlm``). Shear produced this way is pure E-mode by construction. Validated
    full-sky round-trip in ``tests/lensing/purification_study/ks_forward_study.py``.

Flat-sky (Cartesian)
    The spin-2 operator is the pure phase ``gamma~ = e^{2i phi} kappa~`` with
    ``phi = atan2(l_y, l_x)``. Writing ``A = cos 2phi = (lx^2 - ly^2) / l^2`` and
    ``B = sin 2phi = 2 lx ly / l^2`` (both zeroed at ``l = 0``), the forward map is
    ``gamma1 = Re ifft2(A kappa~)``, ``gamma2 = Re ifft2(B kappa~)`` and the inverse is
    ``kappa = Re ifft2(A gamma1~ + B gamma2~)``. Because ``A^2 + B^2 = 1`` away from ``l = 0`` the
    round-trip reproduces every reconstructable mode of ``kappa``. Two are not recovered: the DC mode
    (set to zero) and, on an even grid, the Nyquist row/column (there ``B`` is sign-ambiguous under
    ``k -> -k``, so its imaginary part is dropped by the real ``gamma`` representation) — both
    negligible for band-limited (smooth) convergence. The factors are scale-invariant (pixel size
    cancels in the ratio), so no field/pixel size is needed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_healpy as jhp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from ._sharding import _shear_spec, lensing_axes

__all__ = [
    "kappa2shear_flat",
    "kappa2shear_spherical",
    "shear2kappa_flat",
    "shear2kappa_spherical",
]


# --------------------------------------------------------------------------- #
# Spherical (HEALPix)                                                         #
# --------------------------------------------------------------------------- #
def _ks_factors(lmax: int):
    """(kappa->E, E->kappa) per-l prefactors, zeroed at l=0,1."""
    ell = np.arange(lmax + 1)
    fk2g = np.zeros(lmax + 1)
    fk2g[2:] = np.sqrt((ell[2:] + 2.0) * (ell[2:] - 1.0) / (ell[2:] * (ell[2:] + 1.0)))
    fg2k = np.where(fk2g > 0, 1.0 / np.where(fk2g == 0, 1.0, fk2g), 0.0)
    return jnp.asarray(fk2g), jnp.asarray(fg2k)


def _k2g_one(kappa, nside: int, lmax: int, method: str, iter: int):
    fk2g, _ = _ks_factors(lmax)
    alm = jhp.map2alm(kappa, lmax=lmax, iter=iter, pol=False, healpy_ordering=True, method=method)
    E = jhp.almxfl(alm, fk2g, healpy_ordering=True)
    g1, g2 = jhp.alm2map([E, jnp.zeros_like(E)], nside, lmax=lmax, pol=True, healpy_ordering=True, method=method)
    return jnp.stack([g1, g2], axis=0)  # (2, npix)


def _g2k_one(gamma, nside: int, lmax: int, method: str, iter: int):
    _, fg2k = _ks_factors(lmax)
    E, _B = jhp.map2alm_spin([gamma[0], gamma[1]], spin=2, lmax=lmax, iter=iter, healpy_ordering=True, method=method)
    return jhp.alm2map(
        jhp.almxfl(E, fg2k, healpy_ordering=True), nside, lmax=lmax, pol=False, healpy_ordering=True, method=method
    )


def _dbg_sharding(tag: str, arr, enabled: bool):
    """Print ``arr``'s shape (static) and resolved sharding when ``enabled`` (jit-safe)."""
    if enabled:
        jax.debug.inspect_array_sharding(
            arr, callback=lambda s: print(f"[get_shear] {tag:34s} shape={tuple(arr.shape)} sharding={s.spec}")
        )


def _kappa2shear_spherical_sharded(
    kappa, sharding, bins_axis, nside: int, lmax: int, method: str, iter: int, debug_sharding: bool = False
):
    """Kaiser-Squires on the sphere with the SHT sealed inside a ``jax.shard_map``.

    ``map2alm``/``alm2map`` are not partition-aware on ``npix``: the ``shard_map`` manual region keeps
    ``npix`` whole on each device *and* is a hard boundary that stops XLA propagating the npix-sharded
    output back into the FFT. ``bins_axis`` (the mesh's N axis, or ``None`` to replicate) carries the
    bins; the final ``_shear_spec`` constraint puts npix back on the M axis. The shear follows the
    lensing convention ``P([None,] N, None, M)`` (BINS/N, NPIX/M, component replicated).
    """
    mesh = sharding.mesh
    npix = kappa.shape[-1]
    lead = kappa.shape[:-1]
    n_flat = int(np.prod(lead)) if lead else 1

    flat = kappa.reshape((n_flat, npix))
    _dbg_sharding("0 convergence", flat, debug_sharding)
    flat = jax.lax.with_sharding_constraint(flat, NamedSharding(mesh, P(bins_axis, None)))
    _dbg_sharding("1 bins-sharded  P(N, None)", flat, debug_sharding)

    def _block(block):  # block: (maps_per_device, npix) -- full npix present on each device
        if debug_sharding:
            print(f"[get_shear] (inside shard_map) per-device block = {tuple(block.shape)} (maps, npix)")
        return jax.vmap(lambda k: _k2g_one(k, nside, lmax, method, iter))(block)

    out = jax.shard_map(_block, mesh=mesh, in_specs=P(bins_axis, None), out_specs=P(bins_axis, None, None))(flat)
    _dbg_sharding("2 shard_map out P(N, None, None)", out, debug_sharding)
    out = out.reshape((*lead, 2, npix))
    nbins = kappa.shape[-2] if kappa.ndim >= 2 else 1
    out = jax.lax.with_sharding_constraint(out, _shear_spec(sharding, nbins, kappa.ndim))
    _dbg_sharding("3 final         P([None,] N, None, M)", out, debug_sharding)
    return out


def kappa2shear_spherical(
    kappa, *, lmax: int | None = None, method: str = "jax", iter: int = 3, sharding=None, debug_sharding: bool = False
):
    """Convergence -> shear via Kaiser-Squires on the sphere. ``(..., npix)`` -> ``(..., 2, npix)``.

    Leading dimensions are treated as batch. ``lmax``/``method``/``iter`` are static (defaults:
    ``lmax = 3*nside-1``, ``method='jax'``, ``iter=3``); jittable. Shear is pure E-mode. ``iter`` sets
    the ``map2alm`` Jacobi iterations for the forward analysis (``alm2map`` synthesis takes none).

    If ``sharding`` is a multi-device ``NamedSharding`` the per-bin transform runs inside a
    ``jax.shard_map`` and the shear follows the **lensing convention** ``P([None,] N, None, M)``
    (BINS/N, NPIX/M; mesh axes M = first, N = second). The bins distribute over the N axis when it
    divides the bin count; a mis-shaped mesh (``N_size > 1`` that does not divide the bins) raises
    ``ValueError`` (the bad-divisor case — born emits the N=1 *warning*, get_shear stays quiet). A
    single device falls back to the plain ``vmap``. ``debug_sharding`` prints the shape and sharding at
    each stage (works under ``jax.jit``).
    """
    kappa = jnp.asarray(kappa)
    npix = kappa.shape[-1]
    nside = jhp.npix2nside(npix)
    if lmax is None:
        lmax = 3 * nside - 1
    lead = kappa.shape[:-1]
    mesh = getattr(sharding, "mesh", None)
    if sharding is not None and mesh is not None and mesh.size > 1:
        nbins = kappa.shape[-2] if kappa.ndim >= 2 else 1
        _, n_axis, _, _, distribute = lensing_axes(sharding, nbins)  # raises the bad-divisor case
        bins_axis = n_axis if distribute else None
        return _kappa2shear_spherical_sharded(kappa, sharding, bins_axis, nside, lmax, method, iter, debug_sharding)
    flat = kappa.reshape((-1, npix))
    out = jax.vmap(lambda k: _k2g_one(k, nside, lmax, method, iter))(flat)  # (B, 2, npix)
    return out.reshape((*lead, 2, npix))


def shear2kappa_spherical(gamma, *, lmax: int | None = None, method: str = "jax", iter: int = 3):
    """Shear -> convergence via inverse Kaiser-Squires on the sphere. ``(..., 2, npix)`` -> ``(..., npix)``.

    Uses the E-mode only. Leading dimensions are batch (``vmap``). ``lmax``/``method``/``iter`` static.
    ``iter`` sets the ``map2alm_spin`` Jacobi iterations for the spin-2 analysis (its ``jax_healpy``
    default is 0, so raising it sharpens the round-trip).
    """
    gamma = jnp.asarray(gamma)
    if gamma.ndim < 2 or gamma.shape[-2] != 2:
        raise ValueError(f"shear array must have a trailing (2, npix) spin-2 axis, got shape {gamma.shape}.")
    npix = gamma.shape[-1]
    nside = jhp.npix2nside(npix)
    if lmax is None:
        lmax = 3 * nside - 1
    lead = gamma.shape[:-2]
    flat = gamma.reshape((-1, 2, npix))
    out = jax.vmap(lambda g: _g2k_one(g, nside, lmax, method, iter))(flat)  # (B, npix)
    return out.reshape((*lead, npix))


# --------------------------------------------------------------------------- #
# Flat-sky (Cartesian)                                                        #
# --------------------------------------------------------------------------- #
def _flat_ks_factors(n0: int, n1: int):
    """Spin-2 phase factors ``A = cos 2phi``, ``B = sin 2phi`` on the FFT grid, DC zeroed.

    Axis convention matches ``_src/summary_statistics/_compute.py:_flat_cl`` (``meshgrid(..., indexing="ij")``):
    axis -2 carries the first frequency, axis -1 the second. The factors are scale-invariant, so
    plain ``fftfreq`` (unit spacing) is used.
    """
    l0 = jnp.fft.fftfreq(n0)
    l1 = jnp.fft.fftfreq(n1)
    L0, L1 = jnp.meshgrid(l0, l1, indexing="ij")
    l2 = L0**2 + L1**2
    l2_safe = jnp.where(l2 == 0.0, 1.0, l2)
    A = jnp.where(l2 == 0.0, 0.0, (L0**2 - L1**2) / l2_safe)
    B = jnp.where(l2 == 0.0, 0.0, (2.0 * L0 * L1) / l2_safe)
    return A, B


def _k2g_flat_one(kappa):
    n0, n1 = kappa.shape
    A, B = _flat_ks_factors(n0, n1)
    kf = jnp.fft.fft2(kappa)
    g1 = jnp.fft.ifft2(A * kf).real
    g2 = jnp.fft.ifft2(B * kf).real
    return jnp.stack([g1, g2], axis=0)  # (2, n0, n1)


def _g2k_flat_one(gamma):
    n0, n1 = gamma.shape[-2:]
    A, B = _flat_ks_factors(n0, n1)
    kf = A * jnp.fft.fft2(gamma[0]) + B * jnp.fft.fft2(gamma[1])
    return jnp.fft.ifft2(kf).real  # (n0, n1)


def kappa2shear_flat(kappa):
    """Convergence -> shear via flat-sky Kaiser-Squires. ``(..., ny, nx)`` -> ``(..., 2, ny, nx)``.

    Leading dimensions are batch (``vmap``). The spin-2 operator is the scale-invariant phase
    ``e^{2i phi}``, so no field/pixel size or band limit is needed; jittable. Shear is pure E-mode.
    """
    kappa = jnp.asarray(kappa)
    n0, n1 = kappa.shape[-2:]
    lead = kappa.shape[:-2]
    flat = kappa.reshape((-1, n0, n1))
    out = jax.vmap(_k2g_flat_one)(flat)  # (B, 2, n0, n1)
    return out.reshape((*lead, 2, n0, n1))


def shear2kappa_flat(gamma):
    """Shear -> convergence via inverse flat-sky Kaiser-Squires. ``(..., 2, ny, nx)`` -> ``(..., ny, nx)``.

    Leading dimensions are batch (``vmap``); jittable. The recovered map has zero mean (the DC mode
    is unconstrained and set to zero), so for band-limited convergence the round-trip reproduces
    ``kappa - mean(kappa)`` (the even-grid Nyquist row/column is not reconstructed; see module docs).
    """
    gamma = jnp.asarray(gamma)
    if gamma.ndim < 3 or gamma.shape[-3] != 2:
        raise ValueError(f"shear array must have a (2, ny, nx) spin-2 axis at -3, got shape {gamma.shape}.")
    n0, n1 = gamma.shape[-2:]
    lead = gamma.shape[:-3]
    flat = gamma.reshape((-1, 2, n0, n1))
    out = jax.vmap(_g2k_flat_one)(flat)  # (B, n0, n1)
    return out.reshape((*lead, n0, n1))
