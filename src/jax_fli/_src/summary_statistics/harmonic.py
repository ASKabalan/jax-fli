"""Harmonic-space (ell-tapered) scale-cut packing for field-level inference.

The scale cut is a smooth per-ell taper ``w_ell`` on a whitened harmonic residual: each retained
``(l, m)`` of ``model - data`` is scaled by ``sqrt(w_ell)`` and packed into a real vector ``rho``
so that ``0.5 * ||rho / sqrt(N_ell)||^2`` reproduces the Gaussian map likelihood at ``w_ell == 1``
(Parseval). White pixel noise gives a flat ``N_ell``, applied by the caller (the noise lives
outside the pack, which is a pure linear map).

Mirrors the ``MCM`` / :func:`compute_mcm` / :func:`anafast_masked` precompute-then-apply pattern:
``compute_harmonic_pack_*`` builds the data-independent :class:`HarmonicPack` weights once (all with
NumPy, so they fold to constants under ``jit``); ``apply_harmonic_pack_*`` runs the fixed,
differentiable transform (``map2alm`` / ``map2alm_spin`` E-mode / flat Kaiser-Squires E-map with
``iter=0, pol=False, healpy_ordering=True``) on a map and packs it. Data and model go through the
same pack, so the quadrature error is common-mode and cancels in the residual.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax_healpy as jhp
import numpy as np
from jaxtyping import Array

__all__ = [
    "HarmonicPack",
    "compute_harmonic_pack_spherical",
    "compute_harmonic_pack_flat",
    "apply_harmonic_pack_spherical",
    "apply_harmonic_pack_flat",
]


class HarmonicPack(eqx.Module):
    """Precomputed ell-taper + Parseval weights for the harmonic scale-cut pack.

    Depends only on geometry, ``l_cut``/``width`` and spin (not on any map), so build once with
    ``<field>.harmonic_pack_precompute(...)`` and reuse for data and model (freeze for inference),
    mirroring :class:`~jax_fli.summary_statistics.decouple.MCM`. Spherical fills
    ``re_weight``/``im_weight``/``im_idx`` (+ ``lmax``); flat fills ``wroot`` (+ ``shape``).
    Consumed by :func:`apply_harmonic_pack_spherical` / :func:`apply_harmonic_pack_flat`.
    """

    re_weight: Array | None = None
    im_weight: Array | None = None
    im_idx: Array | None = None
    wroot: Array | None = None
    spin: int = eqx.field(static=True, default=0)
    lmax: int | None = eqx.field(static=True, default=None)
    shape: tuple[int, int] | None = eqx.field(static=True, default=None)
    method: str = eqx.field(static=True, default="jax")


def compute_harmonic_pack_spherical(nside: int, l_cut: int, width: int, *, spin: int = 0, method: str = "jax"):
    """Build the spherical (HEALPix) :class:`HarmonicPack`. ``lmax = max(l_cut, 2*nside - 1)`` (the
    s2fft floor; the taper zeroes every ell above ``l_cut`` anyway). The real/imag weights encode
    the healpy m-major ``a_lm`` ordering and the ``sqrt(2 - delta_m0)`` real-packing factor, so
    ``||rho||^2`` is the tapered harmonic chi^2 (Parseval-normalized, no noise). Built with NumPy
    → folds to a constant under ``jit``."""
    if not 0 < width <= l_cut:
        raise ValueError(f"width must be in (0, l_cut]; got width={width}, l_cut={l_cut}")
    lmax = max(int(l_cut), 2 * nside - 1)
    ell = np.arange(lmax + 1)
    # cosine scale-cut taper: 1 below l_cut - width, cosine roll-off to 0 at l_cut.
    x = (ell - (l_cut - width)) / width
    w_ell = np.where(ell <= l_cut - width, 1.0, np.where(ell >= l_cut, 0.0, 0.5 * (1.0 + np.cos(np.pi * x))))
    # healpy alm ordering is m-major: for m = 0..lmax, l runs m..lmax contiguously.
    ells = np.concatenate([np.arange(m, lmax + 1) for m in range(lmax + 1)])
    ms = np.concatenate([np.full(lmax + 1 - m, m) for m in range(lmax + 1)])
    wroot = np.sqrt(w_ell[ells])
    return HarmonicPack(
        re_weight=jnp.asarray(wroot * np.where(ms == 0, 1.0, np.sqrt(2.0))),
        im_weight=jnp.asarray(wroot[ms > 0] * np.sqrt(2.0)),
        im_idx=jnp.asarray(np.nonzero(ms > 0)[0]),
        spin=spin,
        lmax=lmax,
        method=method,
    )


def compute_harmonic_pack_flat(shape, field_size, l_cut: int, width: int, *, spin: int = 0):
    """Build the flat-sky :class:`HarmonicPack`. ``field_size`` is in RADIANS so the FFT grid is a
    true multipole grid. Per Parseval, ``rho = sqrt(w_k * field_area / N^2) * delta~_k`` (packed
    ``[Re, Im]``), so ``||rho||^2`` equals the per-pixel chi^2 at ``w == 1``. The full ``fft2``
    over-counts the independent real dof ~2x (fine for value and gradient). Built with NumPy →
    folds to a constant under ``jit``."""
    if not 0 < width <= l_cut:
        raise ValueError(f"width must be in (0, l_cut]; got width={width}, l_cut={l_cut}")
    n0, n1 = shape
    npix = n0 * n1
    field_area = float(field_size[0]) * float(field_size[1])
    l0 = 2.0 * np.pi * np.fft.fftfreq(n0, d=field_size[0] / n0)
    l1 = 2.0 * np.pi * np.fft.fftfreq(n1, d=field_size[1] / n1)
    lx, ly = np.meshgrid(l0, l1, indexing="ij")
    ell = np.sqrt(lx**2 + ly**2)
    # cosine scale-cut taper: 1 below l_cut - width, cosine roll-off to 0 at l_cut.
    x = (ell - (l_cut - width)) / width
    w = np.where(ell <= l_cut - width, 1.0, np.where(ell >= l_cut, 0.0, 0.5 * (1.0 + np.cos(np.pi * x))))
    return HarmonicPack(
        wroot=jnp.asarray(np.sqrt(w * field_area / npix**2)),
        spin=spin,
        shape=(int(n0), int(n1)),
    )


def apply_harmonic_pack_spherical(array: Array, pack: HarmonicPack) -> Array:
    """Transform a HEALPix map to its whitened, ell-tapered harmonic residual vector ``rho`` (no
    noise). ``array`` is ``(..., npix)`` (spin-0) or ``(..., 2, npix)`` (spin-2 shear); leading dims
    are batch. Spin-2 keeps only the E-mode (B is a pure-noise null)."""
    array = jnp.asarray(array)

    def _one(m):
        if pack.spin == 2:
            alm = jhp.map2alm_spin(
                [m[0], m[1]], spin=2, lmax=pack.lmax, iter=0, healpy_ordering=True, method=pack.method
            )[0]
        else:
            alm = jhp.map2alm(m, lmax=pack.lmax, iter=0, pol=False, healpy_ordering=True, method=pack.method)
        return jnp.concatenate([alm.real * pack.re_weight, alm.imag[pack.im_idx] * pack.im_weight])

    if pack.spin == 2:
        lead, flat = array.shape[:-2], array.reshape((-1, 2, array.shape[-1]))
    else:
        lead, flat = array.shape[:-1], array.reshape((-1, array.shape[-1]))
    out = jax.vmap(_one)(flat)
    return out.reshape((*lead, out.shape[-1]))


def apply_harmonic_pack_flat(array: Array, pack: HarmonicPack) -> Array:
    """Transform a flat-sky map to its whitened, ell-tapered harmonic residual vector ``rho`` (no
    noise). ``array`` is ``(..., ny, nx)`` (spin-0) or ``(..., 2, ny, nx)`` (spin-2 shear); leading
    dims are batch. Spin-2 goes through the flat Kaiser-Squires E-map (the spin-2 transfer is unity
    on the flat sky, so spin-0 and spin-2 share one taper)."""
    array = jnp.asarray(array)
    n0, n1 = pack.shape
    if pack.spin == 2:
        # lazy import: `_src.lensing.__init__` pulls `_born -> ...fields` (a cycle at import time).
        from ..lensing._kaiser_squires import _flat_ks_factors

        cos2phi, sin2phi = _flat_ks_factors(n0, n1)

    def _one(m):
        if pack.spin == 2:
            e_fourier = cos2phi * jnp.fft.fft2(m[0]) + sin2phi * jnp.fft.fft2(m[1])
            delta = jnp.fft.ifft2(e_fourier).real
        else:
            delta = m
        f = jnp.fft.fft2(delta) * pack.wroot
        return jnp.concatenate([f.real.reshape(-1), f.imag.reshape(-1)])

    if pack.spin == 2:
        lead, flat = array.shape[:-3], array.reshape((-1, 2, n0, n1))
    else:
        lead, flat = array.shape[:-2], array.reshape((-1, n0, n1))
    out = jax.vmap(_one)(flat)
    return out.reshape((*lead, out.shape[-1]))
