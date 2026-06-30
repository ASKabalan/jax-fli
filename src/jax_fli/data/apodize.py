"""Differentiable HEALPix mask apodization (pure ``jax_healpy``).

Lives next to the survey masks (:mod:`jax_fli.data.masks`). A binary mask rings in harmonic
space, so before computing masked-sky power spectra (see :mod:`jax_fli.summary_statistics.decouple`) the mask
is tapered. The caller is responsible for building this apodized mask and passing it to
``angular_cl(mask=...)`` / ``anafast_masked(mask=...)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_healpy as jhp
import numpy as np

__all__ = ["apodize"]


def apodize(binary_mask, aposize_deg: float = 1.0, *, apotype: str = "C2"):
    """Apodize a binary HEALPix mask with a NaMaster-style C1/C2 window (pure JAX).

    Reimplements NaMaster's **C1/C2** apodization with
    ``x = sqrt((1 - cos d) / (1 - cos theta*))`` where ``d`` is the great-circle distance to the
    nearest masked pixel (capped at ``theta* = aposize_deg``): C2 is ``f = (1 - cos(pi x)) / 2``,
    C1 is ``f = x - sin(2 pi x) / (2 pi)``. The distance is obtained by a
    HEALPix **grassfire** (iterated ``min`` over true neighbour separations), so the whole thing
    is ``jnp`` + ``lax.scan`` and therefore **differentiable** w.r.t. the input mask.

    Parameters
    ----------
    binary_mask : array_like
        Binary (0/1) HEALPix mask in RING ordering, shape ``(npix,)``.
    aposize_deg : float, default=1.0
        Apodization scale ``theta*`` in degrees. Treated as a static config (sets the number of
        grassfire iterations), so keep it concrete under ``jit``.
    apotype : str, default="C2"
        Apodization window: ``"C2"`` (``f = (1 - cos(pi x)) / 2``) or ``"C1"``
        (``f = x - sin(2 pi x) / (2 pi)``). Both vanish with their first derivative at the
        boundary (NaMaster parity), which spin-2 E/B purification requires.

    Returns
    -------
    jax.Array
        Apodized mask in ``[0, 1]``, shape ``(npix,)``, ``float``. Exactly 0 wherever the input
        is 0 (so ``apodized * map`` carries no outside-footprint pixels).

    Notes
    -----
    The neighbour-separation step materializes a ``(3, 8, npix)`` intermediate, so peak memory grows
    with ``npix`` (~10 GB at ``nside=2048`` in float64); apodize at a coarser ``nside`` if memory is
    tight.
    """
    if apotype not in ("C1", "C2"):
        raise NotImplementedError(f"apotype={apotype!r} not implemented; only 'C1' and 'C2' are supported.")

    binary = jnp.asarray(binary_mask)
    if binary.ndim != 1:
        raise ValueError(f"binary_mask must be 1-D (npix,), got shape {binary.shape}.")
    npix = binary.shape[0]
    nside = jhp.npix2nside(npix)

    ipix = jnp.arange(npix)
    neigh = jhp.get_all_neighbours(nside, ipix)
    neigh = neigh if neigh.shape[0] == 8 else neigh.T  # (8, npix), -1 = missing
    vecs = jhp.pix2vec(nside, ipix)
    vecs = vecs if vecs.shape[0] == 3 else vecs.T  # (3, npix)

    theta_star = jnp.deg2rad(aposize_deg)
    nb = jnp.clip(neigh, 0, npix - 1)
    sep = jnp.where(
        neigh >= 0,
        jnp.arccos(jnp.clip((vecs[:, None, :] * vecs[:, nb]).sum(0), -1.0, 1.0)),
        jnp.inf,
    )
    dist = jnp.where(binary <= 0, 0.0, 10.0)
    # number of grassfire sweeps to cover theta* (static: nside known, aposize_deg concrete)
    resol = float(np.sqrt(4.0 * np.pi / npix))  # == healpy.nside2resol(nside)
    niter = int(np.ceil(2.5 * float(np.deg2rad(aposize_deg)) / resol)) + 6
    dist, _ = jax.lax.scan(lambda d, _: (jnp.minimum(d, (d[nb] + sep).min(0)), None), dist, None, length=niter)
    d = jnp.clip(dist, 0.0, theta_star)
    x = jnp.clip(jnp.sqrt((1 - jnp.cos(d)) / (1 - jnp.cos(theta_star))), 0.0, 1.0)
    window = x - jnp.sin(2 * jnp.pi * x) / (2 * jnp.pi) if apotype == "C1" else (1 - jnp.cos(jnp.pi * x)) / 2
    return jnp.where(binary <= 0, 0.0, window)
