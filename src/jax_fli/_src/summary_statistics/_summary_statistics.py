"""Low-level, array-only implementations of map-based summary statistics.

These operate on a single HEALPix map ``(npix,)`` and return plain JAX arrays. Batching and
the result-object wrapping live in the public layer (``summary_statistics/{pdf,peak_counts,
starlet}.py``) and in :class:`jax_fli.fields.SphericalDensity`.

- :func:`_pdf` and :func:`_peak_counts` are pure-JAX and jittable. Each takes an explicit
  ``bins`` *edges* array so the histogram is jittable (no data-dependent range). They offer an
  exact path (hard binning) and an optional differentiable ``soft`` path (a fixed-bandwidth
  Gaussian kernel) for use in gradient-based inference.
- :func:`_starlet_transform` wraps the CosmoStat C++ spherical starlet (``pycs``); it is
  concrete-only (NumPy) and is guarded by ``require_cosmostat`` in the public layer.
"""

from __future__ import annotations

import jax.core
import jax.numpy as jnp
import jax_healpy as jhp
import numpy as np
from jax.scipy.stats import norm

__all__ = ["_pdf", "_peak_counts", "_starlet_transform"]


def _bin_centers(edges: jnp.ndarray) -> jnp.ndarray:
    """Midpoints of consecutive bin ``edges`` (length ``len(edges) - 1``)."""
    return 0.5 * (edges[..., :-1] + edges[..., 1:])


def _soft_histogram(
    values: jnp.ndarray, centers: jnp.ndarray, bandwidth: float | jnp.ndarray, *, weights=None
) -> jnp.ndarray:
    """Differentiable soft histogram: each sample spreads a unit (or ``weights``) of mass over
    ``centers`` with a normalised Gaussian kernel of width ``bandwidth``.

    Smooth in the sample *values* (so ``jax.grad`` flows through them), unlike a hard histogram.
    Returns one value per bin center.
    """
    # kernel[b, i] = exp(-0.5 ((center_b - value_i) / bw)^2)
    kernel = jnp.exp(-0.5 * ((centers[:, None] - values[None, :]) / bandwidth) ** 2)
    # Normalise each sample's mass to sum to 1 across bins, then sum (optionally weighted).
    kernel = kernel / (kernel.sum(axis=0, keepdims=True) + 1e-30)
    if weights is not None:
        kernel = kernel * weights[None, :]
    return kernel.sum(axis=1)


def _pdf(
    values: jnp.ndarray,
    *,
    bins: jnp.ndarray,
    density: bool = True,
    soft: bool = False,
    bandwidth: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One-point PDF of pixel ``values`` over the bin *edges* ``bins``.

    Returns ``(centers, pdf)``. ``density=True`` normalises so the histogram integrates to 1.
    With ``soft=True`` a fixed-bandwidth Gaussian KDE is evaluated at the bin centers instead
    (differentiable; ``bandwidth`` defaults to the mean bin width).
    """
    values = jnp.ravel(values)
    centers = _bin_centers(bins)
    if soft:
        # Keep the default bandwidth as a JAX scalar (no float()): _pdf/_peak_counts run inside
        # jax.lax.map for batched maps, where float() on a bins-derived tracer would raise.
        bw = bandwidth if bandwidth is not None else jnp.mean(jnp.diff(bins))
        pdf = norm.pdf(centers[:, None], loc=values[None, :], scale=bw).mean(axis=1)
        return centers, pdf
    counts, _ = jnp.histogram(values, bins=bins, density=density)
    return centers, counts


def _peak_counts(
    m: jnp.ndarray,
    *,
    nside: int,
    bins: jnp.ndarray,
    normalize: bool = False,
    soft: bool = False,
    bandwidth: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Histogram of local-maxima heights of a HEALPix map ``m`` over bin *edges* ``bins``.

    A pixel is a peak iff it is strictly greater than all of its (up to 8) HEALPix neighbours.
    The neighbour topology depends only on ``nside`` (static), so it is computed once with
    ``healpy`` at trace time and embedded as a constant — detection, comparison and binning are
    then pure JAX. With ``normalize=True`` the map is converted to S/N (``(m - mean) / std``)
    before detection and binning. ``soft=True`` replaces the hard histogram with a differentiable
    Gaussian soft-binning of the peak heights (peak *detection* stays hard). Returns
    ``(centers, counts)``.

    Assumes RING ordering (the repo default).
    """
    m = jnp.ravel(m)
    if normalize:
        m = (m - jnp.mean(m)) / (jnp.std(m) + 1e-30)

    npix = jhp.nside2npix(nside)
    # (8, npix) neighbour indices; -1 marks a missing neighbour. Depends only on nside (static).
    neighbours = jhp.get_all_neighbours(nside, jnp.arange(npix))
    valid = neighbours >= 0
    neigh_safe = jnp.where(valid, neighbours, 0)

    neigh_vals = jnp.where(valid, m[neigh_safe], -jnp.inf)  # (8, npix)
    is_peak = m > jnp.max(neigh_vals, axis=0)  # (npix,) bool

    centers = _bin_centers(bins)
    if soft:
        # Keep the default bandwidth as a JAX scalar (no float()): _pdf/_peak_counts run inside
        # jax.lax.map for batched maps, where float() on a bins-derived tracer would raise.
        bw = bandwidth if bandwidth is not None else jnp.mean(jnp.diff(bins))
        counts = _soft_histogram(m, centers, bw, weights=is_peak.astype(m.dtype))
        return centers, counts
    counts, _ = jnp.histogram(m, bins=bins, weights=is_peak.astype(m.dtype))
    return centers, counts


def _starlet_transform(m, *, nside: int, nscales: int) -> tuple[np.ndarray, np.ndarray]:
    """Spherical isotropic-undecimated (starlet) wavelet transform of a HEALPix map.

    Uses the CosmoStat C++ backend (``pycs.sparsity.mrs.mrs_starlet.CMRStarlet``). Concrete-only
    (NumPy in/out) — **not** jittable. Returns ``(coef, tab_norm)`` where ``coef`` has shape
    ``(nscales, npix)`` (one wavelet map per scale) and ``tab_norm`` is the per-scale
    normalisation (length ``nscales``).
    """
    from pycs.sparsity.mrs.mrs_starlet import CMRStarlet

    if isinstance(m, jax.core.Tracer):
        raise ValueError("starlet transform requires concrete (non-traced) arrays")

    m_np = np.asarray(m, dtype=np.float64)
    transform = CMRStarlet()
    transform.init_starlet(nside, nscale=nscales)
    transform.transform(m_np)
    coef = np.stack([np.asarray(transform.coef[i]) for i in range(nscales)], axis=0)
    tab_norm = np.asarray(transform.TabNorm)[:nscales]
    return coef, tab_norm
