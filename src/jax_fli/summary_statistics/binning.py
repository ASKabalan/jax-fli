"""Bandpower binning for angular / radial spectra.

A light, **eager** post-processing utility: collapse a finely sampled spectrum into a
handful of (mode-weighted) bandpowers so that spectra computed at different native
resolutions can be put on a common grid for plotting and comparison.

This is *not* mask decoupling — for the mode-coupling-matrix machinery see
:mod:`jax_fli.summary_statistics.decouple`. Here we just take a plain weighted average
over multipole/wavenumber bins, matching the canonical ``(2l+1)``-weighted ``log_bin``.
"""

from __future__ import annotations

from collections.abc import Iterable

import jax.numpy as jnp
import numpy as np


def linear_edges(nlb: int, lmin: float, lmax: float) -> np.ndarray:
    """Linear bin edges: bands of ``nlb`` consecutive multipoles spanning ``[lmin, lmax]``.

    Returns ``nbands + 1`` edges ``[lmin, lmin+nlb, ...]`` where
    ``nbands = (lmax + 1 - lmin) // nlb``; the trailing partial band is dropped (the
    ``nmt.NmtBin.from_nside_linear`` convention). Bin ``q`` covers ``[edges[q], edges[q+1])``.
    """
    lmin = int(np.ceil(lmin))
    lmax = int(np.floor(lmax))
    nbands = (lmax + 1 - lmin) // int(nlb)
    if nbands < 1:
        raise ValueError(f"nlb={nlb} is too large for the multipole range [{lmin}, {lmax}].")
    return np.asarray([lmin + q * int(nlb) for q in range(nbands + 1)], dtype=float)


def log_edges(nbins: int, lmin: float, lmax: float) -> np.ndarray:
    """``nbins`` log-spaced integer bin edges spanning ``[lmin, lmax]``.

    Mirrors ``np.unique(np.geomspace(lmin, lmax, nbins + 1).astype(int))`` — note that
    duplicate integer edges at low multipole collapse, so fewer than ``nbins`` bands may
    result.
    """
    lmin = max(int(np.ceil(lmin)), 1)
    lmax = int(np.floor(lmax))
    if lmax <= lmin:
        raise ValueError(f"need lmax ({lmax}) > lmin ({lmin}) to build log bins.")
    return np.unique(np.geomspace(lmin, lmax, int(nbins) + 1).astype(int)).astype(float)


def bin_bandpowers(
    wavenumber,
    spectra,
    *,
    edges: Iterable[float],
    weight: str = "modes",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Weighted bandpowers of ``spectra`` over the bins defined by ``edges``.

    Parameters
    ----------
    wavenumber : (n,) array_like
        Multipole / wavenumber grid; it is the trailing axis of ``spectra``.
    spectra : (..., n) array_like
        Spectrum values. Binning is over the **last** axis; any leading axes (batch,
        components) are preserved.
    edges : (nbins+1,) array_like
        Bin edges; bin ``i`` covers ``[edges[i], edges[i+1])``. Empty bins are dropped.
    weight : {"modes", "uniform"}, default "modes"
        ``"modes"`` weights each multipole by ``2*ell + 1`` (the number of ``m`` modes —
        the standard ``log_bin``); ``"uniform"`` weights every multipole equally.

    Returns
    -------
    (leff, binned, nmodes)
        ``leff`` ``(n_kept,)`` weighted-mean wavenumber per bin, ``binned``
        ``(..., n_kept)`` weighted-mean spectra, ``nmodes`` ``(n_kept,)`` summed weights.

    Notes
    -----
    Eager only: the number of surviving (non-empty) bins depends on the data, so this is
    not ``jit``-traceable. It is a plotting / comparison convenience, not a forward-model op.
    """
    ell = np.asarray(wavenumber, dtype=float)
    sp = jnp.asarray(spectra)
    edges = np.asarray(list(edges), dtype=float)
    if ell.ndim != 1:
        raise ValueError("wavenumber must be 1-D.")
    if sp.shape[-1] != ell.shape[0]:
        raise ValueError(f"spectra last axis {sp.shape[-1]} does not match wavenumber {ell.shape[0]}.")
    if edges.ndim != 1 or edges.shape[0] < 2:
        raise ValueError("edges must be a 1-D array of at least two bin edges.")

    if weight == "modes":
        w = 2.0 * ell + 1.0
    elif weight == "uniform":
        w = np.ones_like(ell)
    else:
        raise ValueError(f"weight must be 'modes' or 'uniform', got {weight!r}.")
    w_j = jnp.asarray(w)

    leff: list[float] = []
    binned: list[jnp.ndarray] = []
    nmodes: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ell >= lo) & (ell < hi)
        if not m.any():
            continue
        idx = jnp.asarray(np.nonzero(m)[0])
        ww = w[m]
        wsum = float(ww.sum())
        leff.append(float((ww * ell[m]).sum() / wsum))
        binned.append((sp[..., idx] * w_j[idx]).sum(axis=-1) / wsum)
        nmodes.append(wsum)

    if not binned:
        raise ValueError("No non-empty bins; check `edges` against the wavenumber range.")
    return jnp.asarray(leff), jnp.stack(binned, axis=-1), jnp.asarray(nmodes)
