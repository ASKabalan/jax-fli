from __future__ import annotations

from collections.abc import Iterable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .._src.summary_statistics import (
    _coherence,
    _cross_spherical_cl,
    _deconvolve_spherical,
    _flat_cl,
    _power,
    _spherical_cl,
    _transfer,
)


@partial(
    jax.jit,
    static_argnames=["box_shape", "multipoles", "los", "kmax", "dk", "compensate_order", "shotnoise"],
)
def power(
    mesh,
    mesh2=None,
    *,
    box_shape: tuple[float, float, float],
    kedges: int | float | Iterable[float] | None = None,
    dk: float | None = None,
    kmax: float | None = None,
    multipoles: int | Iterable[int] = 0,
    los: jnp.ndarray | Iterable[float] = jnp.array([0.0, 0.0, 1.0]),
    compensate_order: int | str | None = None,
    shotnoise: tuple[int | str, float] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Array-only 3D power spectrum. Returns (wavenumber, spectra).

    ``compensate_order`` deconvolves the mass-assignment window; ``shotnoise``
    (``(order, nbar)``) subtracts the aliased shot noise (auto-spectrum only).
    """
    box_shape = tuple(box_shape)
    los_array = None if multipoles == 0 else tuple(np.asarray(los))
    wavenumber, spectra = _power(
        mesh,
        mesh2,
        box_shape=box_shape,
        kedges=kedges,
        dk=dk,
        kmax=kmax,
        multipoles=multipoles,
        los=los_array,
        compensate_order=compensate_order,
        shotnoise=shotnoise,
    )
    return wavenumber, spectra


def transfer(
    mesh0,
    mesh1,
    *,
    box_shape: tuple[float, float, float],
    kedges: int | float | Iterable[float] | None = None,
    dk: float | None = None,
    kmax: float | None = None,
    compensate_order: int | str | None = None,
    shotnoise: tuple[int | str, float] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Monopole transfer function sqrt(P1/P0).

    ``compensate_order``/``shotnoise`` are applied to both auto-spectra.
    """
    wavenumber, tr = _transfer(
        mesh0,
        mesh1,
        box_shape=box_shape,
        kedges=kedges,
        dk=dk,
        kmax=kmax,
        compensate_order=compensate_order,
        shotnoise=shotnoise,
    )
    return wavenumber, tr


def coherence(
    mesh0,
    mesh1,
    *,
    box_shape: tuple[float, float, float],
    kedges: int | float | Iterable[float] | None = None,
    dk: float | None = None,
    kmax: float | None = None,
    compensate_order: int | str | None = None,
    shotnoise: tuple[int | str, float] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Monopole coherence pk01 / sqrt(pk0 pk1).

    ``compensate_order`` deconvolves every spectrum; ``shotnoise`` is subtracted
    from the two auto-spectra only.
    """
    wavenumber, coh = _coherence(
        mesh0,
        mesh1,
        box_shape=box_shape,
        kedges=kedges,
        dk=dk,
        kmax=kmax,
        compensate_order=compensate_order,
        shotnoise=shotnoise,
    )
    return wavenumber, coh


def angular_cl_flat(
    map1,
    map2=None,
    *,
    pixel_size: float | None = None,
    field_size: float | None = None,
    ell_edges: Iterable[float] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Flat-sky angular Cl. Returns (ell, spectra)."""
    ell, spectra = _flat_cl(
        map1,
        map2,
        pixel_size=pixel_size,
        field_size=field_size,
        ell_edges=ell_edges,
    )
    return ell, spectra


def angular_cl_spherical(
    map_sphere,
    map_sphere2=None,
    *,
    lmax: int | None = None,
    method: str = "jax",
    mask=None,
    pol: bool = False,
    purify_e: bool = False,
    purify_b: bool = False,
    mcm=None,
    nlb: int = 16,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Spherical (HEALPix) angular Cl. Returns (ell, spectra).

    With ``mask=None`` and ``pol=False`` this is the plain scalar spectrum (unchanged). Otherwise
    it delegates to :func:`jax_fli.power.anafast_masked` — ``pol=True`` for a spin-2 ``(2, npix)``
    map (returns EE, EB, BB) and/or ``mask`` for masked, **decoupled** bandpowers (with optional
    ``purify_e``/``purify_b``).
    """
    if mask is None and not pol:
        ell_out, spectra = _spherical_cl(map_sphere, map_sphere2, lmax=lmax, method=method)
        return ell_out, spectra
    from .decouple import anafast_masked

    return anafast_masked(
        map_sphere,
        map_sphere2,
        mask=mask,
        lmax=lmax,
        method=method,
        pol=pol,
        purify_e=purify_e,
        purify_b=purify_b,
        mcm=mcm,
        nlb=nlb,
    )


def cross_angular_cl_spherical(
    maps,
    *,
    lmax: int | None = None,
    method: str = "healpy",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Cross-spherical (HEALPix) angular Cl for all pairs. Returns (ell, spectra)."""
    ell_out, spectra = _cross_spherical_cl(maps, lmax=lmax, method=method)
    return ell_out, spectra


def deconvolve_spherical(
    map_sphere,
    *,
    method: str,
    nside: int,
    lmax: int | None = None,
    lcut: int | None = None,
    kernel_width_arcmin: float | None = None,
    kernel_width_pixels: float | None = None,
    smoothing_interpretation: str = "fwhm",
    iter: int = 0,
    w_floor: float = 1e-8,
) -> jnp.ndarray:
    """Deconvolve the HEALPix mass-assignment window from a single painted map.

    Removes one factor of the assignment window ``W_l`` at the ``a_lm`` level.
    ``method`` selects the window: ``'ngp'`` (pixel window), ``'rbf_neighbor'``
    (pixel window times a Gaussian beam set by ``kernel_width_*``), or
    ``'bilinear'`` (raises ``NotImplementedError``). Expects RING ordering.
    """
    return _deconvolve_spherical(
        map_sphere,
        method=method,
        nside=nside,
        lmax=lmax,
        lcut=lcut,
        kernel_width_arcmin=kernel_width_arcmin,
        kernel_width_pixels=kernel_width_pixels,
        smoothing_interpretation=smoothing_interpretation,
        iter=iter,
        w_floor=w_floor,
    )


__all__ = [
    "power",
    "transfer",
    "coherence",
    "angular_cl_flat",
    "angular_cl_spherical",
    "cross_angular_cl_spherical",
    "deconvolve_spherical",
]
