from __future__ import annotations

from collections.abc import Iterable

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


@jax.jit(
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


@jax.jit(static_argnames=["box_shape", "kmax", "dk", "compensate_order", "shotnoise"])
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


@jax.jit(static_argnames=["box_shape", "kmax", "dk", "compensate_order", "shotnoise"])
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


@jax.jit(static_argnames=["pixel_size", "field_size"])
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


def angular_cl_spherical_batched(
    maps,
    maps2=None,
    *,
    lmax: int | None = None,
    method: str = "jax",
    mask=None,
    pol: bool = False,
    purify_e: bool = False,
    purify_b: bool = False,
    mcm=None,
    nlb: int = 16,
    batch_size: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Batched spherical angular Cl. Returns ``(wavenumber, spectra)``.

    Thin batching layer over the single-map :func:`angular_cl_spherical`: it owns the
    leading-batch loop and the backend branching that used to live in the field methods.
    ``method="healpy"`` loops in Python (healpy needs concrete arrays); any other method
    maps with :func:`jax.lax.map` (honouring ``batch_size``).

    ``maps`` is ``(npix,)`` / ``(B, npix)`` for a scalar field, or ``(2, npix)`` /
    ``(B, 2, npix)`` when ``pol=True``; a single (unbatched) input drops the batch axis on
    the way out. ``maps2`` is an optional second map (stack) for scalar cross-spectra.

    With ``mask`` (or a precomputed ``mcm``) the result is mask-decoupled into bandpowers
    and ``wavenumber`` is the effective bandpower multipoles; otherwise it is the full
    ``0..lmax`` grid. The MCM is built **once** here and reused across the batch.
    """
    maps = jnp.asarray(maps)
    single_ndim = 2 if pol else 1
    if maps.ndim not in (single_ndim, single_ndim + 1):
        raise ValueError(
            f"angular_cl_spherical_batched expected a single map (ndim {single_ndim}) or a "
            f"batched stack (ndim {single_ndim + 1}), got array shape {tuple(maps.shape)}."
        )
    single = maps.ndim == single_ndim

    # Build the mode-coupling matrix once so it is reused across the batch (compute_mcm
    # defaults lmax to 3*nside-1 internally when lmax is None).
    _mcm = mcm
    if mask is not None and _mcm is None:
        from .decouple import compute_mcm

        _mcm = compute_mcm(mask, lmax=lmax, nlb=nlb, pol=pol, method=method)

    def _kern(m1, m2):
        return angular_cl_spherical(
            m1,
            m2,
            lmax=lmax,
            method=method,
            mask=mask,
            pol=pol,
            purify_e=purify_e,
            purify_b=purify_b,
            mcm=_mcm,
            nlb=nlb,
        )

    if single:
        return _kern(maps, maps2)

    if maps2 is not None and jnp.asarray(maps2).shape != maps.shape:
        raise ValueError("maps2 must match maps shape for a cross spectrum.")

    if method == "healpy":
        # healpy needs concrete arrays -> loop in Python (cannot run under jax transforms).
        ell = None
        spectra = []
        for i in range(maps.shape[0]):
            m2 = None if maps2 is None else maps2[i]
            ell, sp = _kern(maps[i], m2)
            spectra.append(sp)
        assert ell is not None, "batch must be non-empty"
        return ell, jnp.stack(spectra, axis=0)

    ell_stack, spectra_stack = jax.lax.map(lambda pair: _kern(pair[0], pair[1]), (maps, maps2), batch_size=batch_size)
    return ell_stack[0], spectra_stack


def cross_angular_cl_spherical(
    maps,
    *,
    lmax: int | None = None,
    method: str = "healpy",
    mask=None,
    mcm=None,
    nlb: int = 16,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Cross-spherical (HEALPix) angular Cl for all unique pairs ``(i <= j)``. Returns ``(ell, spectra)``.

    With ``mask=None`` this is the plain healpy all-pairs spectrum (unbinned, full
    ``0..lmax``). With an apodized ``mask`` (or a precomputed ``mcm``) each pair is masked
    and **decoupled** into bandpowers (``ell`` = effective bandpower multipoles), looped in
    upper-triangular order ``(0,0), (0,1), ..., (B-1,B-1)`` to match
    :meth:`FlatDensity.cross_angular_cl`.
    """
    if mask is None:
        ell_out, spectra = _cross_spherical_cl(maps, lmax=lmax, method=method)
        return ell_out, spectra

    from .decouple import anafast_masked, compute_mcm

    maps = jnp.asarray(maps)
    n_maps = maps.shape[0]
    _mcm = mcm if mcm is not None else compute_mcm(mask, lmax=lmax, nlb=nlb, pol=False, method=method)
    spectra = [
        anafast_masked(maps[i], maps[j], mask=mask, lmax=lmax, method=method, pol=False, mcm=_mcm, nlb=nlb)[1]
        for i in range(n_maps)
        for j in range(i, n_maps)
    ]
    return _mcm.ell_eff, jnp.stack(spectra, axis=0)


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
    "angular_cl_spherical_batched",
    "cross_angular_cl_spherical",
    "deconvolve_spherical",
]
