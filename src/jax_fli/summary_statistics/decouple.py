"""Masked-sky angular power spectra: mode decoupling + E/B purification (pure ``jax_healpy``).

A binary/apodized mask couples multipoles and (for spin-2) mixes E and B. The unbiased
estimator is the pseudo-``C_l`` MASTER method: build a **mode-coupling matrix (MCM)** from the
mask power spectrum ``W_l`` and **decouple** the measured pseudo-spectrum; for spin-2 fields,
optionally **purify** the B-mode (Smith 2006) to remove E->B leakage.

This is a pure-``jax_healpy`` port of the pipeline validated against NaMaster in
``tests/lensing/purification_study/`` (``masked_cl_study.py``). NaMaster is **not** a runtime
dependency — it is only used as the test oracle.

The public entry point is :func:`anafast_masked`:

* ``mask=None``  -> plain ``jhp.anafast`` (also serves as the *coupled* pseudo-``C_l`` when the
  caller has pre-multiplied the map by the apodized mask, e.g. for forward-model inference).
* ``mask`` given -> masked pseudo-``C_l``, **decoupled** (decoupling is implied by a non-None
  mask; there is no separate ``decouple`` flag). Spin-2 honours ``purify_e``/``purify_b``.

The MCM depends only on the mask, so it can be precomputed once with :func:`compute_mcm` and
passed back via ``mcm=`` (e.g. frozen for inference).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax_healpy as jhp
import numpy as np
from jaxtyping import Array

__all__ = ["MCM", "compute_mcm", "anafast_masked"]


# --------------------------------------------------------------------------- #
# Legendre / Wigner-d kernels (upward ell-recursions on Gauss-Legendre nodes) #
# --------------------------------------------------------------------------- #
def _legendre_all(x, lmax):
    """P_l(x), l=0..lmax -> (lmax+1, N)."""

    def body(carry, ell):
        Plm1, Pl = carry
        return (Pl, ((2 * ell + 1) * x * Pl - ell * Plm1) / (ell + 1)), Pl

    P0 = jnp.ones_like(x)
    _, Pstk = jax.lax.scan(body, (P0, x), jnp.arange(1, lmax + 1))
    return jnp.concatenate([P0[None], Pstk], axis=0)


def _wigner_d2_all(x, lmax, mp):
    """d^l_{2,mp}(theta), x=cos(theta), l=0..lmax, mp=+/-2 -> (lmax+1, N)."""
    m = 2
    d2 = (1 + x) ** 2 / 4 if mp == 2 else (1 - x) ** 2 / 4

    def body(carry, ell):
        dlm1, dl = carry
        a = (2 * ell + 1) * (ell * (ell + 1) * x - m * mp)
        b = (ell + 1) * jnp.sqrt((ell**2 - m**2) * (ell**2 - mp**2))
        c = ell * jnp.sqrt(((ell + 1) ** 2 - m**2) * ((ell + 1) ** 2 - mp**2))
        return (dl, (a * dl - b * dlm1) / c), dl

    _, dstk = jax.lax.scan(body, (jnp.zeros_like(x), d2), jnp.arange(2, lmax + 1))
    return jnp.concatenate([jnp.zeros((2,) + x.shape), dstk], axis=0)


@jax.jit(static_argnums=(1,))
def _mcm_spin0(Wl, lmax):
    """Spin-0 mode-coupling matrix M[l1,l2]."""
    ells = jnp.arange(lmax + 1)
    x_np, w_np = np.polynomial.legendre.leggauss(3 * lmax + 5)
    x, wq = jnp.asarray(x_np), jnp.asarray(w_np)
    P = _legendre_all(x, lmax)
    G = jnp.sum((2 * ells + 1)[:, None] * Wl[:, None] * P, axis=0)
    return (2 * ells + 1)[None, :] / (8 * jnp.pi) * ((P * (wq * G)[None, :]) @ P.T)


@jax.jit(static_argnums=(1,))
def _mcm_spin2(Wl, lmax):
    """Spin-2 MCM blocks (EE<-EE, EE<-BB) from the +/-2 Wigner-d kernels."""
    ells = jnp.arange(lmax + 1)
    x_np, w_np = np.polynomial.legendre.leggauss(3 * lmax + 5)
    x, wq = jnp.asarray(x_np), jnp.asarray(w_np)
    G = jnp.sum((2 * ells + 1)[:, None] * Wl[:, None] * _legendre_all(x, lmax), axis=0)
    d22, d2m2 = _wigner_d2_all(x, lmax, 2), _wigner_d2_all(x, lmax, -2)
    norm = (2 * ells + 1)[None, :] / (8 * jnp.pi)
    Mpp = norm * ((d22 * (wq * G)[None, :]) @ d22.T)
    Mmm = norm * ((d2m2 * (wq * G)[None, :]) @ d2m2.T)
    return 0.5 * (Mpp + Mmm), 0.5 * (Mpp - Mmm)


# --------------------------------------------------------------------------- #
# Linear bandpower binning (matches nmt.NmtBin.from_nside_linear)             #
# --------------------------------------------------------------------------- #
def _linear_bins(nside, nlb, lmax):
    """Binning operators B (n_bands, lmax+1) and S (lmax+1, n_bands), plus effective ells.

    Replicates ``nmt.NmtBin.from_nside_linear(nside, nlb)``: bins of ``nlb`` consecutive
    multipoles starting at l=2 (uniform weights 1/nlb), trailing partial bin dropped.
    """
    nbands = (lmax + 1 - 2) // nlb
    B = np.zeros((nbands, lmax + 1))
    S = np.zeros((lmax + 1, nbands))
    ell_eff = np.zeros(nbands)
    for q in range(nbands):
        ells = np.arange(2 + q * nlb, 2 + (q + 1) * nlb)
        B[q, ells] = 1.0 / nlb
        S[ells, q] = 1.0
        ell_eff[q] = ells.mean()
    return jnp.asarray(B), jnp.asarray(S), jnp.asarray(ell_eff)


def _decouple_spin0(cl_coupled, M0, B, S):
    return jnp.linalg.solve(B @ M0 @ S, B @ jnp.asarray(cl_coupled))


def _decouple_spin2(cl_EE, cl_BB, EEEE, EEBB, B, S):
    nb = B.shape[0]
    EEEE_b, EEBB_b = B @ EEEE @ S, B @ EEBB @ S
    Mbin = jnp.block([[EEEE_b, EEBB_b], [EEBB_b, EEEE_b]])
    rhs = jnp.concatenate([B @ jnp.asarray(cl_EE), B @ jnp.asarray(cl_BB)])
    dec = jnp.linalg.solve(Mbin, rhs)
    return dec[:nb], dec[nb:]


# --------------------------------------------------------------------------- #
# E/B purification (Smith 2006, pure jax_healpy)                              #
# --------------------------------------------------------------------------- #
def _purify_eb(g1, g2, mask, nside, lmax, method, task=(False, True)):
    """(E_alm, B_alm) purified a la Smith 2006. task=(purify_E, purify_B)."""
    g1, g2, mask = jnp.asarray(g1), jnp.asarray(g2), jnp.asarray(mask)
    ls = jnp.arange(lmax + 1)
    alm_mask = jhp.map2alm(mask, lmax=lmax, pol=False, iter=3, healpy_ordering=True, method=method)
    E, Bb = jhp.map2alm_spin([mask * g1, mask * g2], spin=2, lmax=lmax, healpy_ordering=True, method=method)
    alms = [E, Bb]
    walm0 = jhp.almxfl(alm_mask, -jnp.sqrt((ls + 1.0) * ls), healpy_ordering=True)
    walm1 = jnp.zeros_like(walm0)
    w0, w1 = jhp.alm2map_spin([walm0, walm1], nside, spin=1, lmax=lmax, healpy_ordering=True, method=method)
    p = jhp.map2alm_spin([w0 * g1 + w1 * g2, w0 * g2 - w1 * g1], spin=1, lmax=lmax, healpy_ordering=True, method=method)
    f1 = jnp.zeros(lmax + 1).at[2:].set(2.0 / jnp.sqrt((ls[2:] + 2.0) * (ls[2:] - 1.0)))
    for i, do in enumerate(task):
        if do:
            alms[i] = alms[i] + jhp.almxfl(p[i], f1, healpy_ordering=True)
    f2 = jnp.zeros(lmax + 1).at[2:].set(-jnp.sqrt((ls[2:] + 2.0) * (ls[2:] - 1.0)))
    walm0b = jhp.almxfl(walm0, f2, healpy_ordering=True)
    w0, w1 = jhp.alm2map_spin([walm0b, walm1], nside, spin=2, lmax=lmax, healpy_ordering=True, method=method)
    p0 = jhp.map2alm(w0 * g1 + w1 * g2, lmax=lmax, pol=False, iter=0, healpy_ordering=True, method=method)
    p1 = jhp.map2alm(w0 * g2 - w1 * g1, lmax=lmax, pol=False, iter=0, healpy_ordering=True, method=method)
    p = [p0, p1]
    f3 = jnp.zeros(lmax + 1).at[2:].set(1.0 / jnp.sqrt((ls[2:] + 2.0) * (ls[2:] + 1.0) * ls[2:] * (ls[2:] - 1.0)))
    for i, do in enumerate(task):
        if do:
            alms[i] = alms[i] + jhp.almxfl(p[i], f3, healpy_ordering=True)
    return alms[0], alms[1]


# --------------------------------------------------------------------------- #
# MCM container + public API                                                  #
# --------------------------------------------------------------------------- #
class MCM(eqx.Module):
    """Precomputed mode-coupling data for a fixed (apodized) mask.

    Depends only on the mask, so build once and reuse (freeze for inference). Built by
    :func:`compute_mcm` and accepted by :func:`anafast_masked` via ``mcm=``.
    """

    spin0: Array | None
    eeee: Array | None
    eebb: Array | None
    B: Array
    S: Array
    ell_eff: Array
    lmax: int = eqx.field(static=True)
    nlb: int = eqx.field(static=True)
    pol: bool = eqx.field(static=True)


def compute_mcm(mask, *, lmax: int | None = None, nlb: int = 16, pol: bool = False, method: str = "jax") -> MCM:
    """Build the mode-coupling matrix + bandpower binning for an (apodized) ``mask``.

    Parameters
    ----------
    mask : array_like
        Apodized HEALPix mask (RING), shape ``(npix,)``.
    lmax : int, optional
        Max multipole; defaults to ``3*nside-1``.
    nlb : int, default=16
        Multipoles per bandpower (linear binning, matches NaMaster).
    pol : bool, default=False
        If True also build the spin-2 (EE<-EE, EE<-BB) blocks.
    """
    mask = jnp.asarray(mask)
    nside = jhp.npix2nside(mask.shape[0])
    if lmax is None:
        lmax = 3 * nside - 1
    Wl = jhp.anafast(mask, lmax=lmax, pol=False, method=method)
    spin0 = _mcm_spin0(Wl, lmax)
    eeee, eebb = _mcm_spin2(Wl, lmax) if pol else (None, None)
    B, S, ell_eff = _linear_bins(nside, nlb, lmax)
    return MCM(spin0=spin0, eeee=eeee, eebb=eebb, B=B, S=S, ell_eff=ell_eff, lmax=lmax, nlb=nlb, pol=pol)


def anafast_masked(
    map1,
    map2=None,
    *,
    mask=None,
    lmax: int | None = None,
    method: str = "jax",
    pol: bool = False,
    purify_e: bool = False,
    purify_b: bool = False,
    mcm: MCM | None = None,
    nlb: int = 16,
):
    """Angular power spectrum of a (optionally masked) HEALPix map. Returns ``(ell, cl)``.

    * ``mask=None`` -> plain ``jhp.anafast``. ``pol=False`` returns one ``C_l`` (shape
      ``(lmax+1,)``); ``pol=True`` takes ``map1=(2, npix)`` shear and returns ``(3, lmax+1)``
      = (EE, EB, BB). (Pre-multiply the map by the apodized mask and pass ``mask=None`` to get the
      *coupled* pseudo-``C_l`` used in forward-model inference.)
    * ``mask`` given -> masked pseudo-``C_l``, **decoupled** into bandpowers (``ell`` =
      effective bandpower multipoles). ``pol=False`` returns decoupled ``C_l^{kappa kappa}``
      (shape ``(n_bands,)``); ``pol=True`` returns ``(3, n_bands)`` with EE and BB **decoupled**
      and EB the **binned (coupled) pseudo** (its decoupling block is out of scope). ``purify_e``
      / ``purify_b`` apply the spin-2 purification before estimation.

    ``purify_*`` require a ``mask``. The MCM is built from ``mask`` unless a precomputed ``mcm``
    is supplied.
    """
    map1 = jnp.asarray(map1)
    if pol and map2 is not None:
        raise NotImplementedError(
            "Spin-2 cross-spectra (pol=True with map2) are not supported; map2 is for scalar "
            "cross-spectra only. Pass a single (2, npix) shear map as map1."
        )
    if mask is None:
        if purify_e or purify_b:
            raise ValueError("purify_e/purify_b require a mask (purification needs the mask).")
        if not pol:
            cl = jhp.anafast(map1, None if map2 is None else jnp.asarray(map2), lmax=lmax, pol=False, method=method)
            return jnp.arange(cl.shape[-1]) * 1.0, jnp.asarray(cl)
        g1, g2 = map1[0], map1[1]
        _lmax = lmax if lmax is not None else 3 * jhp.npix2nside(g1.shape[-1]) - 1
        E, Bb = jhp.map2alm_spin([g1, g2], spin=2, lmax=_lmax, healpy_ordering=True, method=method)
        ee = jhp.alm2cl(E, healpy_ordering=True)
        eb = jhp.alm2cl(E, Bb, healpy_ordering=True)
        bb = jhp.alm2cl(Bb, healpy_ordering=True)
        return jnp.arange(ee.shape[-1]) * 1.0, jnp.stack([ee, eb, bb], axis=0)

    mask = jnp.asarray(mask)
    nside = jhp.npix2nside(mask.shape[0])
    _lmax = lmax if lmax is not None else 3 * nside - 1
    if mcm is None:
        mcm = compute_mcm(mask, lmax=_lmax, nlb=nlb, pol=pol, method=method)

    if not pol:
        masked1 = mask * map1
        masked2 = None if map2 is None else mask * jnp.asarray(map2)
        ps = jhp.anafast(masked1, masked2, lmax=mcm.lmax, pol=False, method=method)
        dec = _decouple_spin0(ps, mcm.spin0, mcm.B, mcm.S)
        return mcm.ell_eff, jnp.asarray(dec)

    g1, g2 = map1[0], map1[1]
    if purify_e or purify_b:
        E, Bb = _purify_eb(g1, g2, mask, nside, mcm.lmax, method, task=(purify_e, purify_b))
    else:
        E, Bb = jhp.map2alm_spin([mask * g1, mask * g2], spin=2, lmax=mcm.lmax, healpy_ordering=True, method=method)
    psEE = jhp.alm2cl(E, healpy_ordering=True)
    psEB = jhp.alm2cl(E, Bb, healpy_ordering=True)
    psBB = jhp.alm2cl(Bb, healpy_ordering=True)
    dEE, dBB = _decouple_spin2(psEE, psBB, mcm.eeee, mcm.eebb, mcm.B, mcm.S)
    dEB = mcm.B @ psEB  # EB: binned (coupled) pseudo only; its decoupling block is out of scope
    return mcm.ell_eff, jnp.stack([jnp.asarray(dEE), jnp.asarray(dEB), jnp.asarray(dBB)], axis=0)
