"""Masked-sky power spectra: apodize + anafast_masked (decoupling/purification) vs NaMaster.

NaMaster is the oracle only; tests that need it skip cleanly if pymaster is unavailable.
"""

from __future__ import annotations

import healpy as hp
import jax
import jax.numpy as jnp
import jax_healpy as jhp
import numpy as np
import pytest

import jax_fli as jfli
from jax_fli.summary_statistics.decouple import anafast_masked, compute_mcm

NS = 64
LMAX = 3 * NS - 1
NLB = 16
M = "jax"


def _quadrant(nside):
    npix = hp.nside2npix(nside)
    m = np.ones(npix)
    th, ph = hp.pix2ang(nside, np.arange(npix))
    m[th < np.pi / 2] = 0.0
    m[(ph < 3 * np.pi / 2) & (ph > np.pi / 2)] = 0.0
    return m


def _kappa(seed=3):
    ell = np.arange(LMAX + 1)
    cl = np.zeros(LMAX + 1)
    cl[2:] = 1.0 / (ell[2:] + 10.0) ** 2.5
    np.random.seed(seed)
    return jnp.asarray(hp.synfast(cl, NS, lmax=LMAX, pol=False))


def _pure_e_shear(kappa):
    ell = np.arange(LMAX + 1)
    fk = np.zeros(LMAX + 1)
    fk[2:] = np.sqrt((ell[2:] + 2) * (ell[2:] - 1) / (ell[2:] * (ell[2:] + 1)))
    E = jhp.almxfl(
        jhp.map2alm(kappa, lmax=LMAX, pol=False, healpy_ordering=True, method=M), jnp.asarray(fk), healpy_ordering=True
    )
    g1, g2 = jhp.alm2map([E, jnp.zeros_like(E)], NS, lmax=LMAX, pol=True, healpy_ordering=True, method=M)
    return jnp.stack([g1, g2], axis=0)


@pytest.fixture(scope="module")
def apo():
    return jfli.data.apodize(jnp.asarray(_quadrant(NS)), 5.0)


# --------------------------------------------------------------------------- apodize
def test_apodize_zero_outside_and_range(apo):
    binary = _quadrant(NS)
    apo_np = np.asarray(apo)
    assert apo_np.min() >= 0.0 and apo_np.max() <= 1.0
    assert np.max(np.abs(apo_np[binary <= 0])) == 0.0  # exactly zero outside the footprint


def test_apodize_jittable():
    binary = jnp.asarray(_quadrant(NS))
    out = jax.jit(lambda m: jfli.data.apodize(m, 5.0))(binary)
    assert out.shape == binary.shape


def test_apodize_wl_matches_namaster(apo):
    nmt = pytest.importorskip("pymaster")
    binary = _quadrant(NS)
    apo_nmt = nmt.mask_apodization(binary, 5.0, apotype="C2")
    Wl = np.asarray(jhp.anafast(apo, lmax=LMAX, pol=False, method=M))
    Wl_nmt = hp.anafast(apo_nmt, lmax=LMAX)
    sl = slice(2, 2 * NS)
    rel = np.max(np.abs(Wl[sl] - Wl_nmt[sl])) / np.max(np.abs(Wl_nmt[sl]))
    assert rel < 5e-3  # W_l drives the MCM; study reports <=2e-3


# --------------------------------------------------------------------------- mask=None
def test_mask_none_scalar_equals_anafast():
    kap = _kappa()
    _, cl = anafast_masked(kap, lmax=LMAX, method=M)
    assert np.allclose(np.asarray(cl), np.asarray(jhp.anafast(kap, lmax=LMAX, pol=False, method=M)))


def test_mask_none_pol_three_spectra():
    g = _pure_e_shear(_kappa())
    ell, cl = anafast_masked(g, lmax=LMAX, method=M, pol=True)
    assert cl.shape == (3, LMAX + 1)  # EE, EB, BB


def test_purify_requires_mask():
    g = _pure_e_shear(_kappa())
    with pytest.raises(ValueError):
        anafast_masked(g, lmax=LMAX, pol=True, purify_b=True)


def test_pol_cross_spectrum_not_supported():
    # map2 is only for scalar cross-spectra; spin-2 cross-spectra are not implemented (would be
    # silently ignored otherwise).
    g = _pure_e_shear(_kappa())
    with pytest.raises(NotImplementedError):
        anafast_masked(g, g, lmax=LMAX, pol=True)


# --------------------------------------------------------------------------- coupled pseudo
def test_coupled_pseudo_via_premask_scalar(apo):
    kap = _kappa()
    _, cl = anafast_masked(apo * kap, lmax=LMAX, method=M)  # premasked + mask=None
    manual = jhp.anafast(apo * kap, lmax=LMAX, pol=False, method=M)
    assert np.allclose(np.asarray(cl), np.asarray(manual))


# --------------------------------------------------------------------------- decoupling vs NaMaster
def test_decoupled_scalar_matches_namaster(apo):
    nmt = pytest.importorskip("pymaster")
    kap = _kappa()
    _, dec = anafast_masked(kap, mask=apo, lmax=LMAX, nlb=NLB, method=M)
    apo_np = np.asarray(apo)
    f0 = nmt.NmtField(apo_np, [np.asarray(kap)], lmax=LMAX)
    dec_nmt = np.asarray(nmt.compute_full_master(f0, f0, nmt.NmtBin.from_nside_linear(NS, NLB)))[0]
    rel = np.max(np.abs(np.asarray(dec)[1:-1] - dec_nmt[1:-1])) / np.max(np.abs(dec_nmt[1:-1]))
    assert rel < 1e-8  # exact MCM/decoupling


def test_decoupled_spin2_EE_matches_namaster(apo):
    nmt = pytest.importorskip("pymaster")
    g = _pure_e_shear(_kappa())
    g1, g2 = np.asarray(g[0]), np.asarray(g[1])
    _, cl = anafast_masked(g, mask=apo, lmax=LMAX, nlb=NLB, method=M, pol=True)
    apo_np = np.asarray(apo)
    bins = nmt.NmtBin.from_nside_linear(NS, NLB)
    fE = nmt.NmtField(apo_np, [g1, g2], spin=2, lmax=LMAX)
    mEE = np.asarray(nmt.compute_full_master(fE, fE, bins))[0]
    rel = np.max(np.abs(np.asarray(cl)[0][1:-1] - mEE[1:-1])) / np.max(np.abs(mEE[1:-1]))
    assert rel < 5e-2  # spin-2 jax_healpy iter=0 floor (~1%)


def test_purify_b_reduces_leakage(apo):
    g = _pure_e_shear(_kappa())  # pure-E sky -> true BB ~ 0
    _, raw = anafast_masked(g, mask=apo, lmax=LMAX, nlb=NLB, method=M, pol=True)
    _, pur = anafast_masked(g, mask=apo, lmax=LMAX, nlb=NLB, method=M, pol=True, purify_b=True)
    assert np.mean(np.abs(np.asarray(pur)[2])) < np.mean(np.abs(np.asarray(raw)[2]))


def test_mcm_precomputed_identical(apo):
    kap = _kappa()
    mcm = compute_mcm(apo, lmax=LMAX, nlb=NLB, pol=False, method=M)
    _, a = anafast_masked(kap, mask=apo, lmax=LMAX, nlb=NLB, method=M)
    _, b = anafast_masked(kap, mask=apo, lmax=LMAX, nlb=NLB, method=M, mcm=mcm)
    assert np.allclose(np.asarray(a), np.asarray(b))


def test_eb_is_binned_pseudo(apo):
    # EB row is the binned (coupled) pseudo, not a decoupled/validated spectrum.
    g = _pure_e_shear(_kappa())
    mcm = compute_mcm(apo, lmax=LMAX, nlb=NLB, pol=True, method=M)
    E, Bb = jhp.map2alm_spin([apo * g[0], apo * g[1]], spin=2, lmax=LMAX, healpy_ordering=True, method=M)
    ps_eb = np.asarray(jhp.alm2cl(E, Bb, healpy_ordering=True))
    expected = np.asarray(mcm.B @ jnp.asarray(ps_eb))
    _, cl = anafast_masked(g, mask=apo, lmax=LMAX, nlb=NLB, method=M, pol=True, mcm=mcm)
    assert np.allclose(np.asarray(cl)[1], expected)


# --------------------------------------------------------------------------- differentiable through the map
def test_decoupled_cl_differentiable_in_map(apo):
    def loss(kappa):
        _, dec = anafast_masked(kappa, mask=apo, lmax=LMAX, nlb=NLB, method=M)
        return jnp.sum(dec)

    g = jax.grad(loss)(_kappa())
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.sum(jnp.abs(g))) > 0.0
