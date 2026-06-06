"""Kaiser-Squires transforms, shear field methods, and spin-2 angular_cl forwarding."""

from __future__ import annotations

from functools import partial

import healpy as hp
import jax
import jax.numpy as jnp
import jax_healpy as jhp
import numpy as np
import pytest

import jax_fli as jfli
from jax_fli._src.base._enums import ConvergenceUnit, FieldStatus
from jax_fli._src.lensing import kappa2shear, shear2kappa
from jax_fli.fields import SphericalKappaField, SphericalShearField

NS = 64
LMAX = 3 * NS - 1
M = "jax"


def _kappa_array(n=None, seed=3):
    ell = np.arange(LMAX + 1)
    cl = np.zeros(LMAX + 1)
    cl[2:] = 1.0 / (ell[2:] + 10.0) ** 2.5
    np.random.seed(seed)
    if n is None:
        return jnp.asarray(hp.synfast(cl, NS, lmax=LMAX, pol=False))
    return jnp.asarray(np.stack([hp.synfast(cl, NS, lmax=LMAX, pol=False) for _ in range(n)]))


def _kappa_field(array):
    return SphericalKappaField(
        array=array, nside=NS, mesh_size=(NS, NS, NS), box_size=(100.0, 100.0, 100.0),
        unit=ConvergenceUnit.DIMENSIONLESS,
    )


# --------------------------------------------------------------------------- KS array level
def test_ks_roundtrip_harmonic():
    kap = _kappa_array()
    g = kappa2shear(kap, lmax=LMAX, method=M)
    assert g.shape == (2, hp.nside2npix(NS))
    rec = shear2kappa(g, lmax=LMAX, method=M)
    a0 = jhp.map2alm(kap, lmax=LMAX, pol=False, healpy_ordering=True, method=M)
    ar = jhp.map2alm(rec, lmax=LMAX, pol=False, healpy_ordering=True, method=M)
    res = np.asarray(jhp.alm2cl(a0 - ar, healpy_ordering=True))
    sig = np.asarray(jhp.alm2cl(a0, healpy_ordering=True))
    assert np.max(res[2:LMAX - 20] / sig[2:LMAX - 20]) < 3e-3


def test_ks_jittable():
    kap = _kappa_array()
    g = kappa2shear(kap, lmax=LMAX, method=M)
    gj = jax.jit(partial(kappa2shear, lmax=LMAX, method=M))(kap)
    assert np.allclose(np.asarray(g), np.asarray(gj))


def test_ks_batched_shapes():
    kap = _kappa_array(n=2)[None]  # (1, 2, npix) ~ (N=1, S=2, npix)
    g = kappa2shear(kap, lmax=LMAX, method=M)
    assert g.shape == (1, 2, 2, hp.nside2npix(NS))
    rec = shear2kappa(g, lmax=LMAX, method=M)
    assert rec.shape == kap.shape


# --------------------------------------------------------------------------- field methods
def test_get_shear_returns_shear_field():
    kf = _kappa_field(_kappa_array(n=2))  # (S=2, npix)
    sh = kf.get_shear(lmax=LMAX)
    assert isinstance(sh, SphericalShearField)
    assert sh.array.shape == (2, 2, hp.nside2npix(NS))
    assert sh.status == FieldStatus.GAMMA
    assert sh.is_batched() is True


def test_get_convergence_inverts():
    kf = _kappa_field(_kappa_array(n=2))
    rec = kf.get_shear(lmax=LMAX).get_convergence(lmax=LMAX)
    assert isinstance(rec, SphericalKappaField)
    assert rec.array.shape == kf.array.shape
    # harmonic round-trip per bin
    for i in range(2):
        a0 = jhp.map2alm(kf.array[i], lmax=LMAX, pol=False, healpy_ordering=True, method=M)
        ar = jhp.map2alm(rec.array[i], lmax=LMAX, pol=False, healpy_ordering=True, method=M)
        res = np.asarray(jhp.alm2cl(a0 - ar, healpy_ordering=True))
        sig = np.asarray(jhp.alm2cl(a0, healpy_ordering=True))
        assert np.max(res[2:LMAX - 20] / sig[2:LMAX - 20]) < 3e-3


def test_shear_angular_cl_components_single():
    kf = _kappa_field(_kappa_array())  # (npix,)
    sh = kf.get_shear(lmax=LMAX)  # (2, npix)
    ps = sh.angular_cl(lmax=LMAX, method=M)
    assert ps.n_components == 3
    assert ps.array.shape == (3, LMAX + 1)
    assert ps.is_batched() is False


def test_shear_angular_cl_components_batched():
    kf = _kappa_field(_kappa_array(n=3))  # (3, npix)
    sh = kf.get_shear(lmax=LMAX)  # (3, 2, npix)
    ps = sh.angular_cl(lmax=LMAX, method=M)
    assert ps.array.shape == (3, 3, LMAX + 1)
    assert ps.n_components == 3 and ps.is_batched()


def test_shear_angular_cl_multibatched_NS():
    """Headline shape (N, S, 2, npix): angular_cl collapses (N, S) -> (N*S, 3, n_ell)."""
    N, S = 2, 3
    kap = _kappa_array(n=N * S).reshape((N, S, hp.nside2npix(NS)))  # (N, S, npix)
    sh = _kappa_field(kap).get_shear(lmax=LMAX)  # (N, S, 2, npix)
    assert sh.array.shape == (N, S, 2, hp.nside2npix(NS))
    assert sh.is_multi_batched() is True
    ps = sh.angular_cl(lmax=LMAX, method=M)
    assert ps.n_components == 3
    assert ps.array.shape == (N * S, 3, LMAX + 1)
    assert ps.is_batched() is True


def test_shear_angular_cl_masked_bandpowers():
    kf = _kappa_field(_kappa_array())
    sh = kf.get_shear(lmax=LMAX)
    mb = jnp.asarray(np.where(np.arange(hp.nside2npix(NS)) < hp.nside2npix(NS) // 3, 1.0, 0.0))
    apo = jfli.data.apodize(mb, 5.0)
    ps = sh.angular_cl(mask=apo, lmax=LMAX, nlb=16, method=M, purify_b=True)
    assert ps.n_components == 3
    assert ps.array.shape[0] == 3 and ps.array.shape[1] < LMAX  # bandpowers


# --------------------------------------------------------------------------- end-to-end (KS preserves EE)
def test_fullsky_EE_equals_factor_squared_kappa():
    """Full-sky EE of KS(kappa) == f_l^2 * C_l^{kappa kappa} (the KS identity, no mask)."""
    kap = _kappa_array()
    sh = _kappa_field(kap).get_shear(lmax=LMAX)
    ps = sh.angular_cl(lmax=LMAX, method=M)  # mask=None -> per-l EE,EB,BB
    EE = np.asarray(ps.array[0])
    ckk = np.asarray(jhp.anafast(kap, lmax=LMAX, pol=False, method=M))
    ell = np.arange(LMAX + 1)
    f2 = np.zeros(LMAX + 1)
    f2[2:] = (ell[2:] + 2) * (ell[2:] - 1) / (ell[2:] * (ell[2:] + 1))
    sl = slice(10, LMAX - 20)
    rel = np.max(np.abs(EE[sl] - f2[sl] * ckk[sl]) / (f2[sl] * ckk[sl]))
    assert rel < 5e-2  # spin-2 floor


def test_scalar_angular_cl_masked_vs_namaster():
    nmt = pytest.importorskip("pymaster")
    kap = _kappa_array()
    kf = _kappa_field(kap)
    mb = np.where(np.arange(hp.nside2npix(NS)) < hp.nside2npix(NS) // 3, 1.0, 0.0)
    apo = jfli.data.apodize(jnp.asarray(mb), 5.0)
    ps = kf.angular_cl(mask=apo, lmax=LMAX, nlb=16, method=M)
    f0 = nmt.NmtField(np.asarray(apo), [np.asarray(kap)], lmax=LMAX)
    dec_nmt = np.asarray(nmt.compute_full_master(f0, f0, nmt.NmtBin.from_nside_linear(NS, 16)))[0]
    rel = np.max(np.abs(np.asarray(ps.array)[1:-1] - dec_nmt[1:-1])) / np.max(np.abs(dec_nmt[1:-1]))
    assert rel < 1e-8
