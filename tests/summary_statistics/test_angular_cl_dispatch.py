"""Spherical angular_cl after the refactor: the field methods now delegate the batch loop
and the healpy/jax branching to ``angular_cl_spherical_batched``. These tests pin the
delegated behaviour (== healpy / == anafast_masked) and the newly mask-aware
``cross_angular_cl`` / ``transfer`` / ``coherence``.
"""

from __future__ import annotations

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_fli as jfli
from jax_fli import SphericalDensity
from jax_fli.fields.lensing_maps import SphericalShearField
from jax_fli.summary_statistics import compute_mcm
from jax_fli.summary_statistics.decouple import anafast_masked

NSIDE = 16
NPIX = hp.nside2npix(NSIDE)
LMAX = 3 * NSIDE - 1
NLB = 8
MESH = (16, 16, 16)
BOX = (256.0, 256.0, 256.0)


@pytest.fixture(scope="module")
def maps3():
    """Three distinct Gaussian HEALPix realisations, shape (3, npix)."""
    ell = np.arange(LMAX + 1)
    cl = np.zeros(LMAX + 1)
    cl[2:] = 1.0 / (ell[2:] + 10.0) ** 2.0
    np.random.seed(1)  # hp.synfast draws from numpy's global RNG
    return jnp.asarray(np.stack([hp.synfast(cl, NSIDE, lmax=LMAX) for _ in range(3)]))


@pytest.fixture(scope="module")
def field1(maps3):
    return SphericalDensity(array=maps3[0], nside=NSIDE, mesh_size=MESH, box_size=BOX, name="rho")


@pytest.fixture(scope="module")
def field3(maps3):
    return SphericalDensity(array=maps3, nside=NSIDE, mesh_size=MESH, box_size=BOX, name="rho")


@pytest.fixture(scope="module")
def apo_mask():
    """An apodized half-sky footprint at NSIDE (RING)."""
    th, _ = hp.pix2ang(NSIDE, np.arange(NPIX))
    binary = np.ones(NPIX)
    binary[th < np.pi / 2] = 0.0
    return jfli.data.apodize(jnp.asarray(binary), 10.0)


@pytest.fixture(scope="module")
def mcm(apo_mask):
    return compute_mcm(apo_mask, lmax=LMAX, nlb=NLB, method="healpy")


# --------------------------------------------------------------------------- unmasked
def test_unmasked_scalar_equals_healpy(field1):
    cl = field1.angular_cl(method="healpy", lmax=LMAX)
    ref = hp.anafast(np.asarray(field1.array), lmax=LMAX)
    assert cl.array.shape == (LMAX + 1,)
    assert np.allclose(np.asarray(cl.array), ref, rtol=1e-4, atol=1e-8)


def test_unmasked_jax_matches_healpy(field1):
    a = field1.angular_cl(method="jax", lmax=LMAX)
    b = field1.angular_cl(method="healpy", lmax=LMAX)
    assert np.allclose(np.asarray(a.array), np.asarray(b.array), rtol=1e-3, atol=1e-8)


def test_batched_scalar_shape(field3):
    cl = field3.angular_cl(method="jax", lmax=LMAX)
    assert cl.array.shape == (3, LMAX + 1)
    assert cl.wavenumber.shape == (LMAX + 1,)


def test_unmasked_cross_all_pairs(field3):
    xc = field3.cross_angular_cl(method="healpy", lmax=LMAX)
    assert xc.array.shape[0] == 6  # 3*(3+1)/2
    assert xc.wavenumber.shape == (LMAX + 1,)


# --------------------------------------------------------------------------- masked / decoupled
def test_masked_scalar_matches_anafast_masked(field1, apo_mask, mcm):
    cl = field1.angular_cl(method="healpy", lmax=LMAX, mask=apo_mask, mcm=mcm, nlb=NLB)
    _, ref = anafast_masked(field1.array, mask=apo_mask, lmax=LMAX, method="healpy", mcm=mcm, nlb=NLB)
    assert np.allclose(np.asarray(cl.wavenumber), np.asarray(mcm.ell_eff))
    assert np.allclose(np.asarray(cl.array), np.asarray(ref))


def test_masked_jax_matches_kernel_and_differentiable(field1, field3, apo_mask, mcm):
    """The jax (lax.map) masked path is the inference path: the old field method used
    jax.vmap, the dispatcher now uses jax.lax.map. Pin that (a) the field method equals the
    anafast_masked kernel element-for-element, and (b) it runs under jit and grad."""
    # single map: field method == direct kernel (exact, same jax backend)
    cj = field1.angular_cl(method="jax", lmax=LMAX, mask=apo_mask, mcm=mcm, nlb=NLB)
    _, ref = anafast_masked(field1.array, mask=apo_mask, lmax=LMAX, method="jax", mcm=mcm, nlb=NLB)
    assert np.allclose(np.asarray(cj.array), np.asarray(ref))

    # batched: element i of the lax.map output == kernel applied to map i
    cjb = field3.angular_cl(method="jax", lmax=LMAX, mask=apo_mask, mcm=mcm, nlb=NLB)
    assert cjb.array.shape == (3, mcm.ell_eff.shape[0])
    for i in range(3):
        _, r = anafast_masked(field3.array[i], mask=apo_mask, lmax=LMAX, method="jax", mcm=mcm, nlb=NLB)
        assert np.allclose(np.asarray(cjb.array[i]), np.asarray(r))

    # jit + grad through the refactored field method on the masked lax.map path
    def loss(maps):
        f = SphericalDensity(array=maps, nside=NSIDE, mesh_size=MESH, box_size=BOX, name="rho")
        return f.angular_cl(method="jax", lmax=LMAX, mask=apo_mask, mcm=mcm, nlb=NLB).array.sum()

    assert bool(jnp.isfinite(jax.jit(loss)(field3.array)))
    g = jax.grad(loss)(field3.array)
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.sum(jnp.abs(g))) > 0.0


def test_masked_cross_matches_pairwise(field3, apo_mask, mcm):
    xc = field3.cross_angular_cl(method="healpy", lmax=LMAX, mask=apo_mask, mcm=mcm, nlb=NLB)
    nbands = mcm.ell_eff.shape[0]
    assert xc.array.shape == (6, nbands)
    assert np.allclose(np.asarray(xc.wavenumber), np.asarray(mcm.ell_eff))

    maps = field3.array
    expected = [
        anafast_masked(maps[i], maps[j], mask=apo_mask, lmax=LMAX, method="healpy", mcm=mcm, nlb=NLB)[1]
        for i in range(3)
        for j in range(i, 3)
    ]
    assert np.allclose(np.asarray(xc.array), np.asarray(jnp.stack(expected)))


def test_masked_transfer_returns_bandpowers(maps3, apo_mask, mcm):
    a = SphericalDensity(array=maps3[0], nside=NSIDE, mesh_size=MESH, box_size=BOX, name="a")
    b = SphericalDensity(array=maps3[1], nside=NSIDE, mesh_size=MESH, box_size=BOX, name="b")
    tr = a.transfer(b, method="healpy", lmax=LMAX, mask=apo_mask, mcm=mcm, nlb=NLB)
    assert tr.wavenumber.shape == mcm.ell_eff.shape
    assert tr.array.shape == mcm.ell_eff.shape


def test_masked_coherence_returns_bandpowers(maps3, apo_mask, mcm):
    a = SphericalDensity(array=maps3[0], nside=NSIDE, mesh_size=MESH, box_size=BOX, name="a")
    b = SphericalDensity(array=maps3[1], nside=NSIDE, mesh_size=MESH, box_size=BOX, name="b")
    co = a.coherence(b, method="healpy", lmax=LMAX, mask=apo_mask, mcm=mcm, nlb=NLB)
    assert co.wavenumber.shape == mcm.ell_eff.shape
    assert co.array.shape == mcm.ell_eff.shape


# --------------------------------------------------------------------------- shear (both backends)
@pytest.fixture(scope="module")
def shear(maps3):
    return SphericalShearField(
        array=jnp.stack([maps3[0], maps3[1]]), nside=NSIDE, mesh_size=MESH, box_size=BOX, name="shear"
    )


def test_shear_angular_cl_jax(shear):
    ee = shear.angular_cl(method="jax", lmax=LMAX)
    assert ee.array.shape == (3, LMAX + 1)  # EE, EB, BB
    assert ee.n_components == 3


def test_shear_angular_cl_healpy(shear):
    """Regression: the old unconditional vmap broke method='healpy' for shear."""
    ee = shear.angular_cl(method="healpy", lmax=LMAX)
    assert ee.array.shape == (3, LMAX + 1)
