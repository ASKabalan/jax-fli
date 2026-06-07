"""Kaiser-Squires shear reconstruction: ``get_shear`` / ``get_convergence`` (flat + spherical).

Inputs are the **real Born convergence maps** from ``conftest`` (spherical maps are cross-validated
against glass/dorian in ``test_born``; flat maps have no such reference, so flat KS is checked for
self-consistency — forward+inverse round-trip — only).

Spherical KS lives in harmonic space and zeroes ``l < 2``, so the round-trip is validated on the
angular power spectrum (the ``l >= 2`` subspace it preserves). Flat KS is an FFT operator that is
exact except the (unconstrained) DC mode, so its round-trip is checked directly on the mean-removed
map.
"""

from __future__ import annotations

import healpy as hp
import jax.numpy as jnp
import numpy as np
from jax_fli._src.base._enums import FieldStatus
from jax_fli.fields import FlatKappaField, FlatShearField, SphericalKappaField, SphericalShearField

from tests.helpers import compare_fields

# Round-trip tolerances (set from the measured Born-map round-trip).
# Spherical: C_l(rec)/C_l(kappa) over l in [LMIN, 2*nside] — median ~1, drifts to ~5% near the band
# limit (nside=64, iter=3). Flat: FFT KS is exact bar DC, so the mean-removed map matches to ~1e-7.
SPH_CL_RTOL = 0.07
SPH_LMIN = 10
FLAT_RTOL = 0.02
FLAT_ATOL = 1e-6
FLAT_MEAN_ATOL = 1e-10


def _assert_cl_roundtrip(rec, kappa):
    """Assert the recovered convergence preserves the angular power spectrum for ``l >= 2``."""
    nside = kappa.nside
    lmax = 3 * nside - 1
    r = np.atleast_2d(np.asarray(rec.array))
    k = np.atleast_2d(np.asarray(kappa.array))
    for i in range(r.shape[0]):
        cl_rec = hp.anafast(r[i], lmax=lmax)
        cl_kap = hp.anafast(k[i], lmax=lmax)
        ratio = cl_rec[SPH_LMIN : 2 * nside] / cl_kap[SPH_LMIN : 2 * nside]
        np.testing.assert_allclose(ratio, 1.0, atol=SPH_CL_RTOL, err_msg="spherical KS round-trip C_l drift")


def _demean(arr, n_spatial):
    """Remove the mean over the trailing ``n_spatial`` spatial axes (the unconstrained DC mode)."""
    a = np.asarray(arr)
    axes = tuple(range(a.ndim - n_spatial, a.ndim))
    return a - a.mean(axis=axes, keepdims=True)


# --------------------------------------------------------------------------- spherical
def test_spherical_shear_roundtrip(born_kappa_single):
    kappa = born_kappa_single[0]  # strip the S=1 axis -> (npix,)
    sh = kappa.get_shear()
    assert isinstance(sh, SphericalShearField)
    assert sh.status == FieldStatus.GAMMA
    assert sh.unit == kappa.unit
    assert sh.array.shape == (2, kappa.array.shape[-1])
    assert sh.is_batched() is False

    rec = sh.get_convergence()
    assert isinstance(rec, SphericalKappaField)
    assert rec.array.shape == kappa.array.shape
    _assert_cl_roundtrip(rec, kappa)


def test_spherical_shear_roundtrip_batched(born_kappa_multi):
    kappa = born_kappa_multi  # (S, npix)
    s, npix = kappa.array.shape
    sh = kappa.get_shear()
    assert isinstance(sh, SphericalShearField)
    assert sh.array.shape == (s, 2, npix)
    assert sh.is_batched() is True

    rec = sh.get_convergence()
    assert rec.array.shape == kappa.array.shape
    _assert_cl_roundtrip(rec, kappa)


def test_spherical_shear_multibatched_NS(born_kappa_multi):
    """(N, S, npix) convergence -> (N, S, 2, npix) shear."""
    kappa = born_kappa_multi
    s, npix = kappa.array.shape
    n = 2
    stacked = kappa.replace(array=jnp.broadcast_to(kappa.array, (n, s, npix)))
    sh = stacked.get_shear()
    assert sh.array.shape == (n, s, 2, npix)
    assert sh.is_multi_batched() is True


def test_spherical_shear_angular_cl_components(born_kappa_single):
    """Spin-2 angular_cl returns 3 components (EE, EB, BB)."""
    ps = born_kappa_single[0].get_shear().angular_cl()
    assert ps.n_components == 3
    assert ps.array.shape[-2] == 3


# --------------------------------------------------------------------------- flat
def test_flat_shear_roundtrip(born_flat_kappa_single):
    kappa = born_flat_kappa_single[0]  # strip the S=1 axis -> (ny, nx)
    sh = kappa.get_shear()
    assert isinstance(sh, FlatShearField)
    assert sh.status == FieldStatus.GAMMA
    assert sh.unit == kappa.unit
    assert sh.array.shape == (2, *kappa.array.shape)
    assert sh.is_batched() is False

    rec = sh.get_convergence()
    assert isinstance(rec, FlatKappaField)
    assert rec.array.shape == kappa.array.shape
    compare_fields(
        _demean(rec.array, 2),
        _demean(kappa.array, 2),
        "flat KS round-trip",
        rtol=FLAT_RTOL,
        atol=FLAT_ATOL,
        mean_atol=FLAT_MEAN_ATOL,
    )


def test_flat_shear_roundtrip_batched(born_flat_kappa_multi):
    kappa = born_flat_kappa_multi  # (S, ny, nx)
    sh = kappa.get_shear()
    assert isinstance(sh, FlatShearField)
    assert sh.array.shape == (kappa.array.shape[0], 2, *kappa.array.shape[1:])
    assert sh.is_batched() is True

    rec = sh.get_convergence()
    assert rec.array.shape == kappa.array.shape
    compare_fields(
        _demean(rec.array, 2),
        _demean(kappa.array, 2),
        "flat KS round-trip (batched)",
        rtol=FLAT_RTOL,
        atol=FLAT_ATOL,
        mean_atol=FLAT_MEAN_ATOL,
    )


def test_flat_shear_multibatched_NS(born_flat_kappa_multi):
    """(N, S, ny, nx) convergence -> (N, S, 2, ny, nx) shear."""
    kappa = born_flat_kappa_multi
    s, ny, nx = kappa.array.shape
    n = 2
    stacked = kappa.replace(array=jnp.broadcast_to(kappa.array, (n, s, ny, nx)))
    sh = stacked.get_shear()
    assert sh.array.shape == (n, s, 2, ny, nx)
    assert sh.is_multi_batched() is True
