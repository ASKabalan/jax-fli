"""Bandpower binning: ``PowerSpectrum.bin`` and the ``bin_bandpowers`` helper.

The reference is the canonical ``(2l+1)``-weighted bandpower average; the oracle is computed
inline in :func:`test_bin_bandpowers_matches_reference` so the test is self-contained.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jax_fli._src.base._enums import FieldStatus, SpectralUnit
from jax_fli.summary_statistics import PowerSpectrum, bin_bandpowers, linear_edges, log_edges

LMAX = 256


@pytest.fixture
def ell():
    return np.arange(LMAX + 1, dtype=float)


@pytest.fixture
def cl_batched():
    """Three positive synthetic spectra of shape (3, LMAX+1)."""
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.random((3, LMAX + 1)))


@pytest.fixture
def angular_ps(cl_batched):
    """A batched angular PowerSpectrum (3, LMAX+1) on the integer-ℓ grid."""
    return PowerSpectrum(
        wavenumber=jnp.arange(LMAX + 1, dtype=float),
        array=cl_batched,
        unit=SpectralUnit.ANGULAR_CL,
        status=FieldStatus.SPECTRA,
        name="cl",
    )


# --------------------------------------------------------------------------- helper
def test_bin_bandpowers_matches_reference(ell, cl_batched):
    """``bin_bandpowers`` reproduces the canonical (2l+1)-weighted ``log_bin``."""
    cl = np.asarray(cl_batched)
    edges = np.unique(np.geomspace(2, LMAX, 21).astype(int)).astype(float)

    # inline oracle (the user's log_bin)
    w = 2.0 * ell + 1.0
    leff_ref, cb_ref, nm_ref = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ell >= lo) & (ell < hi)
        if not m.any():
            continue
        ww = w[m]
        leff_ref.append((ww * ell[m]).sum() / ww.sum())
        cb_ref.append((cl[:, m] * ww).sum(axis=1) / ww.sum())
        nm_ref.append(ww.sum())
    cb_ref = np.stack(cb_ref, axis=1)

    leff, binned, nmodes = bin_bandpowers(ell, cl_batched, edges=edges, weight="modes")
    assert np.allclose(np.asarray(leff), leff_ref)
    assert np.allclose(np.asarray(binned), cb_ref)
    assert np.allclose(np.asarray(nmodes), nm_ref)


def test_uniform_weight_is_plain_mean(ell):
    """A flat spectrum bins to itself; uniform weight gives the arithmetic bin mean."""
    cl = jnp.asarray(np.arange(LMAX + 1, dtype=float))  # cl[ell] = ell
    leff, binned, _ = bin_bandpowers(ell, cl, edges=[2.0, 12.0], weight="uniform")
    assert np.isclose(float(binned[0]), np.arange(2, 12).mean())  # mean of 2..11


def test_edges_drop_empty_bins(ell):
    cl = jnp.ones(LMAX + 1)
    # [2,10) and [10,1000) are populated (ℓ runs 0..256); [1000,2000) is empty -> dropped.
    leff, binned, _ = bin_bandpowers(ell, cl, edges=[2.0, 10.0, 1000.0, 2000.0])
    assert leff.shape[0] == 2 and binned.shape[-1] == 2


# --------------------------------------------------------------------------- edge builders
def test_linear_edges_band_count():
    """nlb=16 over [2, 256] -> 15 bands (16 edges) — the '15 elements' the user observed."""
    edges = linear_edges(16, 2, 256)
    assert edges.shape[0] == 16  # 15 bands
    assert edges[0] == 2.0 and edges[1] == 18.0


def test_linear_edges_too_large_raises():
    with pytest.raises(ValueError):
        linear_edges(10_000, 2, 256)


def test_log_edges_are_unique_and_increasing():
    edges = log_edges(20, 2, 256)
    assert np.all(np.diff(edges) > 0)
    assert edges[0] >= 2.0 and edges[-1] <= 256.0


# --------------------------------------------------------------------------- PowerSpectrum.bin
def test_bin_nlb_band_count_and_shape(angular_ps):
    binned = angular_ps.bin(nlb=16)
    assert binned.wavenumber.shape[0] == 15
    assert binned.array.shape == (3, 15)
    assert binned.array.shape[-1] == binned.wavenumber.shape[0]


def test_bin_single_spectrum(angular_ps):
    single = angular_ps[0]  # (LMAX+1,)
    binned = single.bin(nlb=16)
    assert binned.array.shape == (15,)


def test_bin_edges_mode(angular_ps):
    binned = angular_ps.bin(edges=[10, 50, 100, 200])
    assert binned.wavenumber.shape[0] == 3  # three bins from four edges
    assert binned.array.shape == (3, 3)


def test_bin_wavenumbers_is_edges_alias(angular_ps):
    """The user's literal call ``bin(wavenumbers=[...])`` aliases ``edges=[...]``."""
    a = angular_ps.bin(edges=[10, 50, 100, 200])
    b = angular_ps.bin(wavenumbers=[10, 50, 100, 200])
    assert np.allclose(np.asarray(a.wavenumber), np.asarray(b.wavenumber))
    assert np.allclose(np.asarray(a.array), np.asarray(b.array))
    with pytest.raises(ValueError):
        angular_ps.bin(edges=[10, 50], wavenumbers=[10, 50])


def test_bin_nbins_log_mode(angular_ps):
    binned = angular_ps.bin(nbins=10)
    assert binned.wavenumber.shape[0] <= 10
    assert np.all(np.diff(np.asarray(binned.wavenumber)) > 0)


def test_bin_binned_ratio_no_shape_error(angular_ps):
    """The motivating bug: a binned/binned ratio must line up element-for-element."""
    a = angular_ps.bin(nlb=16)
    b = angular_ps.bin(nlb=16)
    ratio = a.array / b.array
    assert ratio.shape == a.array.shape
    assert np.allclose(np.asarray(ratio), 1.0)


def test_bin_requires_exactly_one_selector(angular_ps):
    with pytest.raises(ValueError):
        angular_ps.bin()
    with pytest.raises(ValueError):
        angular_ps.bin(nlb=16, nbins=10)


def test_bin_metadata_preserved(angular_ps):
    binned = angular_ps.bin(nlb=16)
    assert binned.name == angular_ps.name
    assert binned.unit == angular_ps.unit
    assert binned.n_components == angular_ps.n_components


def test_angular_default_lmin_skips_monopole_dipole(angular_ps):
    """For an ANGULAR_CL spectrum the default low edge is ℓ=2."""
    binned = angular_ps.bin(nlb=16)
    assert float(binned.wavenumber[0]) >= 2.0


def test_explicit_edges_bin_a_pk_grid():
    """``nlb``/``nbins`` are integer-multipole builders; an arbitrary (e.g. P(k)) grid bins
    via explicit ``edges`` with no ℓ=2 assumption."""
    k = jnp.linspace(0.01, 1.0, 100)
    ps = PowerSpectrum(
        wavenumber=k,
        array=jnp.ones(100),
        unit=SpectralUnit.POWER_SPECTRA,
        status=FieldStatus.SPECTRA,
        name="pk",
    )
    binned = ps.bin(edges=[0.01, 0.1, 0.5, 1.01], weight="uniform")
    assert binned.wavenumber.shape[0] == 3
    assert float(binned.wavenumber[0]) < 0.1  # low-k bin retained
