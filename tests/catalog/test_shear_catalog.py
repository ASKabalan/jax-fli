"""Catalog persistence for SphericalShearField and multi-component PowerSpectrum."""

from __future__ import annotations

import os
import tempfile

import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
from jax_fli._src.base._enums import ConvergenceUnit
from jax_fli.fields import SphericalKappaField, SphericalShearField
from jax_fli.io import Catalog
from jax_fli.power import PowerSpectrum

NS = 16
NPIX = 12 * NS * NS


def _cosmo():
    return jc.Planck15()


def _shear_field(n_bins=2):
    meta = dict(
        nside=NS,
        mesh_size=(NS, NS, NS),
        box_size=(100.0, 100.0, 100.0),
        unit=ConvergenceUnit.DIMENSIONLESS,
        z_sources=jnp.asarray(np.linspace(0.5, 1.5, n_bins)),
        scale_factors=jnp.asarray(np.linspace(0.4, 0.7, n_bins)),
        comoving_centers=jnp.asarray(np.linspace(1500.0, 3000.0, n_bins)),
        density_width=jnp.asarray(np.full(n_bins, 100.0)),
    )
    kf = SphericalKappaField(array=jnp.asarray(np.random.RandomState(0).randn(n_bins, NPIX)), **meta)
    return kf.get_shear(lmax=3 * NS - 1)


def _roundtrip(entry):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "cat.parquet")
        Catalog(entry, _cosmo()).to_parquet(p)
        return Catalog.from_parquet(p).field[0]


def test_shear_field_roundtrip():
    sh = _shear_field(2)
    rt = _roundtrip(sh)
    assert isinstance(rt, SphericalShearField)
    assert rt.array.shape == sh.array.shape  # (S, 2, npix)
    assert np.allclose(np.asarray(rt.array), np.asarray(sh.array), atol=1e-5)


def test_three_component_ps_roundtrip():
    ell = jnp.arange(20.0)
    arr = np.random.RandomState(2).randn(3, 20)
    ps = PowerSpectrum(wavenumber=ell, array=jnp.asarray(arr), n_components=3, nside=NS)
    rt = _roundtrip(ps)
    assert rt.n_components == 3
    assert rt.array.shape == (3, 20)
    assert np.allclose(np.asarray(rt.array), arr, atol=1e-5)


def test_batched_3d_ps_roundtrip():
    ell = jnp.arange(20.0)
    arr = np.random.RandomState(3).randn(4, 3, 20)
    ps = PowerSpectrum(wavenumber=ell, array=jnp.asarray(arr), n_components=3, nside=NS)
    rt = _roundtrip(ps)
    assert rt.n_components == 3
    assert rt.array.shape == (4, 3, 20)
    assert rt.is_batched() is True
    assert np.allclose(np.asarray(rt.array), arr, atol=1e-5)
