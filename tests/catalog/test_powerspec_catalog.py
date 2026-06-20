"""Round-trip tests for PowerSpectrum catalogs (Catalog ``power_spec`` backend).

Covers the 4-method API (parquet + dataset) for auto / 2D-cross / explicit 3-component spectra,
array-level batching, scale-factor metadata, and the list-based Catalog edge cases. Field-type
catalogs (densities, kappa/shear maps, particles) live in ``test_catalog.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

datasets = pytest.importorskip("datasets")

import jax_cosmo as jc

import jax_fli as jfli
from jax_fli._src.io._power_spec_catalog import PS_CATALOG_VERSION
from jax_fli.io.catalog import Catalog
from jax_fli.summary_statistics import cross_angular_cl_spherical

jax.config.update("jax_enable_x64", True)  # Use float64 for better precision in tests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NSIDE = 4  # 12 * 4^2 = 192 pixels
PS_SPEC_TYPES = ["auto", "cross_2d"]


# ---------------------------------------------------------------------------
# Error function
# ---------------------------------------------------------------------------
def error(tree1, tree2):
    """Compute max squared error across all leaves of two PyTrees."""
    return jax.tree.reduce(
        lambda x, y: x + y,
        jax.tree_util.tree_map(lambda x, y: jnp.max((x - y) ** 2), tree1, tree2),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_ps(spec_type: str, with_scale_factors: bool, seed: int = 0) -> jfli.PowerSpectrum:
    """Create a PowerSpectrum by computing angular Cl on HEALPix maps.

    spec_type="auto"     : auto-power spectrum of one map → array shape (n_ell,)
    spec_type="cross_2d" : cross-spectra of 3 maps        → array shape (6, n_ell)
    """
    rng = np.random.RandomState(seed)
    npix = 12 * NSIDE**2  # 192

    if spec_type == "auto":
        map1 = jnp.asarray(rng.randn(npix), dtype=jnp.float64)
        ell, cl = jfli.angular_cl_spherical(map1, lmax=3 * NSIDE - 1, method="healpy")
        ps = jfli.PowerSpectrum(wavenumber=ell, array=cl, name="cl")
    elif spec_type == "cross_2d":
        maps = jnp.asarray(rng.randn(3, npix), dtype=jnp.float64)
        ell, cl_2d = cross_angular_cl_spherical(maps, lmax=3 * NSIDE - 1, method="healpy")
        ps = jfli.PowerSpectrum(wavenumber=ell, array=cl_2d, name="cl")
    else:
        raise ValueError(f"Unknown spec_type: {spec_type}")

    if with_scale_factors:
        n_ell = ell.shape[0]
        sf = jnp.linspace(0.5, 1.0, n_ell, dtype=jnp.float64)
        ps = jfli.PowerSpectrum(wavenumber=ps.wavenumber, array=ps.array, name=ps.name, scale_factors=sf)

    return ps


def make_ps_catalog(spec_type: str, n_entries: int, with_scale_factors: bool) -> Catalog:
    """Build a Catalog with n_entries PowerSpectrum + Planck18 cosmologies."""
    spectra = [make_ps(spec_type, with_scale_factors, seed=10 + i) for i in range(n_entries)]
    cosmologies = [jc.Planck18() for _ in range(n_entries)]
    return Catalog(field=spectra, cosmology=cosmologies, version=PS_CATALOG_VERSION)


def _roundtrip_ps(ps: jfli.PowerSpectrum, tmp_path) -> jfli.PowerSpectrum:
    """Write a single PowerSpectrum through a parquet Catalog and read back the entry."""
    path = str(tmp_path / "ps.parquet")
    Catalog(ps, jc.Planck18()).to_parquet(path)
    return Catalog.from_parquet(path).field[0]


# ---------------------------------------------------------------------------
# Main parametrized test: 2 x 2 x 2 = 8 tests, each testing parquet + dataset
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_entries", [1, 3], ids=["single", "multi"])
@pytest.mark.parametrize("with_scale_factors", [True, False], ids=["with_sf", "no_sf"])
@pytest.mark.parametrize("spec_type", PS_SPEC_TYPES)
def test_ps_catalog_roundtrip(tmp_path, spec_type, with_scale_factors, n_entries):
    """Round-trip: PowerSpectrum Catalog -> write -> read -> compare."""
    catalog = make_ps_catalog(spec_type, n_entries=n_entries, with_scale_factors=with_scale_factors)

    assert catalog.backend == "power_spec"
    assert len(catalog) == n_entries

    # Parquet round-trip
    path = str(tmp_path / "test_ps.parquet")
    catalog.to_parquet(path)
    reloaded = Catalog.from_parquet(path)
    assert isinstance(reloaded, Catalog)
    assert len(reloaded) == n_entries
    assert reloaded.backend == "power_spec"
    assert error(catalog, reloaded) < 1e-8

    # Dataset round-trip
    ds = catalog.to_dataset()
    reloaded_ds = Catalog.from_dataset(ds)
    assert isinstance(reloaded_ds, Catalog)
    assert len(reloaded_ds) == n_entries
    assert reloaded_ds.backend == "power_spec"
    assert error(catalog, reloaded_ds) < 1e-8


# ---------------------------------------------------------------------------
# Explicit n_components / array-level batching
# ---------------------------------------------------------------------------
def test_three_component_ps_roundtrip(tmp_path):
    """A 3-component spectrum (e.g. EE/EB/BB) round-trips with n_components preserved."""
    ell = jnp.arange(20.0)
    arr = jnp.asarray(np.random.RandomState(2).randn(3, 20), dtype=jnp.float64)
    ps = jfli.PowerSpectrum(wavenumber=ell, array=arr, n_components=3, nside=NSIDE)
    rt = _roundtrip_ps(ps, tmp_path)
    assert rt.n_components == 3
    assert rt.array.shape == (3, 20)
    assert np.allclose(np.asarray(rt.array), np.asarray(arr), atol=1e-8)


def test_batched_3d_ps_roundtrip(tmp_path):
    """An array-batched (B, n_components, n_ell) spectrum round-trips and stays batched."""
    ell = jnp.arange(20.0)
    arr = jnp.asarray(np.random.RandomState(3).randn(4, 3, 20), dtype=jnp.float64)
    ps = jfli.PowerSpectrum(wavenumber=ell, array=arr, n_components=3, nside=NSIDE)
    rt = _roundtrip_ps(ps, tmp_path)
    assert rt.n_components == 3
    assert rt.array.shape == (4, 3, 20)
    assert rt.is_batched() is True
    assert np.allclose(np.asarray(rt.array), np.asarray(arr), atol=1e-8)


# ---------------------------------------------------------------------------
# PowerSpectrum edge-case tests
# ---------------------------------------------------------------------------
def test_ps_catalog_single_normalization():
    """Single PowerSpectrum/cosmology should be auto-wrapped to list."""
    ps = make_ps("auto", with_scale_factors=False)
    cosmo = jc.Planck18()
    cat = Catalog(field=ps, cosmology=cosmo)
    assert isinstance(cat.field, list)
    assert isinstance(cat.cosmology, list)
    assert len(cat) == 1
    assert cat.backend == "power_spec"


def test_ps_catalog_length_mismatch():
    """Mismatched list lengths should raise ValueError."""
    ps1 = make_ps("auto", with_scale_factors=False, seed=1)
    ps2 = make_ps("auto", with_scale_factors=False, seed=2)
    cosmo = jc.Planck18()
    with pytest.raises(ValueError, match="same length"):
        Catalog(field=[ps1, ps2], cosmology=[cosmo])


def test_ps_catalog_getitem():
    """Indexing a multi-entry PowerSpectrum Catalog returns a sub-Catalog."""
    catalog = make_ps_catalog("auto", n_entries=3, with_scale_factors=False)
    assert len(catalog) == 3

    sub = catalog[0]
    assert isinstance(sub, Catalog)
    assert len(sub) == 1

    sliced = catalog[0:2]
    assert isinstance(sliced, Catalog)
    assert len(sliced) == 2
