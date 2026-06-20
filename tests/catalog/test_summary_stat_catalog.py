"""Round-trip tests for the summary-statistic Catalog backend (PDF / PeakCounts).

Mirrors ``test_powerspec_catalog.py``. Field-type catalogs (densities, kappa maps, ...) live in
``test_catalog.py``; PowerSpectrum in ``test_powerspec_catalog.py``. The data source is the shared
``lpt_spherical`` fixture (a real LPT spherical map) from ``tests/conftest.py``.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

datasets = pytest.importorskip("datasets")

import jax_cosmo as jc

from jax_fli.io.catalog import Catalog


def assert_stat_close(reloaded_obj, original):
    """Assert a reconstructed stat matches the original (class + array + bins).

    ``Catalog.from_parquet`` reads through datasets' numpy formatter, which yields float32-precision
    arrays (the parquet on disk is full float64) — the same behaviour as the field/PowerSpectrum
    backends — so we compare with a relative tolerance rather than an absolute one.
    """
    assert type(reloaded_obj) is type(original)
    assert jnp.allclose(jnp.asarray(reloaded_obj.array), jnp.asarray(original.array), rtol=1e-4, atol=1e-6)
    assert jnp.allclose(jnp.asarray(reloaded_obj.bins), jnp.asarray(original.bins), rtol=1e-4, atol=1e-8)


# ---------------------------------------------------------------------------
# Fixtures: PDF / PeakCounts built from the LPT spherical map
# ---------------------------------------------------------------------------
@pytest.fixture
def cosmo():
    """A fresh Planck18 (no populated cache) for the catalog — matches test_powerspec_catalog."""
    return jc.Planck18()


@pytest.fixture(params=["pdf", "peaks"])
def stat_kind(request):
    return request.param


@pytest.fixture
def edges(lpt_spherical):
    return jnp.linspace(float(lpt_spherical.array.min()), float(lpt_spherical.array.max()), 41)


@pytest.fixture
def stat_single(stat_kind, lpt_spherical, edges):
    """A single-shell statistic (1-D array)."""
    shell = lpt_spherical[0]
    if stat_kind == "pdf":
        return shell.compute_pdf(bins=edges)
    return shell.compute_peak_counts(bins=edges)


@pytest.fixture
def stat_batched(stat_kind, lpt_spherical, edges):
    """A batched (S, n_bins) statistic carrying per-shell metadata."""
    if stat_kind == "pdf":
        return lpt_spherical.compute_pdf(bins=edges)
    return lpt_spherical.compute_peak_counts(bins=edges)


# ---------------------------------------------------------------------------
# Round-trips
# ---------------------------------------------------------------------------
def test_single_entry_parquet_roundtrip(tmp_path, stat_single, cosmo):
    cat = Catalog(stat_single, cosmo)
    assert cat.backend == "summary_stat"
    assert len(cat) == 1

    path = str(tmp_path / "stat.parquet")
    cat.to_parquet(path)
    reloaded = Catalog.from_parquet(path)
    assert reloaded.backend == "summary_stat"
    assert len(reloaded) == 1
    assert_stat_close(reloaded.field[0], stat_single)  # class + array + bins reconstructed


def test_batched_dataset_roundtrip(stat_batched, cosmo):
    cat = Catalog(stat_batched, cosmo)
    ds = cat.to_dataset()
    reloaded = Catalog.from_dataset(ds)
    assert reloaded.field[0].array.shape == stat_batched.array.shape
    assert reloaded.field[0].is_batched()
    assert_stat_close(reloaded.field[0], stat_batched)


def test_multi_entry_roundtrip(tmp_path, stat_kind, lpt_spherical, edges, cosmo):
    """Three single-shell entries -> exercises the multi-row from_dataset path."""
    stats = []
    for i in range(3):
        shell = lpt_spherical[i]
        stats.append(shell.compute_pdf(bins=edges) if stat_kind == "pdf" else shell.compute_peak_counts(bins=edges))
    cat = Catalog(stats, [cosmo, cosmo, cosmo])
    assert len(cat) == 3

    path = str(tmp_path / "multi.parquet")
    cat.to_parquet(path)
    reloaded = Catalog.from_parquet(path)
    assert len(reloaded) == 3
    for got, expected in zip(reloaded.field, stats):
        assert_stat_close(got, expected)


def test_metadata_preserved(tmp_path, stat_batched, cosmo):
    cat = Catalog(stat_batched, cosmo)
    path = str(tmp_path / "meta.parquet")
    cat.to_parquet(path)
    rt = Catalog.from_parquet(path).field[0]

    assert rt.nside == stat_batched.nside
    assert rt.status == stat_batched.status
    assert jnp.allclose(rt.bins, stat_batched.bins)
    assert jnp.allclose(jnp.atleast_1d(rt.scale_factors), jnp.atleast_1d(stat_batched.scale_factors))
    assert jnp.allclose(jnp.atleast_1d(rt.comoving_centers), jnp.atleast_1d(stat_batched.comoving_centers))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_single_normalization(stat_single, cosmo):
    """A single stat + cosmo is auto-wrapped into length-1 lists."""
    cat = Catalog(field=stat_single, cosmology=cosmo)
    assert isinstance(cat.field, list) and isinstance(cat.cosmology, list)
    assert len(cat) == 1
    assert cat.backend == "summary_stat"


def test_length_mismatch_raises(stat_kind, lpt_spherical, edges, cosmo):
    a = lpt_spherical[0].compute_pdf(bins=edges)
    b = lpt_spherical[1].compute_pdf(bins=edges)
    with pytest.raises(ValueError, match="same length"):
        Catalog(field=[a, b], cosmology=[cosmo])


def test_getitem(stat_kind, lpt_spherical, edges, cosmo):
    stats = [lpt_spherical[i].compute_pdf(bins=edges) for i in range(3)]
    cat = Catalog(stats, [cosmo, cosmo, cosmo])
    assert len(cat[0]) == 1
    assert len(cat[0:2]) == 2


# ---------------------------------------------------------------------------
# Homogeneity: summary stat and field cannot mix (different backends)
# ---------------------------------------------------------------------------
def test_stat_then_field_raises(lpt_spherical, edges, cosmo):
    pdf = lpt_spherical[0].compute_pdf(bins=edges)
    with pytest.raises(TypeError, match="homogeneous"):
        Catalog(field=[pdf, lpt_spherical], cosmology=[cosmo, cosmo])


def test_field_then_stat_raises(lpt_spherical, edges, cosmo):
    pdf = lpt_spherical[0].compute_pdf(bins=edges)
    with pytest.raises(TypeError, match="homogeneous"):
        Catalog(field=[lpt_spherical, pdf], cosmology=[cosmo, cosmo])


# ---------------------------------------------------------------------------
# Regression: the existing backends still round-trip
# ---------------------------------------------------------------------------
def test_powerspectrum_still_roundtrips(tmp_path, lpt_spherical, cosmo):
    ps = lpt_spherical[0].angular_cl(method="healpy")
    Catalog(ps, cosmo).to_parquet(str(tmp_path / "ps.parquet"))
    rt = Catalog.from_parquet(str(tmp_path / "ps.parquet")).field[0]
    assert type(rt).__name__ == "PowerSpectrum"
    assert jnp.allclose(rt.array, ps.array, atol=1e-6)


def test_sphericaldensity_still_roundtrips(tmp_path, lpt_spherical, cosmo):
    Catalog(lpt_spherical, cosmo).to_parquet(str(tmp_path / "field.parquet"))
    rt = Catalog.from_parquet(str(tmp_path / "field.parquet")).field[0]
    assert type(rt).__name__ == "SphericalDensity"
    assert jnp.allclose(rt.array, lpt_spherical.array, atol=1e-6)
