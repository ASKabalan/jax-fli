"""SphericalDensity.compute_peak_counts / peak_counts_spherical: API, correctness, regressions."""

from __future__ import annotations

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_fli.summary_statistics import PeakCounts, peak_counts_spherical

# ---- API / shapes --------------------------------------------------------


def test_compute_peak_counts_returns_peakcounts(gaussian_map):
    pk = gaussian_map.compute_peak_counts(bins=40)
    assert isinstance(pk, PeakCounts)
    assert pk.kind == "peak_counts"
    assert pk.array.shape == (40,)


def test_compute_peak_counts_batched_shape(lpt_spherical):
    n_shells = lpt_spherical.array.shape[0]
    pk = lpt_spherical.compute_peak_counts(bins=30, normalize=True)
    assert pk.array.shape == (n_shells, 30)
    assert pk.is_batched()


# ---- correctness ---------------------------------------------------------


def test_single_pixel_is_one_peak(single_pixel_map):
    edges = jnp.linspace(-0.5, 1.5, 21)
    pk = single_pixel_map.compute_peak_counts(bins=edges)
    assert float(jnp.sum(pk.array)) == pytest.approx(1.0)  # exactly one local maximum
    assert pk.bins is not None
    peak_bin = int(jnp.argmax(pk.array))
    assert abs(float(pk.bins[peak_bin]) - 1.0) < 0.15  # at height ~1


def test_peaks_match_neighbour_reference(gaussian_map):
    """Detection + binning match an independent numpy + healpy.get_all_neighbours reference."""
    m = np.asarray(gaussian_map.array)
    nside = gaussian_map.nside
    neigh = hp.get_all_neighbours(nside, np.arange(m.size))  # (8, npix), -1 = missing
    valid = neigh >= 0
    neigh_vals = np.where(valid, m[np.where(valid, neigh, 0)], -np.inf)
    is_peak = m > neigh_vals.max(axis=0)
    edges = jnp.linspace(float(m.min()), float(m.max()), 31)
    ref, _ = np.histogram(m, bins=np.asarray(edges), weights=is_peak.astype(float))

    _, counts = peak_counts_spherical(gaussian_map.array, nside=nside, bins=edges)
    assert np.allclose(np.asarray(counts), ref)


def test_normalize_preserves_total_count(gaussian_map):
    """normalize only rescales heights (S/N); detection — hence total peaks — is unchanged."""
    m = np.asarray(gaussian_map.array)
    nside = gaussian_map.nside
    neigh = hp.get_all_neighbours(nside, np.arange(m.size))
    valid = neigh >= 0
    n_peaks = int((m > np.where(valid, m[np.where(valid, neigh, 0)], -np.inf).max(axis=0)).sum())
    # wide S/N range so every detected peak is binned
    snr = gaussian_map.compute_peak_counts(bins=jnp.linspace(-10.0, 20.0, 121), normalize=True)
    assert int(round(float(jnp.sum(snr.array)))) == n_peaks


def test_soft_mass_conservation(gaussian_map):
    edges = jnp.linspace(float(gaussian_map.array.min()), float(gaussian_map.array.max()), 41)
    _, hard = peak_counts_spherical(gaussian_map.array, nside=gaussian_map.nside, bins=edges, soft=False)
    _, soft = peak_counts_spherical(gaussian_map.array, nside=gaussian_map.nside, bins=edges, soft=True)
    assert float(jnp.sum(soft)) == pytest.approx(float(jnp.sum(hard)), rel=0.05)


def test_peaks_batched_matches_single(lpt_spherical):
    edges = jnp.linspace(float(lpt_spherical.array.min()), float(lpt_spherical.array.max()), 31)
    batched = lpt_spherical.compute_peak_counts(bins=edges)
    for i in range(lpt_spherical.array.shape[0]):
        single = lpt_spherical[i].compute_peak_counts(bins=edges)
        assert jnp.allclose(batched.array[i], single.array)


# ---- jit / regressions ---------------------------------------------------


def test_peaks_jit_with_explicit_edges(gaussian_map):
    edges = jnp.linspace(-1.0, 1.0, 31)
    out = jax.jit(lambda m: peak_counts_spherical(m, nside=gaussian_map.nside, bins=edges)[1])(gaussian_map.array)
    assert out.shape == (30,)


def test_peaks_soft_batched_no_concretization_error(lpt_spherical):
    """Regression: float(bins-derived bw) broke under jax.lax.map for batched soft peak counts."""
    pk = lpt_spherical.compute_peak_counts(bins=30, normalize=True, soft=True)
    assert pk.array.shape[0] == lpt_spherical.array.shape[0]
    assert bool(jnp.all(jnp.isfinite(pk.array)))
