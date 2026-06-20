"""Behaviour of the BinnedStatistic result objects (PDF / PeakCounts)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jax_fli.summary_statistics import PDF


@pytest.fixture(params=["pdf", "peaks"])
def binned_stat(request, gaussian_map):
    """A single-map PDF or PeakCounts (parametrized)."""
    if request.param == "pdf":
        return gaussian_map.compute_pdf(bins=30)
    return gaussian_map.compute_peak_counts(bins=30)


def test_kind_is_static_no_str_leaf(binned_stat):
    """Regression: ``kind`` must be a static field, not a dynamic str pytree leaf."""
    leaves = jax.tree_util.tree_leaves(binned_stat)
    assert not any(isinstance(leaf, str) for leaf in leaves)


def test_jit_on_object(binned_stat):
    """Regression: a str leaf would make jitting a function of the object raise."""
    total = jax.jit(lambda o: o.array.sum())(binned_stat)
    assert bool(jnp.isfinite(total))


def test_stack(gaussian_map):
    a = gaussian_map.compute_pdf(bins=30)
    stacked = PDF.stack([a, a])
    assert stacked.array.shape == (2, 30)
    assert stacked.is_batched()


def test_getitem_batched_keeps_bins(lpt_spherical):
    pdf = lpt_spherical.compute_pdf(bins=30)  # (S, 30)
    one = pdf[0]
    assert one.array.shape == (30,)
    assert not one.is_batched()
    assert jnp.allclose(one.bins, pdf.bins)  # batch indexing keeps the bin grid


def test_getitem_bin_axis(gaussian_map):
    pdf = gaussian_map.compute_pdf(bins=30)  # (30,) non-batched
    sub = pdf[5:10]
    assert sub.array.shape == (5,)
    assert sub.bins is not None and sub.bins.shape == (5,)


def test_full_like(gaussian_map):
    pdf = gaussian_map.compute_pdf(bins=30)
    zeros = PDF.full_like(pdf, 0.0)
    assert zeros.array.shape == pdf.array.shape
    assert float(jnp.sum(jnp.abs(zeros.array))) == 0.0
    assert zeros.kind == "pdf"


def test_plot_smoke(gaussian_map):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pdf = gaussian_map.compute_pdf(bins=30)
    fig, ax, artists = pdf.plot()
    assert len(artists) == 1
    plt.close(fig)
