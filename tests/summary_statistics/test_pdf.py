"""SphericalDensity.compute_pdf / pdf_spherical: API, correctness, and regressions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_fli.summary_statistics import PDF, pdf_spherical

# ---- API / shapes --------------------------------------------------------


def test_compute_pdf_returns_pdf(gaussian_map):
    pdf = gaussian_map.compute_pdf(bins=40)
    assert isinstance(pdf, PDF)
    assert pdf.kind == "pdf"
    assert pdf.array.shape == (40,)
    assert pdf.bins is not None and pdf.bins.shape == (40,)


def test_compute_pdf_batched_shape(lpt_spherical):
    n_shells = lpt_spherical.array.shape[0]
    pdf = lpt_spherical.compute_pdf(bins=30)
    assert isinstance(pdf, PDF)
    assert pdf.array.shape == (n_shells, 30)
    assert pdf.is_batched()


def test_compute_pdf_explicit_edges(gaussian_map):
    edges = jnp.linspace(-1.0, 1.0, 26)
    pdf = gaussian_map.compute_pdf(bins=edges)
    assert pdf.array.shape == (25,)
    assert pdf.bins is not None and pdf.bins.shape == (25,)


# ---- correctness ---------------------------------------------------------


def test_pdf_density_integrates_to_one(gaussian_map):
    pdf = gaussian_map.compute_pdf(bins=50, density=True)
    width = float(jnp.diff(pdf.bins)[0])  # uniform bins -> centre spacing == bin width
    assert float(jnp.sum(pdf.array) * width) == pytest.approx(1.0, abs=1e-6)


def test_pdf_counts_sum_to_npix(gaussian_map):
    edges = jnp.linspace(float(gaussian_map.array.min()), float(gaussian_map.array.max()), 41)
    _, counts = pdf_spherical(gaussian_map.array, bins=edges, density=False)
    assert float(jnp.sum(counts)) == pytest.approx(gaussian_map.array.size, abs=1.0)


def test_pdf_matches_gaussian_analytic():
    """Exact-histogram PDF of a large iid N(0,1) sample matches the analytic Gaussian."""
    vals = jnp.asarray(np.random.RandomState(0).randn(200_000))
    edges = jnp.linspace(-4.0, 4.0, 81)
    centers, pdf = pdf_spherical(vals, bins=edges, density=True)
    analytic = jax.scipy.stats.norm.pdf(centers)
    mask = jnp.abs(centers) < 3.0  # compare where the histogram is well sampled
    assert float(jnp.max(jnp.abs(pdf[mask] - analytic[mask]))) < 0.02


def test_pdf_soft_close_to_exact():
    vals = jnp.asarray(np.random.RandomState(1).randn(100_000))
    edges = jnp.linspace(-4.0, 4.0, 81)
    centers, exact = pdf_spherical(vals, bins=edges, density=True, soft=False)
    _, soft = pdf_spherical(vals, bins=edges, density=True, soft=True)
    mask = jnp.abs(centers) < 3.0
    assert float(jnp.max(jnp.abs(soft[mask] - exact[mask]))) < 0.03


def test_pdf_batched_matches_single(lpt_spherical):
    edges = jnp.linspace(float(lpt_spherical.array.min()), float(lpt_spherical.array.max()), 31)
    batched = lpt_spherical.compute_pdf(bins=edges)
    for i in range(lpt_spherical.array.shape[0]):
        single = lpt_spherical[i].compute_pdf(bins=edges)
        assert jnp.allclose(batched.array[i], single.array)


# ---- jit / grad / regressions -------------------------------------------


def test_pdf_jit_with_explicit_edges(gaussian_map):
    edges = jnp.linspace(-1.0, 1.0, 41)
    out = jax.jit(lambda m: pdf_spherical(m, bins=edges)[1])(gaussian_map.array)
    assert out.shape == (40,)


def test_pdf_soft_batched_no_concretization_error(lpt_spherical):
    """Regression: float(bins-derived bw) broke under jax.lax.map for batched soft PDFs."""
    pdf = lpt_spherical.compute_pdf(bins=40, soft=True)
    assert pdf.array.shape[0] == lpt_spherical.array.shape[0]
    assert bool(jnp.all(jnp.isfinite(pdf.array)))


def test_pdf_soft_grad_finite_and_nonzero():
    vals = jnp.asarray(np.random.RandomState(2).randn(5_000))
    edges = jnp.linspace(-4.0, 4.0, 41)
    grad = jax.grad(lambda x: pdf_spherical(x, bins=edges, soft=True)[1].sum())(vals)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.abs(grad).max()) > 0.0
