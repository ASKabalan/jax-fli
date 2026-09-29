"""Tests for DensityField.ud_sample: band-limited Fourier resampling, checked against a direct 3-D mode copy."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
import pytest

import jax_fli as jfli

jax.config.update("jax_enable_x64", True)

BOX_SIZE = (180.0, 180.0, 180.0)


def _mode_copy(array, n):
    """Reference: copy the modes |n_i| < min(N, n) / 2 of a cube into an n^3 k-grid (numpy, 3-D)."""
    src = np.fft.fftn(np.asarray(array))
    out = np.zeros((n,) * 3, dtype=complex)
    idx_src, idx_tgt = [], []
    for size in src.shape:
        h = min(size, n) // 2
        low, high = np.arange(h), np.arange(1, h)
        idx_src.append(np.concatenate([low, size - high]))
        idx_tgt.append(np.concatenate([low, n - high]))
    out[np.ix_(*idx_tgt)] = src[np.ix_(*idx_src)] * (n**3 / array.size)
    return np.fft.ifftn(out).real


@pytest.fixture(scope="module")
def field():
    return jfli.gaussian_initial_conditions(jax.random.PRNGKey(0), (64, 64, 64), BOX_SIZE, cosmo=jc.Planck18())


@pytest.mark.parametrize("n", [40, 32, 96])  # non-integer ratio, integer ratio, upsampling
def test_matches_mode_copy(field, n):
    out = field.ud_sample(n)
    assert out.mesh_size == (n, n, n)
    assert out.array.shape == (n, n, n)
    np.testing.assert_allclose(np.asarray(out.array), _mode_copy(field.array, n), atol=1e-12)


def test_keeps_mean_and_round_trips(field):
    down = field.ud_sample(40)
    np.testing.assert_allclose(float(down.array.mean()), float(field.array.mean()), atol=1e-12)
    # up then down again: the kept modes survive both ways
    np.testing.assert_allclose(np.asarray(down.ud_sample(64).ud_sample(40).array), np.asarray(down.array), atol=1e-12)


def test_batched(field):
    batched = field.replace(array=jnp.stack([field.array, 2 * field.array]))
    out = batched.ud_sample(32)
    assert out.array.shape == (2, 32, 32, 32)
    np.testing.assert_allclose(np.asarray(out.array[1]), 2 * _mode_copy(field.array, 32), atol=1e-12)
