"""Shared test utilities: tree-aware error metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

_FIELD_RTOL = 1e-5
_FIELD_ATOL = 1e-9
_PK_RTOL = 1e-5
_PK_ATOL = 1e-9
_MEAN_ATOL = 1e-12
_MEAN_RTOL = 1e-10


def MSE(pred, target):
    """Mean squared error, PyTree-aware."""
    return sum(jax.tree.leaves(jax.tree.map(lambda p, t: jnp.mean((p - t) ** 2), pred, target)))


def MSRE(pred, target):
    """Mean squared relative error, PyTree-aware."""
    return sum(jax.tree.leaves(jax.tree.map(lambda p, t: jnp.mean((p - t) ** 2 / (t**2 + 1e-10)), pred, target)))


def compare_fields(pred, target, label, rtol=_FIELD_RTOL, atol=_FIELD_ATOL, mean_rtol=_MEAN_RTOL, mean_atol=_MEAN_ATOL):
    """Compare two arrays: print MSE/MSRE diagnostics, assert mean-level and element-wise closeness.

    Caller is responsible for gathering distributed arrays before calling.
    Mean check uses stricter tolerances than element-wise because the mean
    is less sensitive to isolated outliers.
    """
    mse = float(MSE(pred, target))
    msre = float(MSRE(pred, target))
    print(f"{label}: MSE={mse:.3e}, MSRE={msre:.3e}")
    assert_allclose(mse, 0.0, rtol=mean_rtol, atol=mean_atol, err_msg=f"{label}: MSE exceeds mean tolerance")
    assert_allclose(pred, target, rtol=rtol, atol=atol, err_msg=f"{label}: element-wise tolerance exceeded")
