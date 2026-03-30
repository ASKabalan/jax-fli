"""Shared fixtures for jax_fli tests."""

from __future__ import annotations

import jax
import jax_cosmo as jc
import pytest


@pytest.fixture(scope="session", autouse=True)
def enable_x64():
    """Enable float64 for all tests (required for lossless parquet round-trips)."""
    print("Enabling float64 precision for all tests.")
    jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="session")
def cosmology():
    """Planck18 cosmology instance."""
    return jc.Planck18()


def pytest_collection_modifyitems(items):
    """Auto-mark tests without 'distributed' marker as 'single_device'."""
    for item in items:
        if "distributed" not in item.keywords:
            item.add_marker(pytest.mark.single_device)
