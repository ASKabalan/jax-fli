"""Shared fixtures for lightcone tests."""

from __future__ import annotations

import jax
import pytest

from jax_fli import gaussian_initial_conditions

MESH_SIZE = (16, 16, 16)
BOX_SIZE = (200.0, 200.0, 200.0)
FLATSKY_NPIX = (32, 32)
PAINT_NSIDE = 16


@pytest.fixture(scope="session")
def small_initial_field(cosmology):
    """Small Gaussian IC for lightcone tests (max_comoving_radius ~ 100 Mpc/h)."""
    return gaussian_initial_conditions(
        jax.random.PRNGKey(42),
        MESH_SIZE,
        BOX_SIZE,
        cosmo=cosmology,
        flatsky_npix=FLATSKY_NPIX,
        nside=PAINT_NSIDE,
    )
