"""Fixtures for the summary-statistic tests.

The shared ``lpt_spherical`` fixture (a real LPT spherical map) lives in the top-level
``tests/conftest.py``. Here we add small synthetic maps with *known* one-point statistics, used for
the correctness checks: ``gaussian_map`` (a rich Gaussian realisation) and ``single_pixel_map``
(a deterministic single local maximum).
"""

from __future__ import annotations

import healpy as hp
import jax.numpy as jnp
import numpy as np
import pytest

from jax_fli import SphericalDensity

NSIDE = 16
NPIX = hp.nside2npix(NSIDE)
MESH = (16, 16, 16)
BOX = (256.0, 256.0, 256.0)


@pytest.fixture(scope="module")
def gaussian_map():
    """A ``SphericalDensity`` wrapping a Gaussian HEALPix realisation (rich, varied structure)."""
    lmax = 3 * NSIDE - 1
    ell = np.arange(lmax + 1)
    cl = 1.0 / (ell + 10.0) ** 2
    m = hp.synfast(cl, nside=NSIDE, lmax=lmax, new=True)
    return SphericalDensity(array=jnp.asarray(m), nside=NSIDE, mesh_size=MESH, box_size=BOX)


@pytest.fixture
def single_pixel_map():
    """A zeros map with one pixel set to 1.0 — exactly one local maximum, at height 1."""
    arr = jnp.zeros(NPIX).at[NPIX // 2].set(1.0)
    return SphericalDensity(array=arr, nside=NSIDE, mesh_size=MESH, box_size=BOX)
