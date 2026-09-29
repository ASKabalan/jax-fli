"""The device-side 3D power spectrum reproduces ``jaxpm.utils.power_spectrum`` (the previous implementation)."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from jaxpm.utils import power_spectrum

from jax_fli._src.summary_statistics._compute import _coherence, _power, _transfer

jax.config.update("jax_enable_x64", True)

BOX = (200.0, 180.0, 220.0)  # anisotropic on purpose: each axis must get its own k scaling


@pytest.fixture(scope="module")
def meshes():
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    a = jax.random.normal(k1, (32, 32, 32))
    return a, 0.7 * a + 0.3 * jax.random.normal(k2, (32, 32, 32))


CASES = [
    dict(),
    dict(dk=0.05, kmax=0.4),
    dict(kedges=12),
    dict(kedges=0.03),
    dict(kedges=np.linspace(0.02, 0.4, 9)),
    dict(multipoles=(0, 2, 4)),
    dict(multipoles=2, los=(1.0, 0.0, 0.0)),
    dict(compensate_order=2),
    dict(shotnoise=(2, 0.01), compensate_order=2),
]


@pytest.mark.parametrize("kw", CASES)
@pytest.mark.parametrize("cross", [False, True])
def test_matches_jaxpm(meshes, kw, cross):
    a, b = meshes
    if cross and "shotnoise" in kw:
        pytest.skip("shot noise applies to auto-spectra only")
    mesh2 = b if cross else None
    los = kw.pop("los", (0.0, 0.0, 1.0))
    k, pk = _power(a, mesh2, box_shape=BOX, los=los, **kw)
    k_ref, pk_ref = power_spectrum(a, mesh2, box_shape=BOX, los=list(los), **kw)
    np.testing.assert_allclose(np.asarray(k), k_ref, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(pk), np.asarray(pk_ref), rtol=1e-10, atol=1e-12)


def test_transfer_coherence(meshes):
    a, b = meshes
    _, pa = power_spectrum(a, box_shape=BOX)
    _, pb = power_spectrum(b, box_shape=BOX)
    _, pab = power_spectrum(a, b, box_shape=BOX)
    np.testing.assert_allclose(np.asarray(_transfer(a, b, box_shape=BOX)[1]), np.sqrt(pb / pa), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(_coherence(a, b, box_shape=BOX)[1]), pab / np.sqrt(pa * pb), rtol=1e-10)
