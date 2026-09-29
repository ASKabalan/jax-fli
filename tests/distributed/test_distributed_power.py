"""3D power / transfer / coherence on a sharded field: same spectra as on one device, and no N^3 gather."""

from __future__ import annotations

import re

import jax
import numpy as np
import pytest

from .conftest import _PDIMS, make_sharded_ics


@pytest.mark.distributed
@pytest.mark.parametrize("pdims", _PDIMS)
def test_power_transfer_coherence_sharded(single_device_ics, pdims):
    sharded, _, _ = make_sharded_ics(single_device_ics, pdims)
    other = single_device_ics.replace(array=0.8 * single_device_ics.array + 0.1 * single_device_ics.array**2)
    other_sharded, _, _ = make_sharded_ics(other, pdims)
    for method in ("power", "transfer", "coherence"):
        args_single = () if method == "power" else (other,)
        args_sharded = () if method == "power" else (other_sharded,)
        expected = getattr(single_device_ics, method)(*args_single)
        out = getattr(sharded, method)(*args_sharded)
        np.testing.assert_allclose(np.asarray(out.wavenumber), np.asarray(expected.wavenumber), rtol=1e-12)
        np.testing.assert_allclose(np.asarray(out.array), np.asarray(expected.array), rtol=1e-10, atol=1e-14)


@pytest.mark.distributed
def test_power_does_not_gather_the_field(single_device_ics):
    if jax.device_count() < 4:
        pytest.skip("needs a 2-D device grid")
    sharded, _, _ = make_sharded_ics(single_device_ics, (jax.device_count() // 2, 2))
    hlo = jax.jit(lambda f: f.power().array).lower(sharded).compile().as_text()
    n_cells = int(np.prod(sharded.mesh_size))
    for line in hlo.splitlines():
        if re.search(r"= \S+ all-gather\(", line):
            shape = re.search(r"\[([\d,]+)\]", line).group(1)
            assert np.prod([int(s) for s in shape.split(",")]) < n_cells, line
