"""DensityField.ud_sample on a sharded field: same values as on one device, output sharded like the input."""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import _PDIMS, make_sharded_ics


@pytest.mark.distributed
@pytest.mark.parametrize("pdims", _PDIMS)
@pytest.mark.parametrize("n", [24, 16, 48])  # 32 -> 24 (non-integer ratio), 32 -> 16, 32 -> 48 (up)
def test_ud_sample_sharded(single_device_ics, pdims, n):
    sharded, _, sharding = make_sharded_ics(single_device_ics, pdims)
    expected = single_device_ics.ud_sample(n).array
    out = sharded.ud_sample(n)
    assert out.array.sharding.is_equivalent_to(sharding, 3)
    np.testing.assert_allclose(np.asarray(out.array), np.asarray(expected), atol=1e-12)
