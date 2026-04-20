import numpy as np

import jax_fli.data as jfli_data

FULL_SKY_DEG2 = 4 * np.pi * (180 / np.pi) ** 2  # ~41253 deg²


def test_desy3_mask_fsky_and_area():
    mask = jfli_data.get_desy3_mask(nside=2048)
    n_pix = 12 * 2048**2
    fsky = float(mask.sum()) / n_pix
    area_deg2 = fsky * FULL_SKY_DEG2

    assert abs(fsky - 0.1200) < 1e-3, f"fsky={fsky:.4f}, expected ~0.1200"
    assert abs(area_deg2 - 4952) < 10, f"area={area_deg2:.1f} deg², expected ~4952"


def test_desy3_mask_udgrade():
    mask_512 = jfli_data.get_desy3_mask(nside=512)
    assert mask_512.shape == (12 * 512**2,)
    assert mask_512.dtype == np.uint8
