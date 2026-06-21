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


def test_appodize_mask():
    """C2 apodization of the DES Y3 footprint: tapered, in [0, 1], 0 outside, positive inside."""
    mask = jfli_data.get_desy3_mask(nside=128)
    apodized = np.asarray(jfli_data.apodize(mask, aposize_deg=0.5))  # 30 arcmin taper

    inside = mask > 0

    # Shape / dtype / range
    assert apodized.shape == mask.shape
    assert np.issubdtype(apodized.dtype, np.floating)
    assert np.all(apodized >= 0) and np.all(apodized <= 1), "Apodized mask values should be in [0, 1]"

    # Support is exactly the footprint: identically 0 outside, strictly positive inside
    assert np.all(apodized[~inside] == 0.0), "Apodized mask must be exactly 0 outside the footprint"
    assert np.all(apodized[inside] > 0.0), "Apodized mask must be positive everywhere inside the footprint"

    # Genuinely tapered (not just a copy of the binary mask): the interior saturates to 1
    # while a band of edge pixels takes fractional values.
    assert np.isclose(apodized.max(), 1.0), "Interior pixels should saturate to 1"
    assert np.any((apodized > 0) & (apodized < 1)), "Apodization should produce a fractional taper band"

    # Tapering only removes weight at the edges, so the total stays below the binary footprint area.
    assert 0 < apodized.sum() < float(mask.sum())


def test_apodize_taper_widens_with_radius():
    """A larger apodization radius tapers more edge pixels, lowering the total weight monotonically."""
    mask = jfli_data.get_desy3_mask(nside=128)
    binary_sum = float(mask.sum())
    sum_small = float(np.asarray(jfli_data.apodize(mask, aposize_deg=0.5)).sum())
    sum_large = float(np.asarray(jfli_data.apodize(mask, aposize_deg=2.0)).sum())
    assert sum_large < sum_small < binary_sum


def test_apodize_c1_window():
    """C1 apodization: a valid taper (in [0, 1], 0 outside, saturating inside) distinct from C2."""
    mask = jfli_data.get_desy3_mask(nside=128)
    inside = mask > 0
    c1 = np.asarray(jfli_data.apodize(mask, aposize_deg=0.5, apotype="C1"))
    c2 = np.asarray(jfli_data.apodize(mask, aposize_deg=0.5, apotype="C2"))

    assert np.all(c1 >= 0) and np.all(c1 <= 1), "C1 values should be in [0, 1]"
    assert np.all(c1[~inside] == 0.0), "C1 must be exactly 0 outside the footprint"
    assert np.all(c1[inside] > 0.0), "C1 must be positive everywhere inside the footprint"
    assert np.isclose(c1.max(), 1.0), "Interior pixels should saturate to 1"
    assert np.any((c1 > 0) & (c1 < 1)), "C1 should produce a fractional taper band"
    assert np.any(c1 != c2), "C1 and C2 share the support but differ in the taper profile"
