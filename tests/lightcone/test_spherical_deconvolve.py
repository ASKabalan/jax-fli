"""Tests for ``SphericalDensity.deconvolve()`` — HEALPix window deconvolution.

Added in response to the Copilot review on PR #44: the new public deconvolution API
(``jax_fli.power.deconvolve_spherical`` and ``SphericalDensity.deconvolve``) was
introduced without automated tests. All coverage here drives the method
``SphericalDensity.deconvolve()`` directly (the functional ``power`` wrapper is not
called).

The headline behavior test mirrors JaxPM's
``test_deconvolve_painted_map_boosts_high_ell``: dividing a *painted* map by its
mass-assignment window ``W_l`` (which is ``< 1`` for ``l > 0``) must raise high-``l``
power while keeping the map finite. The check compares ``C_l`` before/after as a ratio
over the same realization, so cosmic variance cancels and the boost is deterministic.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_fli as jfli

NSIDE = 16
LMAX = 2 * NSIDE - 1
NPIX = 12 * NSIDE * NSIDE

# 1-pixel-FWHM Gaussian beam for the RBF scheme, in arcmin. ``PaintingOptions`` exposes
# ``kernel_width_arcmin`` (not ``kernel_width_pixels``), so we paint and deconvolve with
# the same arcmin width to keep the deconvolution window matched to the painting window.
RBF_KERNEL_ARCMIN = float(np.sqrt(4.0 * np.pi / NPIX) * (180.0 / np.pi) * 60.0)

# (scheme, kwargs shared by painting and deconvolution). RBF carries a matched Gaussian
# beam; NGP uses the bare HEALPix pixel window.
METHOD_KW = [
    ("ngp", {}),
    ("rbf_neighbor", dict(kernel_width_arcmin=RBF_KERNEL_ARCMIN, smoothing_interpretation="fwhm")),
]


# Scale factor / shell width chosen so the comoving shell lands *inside* the
# small_initial_field box (200 Mpc/h, observer at the centre -> radius must stay below
# ~100 Mpc/h). At ts=0.97 the comoving distance is ~92 Mpc/h, so the shell is well
# populated; larger ts pushes the shell outside the box and paints an empty map.
TS = 0.97
DENSITY_WIDTH = 40.0


@pytest.fixture(scope="session")
def painted_sphere(cosmology, small_initial_field):
    """A single ngp-painted ``SphericalDensity`` lightcone shell (nside=16)."""
    painting = jfli.PaintingOptions(target="spherical", scheme="ngp")
    field, _ = jfli.lpt(
        cosmology,
        small_initial_field,
        ts=TS,
        density_widths=DENSITY_WIDTH,
        painting=painting,
    )
    return field


@pytest.mark.parametrize("method,kw", METHOD_KW)
def test_deconvolve_spherical_boosts_high_ell(cosmology, small_initial_field, method, kw):
    """Deconvolving a painted map via ``SphericalDensity.deconvolve`` boosts high-l power."""
    pytest.importorskip("s2fft")

    # Paint a real lightcone shell with the scheme whose window we will remove.
    painting = jfli.PaintingOptions(target="spherical", scheme=method, **kw)
    field, _ = jfli.lpt(
        cosmology,
        small_initial_field,
        ts=TS,
        density_widths=DENSITY_WIDTH,
        painting=painting,
    )
    assert isinstance(field, jfli.SphericalDensity)
    assert jnp.mean(field.array) > 0, "painted shell is empty; geometry is degenerate"

    # Overdensity (deconvolution is meant for an overdensity map). Build it directly to
    # keep the test focused on deconvolution rather than unit conversion.
    od = field.replace(array=field.array / jnp.mean(field.array) - 1.0)

    dec = od.deconvolve(method, lmax=LMAX, **kw)

    # API / shape contract.
    assert isinstance(dec, jfli.SphericalDensity)
    assert dec.array.shape == od.array.shape
    assert dec.nside == NSIDE
    assert bool(jnp.all(jnp.isfinite(dec.array))), method

    # Behavior: window deconvolution amplifies high-l power.
    ell, cl_raw = jfli.angular_cl_spherical(od.array, lmax=LMAX, method="healpy")
    _, cl_dec = jfli.angular_cl_spherical(dec.array, lmax=LMAX, method="healpy")
    ell = np.asarray(ell)
    cl_raw = np.asarray(cl_raw)
    cl_dec = np.asarray(cl_dec)

    hi = (ell >= int(0.4 * NSIDE)) & (ell <= int(0.7 * NSIDE))
    assert np.median(cl_dec[hi]) > np.median(cl_raw[hi]), method


def test_spherical_density_deconvolve_invalid_method(painted_sphere):
    """``method`` is forwarded: bilinear has no closed-form window; unknown -> ValueError."""
    # Both raise during method dispatch, before reaching the spherical-transform path.
    with pytest.raises(NotImplementedError):
        painted_sphere.deconvolve("bilinear")
    with pytest.raises(ValueError):
        painted_sphere.deconvolve("not_a_method")


def test_spherical_density_deconvolve_batched(painted_sphere):
    """Batched (S, npix) maps deconvolve map-by-map via ``jax.lax.map`` and keep shape."""
    pytest.importorskip("s2fft")

    arr = jax.random.normal(jax.random.PRNGKey(0), (3, NPIX))
    sd = painted_sphere.replace(array=arr)

    out = sd.deconvolve("ngp", lmax=LMAX)

    assert isinstance(out, jfli.SphericalDensity)
    assert out.array.shape == (3, NPIX)
    assert bool(jnp.all(jnp.isfinite(out.array)))
