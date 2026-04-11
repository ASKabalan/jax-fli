"""Lightcone-mode tests for LPT."""

from __future__ import annotations

import jax.numpy as jnp
import jax_cosmo as jc
import jax_fli as jfli
import pytest
from numpy.testing import assert_allclose

PAINT_NSIDE = 16


@pytest.mark.parametrize("target", ["flat", "spherical", "particles", "density"])
@pytest.mark.parametrize("a_target", [0.5, 0.8])
def test_scalar_ts(cosmology, small_initial_field, target, a_target):
    """Scalar ts → snapshot mode. flat/spherical should error; particles/density should succeed."""
    painting = jfli.PaintingOptions(target=target, paint_nside=PAINT_NSIDE)

    dx_field, _ = jfli.lpt(
        cosmology,
        small_initial_field,
        ts=a_target,
        painting=painting,
        density_widths=20.0,
    )
    expected_density = 20.0 if target in ["flat", "spherical"] else small_initial_field.box_size[2]

    r_comoving = jc.background.radial_comoving_distance(cosmology, a_target)
    z_target = jc.utils.a2z(a_target)

    assert not dx_field.is_batched()
    assert_allclose(dx_field.scale_factors, a_target, rtol=1e-15)
    assert_allclose(dx_field.comoving_centers, r_comoving, rtol=1e-15)
    assert_allclose(dx_field.z_sources, z_target, rtol=1e-15)
    assert_allclose(dx_field.density_width, expected_density, rtol=1e-15)

    if target == "particles":
        assert isinstance(dx_field, jfli.ParticleField)
    elif target == "density":
        assert isinstance(dx_field, jfli.DensityField)
        assert dx_field.status == jfli.FieldStatus.DENSITY_FIELD
    elif target == "flat":
        assert isinstance(dx_field, jfli.FlatDensity)
        assert dx_field.status == jfli.FieldStatus.LIGHTCONE
    elif target == "spherical":
        assert isinstance(dx_field, jfli.SphericalDensity)
        assert dx_field.status == jfli.FieldStatus.LIGHTCONE
    else:
        raise ValueError(f"Unexpected target: {target}")


@pytest.mark.parametrize("target", ["flat", "spherical", "particles", "density"])
def test_1d_ts(cosmology, small_initial_field, target):
    """1-D ts array → multi-shell lightcone."""
    ts = jnp.linspace(0.5, 0.9, 4)
    widths = jnp.full(4, 20.0)
    painting = jfli.PaintingOptions(target=target, paint_nside=PAINT_NSIDE)

    dx_field, _ = jfli.lpt(
        cosmology,
        small_initial_field,
        ts=ts,
        density_widths=widths,
        painting=painting,
    )

    if target == "flat":
        assert isinstance(dx_field, jfli.FlatDensity)
        assert dx_field.status == jfli.FieldStatus.LIGHTCONE
        assert dx_field.is_batched()
        assert jnp.atleast_1d(dx_field.scale_factors).shape[0] == 4
    elif target == "spherical":
        assert isinstance(dx_field, jfli.SphericalDensity)
        assert dx_field.status == jfli.FieldStatus.LIGHTCONE
        assert dx_field.is_batched()
        assert jnp.atleast_1d(dx_field.scale_factors).shape[0] == 4
    elif target == "density":
        # paint() collapses per-particle displacements into a single density grid
        assert isinstance(dx_field, jfli.DensityField)
        assert dx_field.status == jfli.FieldStatus.DENSITY_FIELD
    else:
        # particles: per-particle scale factors, not batched shells
        assert isinstance(dx_field, jfli.ParticleField)
        assert dx_field.status == jfli.FieldStatus.LIGHTCONE


@pytest.mark.parametrize("target", ["flat", "spherical", "particles", "density"])
def test_2d_ts(cosmology, small_initial_field, target):
    """2-D ts array (near/far) → multi-shell lightcone."""
    if target == "spherical":
        pytest.importorskip("jax_healpy")

    a_near = jnp.array([0.9, 0.8, 0.7])
    a_far = jnp.array([0.85, 0.75, 0.65])
    ts = jnp.stack([a_near, a_far])
    painting = jfli.PaintingOptions(target=target, paint_nside=PAINT_NSIDE)

    dx_field, _ = jfli.lpt(
        cosmology,
        small_initial_field,
        ts=ts,
        painting=painting,
    )

    if target == "flat":
        assert isinstance(dx_field, jfli.FlatDensity)
        assert dx_field.status == jfli.FieldStatus.LIGHTCONE
        assert dx_field.is_batched()
        assert jnp.atleast_1d(dx_field.scale_factors).shape[0] == 3
    elif target == "spherical":
        assert isinstance(dx_field, jfli.SphericalDensity)
        assert dx_field.status == jfli.FieldStatus.LIGHTCONE
        assert dx_field.is_batched()
        assert jnp.atleast_1d(dx_field.scale_factors).shape[0] == 3
    elif target == "density":
        assert isinstance(dx_field, jfli.DensityField)
        assert dx_field.status == jfli.FieldStatus.DENSITY_FIELD
    else:
        assert isinstance(dx_field, jfli.ParticleField)
        assert dx_field.status == jfli.FieldStatus.LIGHTCONE
