"""Tests for jax_fli.fields.units — unit conversion via the .to() API."""

from __future__ import annotations

import jax.numpy as jnp
import jax_fli as jfli
import jax_healpy as jhp
import numpy as np
import pytest
from jax_fli.fields.units import DensityUnit

# ---------------------------------------------------------------------------
# Tiny field geometry shared across tests
# ---------------------------------------------------------------------------
MESH_SIZE = (8, 8, 8)
BOX_SIZE = (200.0, 200.0, 200.0)
NSIDE = 4
N_SHELLS = 3
FLATSKY_NPIX = (16, 16)

COMOVING_CENTERS = jnp.array([25.0, 75.0, 125.0])
DENSITY_WIDTH = jnp.array([50.0, 50.0, 50.0])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def density_field():
    """3-D DensityField with a simple gradient density."""
    rng = jnp.arange(np.prod(MESH_SIZE), dtype=jnp.float64).reshape(MESH_SIZE) + 1.0
    return jfli.DensityField(
        array=rng,
        mesh_size=MESH_SIZE,
        box_size=BOX_SIZE,
        unit=DensityUnit.DENSITY,
    )


@pytest.fixture(scope="module")
def spherical_density():
    """SphericalDensity with N_SHELLS shells, each shell having a different constant."""
    npix = jhp.nside2npix(NSIDE)
    # Shell 0: value 1, shell 1: value 2, shell 2: value 3
    arrays = jnp.stack([jnp.full(npix, float(i + 1)) for i in range(N_SHELLS)])
    return jfli.SphericalDensity(
        array=arrays,
        mesh_size=MESH_SIZE,
        box_size=BOX_SIZE,
        nside=NSIDE,
        unit=DensityUnit.DENSITY,
        comoving_centers=COMOVING_CENTERS,
        density_width=DENSITY_WIDTH,
    )


@pytest.fixture(scope="module")
def spherical_counts():
    """SphericalDensity in COUNTS unit."""
    npix = jhp.nside2npix(NSIDE)
    arrays = jnp.stack([jnp.full(npix, float(i + 1)) for i in range(N_SHELLS)])
    return jfli.SphericalDensity(
        array=arrays,
        mesh_size=MESH_SIZE,
        box_size=BOX_SIZE,
        nside=NSIDE,
        unit=DensityUnit.COUNTS,
        comoving_centers=COMOVING_CENTERS,
        density_width=DENSITY_WIDTH,
    )


@pytest.fixture(scope="module")
def flat_density():
    """FlatDensity with N_SHELLS shells."""
    ny, nx = FLATSKY_NPIX
    # Shell i has all pixels = i+1
    arrays = jnp.stack([jnp.full((ny, nx), float(i + 1)) for i in range(N_SHELLS)])
    return jfli.FlatDensity(
        array=arrays,
        mesh_size=MESH_SIZE,
        box_size=BOX_SIZE,
        flatsky_npix=FLATSKY_NPIX,
        unit=DensityUnit.DENSITY,
        comoving_centers=COMOVING_CENTERS,
        density_width=DENSITY_WIDTH,
    )


# ---------------------------------------------------------------------------
# DensityField tests
# ---------------------------------------------------------------------------


class TestDensityField:
    def test_identity(self, density_field):
        """to(same unit) returns the same object."""
        result = density_field.to(DensityUnit.DENSITY)
        assert result is density_field

    def test_density_to_overdensity_global(self, density_field):
        """Global overdensity: mean of result is 0."""
        od = density_field.to(DensityUnit.OVERDENSITY)
        assert od.unit == DensityUnit.OVERDENSITY
        np.testing.assert_allclose(float(jnp.mean(od.array)), 0.0, atol=1e-10)

    def test_density_to_overdensity_explicit_mean(self, density_field):
        """Explicit mean_density overrides internal computation."""
        mean = float(jnp.mean(density_field.array))
        od_auto = density_field.to(DensityUnit.OVERDENSITY)
        od_explicit = density_field.to(DensityUnit.OVERDENSITY, mean_density=mean)
        np.testing.assert_allclose(od_explicit.array, od_auto.array, rtol=1e-10)

    def test_density_to_counts_roundtrip(self, density_field):
        """DENSITY → COUNTS → DENSITY round-trip."""
        counts = density_field.to(DensityUnit.COUNTS)
        assert counts.unit == DensityUnit.COUNTS
        recovered = counts.to(DensityUnit.DENSITY)
        np.testing.assert_allclose(recovered.array, density_field.array, rtol=1e-10)

    def test_wrong_unit_type_raises(self, density_field):
        with pytest.raises(TypeError):
            density_field.to("overdensity")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SphericalDensity tests
# ---------------------------------------------------------------------------


class TestSphericalDensity:
    def test_identity(self, spherical_density):
        result = spherical_density.to(DensityUnit.DENSITY)
        assert result is spherical_density

    def test_counts_to_density(self, spherical_counts):
        """COUNTS → DENSITY divides by the volume element."""
        density = spherical_counts.to(DensityUnit.DENSITY)
        assert density.unit == DensityUnit.DENSITY
        # All pixels within a shell should have the same value
        for i in range(N_SHELLS):
            shell = density.array[i]
            np.testing.assert_allclose(shell, shell[0], rtol=1e-10)

    def test_density_to_counts_roundtrip(self, spherical_density):
        counts = spherical_density.to(DensityUnit.COUNTS)
        recovered = counts.to(DensityUnit.DENSITY)
        np.testing.assert_allclose(recovered.array, spherical_density.array, rtol=1e-10)

    def test_overdensity_global_mean_is_zero(self, spherical_density):
        """Global overdensity: mean over entire array is 0."""
        od = spherical_density.to(DensityUnit.OVERDENSITY, normalization="global")
        assert od.unit == DensityUnit.OVERDENSITY
        np.testing.assert_allclose(float(jnp.mean(od.array)), 0.0, atol=1e-10)

    def test_overdensity_per_plane_each_shell_mean_is_zero(self, spherical_density):
        """Per-plane overdensity: each shell's mean is 0, but global mean need not be."""
        od = spherical_density.to(DensityUnit.OVERDENSITY, normalization="per_plane")
        assert od.unit == DensityUnit.OVERDENSITY
        for i in range(N_SHELLS):
            np.testing.assert_allclose(float(jnp.mean(od.array[i])), 0.0, atol=1e-10)

    def test_overdensity_per_plane_vs_global_differ_for_nonuniform(self, spherical_density):
        """For shells with different mean densities, global and per_plane give different results."""
        od_global = spherical_density.to(DensityUnit.OVERDENSITY, normalization="global")
        od_per_plane = spherical_density.to(DensityUnit.OVERDENSITY, normalization="per_plane")
        # per_plane gives all zeros (constant-density shells / constant per shell)
        # global gives non-zero values for shells deviating from the global mean
        assert not jnp.allclose(od_global.array, od_per_plane.array)

    def test_overdensity_uniform_shell_is_zero_per_plane(self, spherical_density):
        """A uniform shell (constant pixel values) has overdensity=0 per_plane."""
        od = spherical_density.to(DensityUnit.OVERDENSITY, normalization="per_plane")
        # Each shell has constant pixel values → per-plane δ = const/mean - 1 = 0
        np.testing.assert_allclose(od.array, 0.0, atol=1e-10)

    def test_overdensity_per_plane_preserves_shape(self, spherical_density):
        od = spherical_density.to(DensityUnit.OVERDENSITY, normalization="per_plane")
        assert od.array.shape == spherical_density.array.shape

    def test_overdensity_single_shell(self, spherical_density):
        """Single-shell (1D npix) field: per_plane falls back to global."""
        single = jfli.SphericalDensity(
            array=spherical_density.array[0],  # (npix,)
            mesh_size=MESH_SIZE,
            box_size=BOX_SIZE,
            nside=NSIDE,
            unit=DensityUnit.DENSITY,
            comoving_centers=COMOVING_CENTERS[:1],
            density_width=DENSITY_WIDTH[:1],
        )
        od = single.to(DensityUnit.OVERDENSITY, normalization="per_plane")
        # 1D constant array → global mean = constant → δ = 0 everywhere
        np.testing.assert_allclose(od.array, 0.0, atol=1e-10)

    def test_missing_metadata_raises(self):
        """to() raises when comoving_centers/density_width are missing."""
        npix = jhp.nside2npix(NSIDE)
        field = jfli.SphericalDensity(
            array=jnp.ones(npix),
            mesh_size=MESH_SIZE,
            box_size=BOX_SIZE,
            nside=NSIDE,
            unit=DensityUnit.COUNTS,
        )
        with pytest.raises(ValueError, match="comoving_centers"):
            field.to(DensityUnit.DENSITY)


# ---------------------------------------------------------------------------
# FlatDensity tests
# ---------------------------------------------------------------------------


class TestFlatDensity:
    def test_identity(self, flat_density):
        result = flat_density.to(DensityUnit.DENSITY)
        assert result is flat_density

    def test_density_to_counts_roundtrip(self, flat_density):
        counts = flat_density.to(DensityUnit.COUNTS)
        assert counts.unit == DensityUnit.COUNTS
        recovered = counts.to(DensityUnit.DENSITY)
        np.testing.assert_allclose(recovered.array, flat_density.array, rtol=1e-10)

    def test_overdensity_global_mean_is_zero(self, flat_density):
        od = flat_density.to(DensityUnit.OVERDENSITY, normalization="global")
        assert od.unit == DensityUnit.OVERDENSITY
        np.testing.assert_allclose(float(jnp.mean(od.array)), 0.0, atol=1e-10)

    def test_overdensity_per_plane_each_shell_mean_is_zero(self, flat_density):
        od = flat_density.to(DensityUnit.OVERDENSITY, normalization="per_plane")
        for i in range(N_SHELLS):
            np.testing.assert_allclose(float(jnp.mean(od.array[i])), 0.0, atol=1e-10)

    def test_overdensity_per_plane_uniform_shell_is_zero(self, flat_density):
        """Constant shells → per-plane δ = 0."""
        od = flat_density.to(DensityUnit.OVERDENSITY, normalization="per_plane")
        np.testing.assert_allclose(od.array, 0.0, atol=1e-10)

    def test_overdensity_per_plane_preserves_shape(self, flat_density):
        od = flat_density.to(DensityUnit.OVERDENSITY, normalization="per_plane")
        assert od.array.shape == flat_density.array.shape

    def test_overdensity_per_plane_nonuniform(self):
        """Non-uniform flat shells: per_plane normalises each shell independently."""
        ny, nx = FLATSKY_NPIX
        # Shell 0: [1,2,3,...] pixels; shell 1: constant 5
        row = jnp.arange(1, ny * nx + 1, dtype=jnp.float64).reshape(ny, nx)
        arr = jnp.stack([row, jnp.full((ny, nx), 5.0)])

        field = jfli.FlatDensity(
            array=arr,
            mesh_size=MESH_SIZE,
            box_size=BOX_SIZE,
            flatsky_npix=FLATSKY_NPIX,
            unit=DensityUnit.DENSITY,
            comoving_centers=COMOVING_CENTERS[:2],
            density_width=DENSITY_WIDTH[:2],
        )

        od = field.to(DensityUnit.OVERDENSITY, normalization="per_plane")

        # Each shell must have zero mean
        for i in range(2):
            np.testing.assert_allclose(float(jnp.mean(od.array[i])), 0.0, atol=1e-10)

        # Shell 1 (constant 5) → δ = 0 everywhere
        np.testing.assert_allclose(od.array[1], 0.0, atol=1e-10)

        # Shell 0 should have non-zero pixel variation
        assert float(jnp.std(od.array[0])) > 0.0
