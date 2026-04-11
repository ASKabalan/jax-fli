"""Tests for resolve_geometry with flexible shell-spacing modes."""

from __future__ import annotations

import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
import pytest
from jax_fli.pm._resolve_geometry import (
    _VALID_SHELL_SPACINGS,
    compute_n_steps_for_shells,
    resolve_geometry,
    simulation_stepping,
)
from jax_fli.utils import distances
from jaxpm.growth import growth_factor

# ---------------------------------------------------------------------------
# Constants & fixtures
# ---------------------------------------------------------------------------

MAX_BOX = 1000.0  # max_comoving_radius for box_size=2000
NB_SHELLS = 8
A_INI = 0.1


@pytest.fixture(scope="session")
def r_max_sim(cosmology):
    """Comoving distance at a=A_INI — used by n_steps heuristic tests."""
    return float(jc.background.radial_comoving_distance(cosmology, jnp.array(A_INI)).squeeze())


# ---------------------------------------------------------------------------
# TestManualTs
# ---------------------------------------------------------------------------


class TestManualTs:
    def test_ts_scalar(self, cosmology):
        """Scalar ts produces single-element outputs with zero density width."""
        ts_out, r_centers, dw = resolve_geometry(cosmology, MAX_BOX, ts=0.5, box_size_z=MAX_BOX * 2)
        assert ts_out.shape == (1,)
        assert r_centers.shape == (1,)
        assert dw.shape == (1,)
        assert float(dw[0]) == MAX_BOX * 2

    def test_ts_2d(self, cosmology):
        """2-D (2, N) near/far ts produces correct output."""
        a_near = jnp.array([0.9, 0.8, 0.7])
        a_far = jnp.array([0.85, 0.75, 0.65])
        ts_input = jnp.stack([a_near, a_far])
        ts_out, r_centers, dw = resolve_geometry(cosmology, MAX_BOX, ts=ts_input, box_size_z=MAX_BOX * 2)
        assert ts_out.shape == (3,)
        assert r_centers.shape == (3,)
        assert dw.shape == (3,)
        # Widths should be positive
        assert np.all(np.asarray(dw) > 0)

    def test_ts_1d_with_widths(self, cosmology):
        """1-D ts with explicit density_widths succeeds."""
        ts_input = jnp.linspace(0.3, 0.9, 5)
        widths = jnp.full(5, 100.0)
        ts_out, r_centers, dw = resolve_geometry(
            cosmology, MAX_BOX, ts=ts_input, box_size_z=MAX_BOX * 2, density_widths=widths
        )
        assert ts_out.shape == (5,)
        assert r_centers.shape == (5,)
        assert dw.shape == (5,)
        np.testing.assert_allclose(ts_out, ts_input)
        np.testing.assert_allclose(dw, 100.0)

    def test_ts_1d_without_widths(self, cosmology):
        """1-D ts without density_widths auto-computes Voronoi widths."""
        ts_input = jnp.linspace(0.3, 0.9, 5)
        ts_out, r_centers, dw = resolve_geometry(cosmology, MAX_BOX, ts=ts_input, box_size_z=MAX_BOX * 2)

        dw_expected = distances(r_centers, MAX_BOX)

        np.testing.assert_allclose(ts_out, ts_input)
        np.testing.assert_allclose(dw, dw_expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# TestNbShells
# ---------------------------------------------------------------------------


class TestNbShells:
    def test_comoving(self, cosmology):
        ts, r_centers, dw = resolve_geometry(
            cosmology,
            MAX_BOX,
            nb_shells=NB_SHELLS,
            shell_spacing="comoving",
            box_size_z=MAX_BOX * 2,
        )
        assert ts.shape == (NB_SHELLS,)
        assert r_centers.shape == (NB_SHELLS,)
        assert dw.shape == (NB_SHELLS,)

        # Comoving widths should be uniform
        np.testing.assert_allclose(dw, dw[0], rtol=1e-5)

        # Descending r order
        assert np.all(np.diff(np.asarray(r_centers)) <= 0)

    def test_a(self, cosmology):
        ts, r_centers, dw = resolve_geometry(
            cosmology,
            MAX_BOX,
            nb_shells=NB_SHELLS,
            shell_spacing="a",
            box_size_z=MAX_BOX * 2,
        )
        assert ts.shape == (NB_SHELLS,)
        assert r_centers.shape == (NB_SHELLS,)

        # Scale factors at centers should have roughly uniform spacing in a
        a_sorted = np.sort(np.asarray(ts))
        a_diffs = np.diff(a_sorted)
        np.testing.assert_allclose(a_diffs, a_diffs[0], rtol=0.15)

    def test_growth(self, cosmology):
        ts, r_centers, dw = resolve_geometry(
            cosmology,
            MAX_BOX,
            nb_shells=NB_SHELLS,
            shell_spacing="growth",
            box_size_z=MAX_BOX * 2,
        )
        assert ts.shape == (NB_SHELLS,)
        assert r_centers.shape == (NB_SHELLS,)

        # Growth factor at centers should have uniform spacing
        D_centers = np.asarray(growth_factor(cosmology, ts))
        D_sorted = np.sort(D_centers)
        D_diffs = np.diff(D_sorted)
        np.testing.assert_allclose(D_diffs, D_diffs[0], rtol=0.15)

    def test_equal_vol(self, cosmology):
        ts, r_centers, dw = resolve_geometry(
            cosmology,
            MAX_BOX,
            nb_shells=NB_SHELLS,
            shell_spacing="equal_vol",
            min_width=40.0,
            box_size_z=MAX_BOX * 2,
        )
        assert ts.shape == (NB_SHELLS,)
        assert r_centers.shape == (NB_SHELLS,)

        # Volume increments should be uniform
        r_sorted = np.sort(np.asarray(r_centers))
        dw_sorted = np.asarray(dw)[np.argsort(np.asarray(r_centers))]
        r_edges_lo = r_sorted - dw_sorted / 2
        r_edges_hi = r_sorted + dw_sorted / 2
        vol_increments = r_edges_hi**3 - r_edges_lo**3
        np.testing.assert_allclose(vol_increments, vol_increments[0], rtol=1e-5)


# ---------------------------------------------------------------------------
# TestNStepsForShells
# ---------------------------------------------------------------------------


class TestNStepsForShells:
    """Verify the public compute_n_steps_for_shells heuristic."""

    @pytest.mark.parametrize("shell_spacing", _VALID_SHELL_SPACINGS)
    def test_more_steps_than_shells(self, cosmology, r_max_sim, shell_spacing):
        n_steps = compute_n_steps_for_shells(cosmology, shell_spacing, NB_SHELLS, MAX_BOX, r_max_sim)
        assert n_steps > NB_SHELLS

    @pytest.mark.parametrize("shell_spacing", _VALID_SHELL_SPACINGS)
    def test_positive(self, cosmology, r_max_sim, shell_spacing):
        n_steps = compute_n_steps_for_shells(cosmology, shell_spacing, NB_SHELLS, MAX_BOX, r_max_sim)
        assert n_steps > 0


# ---------------------------------------------------------------------------
# TestMinWidthNbShells
# ---------------------------------------------------------------------------


class TestMinWidthNbShells:
    """Verify error is raised when too many shells force widths below min_width."""

    def _run_expect_error(self, cosmology, shell_spacing):
        with pytest.raises(ValueError, match="min_width"):
            resolve_geometry(
                cosmology,
                MAX_BOX,
                nb_shells=200,
                shell_spacing=shell_spacing,
                min_width=50.0,
                box_size_z=MAX_BOX * 2,
            )

    def test_comoving_min_width(self, cosmology):
        self._run_expect_error(cosmology, "comoving")

    def test_a_min_width(self, cosmology):
        self._run_expect_error(cosmology, "a")

    def test_growth_min_width(self, cosmology):
        self._run_expect_error(cosmology, "growth")

    def test_equal_vol_min_width(self, cosmology):
        self._run_expect_error(cosmology, "equal_vol")


# ---------------------------------------------------------------------------
# TestSimulationStepping
# ---------------------------------------------------------------------------


class TestSimulationStepping:
    """Test the simulation_stepping visualization helper."""

    def test_a_stepping_no_shells(self, cosmology):
        """Without shells, returns n_steps+1 uniform a-steps."""
        a_vals = simulation_stepping(cosmology, 0.1, 1.0, 10, time_stepping="a")
        assert a_vals.shape == (11,)
        np.testing.assert_allclose(float(a_vals[0]), 0.1, atol=1e-10)
        np.testing.assert_allclose(float(a_vals[-1]), 1.0, atol=1e-10)
        assert np.all(np.diff(np.asarray(a_vals)) > 0)

    def test_D_stepping_no_shells(self, cosmology):
        a_vals = simulation_stepping(cosmology, 0.1, 1.0, 10, time_stepping="D")
        assert a_vals.shape == (11,)
        np.testing.assert_allclose(float(a_vals[0]), 0.1, atol=1e-10)
        np.testing.assert_allclose(float(a_vals[-1]), 1.0, atol=1e-6)
        assert np.all(np.diff(np.asarray(a_vals)) > 0)

    def test_log_a_stepping_no_shells(self, cosmology):
        a_vals = simulation_stepping(cosmology, 0.1, 1.0, 10, time_stepping="log_a")
        assert a_vals.shape == (11,)
        np.testing.assert_allclose(float(a_vals[0]), 0.1, atol=1e-10)
        np.testing.assert_allclose(float(a_vals[-1]), 1.0, atol=1e-10)
        assert np.all(np.diff(np.asarray(a_vals)) > 0)

    def test_invalid_stepping(self, cosmology):
        with pytest.raises(ValueError, match="Unknown time_stepping"):
            simulation_stepping(cosmology, 0.1, 1.0, 10, time_stepping="invalid")

    def test_with_explicit_ts(self, cosmology):
        """With explicit ts targets, steps clip to shell boundaries."""
        ts_shells = jnp.array([0.4, 0.7, 1.0])
        a_vals = simulation_stepping(cosmology, 0.1, 1.0, 30, ts=ts_shells, time_stepping="a")
        a_np = np.asarray(a_vals)
        # Must be monotonically increasing
        assert np.all(np.diff(a_np) > -1e-12)
        # First value is t0
        np.testing.assert_allclose(a_np[0], 0.1, atol=1e-10)
        # Last value should be last shell target
        np.testing.assert_allclose(a_np[-1], 1.0, atol=1e-10)
        # Shell targets must appear in the output
        for t in [0.4, 0.7, 1.0]:
            assert np.any(np.abs(a_np - t) < 1e-10), f"Shell target {t} not in output"

    def test_with_nb_shells(self, cosmology):
        """nb_shells + max_comoving_distance triggers resolve_geometry."""
        a_vals = simulation_stepping(
            cosmology,
            0.1,
            1.0,
            30,
            nb_shells=NB_SHELLS,
            max_comoving_distance=MAX_BOX,
            shell_spacing="comoving",
            time_stepping="a",
        )
        a_np = np.asarray(a_vals)
        assert np.all(np.diff(a_np) > -1e-12)
        np.testing.assert_allclose(a_np[0], 0.1, atol=1e-10)
        # Should contain the resolved shell scale factors
        shell_ts, _, _ = resolve_geometry(
            cosmology, MAX_BOX, nb_shells=NB_SHELLS, shell_spacing="comoving", box_size_z=MAX_BOX * 2
        )
        for t in np.asarray(shell_ts):
            assert np.any(np.abs(a_np - t) < 1e-10), f"Shell target {t} not in output"

    def test_nb_shells_without_max_comoving_raises(self, cosmology):
        with pytest.raises(ValueError, match="max_comoving_distance"):
            simulation_stepping(cosmology, 0.1, 1.0, 10, nb_shells=5)
