"""Validate fli-simulate lpt CLI arg hookup against the jax_fli Python API.

Three test functions:
  test_lpt_density_script_vs_api   — density output, sweeps lpt_order/dealiased/exact_growth
  test_lpt_particles_script_vs_api — particles output (positions + scale_factors), same sweep
  test_lpt_spherical_script_vs_api — spherical output (nside=16), sweeps scheme
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.scripts

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

import jax_fli as jfli
from tests.helpers import compare_fields
from tests.scripts.conftest import (
    _BASE_CLI,
    _BOX,
    _HALO_SIZE,
    _RES,
    _SEED,
    LPT_ORDER_IDS,
    LPT_ORDER_PARAMS,
    SCHEME_IDS,
    SCHEME_PARAMS,
    lpt_order_cli,
    run_sim,
    scheme_cli,
)

_NSIDE = 16
_FLATSKY_NPIX = (_RES, _RES)
_FIELD_SIZE = (10, 10)
_NB_SHELLS = 2  # minimum to trigger lightcone mode (ts_resolved.size > 1)

_COMMON_LPT_CLI = [
    "--interp",
    "none",
    "--gradient-order",
    "1",
    "--nb-shells",
    str(_NB_SHELLS),
    "--shell-spacing",
    "comoving",
]


# ---------------------------------------------------------------------------
# Shared API helper
# ---------------------------------------------------------------------------


def _api_lpt(cosmo, initial_field, lpt_order, dealiased, exact_growth, painting):
    """Call jfli.lpt and return the displacement field dx."""
    dx, _p = jfli.lpt(
        cosmo,
        initial_field,
        nb_shells=_NB_SHELLS,
        shell_spacing="comoving",
        order=lpt_order,
        painting=painting,
        gradient_order=1,
        laplace_fd=False,
        dealiased=dealiased,
        exact_growth=exact_growth,
    )
    return dx


def _make_initial_field(cosmo, *, nside=None, flatsky_npix=None, field_size=None):
    """Generate Gaussian ICs matching fli-simulate defaults."""
    mesh, box = (_RES,) * 3, (_BOX,) * 3
    key = jax.random.key(_SEED)
    return jfli.gaussian_initial_conditions(
        key,
        mesh,
        box,
        cosmo=cosmo,
        observer_position=(0.5, 0.5, 0.5),
        nside=nside,
        flatsky_npix=flatsky_npix,
        field_size=field_size,
        halo_size=_HALO_SIZE,
    )


# ---------------------------------------------------------------------------
# Test 1: density output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lpt_order,dealiased,exact_growth", LPT_ORDER_PARAMS, ids=LPT_ORDER_IDS)
def test_lpt_density_script_vs_api(tmp_path, cosmo, lpt_order, dealiased, exact_growth):
    """fli-simulate lpt --density must match direct jax_fli.lpt() call."""
    # TODO also test with laplace_fd=True and False and gradient_order=0 and 1
    out_file = str(tmp_path / "output.parquet")

    # --- subprocess ---
    cmd = (
        ["fli-simulate", "--sim-mode", "lpt", "--density"]
        + _BASE_CLI
        + _COMMON_LPT_CLI
        + [
            "--output",
            out_file,
        ]
        + lpt_order_cli(lpt_order, dealiased, exact_growth)
    )
    run_sim(cmd)

    # --- load ---
    catalog = jfli.io.Catalog.from_parquet(out_file)
    script_field = catalog.field[0].array.squeeze()

    # --- API ---
    painting = jfli.PaintingOptions(target="density")
    initial_field = _make_initial_field(cosmo)
    dx = _api_lpt(cosmo, initial_field, lpt_order, dealiased, exact_growth, painting)
    api_field = dx.array.squeeze()

    # --- compare ---
    label = f"lpt density order={lpt_order}/dealiased={dealiased}/exact_growth={exact_growth}"
    compare_fields(script_field, api_field, label=label, rtol=1e-5, atol=1e-10, mean_atol=1e-12)


# ---------------------------------------------------------------------------
# Test 2: particles output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lpt_order,dealiased,exact_growth", LPT_ORDER_PARAMS, ids=LPT_ORDER_IDS)
def test_lpt_particles_script_vs_api(tmp_path, cosmo, lpt_order, dealiased, exact_growth):
    """fli-simulate lpt (default particles) must match direct jax_fli.lpt() call.

    Compares particle positions (array) and scale_factors metadata.
    """
    # TODO also test with laplace_fd=True and False and gradient_order=0 and 1
    out_file = str(tmp_path / "output.parquet")

    # --- subprocess ---
    # No output-target flag → default is particles
    cmd = (
        ["fli-simulate", "--sim-mode", "lpt"]
        + _BASE_CLI
        + _COMMON_LPT_CLI
        + [
            "--output",
            out_file,
        ]
        + lpt_order_cli(lpt_order, dealiased, exact_growth)
    )
    run_sim(cmd)

    # --- load ---
    catalog = jfli.io.Catalog.from_parquet(out_file)
    script_field = catalog.field[0]

    # --- API ---
    painting = jfli.PaintingOptions(target="particles")
    initial_field = _make_initial_field(cosmo)
    dx = _api_lpt(cosmo, initial_field, lpt_order, dealiased, exact_growth, painting)

    # --- compare positions ---
    label = f"lpt particles order={lpt_order}/dealiased={dealiased}/exact_growth={exact_growth}"
    compare_fields(script_field.array, dx.array, label=label, rtol=1e-5, atol=1e-10, mean_atol=1e-12)

    # --- compare scale_factors metadata ---
    assert_allclose(
        jnp.atleast_1d(script_field.scale_factors),
        jnp.atleast_1d(dx.scale_factors),
        rtol=1e-6,
        atol=1e-10,
        err_msg=f"{label}: scale_factors mismatch",
    )


# ---------------------------------------------------------------------------
# Test 3: spherical output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme,kernel_width", SCHEME_PARAMS, ids=SCHEME_IDS)
def test_lpt_spherical_script_vs_api(tmp_path, cosmo, scheme, kernel_width):
    """fli-simulate lpt --nside must match direct jax_fli.lpt() call across all schemes."""
    out_file = str(tmp_path / "output.parquet")

    # --- subprocess ---
    cmd = (
        ["fli-simulate", "--sim-mode", "lpt", "--nside", str(_NSIDE)]
        + _BASE_CLI
        + _COMMON_LPT_CLI
        + [
            "--lpt-order",
            "1",
            "--output",
            out_file,
        ]
        + scheme_cli(scheme, kernel_width)
    )
    run_sim(cmd)

    # --- load ---
    catalog = jfli.io.Catalog.from_parquet(out_file)
    script_field = catalog.field[0].array.squeeze()

    # --- API ---
    painting = jfli.PaintingOptions(target="spherical", scheme=scheme, kernel_width_arcmin=kernel_width)
    initial_field = _make_initial_field(cosmo, nside=_NSIDE)
    dx = _api_lpt(cosmo, initial_field, lpt_order=1, dealiased=False, exact_growth=False, painting=painting)
    api_field = dx.array.squeeze()

    # --- compare ---
    label = f"lpt spherical nside={_NSIDE}/{scheme}/kw={kernel_width}"
    compare_fields(script_field, api_field, label=label, rtol=1e-5, atol=1e-10, mean_atol=1e-12)


def test_lpt_flat_script_vs_api(tmp_path, cosmo):
    """fli-simulate lpt --flatsky-npix must match direct jax_fli.lpt() call."""
    out_file = str(tmp_path / "output.parquet")

    # --- subprocess ---
    cmd = (
        [
            "fli-simulate",
            "--sim-mode",
            "lpt",
            "--flatsky-npix",
            str(_FLATSKY_NPIX[0]),
            str(_FLATSKY_NPIX[1]),
            "--field-size",
            str(_FIELD_SIZE[0]),
            str(_FIELD_SIZE[1]),
        ]
        + _BASE_CLI
        + _COMMON_LPT_CLI
        + [
            "--lpt-order",
            "1",
            "--output",
            out_file,
        ]
    )
    run_sim(cmd)

    # --- load ---
    catalog = jfli.io.Catalog.from_parquet(out_file)
    script_field = catalog.field[0]

    # --- API ---
    painting = jfli.PaintingOptions(target="flat")
    initial_field = _make_initial_field(cosmo, flatsky_npix=_FLATSKY_NPIX, field_size=_FIELD_SIZE)
    dx = _api_lpt(cosmo, initial_field, lpt_order=1, dealiased=False, exact_growth=False, painting=painting)

    # --- compare array ---
    label = "lpt flat"
    compare_fields(
        script_field.array.squeeze(), dx.array.squeeze(), label=label, rtol=1e-5, atol=1e-10, mean_atol=1e-12
    )

    # --- check all 4 metadata fields are set ---
    assert script_field.scale_factors is not None, f"{label}: scale_factors is None"
    assert script_field.z_sources is not None, f"{label}: z_sources is None"
    assert script_field.comoving_centers is not None, f"{label}: comoving_centers is None"
    assert script_field.density_width is not None, f"{label}: density_width is None"
    assert_allclose(
        jnp.atleast_1d(script_field.comoving_centers),
        jnp.atleast_1d(dx.comoving_centers),
        rtol=1e-6,
        atol=1e-10,
        err_msg=f"{label}: comoving_centers mismatch",
    )
    assert_allclose(
        jnp.atleast_1d(script_field.density_width),
        jnp.atleast_1d(dx.density_width),
        rtol=1e-6,
        atol=1e-10,
        err_msg=f"{label}: density_width mismatch",
    )
