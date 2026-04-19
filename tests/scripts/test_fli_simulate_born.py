"""Validate fli-simulate lensing (born) CLI arg hookup against the jax_fli Python API.

Two test functions:
  test_born_spherical_script_vs_api — kappa on HEALPix sphere, sweeps scheme
  test_born_flatsky_script_vs_api   — kappa on flat-sky grid (npix = mesh res)
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.scripts

import jax
import jax.numpy as jnp
import jax_fli as jfli

from tests.helpers import compare_fields
from tests.scripts.conftest import (
    _BASE_CLI,
    _BOX,
    _HALO_SIZE,
    _RES,
    _SEED,
    SCHEME_IDS,
    SCHEME_PARAMS,
    run_sim,
    scheme_cli,
)

_T0 = 0.1
_T1 = 1.0
_N_STEPS = 10
_N_SHELLS = 2
_NSIDE = 16
_NZ_SHEAR = [0.5, 1.0]  # explicit source redshifts
_MIN_Z = 0.1
_MAX_Z = 1.5
_N_INTEGRATE = 32

# Flat-sky: same pixel count as mesh resolution, 10×10 degree patch
_FLATSKY_NPIX = _RES
_FIELD_SIZE = 10  # degrees

_COMMON_BORN_CLI = [
    "--solver",
    "bf",
    "--time-stepping",
    "D",
    "--gradient-order",
    "1",
    "--lpt-order",
    "1",
    "--t0",
    str(_T0),
    "--t1",
    str(_T1),
    "--nb-steps",
    str(_N_STEPS),
    "--interp",
    "none",
    "--nb-shells",
    str(_N_SHELLS),
    "--shell-spacing",
    "comoving",
    "--nz-shear",
    *[str(z) for z in _NZ_SHEAR],
    "--min-z",
    str(_MIN_Z),
    "--max-z",
    str(_MAX_Z),
    "--n-integrate",
    str(_N_INTEGRATE),
]


# ---------------------------------------------------------------------------
# Shared API helper
# ---------------------------------------------------------------------------


def _api_born(cosmo, initial_field, *, painting):
    """Run LPT → nbody → born via the jax_fli API, matching run_simulations()."""
    solver = jfli.BullFrog(
        interp_kernel=jfli.NoInterp(painting=painting),
        gradient_order=1,
        laplace_fd=False,
        time_stepping="D",
        t0=_T0,
        t1=_T1,
        n_steps=_N_STEPS,
    )
    dx, p = jfli.lpt(
        cosmo,
        initial_field,
        ts=_T0,
        order=1,
        painting=jfli.PaintingOptions(target="particles"),
        gradient_order=1,
        laplace_fd=False,
    )
    lightcone = jfli.nbody(
        cosmo,
        dx,
        p,
        solver=solver,
        nb_shells=_N_SHELLS,
        shell_spacing="comoving",
        min_width=50.0,
    )
    nz_shear = jnp.array(_NZ_SHEAR, dtype=jnp.float32)
    return jfli.born(cosmo, lightcone, nz_shear, min_z=_MIN_Z, max_z=_MAX_Z, n_integrate=_N_INTEGRATE)


# ---------------------------------------------------------------------------
# Test 1: spherical kappa output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme,kernel_width", SCHEME_PARAMS, ids=SCHEME_IDS)
def test_born_spherical_script_vs_api(tmp_path, cosmo, scheme, kernel_width):
    """fli-simulate lensing --nside must match direct jax_fli born pipeline."""
    out_file = str(tmp_path / "output.parquet")

    # --- subprocess ---
    cmd = (
        ["fli-simulate", "--sim-mode", "lensing", "--nside", str(_NSIDE)]
        + _BASE_CLI
        + _COMMON_BORN_CLI
        + [
            "--output",
            out_file,
        ]
        + scheme_cli(scheme, kernel_width)
    )
    run_sim(cmd)

    # --- load ---
    catalog = jfli.io.Catalog.from_parquet(out_file)
    script_kappa = catalog.field[0].array

    # --- API ---
    mesh, box = (_RES,) * 3, (_BOX,) * 3
    key = jax.random.key(_SEED)
    initial_field = jfli.gaussian_initial_conditions(
        key,
        mesh,
        box,
        cosmo=cosmo,
        observer_position=(0.5, 0.5, 0.5),
        nside=_NSIDE,
        halo_size=_HALO_SIZE,
    )
    painting = jfli.PaintingOptions(target="spherical", scheme=scheme, kernel_width_arcmin=kernel_width)
    kappa = _api_born(cosmo, initial_field, painting=painting)
    api_kappa = kappa.array

    # --- compare ---
    label = f"born spherical nside={_NSIDE}/{scheme}/kw={kernel_width}"
    compare_fields(script_kappa, api_kappa, label=label, rtol=1e-5, atol=1e-10, mean_atol=1e-12)


# ---------------------------------------------------------------------------
# Test 2: flat-sky kappa output
# ---------------------------------------------------------------------------


def test_born_flatsky_script_vs_api(tmp_path, cosmo):
    """fli-simulate lensing --flatsky-npix must match direct jax_fli born pipeline."""
    out_file = str(tmp_path / "output.parquet")
    npix = str(_FLATSKY_NPIX)
    fsize = str(_FIELD_SIZE)

    # --- subprocess ---
    cmd = (
        ["fli-simulate", "--sim-mode", "lensing", "--flatsky-npix", npix, npix, "--field-size", fsize, fsize]
        + _BASE_CLI
        + _COMMON_BORN_CLI
        + ["--output", out_file]
    )
    run_sim(cmd)

    # --- load ---
    catalog = jfli.io.Catalog.from_parquet(out_file)
    script_kappa = catalog.field[0].array

    # --- API ---
    mesh, box = (_RES,) * 3, (_BOX,) * 3
    key = jax.random.key(_SEED)
    initial_field = jfli.gaussian_initial_conditions(
        key,
        mesh,
        box,
        cosmo=cosmo,
        observer_position=(0.5, 0.5, 0.5),
        flatsky_npix=(_FLATSKY_NPIX, _FLATSKY_NPIX),
        field_size=(_FIELD_SIZE, _FIELD_SIZE),
        halo_size=_HALO_SIZE,
    )
    painting = jfli.PaintingOptions(target="flat")
    kappa = _api_born(cosmo, initial_field, painting=painting)
    api_kappa = kappa.array

    # --- compare ---
    compare_fields(
        script_kappa,
        api_kappa,
        label=f"born flatsky {_FLATSKY_NPIX}x{_FLATSKY_NPIX}/{_FIELD_SIZE}deg",
        rtol=1e-5,
        atol=1e-10,
        mean_atol=1e-12,
    )
