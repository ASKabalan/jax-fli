"""Validate fli-simulate nbody CLI arg hookup against the jax_fli Python API.

Two test functions:
  test_nbody_density_script_vs_api  — density output, sweeps solver × lpt_order
  test_nbody_spherical_script_vs_api — spherical output, sweeps solver × scheme
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.scripts

import jax
import jax.numpy as jnp
import jax_fli as jfli
from numpy.testing import assert_allclose

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

_A_INI = 0.01
_A_END = 1.0
_N_STEPS = 10
_NSIDE = 16
_FLATSKY_NPIX = (_RES, _RES)
_FIELD_SIZE = (10, 10)

# ---------------------------------------------------------------------------
# Solver parametrize (name, time_stepping)
# Note: DoubleKickDrift (kdk) only supports time_stepping='a'
# ---------------------------------------------------------------------------

_SOLVER_PARAMS = [
    ("bf", "D"),
    ("dkd", "a"),
    ("kdk", "a"),
]
_SOLVER_IDS = ["bf-D", "dkd-a", "kdk-a"]


def _solver_cli(solver_name: str, time_stepping: str) -> list[str]:
    return ["--solver", solver_name, "--time-stepping", time_stepping]


def _build_solver(solver_name: str, time_stepping: str, painting) -> jfli.AbstractNBodySolver:
    cls = {"bf": jfli.BullFrog, "dkd": jfli.DriftKickDrift, "kdk": jfli.DoubleKickDrift}[solver_name]
    return cls(
        interp_kernel=jfli.NoInterp(painting=painting),
        gradient_order=1,
        laplace_fd=False,
        time_stepping=time_stepping,
        t0=_A_INI,
        t1=_A_END,
        n_steps=_N_STEPS,
    )


@pytest.mark.parametrize("lpt_order,dealiased,exact_growth", LPT_ORDER_PARAMS, ids=LPT_ORDER_IDS)
@pytest.mark.parametrize(
    "solver_name,time_stepping,gradient_order,laplace_fd",
    [
        ("bf", "D", 1, False),
        ("bf", "a", 1, True),
        ("dkd", "log_a", 0, False),
        ("kdk", "a", 1, False),
    ],
    ids=["bf-D-no-fd-laplace", "bf-a-fd-laplace", "dkd-loga-exact-nolaplace", "kdk-a-no-fd-laplace"],
)
def test_nbody_particles_script_vs_api(
    tmp_path,
    cosmo,
    solver_name,
    time_stepping,
    gradient_order,
    laplace_fd,
    lpt_order,
    dealiased,
    exact_growth,
):
    """fli-simulate nbody (default particles) must match direct jax_fli API call."""
    out_file = str(tmp_path / "output.parquet")

    # --- subprocess ---
    cmd = (
        ["fli-simulate", "nbody", "--ts", str(_A_END)]
        + _BASE_CLI
        + [
            "--t0",
            str(_A_INI),
            "--t1",
            str(_A_END),
            "--nb-steps",
            str(_N_STEPS),
            "--interp",
            "none",
            "--scheme",
            "bilinear",
            "--gradient-order",
            str(gradient_order),
            "--time-stepping",
            time_stepping,
            "--solver",
            solver_name,
            "--output",
            out_file,
        ]
    )
    if laplace_fd:
        cmd.append("--laplace-fd")
    cmd += lpt_order_cli(lpt_order, dealiased, exact_growth)
    run_sim(cmd)

    # --- load ---
    catalog = jfli.io.Catalog.from_parquet(out_file)
    script_field = catalog.field[0]

    # --- API ---
    mesh, box = (_RES,) * 3, (_BOX,) * 3
    key = jax.random.key(_SEED)
    initial_field = jfli.gaussian_initial_conditions(
        key,
        mesh,
        box,
        cosmo=cosmo,
        observer_position=(0.5, 0.5, 0.5),
        halo_size=_HALO_SIZE,
    )
    painting = jfli.PaintingOptions(target="particles")
    cls = {"bf": jfli.BullFrog, "dkd": jfli.DriftKickDrift, "kdk": jfli.DoubleKickDrift}[solver_name]
    solver = cls(
        interp_kernel=jfli.NoInterp(painting=painting),
        gradient_order=gradient_order,
        laplace_fd=laplace_fd,
        time_stepping=time_stepping,
        t0=_A_INI,
        t1=_A_END,
        n_steps=_N_STEPS,
    )
    dx, p = jfli.lpt(
        cosmo,
        initial_field,
        ts=_A_INI,
        order=lpt_order,
        painting=jfli.PaintingOptions(target="particles"),
        gradient_order=gradient_order,
        laplace_fd=laplace_fd,
        dealiased=dealiased,
        exact_growth=exact_growth,
    )
    result = jfli.nbody(cosmo, dx, p, solver=solver, ts=jnp.array([_A_END]))

    label = (
        f"nbody particles {solver_name}/{time_stepping}"
        f"/grad={gradient_order}/laplace_fd={laplace_fd}"
        f"/lpt_order={lpt_order}/dealiased={dealiased}/exact_growth={exact_growth}"
    )

    # --- compare positions ---
    compare_fields(script_field.array, result.array, label=label, rtol=1e-5, atol=1e-10, mean_atol=1e-12)

    # --- compare metadata ---
    assert_allclose(
        jnp.atleast_1d(script_field.scale_factors),
        jnp.atleast_1d(result.scale_factors),
        rtol=1e-6,
        atol=1e-10,
        err_msg=f"{label}: scale_factors mismatch",
    )
    assert_allclose(
        jnp.atleast_1d(script_field.comoving_centers),
        jnp.atleast_1d(result.comoving_centers),
        rtol=1e-6,
        atol=1e-10,
        err_msg=f"{label}: comoving_centers mismatch",
    )


# ---------------------------------------------------------------------------
# Test 1: density output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lpt_order,dealiased,exact_growth", LPT_ORDER_PARAMS, ids=LPT_ORDER_IDS)
@pytest.mark.parametrize(
    "solver_name,time_stepping,gradient_order,laplace_fd",
    [
        ("bf", "D", 1, False),
        ("bf", "a", 1, True),
        ("dkd", "log_a", 0, False),
        ("kdk", "a", 1, False),
    ],
    ids=["bf-D-no-fd-laplace", "bf-a-fd-laplace", "dkd-loga-exact-nolaplace", "kdk-a-no-fd-laplace"],
)
def test_nbody_density_script_vs_api(
    tmp_path,
    cosmo,
    solver_name,
    time_stepping,
    gradient_order,
    laplace_fd,
    lpt_order,
    dealiased,
    exact_growth,
):
    """fli-simulate nbody --density must match direct jax_fli API call."""
    out_file = str(tmp_path / "output.parquet")

    # --- subprocess ---
    cmd = (
        ["fli-simulate", "nbody", "--density", "--ts", str(_A_END)]
        + _BASE_CLI
        + [
            "--t0",
            str(_A_INI),
            "--t1",
            str(_A_END),
            "--nb-steps",
            str(_N_STEPS),
            "--interp",
            "none",
            "--scheme",
            "bilinear",
            "--gradient-order",
            str(gradient_order),
            "--time-stepping",
            time_stepping,
            "--solver",
            solver_name,
            "--output",
            out_file,
        ]
    )
    if laplace_fd:
        cmd.append("--laplace-fd")
    cmd += lpt_order_cli(lpt_order, dealiased, exact_growth)
    run_sim(cmd)

    # --- load ---
    catalog = jfli.io.Catalog.from_parquet(out_file)
    script_field = catalog.field[0].array.squeeze()

    # --- API ---
    mesh, box = (_RES,) * 3, (_BOX,) * 3
    key = jax.random.key(_SEED)
    initial_field = jfli.gaussian_initial_conditions(
        key,
        mesh,
        box,
        cosmo=cosmo,
        observer_position=(0.5, 0.5, 0.5),
        halo_size=_HALO_SIZE,
    )
    painting = jfli.PaintingOptions(target="density")
    cls = {"bf": jfli.BullFrog, "dkd": jfli.DriftKickDrift, "kdk": jfli.DoubleKickDrift}[solver_name]
    solver = cls(
        interp_kernel=jfli.NoInterp(painting=painting),
        gradient_order=gradient_order,
        laplace_fd=laplace_fd,
        time_stepping=time_stepping,
        t0=_A_INI,
        t1=_A_END,
        n_steps=_N_STEPS,
    )
    dx, p = jfli.lpt(
        cosmo,
        initial_field,
        ts=_A_INI,
        order=lpt_order,
        painting=jfli.PaintingOptions(target="particles"),
        gradient_order=gradient_order,
        laplace_fd=laplace_fd,
        dealiased=dealiased,
        exact_growth=exact_growth,
    )
    result = jfli.nbody(cosmo, dx, p, solver=solver, ts=jnp.array([_A_END]))
    api_field = result.array.squeeze()

    # --- compare ---
    label = (
        f"nbody density {solver_name}/{time_stepping}"
        f"/grad={gradient_order}/laplace_fd={laplace_fd}"
        f"/lpt_order={lpt_order}/dealiased={dealiased}/exact_growth={exact_growth}"
    )
    compare_fields(script_field, api_field, label=label, rtol=1e-5, atol=1e-10, mean_atol=1e-12)


# ---------------------------------------------------------------------------
# Test 2: spherical output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme,kernel_width", SCHEME_PARAMS, ids=SCHEME_IDS)
@pytest.mark.parametrize("solver_name,time_stepping", _SOLVER_PARAMS, ids=_SOLVER_IDS)
def test_nbody_spherical_script_vs_api(
    tmp_path,
    cosmo,
    solver_name,
    time_stepping,
    scheme,
    kernel_width,
):
    # TODO also test with laplace_fd=True and False and gradient_order=0 and 1
    """fli-simulate nbody --nside must match direct jax_fli API call across all solvers and schemes."""
    out_file = str(tmp_path / "output.parquet")

    # --- subprocess ---
    cmd = (
        ["fli-simulate", "nbody", "--nside", str(_NSIDE), "--ts", str(_A_END)]
        + _BASE_CLI
        + [
            "--t0",
            str(_A_INI),
            "--t1",
            str(_A_END),
            "--nb-steps",
            str(_N_STEPS),
            "--interp",
            "none",
            "--gradient-order",
            "1",
            "--time-stepping",
            time_stepping,
            "--solver",
            solver_name,
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
    cls = {"bf": jfli.BullFrog, "dkd": jfli.DriftKickDrift, "kdk": jfli.DoubleKickDrift}[solver_name]
    solver = cls(
        interp_kernel=jfli.NoInterp(painting=painting),
        gradient_order=1,
        laplace_fd=False,
        time_stepping=time_stepping,
        t0=_A_INI,
        t1=_A_END,
        n_steps=_N_STEPS,
    )
    dx, p = jfli.lpt(
        cosmo,
        initial_field,
        ts=_A_INI,
        order=1,
        painting=jfli.PaintingOptions(target="particles"),
        gradient_order=1,
        laplace_fd=False,
    )
    result = jfli.nbody(cosmo, dx, p, solver=solver, ts=jnp.array([_A_END]))
    api_field = result.array.squeeze()

    # --- compare ---
    label = f"nbody spherical {solver_name}/{time_stepping}/{scheme}/kw={kernel_width}"
    compare_fields(script_field, api_field, label=label, rtol=1e-5, atol=1e-10, mean_atol=1e-12)


@pytest.mark.parametrize("solver_name,time_stepping", _SOLVER_PARAMS, ids=_SOLVER_IDS)
def test_nbody_flat_script_vs_api(
    tmp_path,
    cosmo,
    solver_name,
    time_stepping,
):
    """fli-simulate nbody --flatsky-npix must match direct jax_fli API call across all solvers."""
    out_file = str(tmp_path / "output.parquet")

    # --- subprocess ---
    cmd = (
        [
            "fli-simulate",
            "nbody",
            "--flatsky-npix",
            str(_FLATSKY_NPIX[0]),
            str(_FLATSKY_NPIX[1]),
            "--field-size",
            str(_FIELD_SIZE[0]),
            str(_FIELD_SIZE[1]),
            "--ts",
            str(_A_END),
        ]
        + _BASE_CLI
        + [
            "--t0",
            str(_A_INI),
            "--t1",
            str(_A_END),
            "--nb-steps",
            str(_N_STEPS),
            "--interp",
            "none",
            "--gradient-order",
            "1",
            "--time-stepping",
            time_stepping,
            "--solver",
            solver_name,
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
    mesh, box = (_RES,) * 3, (_BOX,) * 3
    key = jax.random.key(_SEED)
    initial_field = jfli.gaussian_initial_conditions(
        key,
        mesh,
        box,
        cosmo=cosmo,
        observer_position=(0.5, 0.5, 0.5),
        flatsky_npix=_FLATSKY_NPIX,
        field_size=_FIELD_SIZE,
        halo_size=_HALO_SIZE,
    )
    painting = jfli.PaintingOptions(target="flat")
    cls = {"bf": jfli.BullFrog, "dkd": jfli.DriftKickDrift, "kdk": jfli.DoubleKickDrift}[solver_name]
    solver = cls(
        interp_kernel=jfli.NoInterp(painting=painting),
        gradient_order=1,
        laplace_fd=False,
        time_stepping=time_stepping,
        t0=_A_INI,
        t1=_A_END,
        n_steps=_N_STEPS,
    )
    dx, p = jfli.lpt(
        cosmo,
        initial_field,
        ts=_A_INI,
        order=1,
        painting=jfli.PaintingOptions(target="particles"),
        gradient_order=1,
        laplace_fd=False,
    )
    result = jfli.nbody(cosmo, dx, p, solver=solver, ts=jnp.array([_A_END]))

    label = f"nbody flat {solver_name}/{time_stepping}"

    # --- compare array ---
    compare_fields(
        script_field.array.squeeze(), result.array.squeeze(), label=label, rtol=1e-5, atol=1e-10, mean_atol=1e-12
    )

    # --- check all 4 metadata fields are set ---
    assert script_field.scale_factors is not None, f"{label}: scale_factors is None"
    assert script_field.z_sources is not None, f"{label}: z_sources is None"
    assert script_field.comoving_centers is not None, f"{label}: comoving_centers is None"
    assert script_field.density_width is not None, f"{label}: density_width is None"
    assert_allclose(
        jnp.atleast_1d(script_field.comoving_centers),
        jnp.atleast_1d(result.comoving_centers),
        rtol=1e-6,
        atol=1e-10,
        err_msg=f"{label}: comoving_centers mismatch",
    )
    assert_allclose(
        jnp.atleast_1d(script_field.density_width),
        jnp.atleast_1d(result.density_width),
        rtol=1e-6,
        atol=1e-10,
        err_msg=f"{label}: density_width mismatch",
    )
