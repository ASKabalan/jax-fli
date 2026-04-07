"""Validation tests: compare jax_fli N-body solvers against disco-dj.

Test matrix:
- test_dkd_vs_discodj:      DriftKickDrift  vs disco-dj "fastpm"     (64 cases)
- test_bullfrog_vs_discodj:  BullFrog        vs disco-dj "bullfrog"   (32 cases)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import pytest

jax.config.update("jax_enable_x64", True)

import jax_fli as jfli
from discodj import DiscoDJ
from discodj.cosmology.cosmology import Cosmology as DiscoDJCosmology
from jax_fli.fields import DensityField, FieldStatus
from jax_fli.fields.units import DensityUnit

from tests.helpers import compare_fields

_FIELD_RTOL = 1e-5
_FIELD_ATOL = 2e-6
_PK_RTOL = 1e-5
_PK_ATOL = 1e-7


@pytest.fixture(scope="module")
def cosmo():
    return jc.Planck18()


@pytest.fixture(scope="module")
def ddj_cosmo(cosmo):
    """Create a disco-dj Cosmology matching the jax_cosmo Planck18."""
    ddj = DiscoDJCosmology(
        Omega_c=float(cosmo.Omega_m - cosmo.Omega_b),
        Omega_b=float(cosmo.Omega_b),
        h=float(cosmo.h),
        sigma8=float(cosmo.sigma8),
        n_s=float(cosmo.n_s),
        dtype_num=64,
    )
    return ddj.compute_timetables({"steps": 32768, "a_min": 1e-7})


@pytest.fixture(scope="module")
def shared_config():
    return {"res": 32, "boxsize": 256.0, "seed": 42, "a_end": 1.0}


@pytest.fixture(scope="module")
def ddj_instance(ddj_cosmo, shared_config):
    """Create a fully initialized DiscoDJ instance."""
    cosmo_dict = {
        "Omega_c": float(ddj_cosmo.Omega_c),
        "Omega_b": float(ddj_cosmo.Omega_b),
        "h": float(ddj_cosmo.h),
        "sigma8": float(ddj_cosmo.sigma8),
        "n_s": float(ddj_cosmo.n_s),
    }

    res = shared_config["res"]
    boxsize = shared_config["boxsize"]

    dj = DiscoDJ(dim=3, res=res, boxsize=boxsize, cosmo=cosmo_dict, precision="double")
    dj = dj.with_timetables()
    dj = dj.with_linear_ps(transfer_function="Eisenstein-Hu")

    key = jax.random.PRNGKey(shared_config["seed"])
    white_noise = jax.random.normal(key, shape=(res, res, res), dtype=jnp.float64)
    dj = dj.with_ics(white_noise_field=white_noise)
    dj = dj.with_lpt(n_order=2, exact_growth=True)

    return dj


@pytest.fixture(scope="module")
def jfli_initial_field(ddj_instance, shared_config):
    """Create jax_fli initial conditions from DISCO-DJ's linear field."""
    res = shared_config["res"]
    boxsize = shared_config["boxsize"]

    delta_r = ddj_instance.delta_ini

    return DensityField(
        array=delta_r,
        mesh_size=(res, res, res),
        box_size=(boxsize, boxsize, boxsize),
        status=FieldStatus.INITIAL_FIELD,
        unit=DensityUnit.DENSITY,
    )


# ---------------------------------------------------------------------------
# Test 1: DriftKickDrift (DKD) vs disco-dj "fastpm" — 64 cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("time_stepping", ["a", "D", "log_a"])
@pytest.mark.parametrize("a_ini", [0.01, 0.05])
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("grad_order", [0, 4])
@pytest.mark.parametrize("n_steps", [10, 20])
def test_dkd_vs_discodj(
    ddj_instance, jfli_initial_field, cosmo, shared_config, time_stepping, a_ini, order, grad_order, n_steps
):
    """Compare DriftKickDrift (FastPM DKD pi-integrator) against DISCO-DJ's fastpm stepper."""
    res = shared_config["res"]
    boxsize = shared_config["boxsize"]
    a_end = shared_config["a_end"]
    box_shape = (boxsize,) * 3

    # jax_fli: LPT with gradient_order=0 (exact ik) to match disco-dj defaults
    dx, p = jfli.lpt(
        cosmo,
        jfli_initial_field,
        ts=a_ini,
        order=order,
        gradient_order=0,
        dealiased=(order == 2),
        exact_growth=(order == 2),
    )

    solver = jfli.DriftKickDrift(
        interp_kernel=jfli.NoInterp(painting=jfli.PaintingOptions(target="density")),
        gradient_order=grad_order,
        time_stepping=time_stepping,
        t0=a_ini,
        t1=a_end,
        n_steps=n_steps,
    )

    result = jfli.nbody(
        cosmo,
        dx,
        p,
        solver=solver,
    )

    # DISCO-DJ N-body with fastpm stepper
    X, P, a = ddj_instance.run_nbody(
        a_ini=a_ini,
        a_end=a_end,
        n_steps=n_steps,
        stepper="fastpm",
        method="pm",
        res_pm=res,
        time_var=time_stepping,
        ic_method="lpt",
        nlpt_order_ics=order,
        grad_kernel_order=grad_order,
        exact_growth=(order == 2),
    )
    ddj_delta = ddj_instance.get_delta_from_pos(X, res=res)
    ddj_field = ddj_delta + 1.0

    jfli_field = result.array.squeeze()

    _, pk_jfli = jfli.power(jfli_field, box_shape=box_shape)
    _, pk_ddj = jfli.power(ddj_field, box_shape=box_shape)

    compare_fields(
        jfli_field,
        ddj_field,
        f"DKD order={order} grad={grad_order} a_ini={a_ini} n_steps={n_steps} time={time_stepping}",
        rtol=_FIELD_RTOL,
        atol=_FIELD_ATOL,
        mean_atol=1e-9,
    )
    compare_fields(
        pk_jfli,
        pk_ddj,
        f"DKD order={order} grad={grad_order} a_ini={a_ini} n_steps={n_steps} time={time_stepping} Pk",
        rtol=_PK_RTOL,
        atol=_PK_ATOL,
        mean_atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Test 2: BullFrog vs disco-dj "bullfrog" — 32 cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("time_stepping", ["a", "D", "log_a"])
@pytest.mark.parametrize("a_ini", [0.01, 0.05])
@pytest.mark.parametrize("grad_order", [0, 4])
@pytest.mark.parametrize("n_steps", [10, 20])
def test_bullfrog_vs_discodj(
    ddj_instance, jfli_initial_field, cosmo, shared_config, time_stepping, a_ini, grad_order, n_steps
):
    """Compare BullFrog against DISCO-DJ's bullfrog stepper (LPT1 only)."""
    res = shared_config["res"]
    boxsize = shared_config["boxsize"]
    a_end = shared_config["a_end"]
    box_shape = (boxsize,) * 3

    # jax_fli: BullFrog with LPT1 only, gradient_order=0 (exact ik)
    dx, p = jfli.lpt(cosmo, jfli_initial_field, ts=a_ini, order=1, gradient_order=0)

    solver = jfli.BullFrog(
        interp_kernel=jfli.NoInterp(painting=jfli.PaintingOptions(target="density")),
        gradient_order=grad_order,
        time_stepping=time_stepping,
        t0=a_ini,
        t1=a_end,
        n_steps=n_steps,
    )

    result = jfli.nbody(
        cosmo,
        dx,
        p,
        solver=solver,
    )

    # DISCO-DJ N-body with bullfrog stepper
    X, P, a = ddj_instance.run_nbody(
        a_ini=a_ini,
        a_end=a_end,
        n_steps=n_steps,
        stepper="bullfrog",
        method="pm",
        res_pm=res,
        time_var=time_stepping,
        ic_method="lpt",
        nlpt_order_ics=1,
        grad_kernel_order=grad_order,
    )
    ddj_delta = ddj_instance.get_delta_from_pos(X, res=res)
    ddj_field = ddj_delta + 1.0

    jfli_field = result.array.squeeze()

    _, pk_jfli = jfli.power(jfli_field, box_shape=box_shape)
    _, pk_ddj = jfli.power(ddj_field, box_shape=box_shape)

    compare_fields(
        jfli_field,
        ddj_field,
        f"BullFrog grad={grad_order} a_ini={a_ini} n_steps={n_steps} time={time_stepping}",
        rtol=_FIELD_RTOL,
        atol=_FIELD_ATOL,
        mean_atol=1e-9,
    )
    compare_fields(
        pk_jfli,
        pk_ddj,
        f"BullFrog grad={grad_order} a_ini={a_ini} n_steps={n_steps} time={time_stepping} Pk",
        rtol=_PK_RTOL,
        atol=_PK_ATOL,
        mean_atol=1e-5,
    )
