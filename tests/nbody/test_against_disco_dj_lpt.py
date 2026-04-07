"""Validation tests: compare jax_fli LPT against disco-dj."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import jax_fli as jfli
import pytest
from discodj.cosmology.cosmology import Cosmology as DiscoDJCosmology
from jax_fli.fields import DensityField, FieldStatus
from jax_fli.fields.units import DensityUnit

from tests.helpers import compare_fields

_FIELD_RTOL = 1e-5
_FIELD_ATOL = 1e-9
_PK_RTOL = 1e-5
_PK_ATOL = 1e-9


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
    return ddj.compute_timetables({"steps": 2048, "a_min": 1e-7})


@pytest.fixture(scope="module")
def shared_config():
    """Shared configuration for LPT comparison."""
    return {"res": 32, "boxsize": 256.0, "seed": 42}


@pytest.fixture(scope="module")
def ddj_instance(ddj_cosmo, shared_config):
    """Create a DiscoDJ instance with matching ICs."""
    from discodj import DiscoDJ

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


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("a_ini", [0.01, 0.1])
def test_lpt(ddj_instance, jfli_initial_field, cosmo, shared_config, order, a_ini):
    """Compare LPT density fields and power spectra between jax_fli and disco-dj."""
    res = shared_config["res"]
    boxsize = shared_config["boxsize"]
    box_shape = (boxsize,) * 3

    # jax_fli LPT with exact Fourier kernels (gradient_order=0) to match disco-dj
    # For order=2, use dealiasing and exact growth factor to match disco-dj's conventions
    dx, _ = jfli.lpt(
        cosmo,
        jfli_initial_field,
        ts=a_ini,
        order=order,
        gradient_order=0,
        dealiased=(order == 2),
        exact_growth=(order == 2),
    )

    # DISCO-DJ LPT — psi is displacement in Mpc/h
    psi = ddj_instance.evaluate_lpt_psi_at_a(a_ini, n_order=order, exact_growth=True)

    # Compare displacements: convert psi (Mpc/h) → grid cells
    psi_grid = psi * res / boxsize

    # Paint and compare density fields
    jfli_density = dx.paint()
    ddj_delta = ddj_instance.get_delta_from_psi(psi, res=res)

    # Compare in density space (always > 0) so MSRE is well-defined
    jfli_field = jfli_density.array
    ddj_field = ddj_delta + 1.0

    # Power spectra
    pk_jfli = jfli_density.power().spectra
    _, pk_ddj = jfli.power(ddj_field, box_shape=box_shape)

    compare_fields(psi_grid, dx.array, f"DDJ LPT{order} a={a_ini} displacement", rtol=_FIELD_RTOL, atol=_FIELD_ATOL)
    compare_fields(jfli_field, ddj_field, f"DDJ LPT{order} a={a_ini} density", rtol=_FIELD_RTOL, atol=_FIELD_ATOL)
    compare_fields(pk_jfli, pk_ddj, f"DDJ LPT{order} a={a_ini} Pk", rtol=_PK_RTOL, atol=_PK_ATOL, mean_atol=1e-5)
