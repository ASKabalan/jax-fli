"""Validation tests: compare jax_fli N-body against fastpm-python."""

from __future__ import annotations

import jax_fli as jfli
import pytest

from tests.helpers import compare_fields

_FIELD_RTOL = 1e-5
_FIELD_ATOL = 2e-6
_PK_RTOL = 1e-5
_PK_ATOL = 1e-7


@pytest.mark.parametrize("order", [1, 2])
def test_nbody(
    simulation_config,
    initial_conditions,
    lpt_scale_factor,
    nbody_from_lpt1,
    nbody_from_lpt2,
    cosmology,
    order,
    steps,
):
    mesh_shape, box_shape = simulation_config

    dx, p = jfli.lpt(cosmology, initial_conditions, ts=lpt_scale_factor, order=order)

    solver = jfli.DoubleKickDrift(
        interp_kernel=jfli.NoInterp(painting=jfli.PaintingOptions(target="density")),
        t0=lpt_scale_factor,
        t1=1.0,
        n_steps=steps - 1,
    )

    result = jfli.nbody(
        cosmology,
        dx,
        p,
        solver=solver,
    )

    fpm_ref_field = nbody_from_lpt1 if order == 1 else nbody_from_lpt2

    _, pk_jfli = jfli.power(result.array.squeeze(), box_shape=box_shape)
    _, pk_fpm = jfli.power(fpm_ref_field, box_shape=box_shape)

    compare_fields(result.array, fpm_ref_field, f"fpm N-body LPT{order} density", rtol=_FIELD_RTOL, atol=_FIELD_ATOL)
    compare_fields(pk_jfli, pk_fpm, f"fpm N-body LPT{order} Pk", rtol=_PK_RTOL, atol=_PK_ATOL, mean_atol=1e-5)
