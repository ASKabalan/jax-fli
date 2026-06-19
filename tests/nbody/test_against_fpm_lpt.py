"""Validation tests: compare jax_fli LPT against fastpm-python."""

from __future__ import annotations

import pytest

import jax_fli as jfli
from tests.helpers import compare_fields

_FIELD_RTOL = 1e-5
_FIELD_ATOL = 1e-9
_PL_RTOL = 1e-5
_PK_ATOL = 1e-9


@pytest.mark.parametrize("order", [1, 2])
def test_lpt(
    simulation_config,
    initial_conditions,
    lpt_scale_factor,
    fpm_lpt1_field,
    fpm_lpt2_field,
    cosmology,
    order,
):
    mesh_shape, box_shape = simulation_config

    # FastPM LPT implementation uses non-dealiased kernels and approximates growth factors, so we match those settings here for a fair comparison.
    dx, p = jfli.lpt(
        cosmology, initial_conditions, ts=lpt_scale_factor, order=order, dealiased=False, exact_growth=False
    )

    jfli_density = dx.paint()

    fpm_ref_field = fpm_lpt1_field if order == 1 else fpm_lpt2_field

    pk_jfli = jfli_density.power().spectra
    _, pk_fpm = jfli.power(fpm_ref_field, box_shape=box_shape)

    compare_fields(jfli_density.array, fpm_ref_field, f"fpm LPT{order} density", rtol=_FIELD_RTOL, atol=_FIELD_ATOL)
    compare_fields(pk_jfli, pk_fpm, f"fpm LPT{order} Pk", rtol=_PL_RTOL, atol=_PK_ATOL, mean_atol=1e-5)
