"""Forward comparison tests: single-device vs multi-device jax_fli runs.

Tests LPT and N-body pipelines across multiple painting targets and solver types,
verifying bitwise equivalence between single-device and multi-device execution.
"""

from __future__ import annotations

import jax_fli as jfli
import numpy as np
import pytest
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from tests.helpers import compare_fields

from .conftest import _PDIMS, N_STEPS, NB_SHELLS, PAINT_NSIDE, T0, T1, Z_SOURCE_BORN, make_sharded_ics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gather(field):
    """Gather a distributed field array to a single-device numpy array."""
    return process_allgather(field.array, tiled=True)


def _compare_fields(single, multi, label):
    """Gather distributed array then compare with shared compare_fields helper."""
    single_arr = np.asarray(single.array)
    multi_arr = _gather(multi)
    compare_fields(single_arr, multi_arr, label)


def _make_solver(solver_name, target):
    """Create a solver with the given name and painting target."""
    painting = jfli.PaintingOptions(target=target, paint_nside=PAINT_NSIDE)
    interp = jfli.NoInterp(painting=painting)
    kwargs = dict(
        interp_kernel=interp,
        t0=T0,
        t1=T1,
        n_steps=N_STEPS,
        shell_spacing="growth",
        min_width=1.0,
    )
    if solver_name == "KKD":
        return jfli.DoubleKickDrift(**kwargs)
    elif solver_name == "DKD":
        return jfli.DriftKickDrift(time_stepping="a", **kwargs)
    elif solver_name == "BullFrog":
        return jfli.BullFrog(time_stepping="a", **kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver_name}")


def _check_sharding(array, input_sharding, field_type):
    if field_type == "spherical":
        expected_sharding = NamedSharding(input_sharding.mesh, P(None, input_sharding.spec[0]))
    else:
        expected_sharding = NamedSharding(input_sharding.mesh, P(None, *input_sharding.spec))

    print(f"Checking sharding for {field_type}: expected {expected_sharding}, got {array.sharding}")

    assert array.sharding.is_equivalent_to(
        expected_sharding, ndim=array.ndim
    ), f"Expected sharding {expected_sharding}, got {array.sharding}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.distributed
@pytest.mark.parametrize("target", ["density", "flat", "spherical", "particles"])
@pytest.mark.parametrize("pdims", _PDIMS)
def test_distributed_lpt_lightcone(single_device_ics, cosmo, target, pdims):
    """LPT lightcone produces identical results single-device vs multi-device."""
    painting = jfli.PaintingOptions(target=target, paint_nside=PAINT_NSIDE)

    # Single-device run
    single_result, _ = jfli.lpt(
        cosmo,
        single_device_ics,
        nb_shells=4,
        order=1,
        painting=painting,
        shell_spacing="growth",
        min_width=1.0,
    )

    # Multi-device run
    sharded_ics, mesh, sharding = make_sharded_ics(single_device_ics, pdims)
    multi_result, _ = jfli.lpt(
        cosmo,
        sharded_ics,
        nb_shells=4,
        order=1,
        painting=painting,
        shell_spacing="growth",
        min_width=1.0,
    )

    if target == "particles" or target == "density":
        multi_result = multi_result.apply_fn(lambda x: x[None, ...])  # Add leading axis for comparison
        single_result = single_result.apply_fn(lambda x: x[None, ...])  # Add leading axis for comparison
    _check_sharding(multi_result.array, sharding, painting.target)

    _compare_fields(single_result, multi_result, f"LPT target={target} pdims={pdims}")


@pytest.mark.distributed
@pytest.mark.parametrize("target", ["density", "flat", "spherical", "particles"])
@pytest.mark.parametrize("solver_name", ["KKD", "DKD", "BullFrog"])
@pytest.mark.parametrize("pdims", _PDIMS)
def test_distributed_nbody(single_device_ics, cosmo, target, solver_name, pdims):
    """N-body produces identical results single-device vs multi-device."""
    solver = _make_solver(solver_name, target)

    # Single-device run
    dx_single, p_single = jfli.lpt(cosmo, single_device_ics, ts=T0, order=1)
    result_single = jfli.nbody(cosmo, dx_single, p_single, solver=solver, nb_shells=4)

    # Multi-device run
    sharded_ics, mesh, sharding = make_sharded_ics(single_device_ics, pdims)
    dx_multi, p_multi = jfli.lpt(cosmo, sharded_ics, ts=T0, order=1)
    result_multi = jfli.nbody(cosmo, dx_multi, p_multi, solver=solver, nb_shells=4)

    _check_sharding(result_multi.array, sharding, target)
    _compare_fields(
        result_single,
        result_multi,
        f"N-body solver={solver_name} target={target} pdims={pdims}",
    )


@pytest.mark.distributed
@pytest.mark.parametrize("pdims", _PDIMS)
@pytest.mark.parametrize("target", ["flat", "spherical"])
def test_distributed_born_lensing(single_device_ics, cosmo, pdims, target):
    """Born lensing produces identical results single-device vs multi-device."""
    solver = _make_solver("KKD", target)

    # Single-device run
    dx_single, p_single = jfli.lpt(cosmo, single_device_ics, ts=T0, order=1)
    lc_single = jfli.nbody(cosmo, dx_single, p_single, solver=solver, nb_shells=NB_SHELLS)
    kappa_single = jfli.born(cosmo, lc_single, nz_shear=Z_SOURCE_BORN)

    # Multi-device run
    sharded_ics, mesh, sharding = make_sharded_ics(single_device_ics, pdims)
    dx_multi, p_multi = jfli.lpt(cosmo, sharded_ics, ts=T0, order=1)
    lc_multi = jfli.nbody(cosmo, dx_multi, p_multi, solver=solver, nb_shells=NB_SHELLS)
    kappa_multi = jfli.born(cosmo, lc_multi, nz_shear=Z_SOURCE_BORN)

    _compare_fields(
        kappa_single,
        kappa_multi,
        f"Born lensing pdims={pdims}",
    )
