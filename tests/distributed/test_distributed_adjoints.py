"""Gradient tests through distributed pipeline.

Verifies that gradients through LPT and N-body pipelines are correctly sharded,
finite, and non-zero when running on multiple devices. Also compares gradient
values between single-device and multi-device execution.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_fli as jfli
import numpy as np
import pytest
from jax.experimental.multihost_utils import process_allgather

from tests.helpers import compare_fields

from .conftest import _PDIMS, N_STEPS, NB_SHELLS, PAINT_NSIDE, T0, T1, Z_SOURCE_BORN, make_sharded_ics

# Spherical: MSE ~1e-9 but element-wise diffs up to ~1e-3 on values of ~1e-6
# (halo exchanges in backward pass cause large relative errors on small gradients)
_GRAD_SPHERICAL_RTOL = 1.0
_GRAD_SPHERICAL_ATOL = 5e-3
_GRAD_SPHERICAL_MEAN_RTOL = 1e-3
_GRAD_SPHERICAL_MEAN_ATOL = 1e-7

# Flat: element-wise diffs up to ~3e-6 on values of ~1e-7
_GRAD_FLAT_RTOL = 1e-1
_GRAD_FLAT_ATOL = 1e-4
_GRAD_FLAT_MEAN_RTOL = 1e-3
_GRAD_FLAT_MEAN_ATOL = 1e-6

# Born (through spherical lightcone): element-wise diffs up to ~8e-8
_GRAD_BORN_RTOL = 1e-1
_GRAD_BORN_ATOL = 1e-6
_GRAD_BORN_MEAN_RTOL = 1e-3
_GRAD_BORN_MEAN_ATOL = 1e-8

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Gradient sharding tests
# ---------------------------------------------------------------------------


@pytest.mark.distributed
@pytest.mark.parametrize("target", ["density", "spherical", "flat"])
@pytest.mark.parametrize("pdims", _PDIMS)
def test_distributed_lpt_gradient_sharding(single_device_ics, cosmo, target, pdims):
    """LPT gradient has correct sharding and is finite/non-zero."""
    sharded_ics, mesh, sharding = make_sharded_ics(single_device_ics, pdims)
    painting = jfli.PaintingOptions(target=target, paint_nside=PAINT_NSIDE)

    @jax.jit
    def forward(ic_array):
        ic = sharded_ics.replace(array=ic_array)
        result, _ = jfli.lpt(
            cosmo,
            ic,
            nb_shells=NB_SHELLS,
            order=1,
            painting=painting,
            shell_spacing="growth",
            min_width=1.0,
        )
        return jnp.sum(result.array**2)

    grads = jax.grad(forward)(sharded_ics.array)

    assert grads.sharding.is_equivalent_to(
        sharding, ndim=3
    ), f"Gradient sharding {grads.sharding} doesn't match input sharding {sharding}"
    assert jnp.all(jnp.isfinite(grads)), "Gradients contain non-finite values"
    assert float(jnp.linalg.norm(grads)) > 0, "Gradients are all zero"
    print(f"LPT gradient target={target} pdims={pdims}: OK, norm={float(jnp.linalg.norm(grads)):.6f}")


@pytest.mark.distributed
@pytest.mark.parametrize("target", ["density", "spherical", "flat"])
@pytest.mark.parametrize("solver_name", ["KKD", "DKD", "BullFrog"])
@pytest.mark.parametrize("pdims", _PDIMS)
def test_distributed_nbody_gradient_sharding(single_device_ics, cosmo, solver_name, target, pdims):
    """N-body gradient has correct sharding and is finite/non-zero."""
    sharded_ics, mesh, sharding = make_sharded_ics(single_device_ics, pdims)
    solver = _make_solver(solver_name, target)

    @jax.jit
    def forward(ic_array):
        ic = sharded_ics.replace(array=ic_array)
        dx, p = jfli.lpt(cosmo, ic, ts=T0, order=1)
        result = jfli.nbody(cosmo, dx, p, solver=solver, nb_shells=NB_SHELLS, adjoint="checkpointed")
        return jnp.sum(result.array**2)

    grads = jax.grad(forward)(sharded_ics.array)

    assert grads.sharding.is_equivalent_to(
        sharding, ndim=3
    ), f"Gradient sharding {grads.sharding} doesn't match input sharding {sharding}"
    assert jnp.all(jnp.isfinite(grads)), "Gradients contain non-finite values"
    assert float(jnp.linalg.norm(grads)) > 0, "Gradients are all zero"
    print(
        f"N-body gradient solver={solver_name} target={target} pdims={pdims}: "
        f"OK, norm={float(jnp.linalg.norm(grads)):.6f}"
    )


@pytest.mark.distributed
@pytest.mark.parametrize("pdims", _PDIMS)
def test_distributed_born_gradient_sharding(single_device_ics, cosmo, pdims):
    """Born lensing gradient has correct sharding and is finite/non-zero."""
    sharded_ics, mesh, sharding = make_sharded_ics(single_device_ics, pdims)
    solver = _make_solver("KKD", "spherical")

    @jax.jit
    def forward(ic_array):
        ic = sharded_ics.replace(array=ic_array)
        dx, p = jfli.lpt(cosmo, ic, ts=T0, order=1)
        lc = jfli.nbody(cosmo, dx, p, solver=solver, nb_shells=NB_SHELLS, adjoint="checkpointed")
        kappa = jfli.born(cosmo, lc, nz_shear=Z_SOURCE_BORN)
        return jnp.sum(kappa.array**2)

    grads = jax.grad(forward)(sharded_ics.array)

    assert grads.sharding.is_equivalent_to(
        sharding, ndim=3
    ), f"Gradient sharding {grads.sharding} doesn't match input sharding {sharding}"
    assert jnp.all(jnp.isfinite(grads)), "Gradients contain non-finite values"
    assert float(jnp.linalg.norm(grads)) > 0, "Gradients are all zero"
    print(f"Born gradient pdims={pdims}: OK, norm={float(jnp.linalg.norm(grads)):.6f}")


@pytest.mark.distributed
@pytest.mark.parametrize("target", ["spherical", "flat"])
@pytest.mark.parametrize("pdims", _PDIMS)
def test_distributed_lpt_gradient_equivalence(single_device_ics, cosmo, target, pdims):
    """LPT gradient values match between single-device and multi-device."""
    painting = jfli.PaintingOptions(target=target, paint_nside=PAINT_NSIDE)

    @jax.jit
    def forward(ic):
        result, _ = jfli.lpt(
            cosmo,
            ic,
            nb_shells=NB_SHELLS,
            order=1,
            painting=painting,
            shell_spacing="growth",
            min_width=1.0,
        )
        return jnp.sum(result.array**2)

    # Single-device gradient
    grad_single = jax.grad(forward)(single_device_ics)

    # Multi-device gradient
    sharded_ics, mesh, sharding = make_sharded_ics(single_device_ics, pdims)
    grad_multi = jax.grad(forward)(sharded_ics)
    gathered_grad = process_allgather(grad_multi, tiled=True)

    tols = dict(
        rtol=_GRAD_SPHERICAL_RTOL if target == "spherical" else _GRAD_FLAT_RTOL,
        atol=_GRAD_SPHERICAL_ATOL if target == "spherical" else _GRAD_FLAT_ATOL,
        mean_rtol=_GRAD_SPHERICAL_MEAN_RTOL if target == "spherical" else _GRAD_FLAT_MEAN_RTOL,
        mean_atol=_GRAD_SPHERICAL_MEAN_ATOL if target == "spherical" else _GRAD_FLAT_MEAN_ATOL,
    )
    compare_fields(
        np.asarray(grad_single),
        np.asarray(gathered_grad),
        f"LPT gradient equiv target={target} pdims={pdims}",
        **tols,
    )


@pytest.mark.distributed
@pytest.mark.parametrize("target", ["spherical", "flat"])
@pytest.mark.parametrize("solver_name", ["KKD", "DKD", "BullFrog"])
@pytest.mark.parametrize("pdims", _PDIMS)
def test_distributed_nbody_gradient_equivalence(single_device_ics, cosmo, solver_name, target, pdims):
    """N-body gradient values match between single-device and multi-device."""
    solver = _make_solver(solver_name, target)

    @jax.jit
    def forward(ic):
        dx, p = jfli.lpt(cosmo, ic, ts=T0, order=1)
        result = jfli.nbody(cosmo, dx, p, solver=solver, nb_shells=NB_SHELLS, adjoint="checkpointed")
        return jnp.sum(result.array**2)

    # Single-device gradient
    grad_single = jax.grad(forward)(single_device_ics)

    # Multi-device gradient
    sharded_ics, mesh, sharding = make_sharded_ics(single_device_ics, pdims)
    grad_multi = jax.grad(forward)(sharded_ics)
    gathered_grad = process_allgather(grad_multi, tiled=True)

    tols = dict(
        rtol=_GRAD_SPHERICAL_RTOL if target == "spherical" else _GRAD_FLAT_RTOL,
        atol=_GRAD_SPHERICAL_ATOL if target == "spherical" else _GRAD_FLAT_ATOL,
        mean_rtol=_GRAD_SPHERICAL_MEAN_RTOL if target == "spherical" else _GRAD_FLAT_MEAN_RTOL,
        mean_atol=_GRAD_SPHERICAL_MEAN_ATOL if target == "spherical" else _GRAD_FLAT_MEAN_ATOL,
    )
    compare_fields(
        np.asarray(grad_single),
        np.asarray(gathered_grad),
        f"N-body gradient equiv solver={solver_name} target={target} pdims={pdims}",
        **tols,
    )


@pytest.mark.distributed
@pytest.mark.parametrize("pdims", _PDIMS)
def test_distributed_born_gradient_equivalence(single_device_ics, cosmo, pdims):
    """Born lensing gradient values match between single-device and multi-device."""
    solver = _make_solver("KKD", "spherical")

    @jax.jit
    def forward(ic):
        dx, p = jfli.lpt(cosmo, ic, ts=T0, order=1)
        lc = jfli.nbody(cosmo, dx, p, solver=solver, nb_shells=NB_SHELLS, adjoint="checkpointed")
        kappa = jfli.born(cosmo, lc, nz_shear=Z_SOURCE_BORN)
        return jnp.sum(kappa.array**2)

    # Single-device gradient
    grad_single = jax.grad(forward)(single_device_ics)

    # Multi-device gradient
    sharded_ics, mesh, sharding = make_sharded_ics(single_device_ics, pdims)
    grad_multi = jax.grad(forward)(sharded_ics)
    gathered_grad = process_allgather(grad_multi, tiled=True)

    compare_fields(
        np.asarray(grad_single),
        np.asarray(gathered_grad),
        f"Born gradient equiv pdims={pdims}",
        rtol=_GRAD_BORN_RTOL,
        atol=_GRAD_BORN_ATOL,
        mean_rtol=_GRAD_BORN_MEAN_RTOL,
        mean_atol=_GRAD_BORN_MEAN_ATOL,
    )
