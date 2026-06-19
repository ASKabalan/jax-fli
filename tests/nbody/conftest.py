"""Fixtures for N-body / LPT tests (FastPM reference data)."""

from __future__ import annotations

import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
import pytest

from jax_fli.fields import DensityField, FieldStatus
from jax_fli.fields.units import DensityUnit

# ---------------------------------------------------------------------------
# Simulation configuration fixtures for fastpm comparison tests
# ---------------------------------------------------------------------------


@pytest.fixture(
    scope="session",
    params=[
        ((32, 32, 32), (256.0, 256.0, 256.0)),
        ((32, 32, 64), (256.0, 256.0, 512.0)),
    ],
)
def simulation_config(request):
    return request.param


@pytest.fixture(scope="session", params=[0.1, 0.2])
def lpt_scale_factor(request):
    return request.param


@pytest.fixture(scope="session", params=[10, 50, 80])
def steps(request):
    """Number of steps for n-body tests."""
    return request.param


@pytest.fixture(scope="session")
def particle_mesh(simulation_config):
    from pmesh.pm import ParticleMesh

    mesh_shape, box_shape = simulation_config
    return ParticleMesh(BoxSize=box_shape, Nmesh=mesh_shape, dtype="f8")


@pytest.fixture(scope="session")
def fpm_initial_conditions(cosmology, particle_mesh):
    """Generate initial conditions using fastpm's pmesh (whitenoise + power spectrum)."""
    grid = particle_mesh.generate_uniform_particle_grid(shift=0).astype(np.float64)

    k = jnp.logspace(-4, 1, 128)
    pk = jc.power.linear_matter_power(cosmology, k)

    def pk_fn(x):
        return jnp.interp(x, k, pk)

    whitec = particle_mesh.generate_whitenoise(42, type="complex", unitary=False)
    lineark = whitec.apply(
        lambda k, v: jnp.sqrt(pk_fn(jnp.sqrt(sum(ki**2 for ki in k)))) * v * jnp.sqrt((1 / v.BoxSize).prod())
    )
    init_mesh = lineark.c2r().value

    return lineark, grid, init_mesh


@pytest.fixture(scope="session")
def initial_conditions(fpm_initial_conditions, simulation_config):
    """Wrap the raw init_mesh into a DensityField with INITIAL_FIELD status."""
    _, _, init_mesh = fpm_initial_conditions
    mesh_shape, box_shape = simulation_config

    return DensityField(
        array=jnp.asarray(init_mesh),
        mesh_size=mesh_shape,
        box_size=box_shape,
        status=FieldStatus.INITIAL_FIELD,
        unit=DensityUnit.DENSITY,
    )


@pytest.fixture(scope="session")
def fpm_solver(cosmology, particle_mesh):
    from fastpm.core import Cosmology as FastPMCosmology
    from fastpm.core import Solver

    ref_cosmo = FastPMCosmology(cosmology)
    return Solver(particle_mesh, ref_cosmo, B=1)


@pytest.fixture(scope="session")
def fpm_lpt1(fpm_solver, fpm_initial_conditions, lpt_scale_factor):
    lineark, grid, _ = fpm_initial_conditions
    return fpm_solver.lpt(lineark, grid, lpt_scale_factor, order=1)


@pytest.fixture(scope="session")
def fpm_lpt1_field(fpm_lpt1, particle_mesh):
    return particle_mesh.paint(fpm_lpt1.X).value


@pytest.fixture(scope="session")
def fpm_lpt2(fpm_solver, fpm_initial_conditions, lpt_scale_factor):
    lineark, grid, _ = fpm_initial_conditions
    return fpm_solver.lpt(lineark, grid, lpt_scale_factor, order=2)


@pytest.fixture(scope="session")
def fpm_lpt2_field(fpm_lpt2, particle_mesh):
    return particle_mesh.paint(fpm_lpt2.X).value


@pytest.fixture(scope="session")
def nbody_from_lpt1(fpm_solver, fpm_lpt1, particle_mesh, lpt_scale_factor, steps):
    from fastpm.core import leapfrog

    stages = np.linspace(lpt_scale_factor, 1.0, steps, endpoint=True)
    state = fpm_lpt1.copy()
    state.a = dict(fpm_lpt1.a)
    finalstate = fpm_solver.nbody(state, leapfrog(stages))
    return particle_mesh.paint(finalstate.X).value


@pytest.fixture(scope="session")
def nbody_from_lpt2(fpm_solver, fpm_lpt2, particle_mesh, lpt_scale_factor, steps):
    from fastpm.core import leapfrog

    stages = np.linspace(lpt_scale_factor, 1.0, steps, endpoint=True)
    state = fpm_lpt2.copy()
    state.a = dict(fpm_lpt2.a)
    finalstate = fpm_solver.nbody(state, leapfrog(stages))
    return particle_mesh.paint(finalstate.X).value
