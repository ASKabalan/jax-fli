"""Distributed test environment: multi-device fixtures and helpers.

Usage (local, CPU):
    JAX_PLATFORM_NAME=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
        pytest tests/distributed/ -v
"""

from __future__ import annotations

import os

# Initialize distributed before any JAX computation
if "SLURM_JOB_ID" in os.environ:
    import jax

    jax.distributed.initialize()
    print(f"initialized distributed with {jax.process_count()} processes and {jax.device_count()}  devices")
else:
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
    import jax

    print(f"starting local distributed on CPU with {jax.device_count()} devices")

import jax_cosmo as jc  # noqa: E402
import jax_fli as jfli  # noqa: E402
import pytest  # noqa: E402
from jax import lax  # noqa: E402
from jax.sharding import AxisType, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_N_DEVICES = jax.device_count()
_PDIMS = [(1, _N_DEVICES), (_N_DEVICES, 1)]
if _N_DEVICES >= 4:
    _PDIMS.append((_N_DEVICES // 2, 2))
    _PDIMS.append((2, _N_DEVICES // 2))

MESH_SIZE = (32, 32, 32)
BOX_SIZE = (256.0, 256.0, 256.0)
FLATSKY_NPIX = (32, 32)
FIELD_SIZE = (10.0, 10.0)
PAINT_NSIDE = 16
TOLERANCE = 1e-12
SEED = 42
T0 = 0.1
T1 = 1.0
N_STEPS = 10
Z_SOURCE_BORN = 0.5
NB_SHELLS = 4

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cosmo():
    return jc.Planck18()


@pytest.fixture(scope="module")
def single_device_ics(cosmo):
    """Generate initial conditions on a single device with lightcone metadata."""
    key = jax.random.PRNGKey(SEED)
    return jfli.gaussian_initial_conditions(
        key,
        MESH_SIZE,
        BOX_SIZE,
        cosmo=cosmo,
        flatsky_npix=FLATSKY_NPIX,
        field_size=FIELD_SIZE,
        nside=PAINT_NSIDE,
    )


# ---------------------------------------------------------------------------
# Helper (not a fixture — imported by test files)
# ---------------------------------------------------------------------------


def make_sharded_ics(single_device_ics, pdims):
    """Shard single-device ICs across a device mesh.

    Returns
    -------
    sharded_ics : DensityField
    mesh : jax.sharding.Mesh
    sharding : NamedSharding
    """
    mesh = jax.make_mesh(pdims, ("x", "y"), axis_types=(AxisType.Auto, AxisType.Auto))
    sharding = NamedSharding(mesh, P("x", "y"))
    halo_size = (single_device_ics.mesh_size[0] // 4, single_device_ics.mesh_size[1] // 4)

    sharded_ics = single_device_ics.replace(
        array=lax.with_sharding_constraint(single_device_ics.array, sharding),
        field_sharding=sharding,
        halo_size=halo_size,
    )
    return sharded_ics, mesh, sharding
