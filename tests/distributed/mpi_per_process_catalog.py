"""Multi-process check of ``Catalog.to_parquet(path, per_process=True)`` / ``from_parquet``.

Run under MPI (4 processes x 2 CPU devices = 8 devices, arrays genuinely not fully addressable)::

    mpirun -n 4 python tests/distributed/mpi_per_process_catalog.py <outdir>

Launched by ``test_per_process_catalog.py``. Every process builds the same reference arrays from a fixed
seed and shards them with ``make_array_from_callback``, so each check compares local shards against the
reference without any gather. Process 0 saves the references to ``<outdir>/ref_*.npy`` for the
single-process checks done by the wrapper test.

The references are float32-representable: the parquet reader decodes through datasets' numpy
formatter, which returns float32 (the float64 bytes on disk are exact). Exact equality then tests the
block bookkeeping, which is what this file is about.
"""

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax  # noqa: E402

jax.config.update("jax_cpu_collectives_implementation", "gloo")
jax.distributed.initialize(cluster_detection_method="mpi4py")
jax.config.update("jax_enable_x64", True)

import jax_cosmo as jc  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import AxisType, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

import jax_fli as jfli  # noqa: E402
from jax_fli._src.base._enums import ConvergenceUnit  # noqa: E402
from jax_fli.io.catalog import Catalog  # noqa: E402

OUT = sys.argv[1]
RANK = jax.process_index()
MESH_SIZE = (32, 32, 32)
BOX_SIZE = (100.0, 100.0, 100.0)
NSIDE = 4
N_SHELLS = 3


def mesh(pdims):
    return jax.make_mesh(pdims, ("x", "y"), axis_types=(AxisType.Auto, AxisType.Auto))


def reference(shape, dtype, seed):
    """Same array on every process; float32-representable (see module docstring)."""
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32).astype(dtype)


def global_array(ref, sharding):
    return jax.make_array_from_callback(ref.shape, sharding, lambda index: ref[index])


def check_shards(array, ref, label):
    """Every addressable shard of ``array`` equals the reference at its index."""
    assert array.shape == ref.shape, f"{label}: shape {array.shape} != {ref.shape}"
    assert array.dtype == ref.dtype, f"{label}: dtype {array.dtype} != {ref.dtype}"
    for shard in array.addressable_shards:
        np.testing.assert_array_equal(np.asarray(shard.data), ref[shard.index], err_msg=label)


def check_metadata(loaded, original, label):
    assert type(loaded) is type(original), f"{label}: {type(loaded).__name__} != {type(original).__name__}"
    for key in ("mesh_size", "box_size", "observer_position", "halo_size", "nside", "status", "unit", "name"):
        assert getattr(loaded, key) == getattr(original, key), f"{label}: {key} differs"
    for key in ("z_sources", "scale_factors", "comoving_centers", "density_width"):
        np.testing.assert_allclose(np.asarray(getattr(loaded, key)), np.asarray(getattr(original, key)), err_msg=label)


def density_field(sharding):
    ref = reference(MESH_SIZE, np.float64, seed=0)
    field = jfli.DensityField(
        array=global_array(ref, NamedSharding(sharding.mesh, P("x", "y"))),
        mesh_size=MESH_SIZE,
        box_size=BOX_SIZE,
        observer_position=(0.5, 0.5, 0.5),
        field_sharding=sharding,
        halo_size=(0, 0),
        z_sources=np.float64(0.0),
        scale_factors=np.float64(0.1),
        comoving_centers=np.float64(0.0),
        density_width=np.float64(0.0),
        status=jfli.FieldStatus.INITIAL_FIELD,
        unit=jfli.DensityUnit.DENSITY,
        name="ic",
    ).apply_sharding()
    return field, ref


def spherical_fields(sharding, cls, seed, replicated=False):
    npix = 12 * NSIDE**2
    ref = reference((N_SHELLS, npix), np.float32, seed=seed)
    spec = P() if replicated else P(None, "x")
    is_kappa = cls is jfli.SphericalKappaField
    field = cls(
        array=global_array(ref, NamedSharding(sharding.mesh, spec)),
        mesh_size=MESH_SIZE,
        box_size=BOX_SIZE,
        observer_position=(0.5, 0.5, 0.5),
        field_sharding=None if replicated else sharding,
        halo_size=(0, 0),
        nside=NSIDE,
        z_sources=np.array([0.5, 1.0, 1.5]),
        scale_factors=np.array([0.9, 0.8, 0.7]),
        comoving_centers=np.array([100.0, 200.0, 300.0]),
        density_width=np.array([10.0, 20.0, 30.0]),
        status=jfli.FieldStatus.KAPPA if is_kappa else jfli.FieldStatus.LIGHTCONE,
        unit=ConvergenceUnit.DIMENSIONLESS if is_kappa else jfli.DensityUnit.DENSITY,
        name=cls.__name__,
    )
    return (field if replicated else field.apply_sharding()), ref


def main():
    cosmo = jc.Planck18()
    base = NamedSharding(mesh((4, 2)), P("x", "y"))

    ic, ic_ref = density_field(base)
    lc, lc_ref = spherical_fields(base, jfli.SphericalDensity, seed=1)
    kappa, kappa_ref = spherical_fields(base, jfli.SphericalKappaField, seed=2)
    kappa_rep, kappa_rep_ref = spherical_fields(base, jfli.SphericalKappaField, seed=3, replicated=True)
    assert not ic.array.is_fully_addressable and not lc.array.is_fully_addressable
    assert kappa_rep.array.is_fully_replicated and not kappa_rep.array.is_fully_addressable

    if RANK == 0:
        for name, ref in (("ic", ic_ref), ("lc", lc_ref), ("kappa", kappa_ref), ("kappa_rep", kappa_rep_ref)):
            np.save(os.path.join(OUT, f"ref_{name}.npy"), ref)

    # 1. write: one folder, 4 parts, header last
    ic_path = os.path.join(OUT, "ic.parquet")
    Catalog(field=[ic], cosmology=[cosmo]).to_parquet(ic_path, per_process=True)
    parts = sorted(f for f in os.listdir(ic_path) if f.startswith("part_"))
    assert parts == [f"part_000_{p:06d}.parquet" for p in range(4)], parts
    assert os.path.isfile(os.path.join(ic_path, "_header.json"))

    # 2-3. reload into the same and into different layouts
    for pdims in [(4, 2), (2, 4), (8, 1), (1, 8)]:
        sharding = NamedSharding(mesh(pdims), P("x", "y"))
        loaded = Catalog.from_parquet(ic_path, sharding=sharding).field[0]
        check_shards(loaded.array, ic_ref, f"ic reload {pdims}")
        check_metadata(loaded, ic, f"ic reload {pdims}")
        expected = ic.replace(field_sharding=sharding).apply_sharding().array.sharding
        assert loaded.array.sharding.is_equivalent_to(expected, 3), f"ic layout {pdims}: {loaded.array.sharding}"

    # 4. reload without sharding: one host array everywhere
    loaded = Catalog.from_parquet(ic_path).field[0]
    np.testing.assert_array_equal(np.asarray(loaded.array), ic_ref)
    check_metadata(loaded, ic, "ic unsharded")

    # 5-6. batched spherical maps (density + kappa with its own transposed layout) and a replicated
    # kappa map in one catalog, float32 preserved
    multi_path = os.path.join(OUT, "multi.parquet")
    Catalog(field=[lc, kappa, kappa_rep], cosmology=[cosmo] * 3).to_parquet(multi_path, per_process=True)
    parts = sorted(f for f in os.listdir(multi_path) if f.startswith("part_"))
    assert [p for p in parts if p.startswith("part_002_")] == ["part_002_000000.parquet"], parts
    for pdims in [(4, 2), (2, 4)]:
        sharding = NamedSharding(mesh(pdims), P("x", "y"))
        fields = Catalog.from_parquet(multi_path, sharding=sharding).field
        for loaded, original, ref, label in zip(
            fields, (lc, kappa, kappa_rep), (lc_ref, kappa_ref, kappa_rep_ref), ("lc", "kappa", "kappa_rep")
        ):
            check_shards(loaded.array, ref, f"{label} reload {pdims}")
            check_metadata(loaded, original, f"{label} reload {pdims}")
    fields = Catalog.from_parquet(multi_path).field
    for loaded, ref in zip(fields, (lc_ref, kappa_ref, kappa_rep_ref)):
        np.testing.assert_array_equal(np.asarray(loaded.array), ref)

    # 7. the default path is untouched: still one gathered file
    single_path = os.path.join(OUT, "ic_single.parquet")
    Catalog(field=[ic], cosmology=[cosmo]).to_parquet(single_path)
    jax.experimental.multihost_utils.sync_global_devices("single-written")
    assert os.path.isfile(single_path)

    # rewriting over an existing folder replaces it
    Catalog(field=[ic], cosmology=[cosmo]).to_parquet(ic_path, per_process=True)
    check_shards(Catalog.from_parquet(ic_path, sharding=base).field[0].array, ic_ref, "ic rewrite")

    print(f"rank {RANK}: OK", flush=True)


if __name__ == "__main__":
    import jax.experimental.multihost_utils  # noqa: F401

    main()
    jax.distributed.shutdown()
