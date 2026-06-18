#!/usr/bin/env python
# ruff: noqa: E402
"""Experiment 00 — distributed Born convergence (κ) from the CosmoGrid density (JAX multi-host).

Loads ``00-cosmogrid-density`` from HuggingFace **sharded across the device mesh** and runs the
fully-JAX, differentiable Born approximation, **saving a parquet** ``SphericalKappaField``. It does
NOT publish — run it on the cluster, then publish the parquet locally with ``publish_local.py``.

Distributed exactly like ``docs/2-advanced-usage/11-multi-host-pm.py``: one process per GPU,
``jax.distributed`` coordinates them, and the lightcone is sharded ``P("x","y")`` (npix across the
mesh). Unlike ray-tracing (replicated numpy+MPI), Born is JAX-native, so the density is *sharded*,
not replicated — each device holds only its npix slab.

    srun -n $SLURM_NTASKS python born_kappa.py --multihost --nz s3 --out kappa_born.parquet

    # smoke test on fake CPU devices — round-trips a tiny density through the sharded LOAD path,
    # then runs distributed Born on it:
    XLA_FLAGS="--xla_force_host_platform_device_count=4" JAX_PLATFORMS=cpu \
        python born_kappa.py --smoke-test --nbins 2
"""

import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ.setdefault("JAX_ENABLE_X64", "False")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.97")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

from datetime import datetime


def _is_multiprocess() -> bool:
    return (
        int(os.environ.get("SLURM_NTASKS", 0)) > 1
        or int(os.environ.get("SLURM_NTASKS_PER_NODE", 0)) > 1
        or int(os.environ.get("OMPI_COMM_WORLD_SIZE", 0)) > 1
        or int(os.environ.get("PMI_SIZE", 0)) > 1
    )


def _maybe_init_distributed() -> None:
    """Initialize JAX distributed only under a real multi-process launch (srun / mpirun).

    Mirrors ``docs/2-advanced-usage/11-multi-host-pm.py`` / ``bin/fli-simulate``: strip the proxy
    variables *before* ``jax.distributed.initialize()`` so the coordinator's gRPC channel connects
    directly to the sibling tasks instead of being routed through an (unreachable) proxy.
    """
    if not _is_multiprocess():
        return
    for key in ("VSCODE_PROXY_URI", "no_proxy", "NO_PROXY"):
        os.environ.pop(key, None)
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Detected multi-host environment, initializing JAX distributed ...",
        flush=True,
    )
    import jax

    jax.distributed.initialize()
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] JAX distributed initialized with "
        f"{jax.process_count()} processes, rank {jax.process_index()}",
        flush=True,
    )


_maybe_init_distributed()

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
from jax.experimental.multihost_utils import sync_global_devices
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P

import jax_fli as jfli
from jax_fli.data import get_des_y3_nz_shear, get_stage3_nz_shear
from jax_fli.io import Catalog

REPO = "ASKabalan/jax-fli-experiments"
DENSITY_CONFIG = "00-cosmogrid-density"
HERE = Path(__file__).resolve().parent
NZ_CHOICES = {"s3": get_stage3_nz_shear, "des": get_des_y3_nz_shear}


def _mesh_sharding(nbins: int):
    """2-D device mesh + the canonical spherical-lightcone sharding (npix on "x", bins on "y").

    Mirrors 11-multi-host-pm.py (P_Y = source bins, P_X = the rest). ``apply_sharding`` trims this
    ``P("x","y")`` to the loaded map's actual shape (npix → "x"); Born then emits κ as ``P("y","x")``.
    """
    n_dev = jax.device_count()
    p_y = nbins if (nbins > 0 and n_dev % nbins == 0) else 1
    p_x = n_dev // p_y
    mesh = jax.make_mesh((p_x, p_y), ("x", "y"), axis_types=(AxisType.Auto, AxisType.Auto))
    return mesh, NamedSharding(mesh, P("x", "y"))


def _synthetic_density(nside: int = 64, n_shells: int = 4):
    """A tiny replicated SphericalDensity, saved+reloaded to exercise the sharded LOAD path."""
    npix = 12 * nside * nside
    z = np.linspace(0.2, 1.0, n_shells).astype(np.float32)
    a = 1.0 / (1.0 + z)
    cosmo = jc.Planck18()
    com = np.asarray(jc.background.radial_comoving_distance(cosmo, a), dtype=np.float32)
    dw = np.full(n_shells, float(np.mean(np.abs(np.diff(com))) or 200.0), dtype=np.float32)
    counts = np.random.default_rng(0).integers(0, 50, size=(n_shells, npix)).astype(np.float32)
    field = jfli.SphericalDensity(
        array=jnp.asarray(counts),
        mesh_size=(128, 128, 128),
        box_size=(2560.0, 2560.0, 2560.0),
        observer_position=(0.5, 0.5, 0.5),
        halo_size=(0, 0),
        nside=nside,
        z_sources=jnp.asarray(z),
        scale_factors=jnp.asarray(a),
        comoving_centers=jnp.asarray(com),
        density_width=jnp.asarray(dw),
        status=jfli.FieldStatus.LIGHTCONE,
        unit=jfli.DensityUnit.COUNTS,
    )
    return cosmo, field


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nz", choices=list(NZ_CHOICES), default="s3", help="source n(z): s3 (Stage-3) or des (DES Y3)")
    ap.add_argument("--nside", type=int, default=None, help="downsample density to this nside (default: native)")
    ap.add_argument("--nbins", type=int, default=4, help="number of source bins (also the mesh 'y' size)")
    ap.add_argument("--out", default=None, help="output parquet path (default: kappa_born.parquet next to this script)")
    ap.add_argument("--multihost", action="store_true", help="(informational) distributed init is auto-detected")
    ap.add_argument("--smoke-test", action="store_true", help="round-trip a tiny sharded density, then Born it")
    args = ap.parse_args()

    lead = jax.process_index() == 0
    nz_shear = NZ_CHOICES[args.nz]()[: args.nbins]
    mesh, sharding = _mesh_sharding(args.nbins)
    if lead:
        print(f"devices={jax.device_count()} mesh={tuple(mesh.devices.shape)} sharding={sharding.spec}", flush=True)

    if args.smoke_test:
        import tempfile

        cosmo, dens = _synthetic_density()
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "dens.parquet")
            Catalog(field=dens, cosmology=cosmo).to_parquet(p)
            catalog = Catalog.from_parquet(p, sharding=sharding)  # the sharded LOAD path under test
        field, cosmo = catalog.field[0], catalog.cosmology[0]
    else:
        from datasets import load_dataset

        ds = load_dataset(REPO, DENSITY_CONFIG, split="train").with_format("numpy")
        catalog = Catalog.from_dataset(ds, sharding=sharding)
        field, cosmo = catalog.field[0], catalog.cosmology[0]
        if args.nside is not None and int(args.nside) != int(field.nside):
            # ud_sample gathers the (smaller, downsampled) map to replicated P(); Born re-shards its
            # output back to P("y","x"). Fine — the downsampled field is smaller than the native one.
            field = field.ud_sample(int(args.nside))

    if lead:
        print(f"   density: {type(field).__name__} {tuple(field.array.shape)} nside={field.nside} "
              f"sharding={field.array.sharding}  | n(z)={args.nz} ({len(nz_shear)} bins)", flush=True)

    kappa = jfli.born(cosmo, field, nz_shear=nz_shear)

    if lead:
        print(f"   born κ: {type(kappa).__name__} {tuple(kappa.array.shape)} nside={kappa.nside} "
              f"sharding={kappa.array.sharding}", flush=True)
        if args.smoke_test:
            assert kappa.array.shape[-1] == 12 * field.nside * field.nside
            print(
                f"[smoke-test] PASS — sharded load + distributed Born across {jax.device_count()} devices.",
                flush=True,
            )
        else:
            out = Path(args.out) if args.out else HERE / "kappa_born.parquet"
            Catalog(field=kappa, cosmology=cosmo).to_parquet(str(out))
            print(
                f"   wrote {out}  ({out.stat().st_size / 1e9:.2f} GB). Publish with: python publish_local.py",
                flush=True,
            )

    if _is_multiprocess():
        sync_global_devices("born_kappa done")
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
