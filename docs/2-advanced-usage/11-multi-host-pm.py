#!/usr/bin/env python
"""Distributed (multi-host) PM lightcone -> Born convergence -> Parquet catalog.

This is the script you launch across nodes. The whole pipeline is identical to the
single-device notebooks; only the device mesh changes. Run it either way:

  # Multi-host on SLURM (one process per GPU, jax.distributed coordinates them):
  srun -n $SLURM_NTASKS python 11-multi-host-pm.py --multihost \
       --mesh 1024 --box 3000 --nside 1024 --nb-shells 16 --out kappa_sim.parquet

  # Single host with fake CPU devices (laptop / CI smoke test):
  XLA_FLAGS="--xla_force_host_platform_device_count=4" JAX_PLATFORMS=cpu \
       python 11-multi-host-pm.py --mesh 64 --nside 64 --out kappa_sim.parquet

The output Parquet holds a ``SphericalKappaField`` (one map per source bin) plus the
cosmology, ready to load with ``jax_fli.io.Catalog.from_parquet`` (see 11-multi-host-pm.md).
"""

import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["JAX_ENABLE_X64"] = "False"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.97"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["--xla_gpu_nccl_termination_timeout_seconds"] = "100"
os.environ["--xla_gpu_executable_warn_stuck_timeout"] = "60"

from datetime import datetime


def _maybe_init_distributed() -> None:
    """Initialize JAX distributed only under a real multi-process launch (srun / mpirun).

    Mirrors ``bin/fli-simulate``: strip the proxy variables *before*
    ``jax.distributed.initialize()`` so the coordinator's gRPC channel connects
    directly to the sibling tasks instead of being routed through an
    (unreachable) HTTP / VSCode-remote proxy. With a proxy left in place the
    init hangs for minutes — reproduced locally: a black-hole ``http_proxy``
    makes ``initialize()`` time out, while stripping it returns in ~1.5 s.
    ``flush=True`` so the timestamps appear immediately under srun's
    block-buffered stdout (otherwise it merely *looks* stuck).
    """
    multi = (
        int(os.environ.get("SLURM_NTASKS", 0)) > 1
        or int(os.environ.get("SLURM_NTASKS_PER_NODE", 0)) > 1
        or int(os.environ.get("OMPI_COMM_WORLD_SIZE", 0)) > 1
        or int(os.environ.get("PMI_SIZE", 0)) > 1
    )
    if not multi:
        return
    # Match bin/fli-simulate exactly — these are the vars that broke init on this cluster (a stale
    # VSCode-remote / no_proxy config makes the coordinator's gRPC retry forever). Kept narrow on
    # purpose: http(s)_proxy is left in place so jobs that fetch data via the proxy still work,
    # exactly like the launcher. If init *still* hangs and `env | grep -i proxy` shows an
    # http(s)_proxy the compute node can't reach, also pop http_proxy/https_proxy/HTTP_PROXY/
    # HTTPS_PROXY here (a black-hole http_proxy reproduces the same hang locally).
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

import jax
import jax_cosmo as jc
from jax.experimental.multihost_utils import sync_global_devices
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P

import jax_fli as jfli


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=int, default=256, help="cells per axis")
    parser.add_argument("--nbins", type=int, default=2, help="source bins")
    parser.add_argument("--nside", type=int, default=256, help="HEALPix nside")
    parser.add_argument("--nb-shells", type=int, default=8, help="lightcone shells")
    parser.add_argument("--out", type=str, default="kappa_sim.parquet")
    parser.add_argument(
        "--multihost",
        action="store_true",
        help="call jax.distributed.initialize() (one process per device across nodes)",
    )
    args = parser.parse_args()

    # Compute the box size from the redshift of the lightcone (z=1.0) and the observer position.
    # Z=1.0 is good for nbins = 2
    cosmo = jc.Planck18()
    box_size = jfli.compute_box_size_from_redshift(cosmo, 1.0, observer_position=(0.5, 0.5, 0.5))
    box_size = tuple(float(b) for b in box_size)

    n_dev = jax.device_count()
    P_X = n_dev // args.nbins
    P_Y = args.nbins
    # 2-D device mesh: partition the first two spatial axes (jaxpm convention).
    mesh = jax.make_mesh((P_X, P_Y), ("x", "y"), axis_types=(AxisType.Auto, AxisType.Auto))
    sharding = NamedSharding(mesh, P("x", "y"))
    print(f"Created device mesh {mesh} with {n_dev} devices, sharding {sharding}")

    key = jax.random.PRNGKey(0)

    # Ghost-cell width for the halo exchange at shard boundaries. Without it
    # (halo_size=0) the PM forces are wrong at the x-slab edges and the lightcone
    # shells show repeated patterns that worsen with more devices. The halo must
    # exceed the largest particle displacement (in cells) that crosses a boundary;
    # mesh//4 is a safe, generous default. Harmless on a single device.
    halo = args.mesh // 4
    initial_field = jfli.gaussian_initial_conditions(
        key,
        (args.mesh,) * 3,
        box_size=box_size,
        cosmo=cosmo,
        nside=args.nside,
        field_sharding=sharding,
        halo_size=(halo, halo),
    )
    print(f"Generated initial conditions with shape {initial_field.shape} and sharding {initial_field.array.sharding}")

    dx, p = jfli.lpt(cosmo, initial_field, ts=0.1, order=1)
    print(f"LPT finished: dx {dx.shape} p {p.shape} | sharding {dx.array.sharding} {p.array.sharding}")

    solver = jfli.DoubleKickDrift(
        interp_kernel=jfli.NoInterp(painting=jfli.PaintingOptions(target="spherical", scheme="rbf_neighbor")),
        t0=0.1,
        t1=1.0,
        n_steps=max(10, args.nb_shells + 2),
    )
    lightcone = jfli.nbody(cosmo, dx, p, nb_shells=args.nb_shells, solver=solver)
    print(f"NBody finished: lightcone {lightcone.shape} | sharding {lightcone.array.sharding}")

    nz_sources = jfli.io.get_stage3_nz_shear()[: args.nbins]
    kappa = jfli.born(cosmo, lightcone, nz_shear=nz_sources)
    print(f"Born finished: kappa {kappa.shape} | sharding {kappa.array.sharding}")
    # In multi host setup we can use jax_cuda for the shear computation
    # In single host setup you have to use jax (due to a bug in s2fft)
    shear = kappa.get_shear(method="jax")
    print(f"Shear finished: shear {shear.shape} | sharding {shear.array.sharding}")

    print("simulation finished")

    jfli.io.Catalog(field=kappa, cosmology=cosmo).to_parquet(args.out.replace(".parquet", "_kappa.parquet"))
    print(f"wrote {args.out}: {type(kappa).__name__} {kappa.shape}")
    jfli.io.Catalog(field=shear, cosmology=cosmo).to_parquet(args.out.replace(".parquet", "_shear.parquet"))
    print(f"wrote {args.out}: {type(shear).__name__} {shear.shape}")

    sync_global_devices("Done")
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
