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

    srun -n $SLURM_NTASKS python born_kappa.py --nz s3 --out kappa_born.parquet

    # local multi-process validation on CPU (a 1-GPU box can't host 8 JAX processes) — needs the
    # explicit CPU platform; one MPI rank per CPU device, true multi-host via jax.distributed:
    uv run mpirun -n 8 -x JAX_PLATFORMS=cpu python born_kappa.py --nside 32

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
# Run fully offline from the warm HuggingFace cache (pre-populated by download.py): the per-shell
# snapshot_download + streaming load below must NEVER touch the network. The cache LOCATION is still
# $HF_HOME (site-specific) — set that separately on offline compute nodes. setdefault, so an explicit
# HF_HUB_OFFLINE=0 can still re-warm the cache if ever needed.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

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
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
from jax.experimental.mesh_utils import create_hybrid_device_mesh
from jax.experimental.multihost_utils import sync_global_devices
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import jax_fli as jfli
from jax_fli.data import get_des_y3_nz_shear, get_stage3_nz_shear
from jax_fli.io import Catalog

REPO = "ASKabalan/jax-fli-experiments"
DENSITY_CONFIG = "00-cosmogrid-density"
DENSITY_GLOB = "00-cosmogrid/catalogs/cosmogrid_density_nside2048_shell*.parquet"
HERE = Path(__file__).resolve().parent
NZ_CHOICES = {"s3": get_stage3_nz_shear, "des": get_des_y3_nz_shear}


def _mesh_sharding(nbins: int, gpus_per_node: int | None = None):
    """2-D device mesh + the canonical spherical-lightcone sharding (npix on "x", bins on "y").

    Mirrors 11-multi-host-pm.py (P_Y = source bins, P_X = the rest). ``apply_sharding`` trims this
    ``P("x","y")`` to the loaded map's actual shape (npix → "x"); Born then emits κ as ``P("y","x")``.

    On a non-uniform interconnect (NVLink intra-node + InfiniBand inter-node, detected via the
    ``slice_index`` attribute) a hybrid mesh is built so the fast NVLink axis stays inside each
    node; ``gpus_per_node`` (``--gpus-per-node`` or ``$SLURM_GPUS_ON_NODE``) is the slice width.
    """
    n_dev = jax.device_count()
    p_y = nbins if (nbins > 0 and n_dev % nbins == 0) else 1
    p_x = n_dev // p_y
    # A hybrid (multi-bandwidth) mesh is only needed when devices span MULTIPLE nodes/slices. On
    # current jax every GPU device carries a ``slice_index`` attribute even on a single GPU, and CPU
    # multi-process devices all report slice_index 0 — so detect multi-node by the number of DISTINCT
    # slice indices, not by the attribute's presence. Single GPU, single-node multi-GPU, and CPU
    # multi-process (one slice) take the uniform mesh; only genuine multi-node GPU takes the hybrid one.
    if len({getattr(d, "slice_index", 0) for d in jax.devices()}) <= 1:
        # Uniform interconnect (single node / NVLink only, or CPU).
        mesh = jax.make_mesh((p_x, p_y), ("x", "y"), axis_types=(AxisType.Auto, AxisType.Auto))
    else:
        # Non-uniform interconnect (NVLink intra-node + InfiniBand inter-node): hybrid mesh.
        if gpus_per_node is None:
            env = os.environ.get("SLURM_GPUS_ON_NODE")
            if env is None:
                raise RuntimeError(
                    "Hybrid (multi-bandwidth) mesh detected but GPUs-per-node is unknown: pass "
                    "--gpus-per-node or set $SLURM_GPUS_ON_NODE."
                )
            gpus_per_node = int(env)
        if gpus_per_node <= 0 or p_x % gpus_per_node != 0:
            raise ValueError(
                f"mesh ({p_x}, {p_y}) with gpus_per_node={gpus_per_node}: the first mesh axis "
                f"({p_x}) must be a positive multiple of gpus_per_node for the hybrid mesh."
            )
        mesh_shape = (gpus_per_node, 1)
        dcn_shape = (p_x // gpus_per_node, p_y)
        mesh = Mesh(create_hybrid_device_mesh(mesh_shape, dcn_shape), axis_names=("x", "y"))
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
    ap.add_argument("--min_z", type=float, default=0.01, help="minimum source redshift for Born integration")
    ap.add_argument("--max_z", type=float, default=1.5, help="maximum source redshift for Born integration")
    ap.add_argument("--n-integrate", type=int, default=32, help="Born integration steps per shell")
    ap.add_argument(
        "--normalization", choices=["global", "per_plane"], default="global", help="Born normalization scheme"
    )
    ap.add_argument("--nside", type=int, default=None, help="downsample density to this nside (default: native)")
    ap.add_argument("--nbins", type=int, default=4, help="number of source bins (also the mesh 'y' size)")
    ap.add_argument(
        "--gpus-per-node",
        type=int,
        default=None,
        dest="gpus_per_node",
        help="GPUs per node (intra-node NVLink slice width) for the hybrid device mesh on "
        "non-uniform interconnects. Default: None (falls back to $SLURM_GPUS_ON_NODE).",
    )
    ap.add_argument("--out", default=None, help="output parquet path (default: kappa_born.parquet next to this script)")
    ap.add_argument("--smoke-test", action="store_true", help="round-trip a tiny sharded density, then Born it")
    args = ap.parse_args()

    lead = jax.process_index() == 0
    nz_shear = NZ_CHOICES[args.nz]()[: args.nbins]
    mesh, sharding = _mesh_sharding(args.nbins, args.gpus_per_node)
    if lead:
        print(f"devices={jax.device_count()} mesh={tuple(mesh.devices.shape)} sharding={sharding.spec}", flush=True)

    if args.smoke_test:
        import tempfile

        cosmo, dens = _synthetic_density()
        # _synthetic_density builds a REPLICATED, process-local array. Under multi-host, to_parquet's
        # process_allgather(tiled=True) would CONCATENATE the per-rank copies (→ "z_sources length N !=
        # batch_size N*nprocs"). Shard it onto the mesh first so it is a proper GLOBAL array the
        # all_gather reconstructs (no-op single-process).
        dens = dens.replace(field_sharding=sharding).apply_sharding()
        # All ranks must call to_parquet (catalog_to_row does a collective all_gather), but only rank 0
        # writes — so use a SHARED path (one node) all ranks can read. A per-rank TemporaryDirectory
        # would give rank 0 a path the others can't see (FileNotFoundError under multi-process).
        p = str(Path(tempfile.gettempdir()) / "born_kappa_smoke_dens.parquet")
        Catalog(field=dens, cosmology=cosmo).to_parquet(p)
        if _is_multiprocess():
            sync_global_devices("smoke-test density written")
        catalog = Catalog.from_parquet(p, sharding=sharding)  # the sharded LOAD path under test
        field, cosmo = catalog.field[0], catalog.cosmology[0]
    else:
        # The density is ONE config 00-cosmogrid-density with a (npix,) row per shell. STREAM it (a
        # plain load_dataset would combine the 56 shell rows into one Arrow column and overflow its
        # int32 list-offset). Two load strategies, decided from the first shell's native nside:
        #   * downsample (--nside < native): load each shell REPLICATED, ud_sample, then shard the
        #     small stacked field ONCE below (ud_grade on a P("x")-sharded nside-2048 shell is unsafe —
        #     RING superpixels are not contiguous per shard);
        #   * native: shard each full shell on load (each rank keeps ~npix/n_dev) — never materialize
        #     the whole 56-shell lightcone replicated, which would OOM every rank.
        from datasets import load_dataset
        from huggingface_hub import snapshot_download

        # Cache the per-shell parquets + README ONCE (idempotent; offline-capable under HF_HUB_OFFLINE=1)
        # and STREAM from the local dir. Under multi-host only the lead rank fetches; a barrier then lets
        # every rank read the populated cache — otherwise N ranks each pull the full ~11 GB lightcone.
        if lead:
            snapshot_download(REPO, repo_type="dataset", allow_patterns=[DENSITY_GLOB, "README.md"])
        if _is_multiprocess():
            sync_global_devices("density cached")
        local = snapshot_download(REPO, repo_type="dataset", allow_patterns=[DENSITY_GLOB, "README.md"])

        fields, cosmo, downsample = [], None, None
        for row in load_dataset(local, DENSITY_CONFIG, split="train", streaming=True):
            if downsample is None:
                cat = Catalog.from_dataset(row)  # replicated peek to read the native nside
                fld = cat.field[0]
                downsample = args.nside is not None and int(args.nside) < int(fld.nside)
                # native: shard this first (peeked) shell now; downsample: shrink it
                fld = (
                    fld.ud_sample(int(args.nside))
                    if downsample
                    else fld.replace(field_sharding=sharding).apply_sharding()
                )
            elif downsample:
                cat = Catalog.from_dataset(row)
                fld = cat.field[0].ud_sample(int(args.nside))
            else:
                cat = Catalog.from_dataset(row, sharding=sharding)
                fld = cat.field[0]
            fields.append(fld)
            cosmo = cat.cosmology[0]
        # Each row is one (npix,) shell; stack adds the leading shell axis → (n_shells, npix).
        field = jfli.SphericalDensity.stack(fields)
        if downsample:
            # Shard the small downsampled lightcone once → a proper GLOBAL array for Born + to_parquet.
            field = field.replace(field_sharding=sharding).apply_sharding()

    if lead:
        print(
            f"   density: {type(field).__name__} {tuple(field.array.shape)} nside={field.nside} "
            f"sharding={field.array.sharding}  | n(z)={args.nz} ({len(nz_shear)} bins)",
            flush=True,
        )

    kappa = jfli.born(
        cosmo,
        field,
        nz_shear=nz_shear,
        min_z=args.min_z,
        max_z=args.max_z,
        n_integrate=args.n_integrate,
        normalization=args.normalization,
    )

    if lead:
        print(
            f"   born κ: {type(kappa).__name__} {tuple(kappa.array.shape)} nside={kappa.nside} "
            f"sharding={kappa.array.sharding}",
            flush=True,
        )
    if args.smoke_test:
        assert kappa.array.shape[-1] == 12 * field.nside * field.nside
        print(
            f"[smoke-test] PASS — sharded load + distributed Born across {jax.device_count()} devices.",
            flush=True,
        )
    else:
        out = Path(args.out) if args.out else HERE / "kappa_born.parquet"
        # to_parquet is collective (process_allgather); only rank 0 actually writes the file. Guard the
        # stat/print so non-lead ranks don't stat a file they never wrote (FileNotFoundError).
        Catalog(field=kappa, cosmology=cosmo).to_parquet(str(out))
        if lead:
            print(
                f"   wrote {out}  ({out.stat().st_size / 1e9:.2f} GB). Publish with: python publish_local.py",
                flush=True,
            )

    if _is_multiprocess():
        sync_global_devices("born_kappa done")
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
