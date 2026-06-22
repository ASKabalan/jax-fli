"""Shared helpers used across multiple fli-* CLI scripts."""

from __future__ import annotations

import math
import os
import re
import warnings
from argparse import Namespace

import jax
from jax.experimental.mesh_utils import create_hybrid_device_mesh
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import jax_fli as jfli

__all__ = [
    "_try_parse_s3",
    "_try_parse_des_y3",
    "_resolve_nz_shear",
    "_resolve_solver_name",
    "_resolve_mask",
    "_resolve_summary_stats_mask",
    "_build_sharding",
    "_save_args_log",
    "_load_lightcone",
    "_resolve_source",
    "_resolve_chain_sources",
]

# CLI --solver token -> Configurations.nbody_solver name (forward_model._SOLVERS keys).
_SOLVER_NAMES = {"kdk": "DoubleKickDrift", "dkd": "DriftKickDrift", "bf": "BullFrog"}


def _resolve_solver_name(solver: str) -> str:
    """Map the CLI ``--solver`` token (kdk/dkd/bf) to ``Configurations.nbody_solver``."""
    try:
        return _SOLVER_NAMES[solver]
    except KeyError as exc:
        raise ValueError(f"Unknown --solver '{solver}'; expected one of {tuple(_SOLVER_NAMES)}.") from exc


def _load_healpix_mask(path: str, nside):
    """Load a HEALPix mask from a ``.npy`` / ``.npz`` / ``.fits`` file.

    For spherical geometry (``nside`` given) the map is ud_graded to the model nside.
    """
    import numpy as np

    if path.endswith(".npy"):
        arr = np.load(path)
    elif path.endswith(".npz"):
        with np.load(path) as npz:
            arr = npz[npz.files[0]]
    elif path.endswith((".fits", ".fits.gz")):
        import healpy as hp

        arr = hp.read_map(path)
    else:
        raise ValueError(f"mask must be 'des_y3' or a .npy/.npz/.fits path, got {path!r}")

    arr = np.asarray(arr)
    if nside is not None:  # spherical geometry: match the model nside
        import healpy as hp

        if hp.npix2nside(arr.shape[-1]) != nside:
            arr = hp.ud_grade(arr, nside)
    return arr


def _resolve_mask(mask_arg, nside):
    """Resolve the forward-model ``--mask`` into a HEALPix survey footprint array (or None).

    Accepts ``None``, the ``des_y3`` keyword (``jfli.data.get_desy3_mask`` at the model
    nside, mirroring ``--nz-shear des_y3``), or a path to a ``.npy`` / ``.npz`` / ``.fits``
    map (ud_graded to the model nside for spherical geometry).
    """
    if mask_arg is None:
        return None
    if mask_arg.lower() in ("des_y3", "desy3"):
        if nside is None:
            raise ValueError("--mask des_y3 requires spherical geometry (a model nside).")
        return jfli.data.get_desy3_mask(nside)
    return _load_healpix_mask(mask_arg, nside)


def _resolve_summary_stats_mask(mask_arg, nside, observer_position, apodization_scale_deg: float = 1.0):
    """Resolve the fli-summary-stats ``--mask`` into an apodized HEALPix footprint (or None).

    Spherical geometry only. Accepts:

    * ``none`` / ``None`` → no mask;
    * ``infer_from_observer_position`` → the apodized observer-visibility mask built from
      ``observer_position`` (a centered observer sees the whole sky, so this returns ``None``);
    * ``des_y3`` → the DES Y3 footprint, apodized with a C2 window;
    * a ``.npy`` / ``.npz`` / ``.fits`` path → loaded, ud_graded, and apodized.
    """
    if mask_arg is None:
        return None
    key = mask_arg.lower()
    if key == "none":
        return None
    if key in ("infer_from_observer_position", "infer", "observer"):
        mask = jfli.data.build_observer_visibility_mask(tuple(observer_position), nside, apodization_scale_deg)
        import numpy as np

        # build_observer_visibility_mask returns the scalar 1 for a centered observer (whole sky);
        # treat that as no footprint restriction.
        return None if np.ndim(mask) == 0 else mask
    if nside is None:
        raise ValueError("--mask requires spherical geometry (a HEALPix nside).")
    if key in ("des_y3", "desy3"):
        binary = jfli.data.get_desy3_mask(nside)
    else:
        binary = _load_healpix_mask(mask_arg, nside)
    return jfli.data.apodize(binary, apodization_scale_deg)


# ---------------------------------------------------------------------------
# nz_shear resolver
# ---------------------------------------------------------------------------


def _try_parse_s3(token: str):
    """Parse s3/stage3 with an optional bin selector. Returns None if token is not s3.

    Supported forms:
      s3            → all 4 Stage-3 bins
      s3[0]         → first bin only (wrapped in a list)
      s3[1:3]       → bins 1 and 2
      s3[:2]        → first two bins
      s3[::2]       → every other bin
    """
    m = re.fullmatch(r"(?:stage3|s3)(?:\[([^\]]*)\])?", token, re.IGNORECASE)
    if m is None:
        return None
    distributions = jfli.io.get_stage3_nz_shear()
    selector = m.group(1)
    if selector is None:
        return distributions
    # Integer index → wrap in list for uniform downstream handling
    if re.fullmatch(r"-?\d+", selector):
        return [distributions[int(selector)]]
    # Slice notation  start:stop  or  start:stop:step
    parts = selector.split(":")
    if 2 <= len(parts) <= 3:
        opt = lambda s: int(s) if s else None  # noqa: E731
        slc = slice(opt(parts[0]), opt(parts[1]), opt(parts[2]) if len(parts) == 3 else None)
        return distributions[slc]
    raise ValueError(f"Cannot parse s3 selector '[{selector}]'. Use s3[i] or s3[start:stop[:step]].")


def _try_parse_des_y3(token: str):
    """Parse des_y3 with an optional bin selector. Returns None if token is not des_y3.

    Supported forms:
      des_y3            → all 4 DES Y3 bins
      des_y3[0]         → first bin only (wrapped in a list)
      des_y3[1:3]       → bins 1 and 2
      des_y3[:2]        → first two bins
      des_y3[::2]       → every other bin
    """
    m = re.fullmatch(r"(?:des_y3|desy3)(?:\[([^\]]*)\])?", token, re.IGNORECASE)
    if m is None:
        return None
    distributions = jfli.io.get_des_y3_nz_shear()
    selector = m.group(1)
    if selector is None:
        return distributions
    # Integer index → wrap in list for uniform downstream handling
    if re.fullmatch(r"-?\d+", selector):
        return [distributions[int(selector)]]
    # Slice notation  start:stop  or  start:stop:step
    parts = selector.split(":")
    if 2 <= len(parts) <= 3:
        opt = lambda s: int(s) if s else None  # noqa: E731
        slc = slice(opt(parts[0]), opt(parts[1]), opt(parts[2]) if len(parts) == 3 else None)
        return distributions[slc]
    raise ValueError(f"Cannot parse des_y3 selector '[{selector}]'. Use des_y3[i] or des_y3[start:stop[:step]].")


def _resolve_nz_shear(args: Namespace):
    """Return nz_shear list from CLI --nz-shear values."""
    nz_shear = getattr(args, "nz_shear", None)
    if nz_shear is None:
        return None
    values = nz_shear
    if len(values) == 1:
        s3 = _try_parse_s3(values[0])
        if s3 is not None:
            return s3
        des_y3 = _try_parse_des_y3(values[0])
        if des_y3 is not None:
            return des_y3
    # Otherwise parse as floats
    try:
        import jax.numpy as jnp

        return jnp.array(values, dtype=jnp.float32)
    except ValueError as exc:
        raise ValueError(
            f"--nz-shear values must be floats or s3/s3[i]/s3[start:stop] or des_y3/des_y3[i]/des_y3[start:stop]: {values}"
        ) from exc


# ---------------------------------------------------------------------------
# Sharding setup
# ---------------------------------------------------------------------------


def _save_args_log(args: Namespace, output_dir: str, prog: str, filename: str = "args.log", mode: str = "w") -> None:
    """Write a formatted args summary to ``{output_dir}/{filename}``.

    ``filename`` lets callers give the log a per-run name (e.g. derived from the output file) so
    concurrent runs sharing a directory don't clobber each other; ``mode="a"`` appends instead of
    overwriting, e.g. to sit after the fli-launcher command already written to the same file.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    width = 60
    lines = ["=" * width, f"  {prog}", "=" * width]
    skip = {"func", "subcommand"}
    for key, val in sorted(vars(args).items()):
        if key in skip:
            continue
        lines.append(f"  {key:<30} {val}")
    lines.append("=" * width)
    log_path = os.path.join(output_dir, filename)
    with open(log_path, mode) as f:
        f.write("\n".join(lines) + "\n")
    print(f"Args saved to {log_path}")


def _resolve_gpus_per_node(args: Namespace) -> int:
    """GPUs per node for the hybrid mesh: ``--gpus-per-node``, else ``$SLURM_GPUS_ON_NODE``."""
    if getattr(args, "gpus_per_node", None) is not None:
        return int(args.gpus_per_node)
    env = os.environ.get("SLURM_GPUS_ON_NODE")
    if env is None:
        raise RuntimeError(
            "Hybrid (multi-bandwidth) mesh detected but GPUs-per-node is unknown: pass "
            "--gpus-per-node or set $SLURM_GPUS_ON_NODE."
        )
    return int(env)


def _build_sharding(args: Namespace):
    """Return sharding or None for single-device runs.

    Warns if the product of pdim dimensions does not match the available device count. On a
    non-uniform interconnect (NVLink intra-node + InfiniBand inter-node, detected via the
    ``slice_index`` attribute on the devices) a hybrid mesh is built so the fast NVLink axis
    stays inside each node; otherwise a plain uniform-bandwidth mesh is used.
    """
    print(f"jax devices: {jax.devices()}")
    pdim = tuple(args.pdim)

    n_devices = jax.device_count()
    if math.prod(pdim) != n_devices:
        warnings.warn(
            f"--pdim {pdim} implies {math.prod(pdim)} devices but jax.device_count() == {n_devices}. "
            "Results may be incorrect on a misconfigured device mesh.",
            stacklevel=2,
        )

    if pdim == (1, 1):
        return None

    if not hasattr(jax.devices()[0], "slice_index"):
        # Uniform interconnect (single node / NVLink only).
        mesh = jax.make_mesh(pdim, ("x", "y"), axis_types=(AxisType.Auto, AxisType.Auto))
    else:
        # Non-uniform interconnect (NVLink intra-node + InfiniBand inter-node): hybrid mesh.
        gpus_per_node = _resolve_gpus_per_node(args)
        if gpus_per_node <= 0 or pdim[0] % gpus_per_node != 0:
            raise ValueError(
                f"--pdim {pdim} with gpus_per_node={gpus_per_node}: the first pdim axis "
                f"({pdim[0]}) must be a positive multiple of gpus_per_node for the hybrid mesh."
            )
        mesh_shape = (gpus_per_node, 1)
        dcn_shape = (pdim[0] // gpus_per_node, pdim[1])
        mesh = Mesh(create_hybrid_device_mesh(mesh_shape, dcn_shape), axis_names=("x", "y"))
    sharding = NamedSharding(mesh, P("x", "y"))
    return sharding


# ---------------------------------------------------------------------------
# Density lightcone loading (add_source_args: local glob or HuggingFace repo)
# ---------------------------------------------------------------------------


def _load_lightcone(args: Namespace, *, sharding=None):
    """Load and stack the density shells named by ``add_source_args`` into one lightcone.

    The source is EITHER a local parquet glob (``--input``) OR a HuggingFace repo (``--repo`` +
    ``--data-files``), streamed row by row. On the cluster the HF cache is pre-warmed out of band, so
    this just calls ``load_dataset(..., streaming=True)`` — no ``snapshot_download``. Each shell is
    ud_grade-downsampled to ``--nside`` when that is smaller than native, the shells are ordered
    nearest→farthest by comoving distance, and stacked into one ``(S, npix)`` ``SphericalDensity``. A
    single row that is already a stacked ``(S, npix)`` lightcone is used as-is. ``sharding`` (Born
    only) distributes the maps: native shells are sharded on load, a downsampled stack is sharded once
    after stacking (ud_grade on a ``P("x")``-sharded shell is unsafe).

    Returns ``(lightcone, cosmology)``.
    """
    from datasets import load_dataset

    from jax_fli.io import Catalog

    if args.input:
        if args.repo or args.data_files:
            raise ValueError("Pass EITHER --input (local glob) OR --repo/--data-files (HuggingFace), not both.")
        ds = load_dataset("parquet", data_files=args.input, split="train", streaming=True)
    elif args.repo and args.data_files:
        ds = load_dataset(args.repo, data_files=args.data_files, split="train", streaming=True)
    else:
        raise ValueError("No source: pass --input (local glob) or --repo + --data-files (HuggingFace).")

    nside = args.nside
    fields, cosmo, downsample = [], None, None
    for row in ds.with_format("numpy"):
        if downsample is None:  # the first shell's native nside decides downsample-vs-native
            cat = Catalog.from_dataset(row)
            fld = cat.field[0]
            downsample = nside is not None and int(nside) < int(fld.nside)
            if downsample:
                fld = fld.ud_sample(int(nside))
            elif sharding is not None:
                fld = fld.replace(field_sharding=sharding).apply_sharding()
        elif downsample:
            cat = Catalog.from_dataset(row)
            fld = cat.field[0].ud_sample(int(nside))
        else:
            cat = Catalog.from_dataset(row, sharding=sharding)
            fld = cat.field[0]
        fields.append(fld)
        cosmo = cat.cosmology[0]

    if not fields:
        raise ValueError("No density shells found in the source.")
    if not isinstance(fields[0], jfli.SphericalDensity):
        raise TypeError(f"Expected SphericalDensity shells, got {type(fields[0]).__name__}.")

    if len(fields) == 1 and fields[0].array.ndim >= 2:
        lightcone = fields[0]  # already a stacked (S, npix) lightcone in one row
    else:
        # nearest→farthest: order-invariant for Born, the required radial order for ray-tracing.
        fields.sort(key=lambda f: float(f.comoving_centers))
        lightcone = jfli.SphericalDensity.stack(fields)

    if downsample and sharding is not None:
        lightcone = lightcone.replace(field_sharding=sharding).apply_sharding()
    return lightcone, cosmo


def _resolve_source(args: Namespace, *, prefix: str = ""):
    """Return ONE streamed ``datasets`` dataset from a single ``add_source_args`` source.

    Reads the (optionally prefixed) ``input`` XOR ``repo`` + ``data_files`` attributes — the same
    branching as :func:`_load_lightcone`, factored out for the single-row consumers (``fli-infer``
    observable and initial condition). ``prefix="ic"`` reads ``ic_input`` / ``ic_repo`` /
    ``ic_data_files``. Streams via ``load_dataset(..., streaming=True)`` (the cluster HF cache is
    pre-warmed out of band, so no ``snapshot_download``).
    """
    from datasets import load_dataset

    under = f"{prefix}_" if prefix else ""
    label = f"--{prefix}-" if prefix else "--"
    input_ = getattr(args, f"{under}input", None)
    repo = getattr(args, f"{under}repo", None)
    data_files = getattr(args, f"{under}data_files", None)

    if input_:
        if repo or data_files:
            raise ValueError(f"Pass EITHER {label}input OR {label}repo/{label}data-files, not both.")
        return load_dataset("parquet", data_files=input_, split="train", streaming=True)
    if repo and data_files:
        return load_dataset(repo, data_files=data_files, split="train", streaming=True)
    raise ValueError(f"No source: pass {label}input (local glob) or {label}repo + {label}data-files (HuggingFace).")


def _resolve_chain_sources(args: Namespace):
    """Resolve a multi-pattern ``add_source_args(multi=True)`` source into ``(repo, patterns)``.

    ``patterns`` is the list of per-chain parquet sources (``fli-extract`` opens one streamed
    ``load_dataset`` per pattern). EITHER local ``--input`` (one or more globs/dirs; ``repo=None``) OR
    HuggingFace ``--repo`` + ``--data-files`` (one or more globs inside the repo); the single-pattern
    forms are just a one-element list. Directory expansion (a local root holding ``chain_N/``) happens
    downstream in :func:`jax_fli.io.extract.extract_catalog`.
    """
    input_ = getattr(args, "input", None)
    repo = getattr(args, "repo", None)
    data_files = getattr(args, "data_files", None)

    if input_:
        if repo or data_files:
            raise ValueError("Pass EITHER --input (local) OR --repo/--data-files (HuggingFace), not both.")
        return None, list(input_)
    if repo and data_files:
        return repo, list(data_files)
    raise ValueError("No source: pass --input (local glob[s]/dir) or --repo + --data-files (HuggingFace).")
