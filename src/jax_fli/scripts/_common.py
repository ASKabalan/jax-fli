"""Shared helpers used across multiple fli-* CLI scripts."""

from __future__ import annotations

import math
import re
import warnings
from argparse import Namespace

import jax
from jax.sharding import AxisType, NamedSharding
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


def _build_sharding(args: Namespace):
    """Return sharding or None for single-device runs.

    Warns if the product of pdim dimensions does not match the available device count.
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

    mesh = jax.make_mesh(pdim, ("x", "y"), axis_types=(AxisType.Auto, AxisType.Auto))
    sharding = NamedSharding(mesh, P("x", "y"))
    return sharding
