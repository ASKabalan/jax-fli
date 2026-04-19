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

__all__ = ["_try_parse_s3", "_try_parse_des_y3", "_resolve_nz_shear", "_build_sharding", "_save_args_log"]


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


def _save_args_log(args: Namespace, output_dir: str, prog: str) -> None:
    """Write a formatted args summary to {output_dir}/args.log."""
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
    log_path = os.path.join(output_dir, "args.log")
    with open(log_path, "w") as f:
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
