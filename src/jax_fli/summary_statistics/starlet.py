"""Spherical starlet (isotropic undecimated wavelet) transform.

Wraps the CosmoStat C++ backend (``pycs``), which is an optional dependency with a non-trivial
build, so the public entry point is guarded by :func:`require_cosmostat`. This module is
array-only and deliberately does **not** import :mod:`jax_fli.fields` — the
``StarletCoefficients`` result object (a ``SphericalDensity`` subclass) lives in
``jax_fli.fields`` to keep the import graph acyclic.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

import numpy as np

from .._src.summary_statistics import _starlet_transform

__all__ = ["require_cosmostat", "starlet_coefficients_spherical"]

Param = ParamSpec("Param")
ReturnType = TypeVar("ReturnType")


def require_cosmostat(func: Callable[Param, ReturnType]) -> Callable[Param, ReturnType]:
    """Decorator that raises a helpful ImportError if the CosmoStat backend is unavailable.

    The spherical starlet uses ``pycs.sparsity.mrs.mrs_starlet.CMRStarlet`` from the CosmoStat
    package, which ships a C++ backend and is installed separately.
    """
    try:
        from pycs.sparsity.mrs.mrs_starlet import CMRStarlet  # noqa: F401

        return func
    except ImportError:
        pass

    @wraps(func)
    def deferred_func(*args: Param.args, **kwargs: Param.kwargs) -> ReturnType:
        msg = "Missing optional library 'cosmostat' (imports as 'pycs'). Install with: pip install jax-fli[starlet]"
        raise ImportError(msg)

    return deferred_func


@require_cosmostat
def starlet_coefficients_spherical(
    map_sphere,
    *,
    nside: int,
    nscales: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Spherical starlet transform of a single HEALPix map (concrete/NumPy, not jittable).

    Returns ``(coef, tab_norm)`` where ``coef`` has shape ``(nscales, npix)`` (one wavelet map
    per scale) and ``tab_norm`` is the per-scale normalisation (length ``nscales``).
    """
    return _starlet_transform(map_sphere, nside=nside, nscales=nscales)
