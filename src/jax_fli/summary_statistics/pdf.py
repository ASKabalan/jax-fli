"""One-point PDF of a HEALPix map.

``pdf_spherical`` is the array-only primitive (jittable; takes explicit bin *edges*). The
:class:`PDF` result object is built by :meth:`jax_fli.fields.SphericalDensity.compute_pdf`.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from .._src.summary_statistics import _pdf
from .binned import BinnedStatistic

__all__ = ["PDF", "pdf_spherical"]


class PDF(BinnedStatistic):
    """One-point probability density of pixel values, over a fixed bin grid."""

    kind: str = eqx.field(static=True, default="pdf")


def pdf_spherical(
    map_sphere,
    *,
    bins: jnp.ndarray,
    density: bool = True,
    soft: bool = False,
    bandwidth: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One-point PDF of a HEALPix map over the bin *edges* ``bins``. Returns ``(centers, pdf)``.

    ``density=True`` normalises to integrate to 1. ``soft=True`` evaluates a differentiable
    fixed-bandwidth KDE at the bin centers (``bandwidth`` defaults to the mean bin width).
    """
    return _pdf(map_sphere, bins=bins, density=density, soft=soft, bandwidth=bandwidth)
