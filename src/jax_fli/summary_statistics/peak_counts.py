"""Peak counts (local-maxima histogram) of a HEALPix map.

``peak_counts_spherical`` is the array-only primitive (jittable; takes explicit bin *edges*).
The :class:`PeakCounts` result object is built by
:meth:`jax_fli.fields.SphericalDensity.compute_peak_counts`.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .._src.summary_statistics import _peak_counts
from .binned import BinnedStatistic

__all__ = ["PeakCounts", "peak_counts_spherical"]


class PeakCounts(BinnedStatistic):
    """Histogram of local-maxima heights of a HEALPix map, over a fixed bin grid."""

    kind: str = eqx.field(static=True, default="peak_counts")


@jax.jit(static_argnames=["nside", "normalize", "soft", "bandwidth"])
def peak_counts_spherical(
    map_sphere,
    *,
    nside: int,
    bins: jnp.ndarray,
    normalize: bool = False,
    soft: bool = False,
    bandwidth: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Peak counts of a HEALPix map over the bin *edges* ``bins``. Returns ``(centers, counts)``.

    A pixel is a peak iff it exceeds all of its HEALPix neighbours. ``normalize=True`` converts
    the map to S/N (``(m - mean) / std``) before detection and binning. ``soft=True`` uses a
    differentiable Gaussian soft-binning of the peak heights (detection stays hard). Assumes
    RING ordering.
    """
    return _peak_counts(map_sphere, nside=nside, bins=bins, normalize=normalize, soft=soft, bandwidth=bandwidth)
