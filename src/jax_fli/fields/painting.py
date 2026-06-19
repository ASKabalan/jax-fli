from __future__ import annotations

from typing import Literal

import equinox as eqx
from jaxtyping import Array

__all__ = ["PaintingOptions"]

SphericalScheme = Literal["ngp", "bilinear", "rbf_neighbor"]
PaintingTarget = Literal["spherical", "flat", "density", "particles"]


class PaintingOptions(eqx.Module):
    """
    Configuration for painting in nbody/lpt pipelines.

    This frozen configuration class specifies what type of output to produce
    and the parameters for painting particles onto density grids.

    Parameters
    ----------
    target : {"spherical", "flat", "density", "particles"}
        Output type:
        - "spherical": HEALPix maps via paint_spherical()
        - "flat": Flat-sky 2D grids via paint_2d()
        - "density": 3D density fields via paint()
        - "particles": Return particle displacements directly (no painting)

    scheme : {"ngp", "bilinear", "rbf_neighbor"}, default="bilinear"
        Interpolation scheme for spherical painting.

    kernel_width_arcmin : float or None, default=None
        Smoothing kernel width in arcminutes for spherical painting.

    kernel_width_pixels : float or None, default=None
        Smoothing kernel width in HEALPix pixels (e.g. 0.5 = half a pixel) for
        spherical painting. Mutually exclusive with ``kernel_width_arcmin``.

    smoothing_interpretation : str, default="fwhm"
        How to interpret kernel_width_arcmin ("fwhm" or "sigma").

    paint_nside : int or None, default=None
        Resolution for spherical painting (if different from field nside).

    ud_grade_power : float, default=0.0
        Power for ud_grade operation in spherical painting.

    ud_grade_order_in : str, default="RING"
        Input ordering for ud_grade.

    ud_grade_order_out : str, default="RING"
        Output ordering for ud_grade.

    ud_grade_pess : bool, default=False
        Pessimistic ud_grade mode.

    order : int or str, default="cic"
        Mass-assignment order for 3D ("density") painting (NGP=1, CIC=2, TSC=3, PCS=4).

    deconvolution : bool, default=False
        Deconvolve the assignment window from the painted 3D ("density") field in
        Fourier space. Sharpens the field and can go NEGATIVE (Gibbs ringing) --
        intended for Fourier / P(k) use, not a density to visualise. ``target="density"``
        only; never used for force computation.

    pixel_window_deconvolution : bool, default=False
        Deconvolve the HEALPix assignment window from the painted ``target="spherical"`` map at
        the a_lm level (one factor of W_l: the pixel window for ``scheme="ngp"``, pixel window x
        Gaussian beam for ``"rbf_neighbor"``) via ``SphericalDensity.deconvolve``. Distinct from
        ``deconvolution`` above (the 3D force/density window). Requires ``scheme`` in
        {"ngp", "rbf_neighbor"}; applied at the ``lpt()`` / ``nbody()`` spherical output.

    weights : array, float, or None, default=None
        Weights for painting.

    batch_size : int or None, default=None
        Batch size for painting operations.

    Examples
    --------
    >>> # Spherical painting with bilinear interpolation
    >>> painting = PaintingOptions(target="spherical", scheme="bilinear")
    >>> lightcone = lpt(cosmo, field, ts=a_centers, painting=painting)

    >>> # Flat-sky painting
    >>> painting = PaintingOptions(target="flat")
    >>> lightcone = nbody(cosmo, dx, p, nb_shells=10, painting=painting)

    >>> # Return particles directly (default behavior when painting=None)
    >>> dx, p = lpt(cosmo, field, ts=0.1)
    """

    # Output target
    target: PaintingTarget = eqx.field(static=True)

    # Spherical-specific
    scheme: SphericalScheme = eqx.field(static=True, default="bilinear")
    kernel_width_arcmin: float | None = eqx.field(static=True, default=None)
    kernel_width_pixels: float | None = eqx.field(static=True, default=None)
    smoothing_interpretation: str = eqx.field(static=True, default="fwhm")
    paint_nside: int | None = eqx.field(static=True, default=None)
    ud_grade_power: float = eqx.field(static=True, default=0.0)
    ud_grade_order_in: str = eqx.field(static=True, default="RING")
    ud_grade_order_out: str = eqx.field(static=True, default="RING")
    ud_grade_pess: bool = eqx.field(static=True, default=False)

    # 3D paint specific
    order: str = eqx.field(static=True, default="cic")
    deconvolution: bool = eqx.field(static=True, default=False)

    # Spherical post-paint window deconvolution
    pixel_window_deconvolution: bool = eqx.field(static=True, default=False)

    # Shared
    weights: Array | float | None = 1.0
    batch_size: int | None = eqx.field(static=True, default=None)

    def __post_init__(self):
        # The post-paint spherical deconvolution removes the HEALPix assignment window at the
        # a_lm level; "bilinear" painting has no closed-form window (see SphericalDensity.deconvolve).
        if (
            self.pixel_window_deconvolution
            and self.target == "spherical"
            and self.scheme not in ("ngp", "rbf_neighbor")
        ):
            raise ValueError(
                "pixel_window_deconvolution requires scheme 'ngp' or 'rbf_neighbor' "
                f"('{self.scheme}' has no closed-form HEALPix pixel window)."
            )
