"""Summary statistics for cosmological maps.

Two-point statistics (power spectra, C_ell, transfer, coherence) plus higher-order,
map-based summary statistics (one-point PDF, peak counts, spherical starlet wavelets).
This package is the new home of what used to live in :mod:`jax_fli.power`, which now
re-exports from here for backward compatibility.
"""

from .binned import BinnedStatistic
from .compute import (
    angular_cl_flat,
    angular_cl_spherical,
    coherence,
    cross_angular_cl_spherical,
    deconvolve_spherical,
    power,
    transfer,
)
from .decouple import MCM, anafast_masked, compute_mcm
from .pdf import PDF, pdf_spherical
from .peak_counts import PeakCounts, peak_counts_spherical
from .power_spec import PowerSpectrum
from .starlet import require_cosmostat, starlet_coefficients_spherical
from .theory import compute_theory_cl, compute_theory_cl_for_density, tophat_z

__all__ = [
    # ---- two-point ----
    "PowerSpectrum",
    "power",
    "transfer",
    "coherence",
    "angular_cl_flat",
    "angular_cl_spherical",
    "cross_angular_cl_spherical",
    "deconvolve_spherical",
    "anafast_masked",
    "compute_mcm",
    "MCM",
    "compute_theory_cl",
    "compute_theory_cl_for_density",
    "tophat_z",
    # ---- higher-order, map-based ----
    "BinnedStatistic",
    "PDF",
    "pdf_spherical",
    "PeakCounts",
    "peak_counts_spherical",
    "require_cosmostat",
    "starlet_coefficients_spherical",
]
