"""Backward-compatibility shim.

The power-spectrum code moved to :mod:`jax_fli.summary_statistics` (which also hosts the
new higher-order map statistics). This module re-exports the original public API so that
``from jax_fli.power import ...`` (and the submodules ``power.compute``, ``power.decouple``,
``power.power_spec``, ``power.theory``) keep working. New code should import from
:mod:`jax_fli.summary_statistics`.
"""

from ..summary_statistics import (
    MCM,
    PowerSpectrum,
    anafast_masked,
    angular_cl_flat,
    angular_cl_spherical,
    coherence,
    compute_mcm,
    compute_theory_cl,
    compute_theory_cl_for_density,
    cross_angular_cl_spherical,
    deconvolve_spherical,
    power,
    tophat_z,
    transfer,
)

__all__ = [
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
]
