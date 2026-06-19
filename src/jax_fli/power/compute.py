"""Backward-compatibility shim for :mod:`jax_fli.summary_statistics.compute`."""

from ..summary_statistics.compute import (
    angular_cl_flat,
    angular_cl_spherical,
    coherence,
    cross_angular_cl_spherical,
    deconvolve_spherical,
    power,
    transfer,
)

__all__ = [
    "power",
    "transfer",
    "coherence",
    "angular_cl_flat",
    "angular_cl_spherical",
    "cross_angular_cl_spherical",
    "deconvolve_spherical",
]
