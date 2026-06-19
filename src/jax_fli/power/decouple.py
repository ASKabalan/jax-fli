"""Backward-compatibility shim for :mod:`jax_fli.summary_statistics.decouple`."""

from ..summary_statistics.decouple import MCM, anafast_masked, compute_mcm

__all__ = ["MCM", "anafast_masked", "compute_mcm"]
