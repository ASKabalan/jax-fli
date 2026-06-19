"""Backward-compatibility shim for :mod:`jax_fli.summary_statistics.theory`."""

from ..summary_statistics.theory import compute_theory_cl, compute_theory_cl_for_density, tophat_z

__all__ = ["compute_theory_cl", "compute_theory_cl_for_density", "tophat_z"]
