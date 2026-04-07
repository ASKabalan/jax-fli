"""Sampling utilities and distributed persistency helpers."""

from .analyze import analyze, requires_arviz
from .batched_sampling import batched_sampling, requires_samplers
from .posterior import build_mcsamples, plot_posterior, requires_getdist
from .sample_converter import default_save, sample2catalog

try:
    from .dist import DistributedIC, DistributedNormal, PreconditionnedUniform
except ImportError:

    def _sampling_required(*args, **kwargs):
        raise ImportError("Missing optional dependency 'numpyro'. Install with: pip install jax-fli[sampling]")

    DistributedIC = _sampling_required  # type: ignore[assignment]
    DistributedNormal = _sampling_required  # type: ignore[assignment]
    PreconditionnedUniform = _sampling_required  # type: ignore[assignment]

__all__ = [
    "DistributedIC",
    "DistributedNormal",
    "PreconditionnedUniform",
    "analyze",
    "batched_sampling",
    "build_mcsamples",
    "plot_posterior",
    "requires_arviz",
    "requires_samplers",
    "requires_getdist",
    "sample2catalog",
    "default_save",
]
