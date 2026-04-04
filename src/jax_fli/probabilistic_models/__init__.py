"""Probabilistic model utilities."""

from ..infer.sample_converter import sample2catalog
from .config import Configurations
from .forward_model import make_full_field_model

try:
    from .full_field_model import full_field_probmodel, mock_probmodel
    from .power_spec_model import make_2pt_model, pixel_window_function, powerspec_probmodel
except ImportError:

    def _sampling_required(*args, **kwargs):
        raise ImportError("Missing optional dependency 'numpyro'. Install with: pip install jax-fli[sampling]")

    full_field_probmodel = _sampling_required  # type: ignore[assignment]
    mock_probmodel = _sampling_required  # type: ignore[assignment]
    make_2pt_model = _sampling_required  # type: ignore[assignment]
    pixel_window_function = _sampling_required  # type: ignore[assignment]
    powerspec_probmodel = _sampling_required  # type: ignore[assignment]

__all__ = [
    "Configurations",
    "make_full_field_model",
    "full_field_probmodel",
    "mock_probmodel",
    "make_2pt_model",
    "pixel_window_function",
    "powerspec_probmodel",
    "sample2catalog",
]
