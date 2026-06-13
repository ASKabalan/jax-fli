"""Data subpackage: n(z) loaders, source redshift utilities, and survey masks."""

from .apodize import apodize
from .masks import build_observer_visibility_mask, get_desy3_mask
from .metadata import _max_z_source
from .nz import get_des_y3_nz_shear, get_stage3_nz_shear, plot_nz

__all__ = [
    "_max_z_source",
    "get_stage3_nz_shear",
    "get_des_y3_nz_shear",
    "plot_nz",
    "get_desy3_mask",
    "build_observer_visibility_mask",
    "apodize",
]
