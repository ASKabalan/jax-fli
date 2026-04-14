"""Data subpackage: n(z) loaders and source redshift utilities."""

from .metadata import _max_z_source
from .nz import get_des_y3_nz_shear, get_stage3_nz_shear, plot_nz

__all__ = ["_max_z_source", "get_stage3_nz_shear", "get_des_y3_nz_shear", "plot_nz"]
