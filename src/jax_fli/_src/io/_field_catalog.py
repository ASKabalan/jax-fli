"""Field-specific catalog serialization (HuggingFace parquet backend)."""

from __future__ import annotations

from functools import partial

import jax
import jax_cosmo as jc
import numpy as np
from jax.experimental.multihost_utils import process_allgather
from jaxtyping import Array

from ...fields import DensityField, FlatDensity, ParticleField, SphericalDensity
from ...fields.lensing_maps import FlatKappaField, SphericalKappaField
from ..base._core import AbstractField
from ..base._enums import ConvergenceUnit, DensityUnit, FieldStatus, PositionUnit

CATALOG_VERSION = 2

all_gather = partial(process_allgather, tiled=True)


def _ensure_batch_dim(array: np.ndarray, field_type: str) -> np.ndarray:
    """Ensure array has batch dimension."""
    if field_type in ("SphericalDensity", "SphericalKappaField"):
        if array.ndim == 1:
            return array[None, ...]
        elif array.ndim != 2:
            raise ValueError(f"Unexpected shape for {field_type}: {array.shape}")
    elif field_type in ("FlatDensity", "FlatKappaField"):
        if array.ndim == 2:
            return array[None, ...]
        elif array.ndim != 3:
            raise ValueError(f"Unexpected shape for FlatDensity: {array.shape}")
    elif field_type == "DensityField":
        if array.ndim == 3:
            return array[None, ...]
        elif array.ndim != 4:
            raise ValueError(f"Unexpected shape for DensityField: {array.shape}")
    elif field_type == "ParticleField":
        if array.ndim == 4:
            return array[None, ...]
        elif array.ndim != 5:
            raise ValueError(f"Unexpected shape for ParticleField: {array.shape}")
    else:
        raise ValueError(f"Unknown field type: {field_type}")
    return array


def _ensure_1d_metadata(value, name: str, batch_size: int) -> np.ndarray:
    """Normalize dynamic metadata to a 1-D array of length ``batch_size``."""
    if value is None:
        raise ValueError(f"Dynamic metadata '{name}' cannot be None.")
    arr = np.asarray(value)
    if arr.ndim == 0:
        arr = arr[None]
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.squeeze(axis=1)
    if arr.ndim != 1:
        raise ValueError(f"Dynamic metadata '{name}' must be 0-d or 1-d, got shape {arr.shape}.")
    # When batch_size == 1 (e.g. LPT lightcone: single non-batched field whose metadata
    # carries per-shell values), we relax the length check so the catalog can store
    # richer provenance without requiring the metadata to be collapsed to a scalar.
    if arr.shape[0] != batch_size and batch_size != 1:
        raise ValueError(f"Dynamic metadata '{name}' length {arr.shape[0]} != batch_size {batch_size}.")
    return arr


def build_features(field: AbstractField):
    """Build HuggingFace Features schema for v2 format (batched arrays per row)."""
    from datasets import Array2D, Array3D, Array4D, Array5D, Features, Sequence, Value

    field_type = type(field).__name__
    array = _ensure_batch_dim(field.array, field_type)
    element_shape = array.shape[1:]
    dtype_str = np.dtype(array.dtype).name

    if field_type in ("SphericalDensity", "SphericalKappaField"):
        array_feature = Array2D(shape=(None, element_shape[0]), dtype=dtype_str)
    elif field_type in ("FlatDensity", "FlatKappaField"):
        array_feature = Array3D(shape=(None, *element_shape), dtype=dtype_str)
    elif field_type == "DensityField":
        array_feature = Array4D(shape=(None, *element_shape), dtype=dtype_str)
    elif field_type == "ParticleField":
        array_feature = Array5D(shape=(None, *element_shape), dtype=dtype_str)
    else:
        raise ValueError(f"Unknown field type: {field_type}")

    feature_dict = {
        "array": array_feature,
        "z_sources": Sequence(Value("float64")),
        "scale_factors": Sequence(Value("float64")),
        "comoving_centers": Sequence(Value("float64")),
        "density_width": Sequence(Value("float64")),
        "mesh_size": Sequence(Value("int32"), length=3),
        "box_size": Sequence(Value("float32"), length=3),
        "observer_position": Sequence(Value("float32"), length=3),
        "halo_size": Sequence(Value("int32"), length=2),
        "status": Value("string"),
        "unit": Value("string"),
        "name": Value("string"),
        "entry_type": Value("string"),
        "Omega_c": Value("float32"),
        "Omega_b": Value("float32"),
        "h": Value("float32"),
        "n_s": Value("float32"),
        "sigma8": Value("float32"),
        "w0": Value("float32"),
        "wa": Value("float32"),
        "Omega_k": Value("float32"),
        "Omega_nu": Value("float32"),
        "version": Value("int32"),
        "field_type": Value("string"),
    }

    if field.nside is not None:
        feature_dict["nside"] = Value("int32")
    if field.flatsky_npix is not None:
        feature_dict["flatsky_npix"] = Sequence(Value("int32"), length=2)
    if field.field_size is not None:
        feature_dict["field_size"] = Sequence(Value("float32"), length=2)

    return Features(feature_dict)


def catalog_to_row(field: AbstractField, cosmology: jc.Cosmology, version: int) -> dict | None:
    """Convert a single field + cosmology into a 1-row column-oriented dict."""
    field_type = type(field).__name__
    array = all_gather(_ensure_batch_dim(field.array, field_type))

    if jax.process_index() != 0:
        return None

    batch_size = array.shape[0]

    z_src = _ensure_1d_metadata(field.z_sources, "z_sources", batch_size)
    scale = _ensure_1d_metadata(field.scale_factors, "scale_factors", batch_size)
    comov = _ensure_1d_metadata(field.comoving_centers, "comoving_centers", batch_size)
    dw = _ensure_1d_metadata(field.density_width, "density_width", batch_size)

    static = {
        "mesh_size": list(field.mesh_size),
        "box_size": list(field.box_size),
        "observer_position": list(field.observer_position),
        "halo_size": list(field.halo_size),
        "status": field.status.name,
        "unit": field.unit.name,
        "name": field.name if field.name is not None else "",
        "entry_type": "field",
        "field_type": field_type,
        "version": int(version),
    }

    if field.nside is not None:
        static["nside"] = int(field.nside)
    if field.flatsky_npix is not None:
        static["flatsky_npix"] = list(field.flatsky_npix)
    if field.field_size is not None:
        static["field_size"] = list(field.field_size)

    cosmo = {
        "Omega_c": float(cosmology.Omega_c),
        "Omega_b": float(cosmology.Omega_b),
        "h": float(cosmology.h),
        "n_s": float(cosmology.n_s),
        "sigma8": float(cosmology.sigma8),
        "w0": float(cosmology.w0),
        "wa": float(cosmology.wa),
        "Omega_k": float(cosmology.Omega_k),
        "Omega_nu": float(cosmology.Omega_nu),
    }

    data = {
        "array": [array],
        "z_sources": [z_src.tolist()],
        "scale_factors": [scale.tolist()],
        "comoving_centers": [comov.tolist()],
        "density_width": [dw.tolist()],
    }
    for k, v in static.items():
        data[k] = [v]
    for k, v in cosmo.items():
        data[k] = [v]

    return data


def row_to_field_cosmo(item: dict, sharding=None) -> tuple[AbstractField, jc.Cosmology, int]:
    """Convert a single v2 dataset row to a (field, cosmology, version) tuple."""
    field_classes = {
        "SphericalDensity": SphericalDensity,
        "SphericalKappaField": SphericalKappaField,
        "FlatDensity": FlatDensity,
        "FlatKappaField": FlatKappaField,
        "DensityField": DensityField,
        "ParticleField": ParticleField,
    }
    field_cls = field_classes[item["field_type"]]
    if field_cls is ParticleField:
        unit = PositionUnit[item["unit"]]
    elif field_cls in (SphericalKappaField, FlatKappaField):
        unit = ConvergenceUnit[item["unit"]]
    else:
        unit = DensityUnit[item["unit"]]

    def _to_static_tuple(v, _type):
        return tuple(_type(x) for x in v)

    array = np.asarray(item["array"])
    unbatched = array.shape[0] == 1

    if unbatched:
        array = array[0]

    def _read_dynamic(key):
        val = np.asarray(item[key], dtype=np.float64)
        if unbatched:
            # Single-value metadata: collapse to scalar for true snapshots.
            # Multi-value metadata on a non-batched field (e.g. LPT lightcone
            # storing per-shell provenance) is preserved as-is.
            return val[0] if val.ndim >= 1 and val.shape[0] == 1 else val
        return val

    z_sources = _read_dynamic("z_sources")
    if not unbatched:
        z_sources = np.atleast_1d(z_sources)

    # Read name: empty string was stored for None
    raw_name = item.get("name", "")
    name = raw_name if raw_name else None

    field = field_cls(
        array=array,
        mesh_size=_to_static_tuple(item["mesh_size"], int),
        box_size=_to_static_tuple(item["box_size"], float),
        observer_position=_to_static_tuple(item["observer_position"], float),
        field_sharding=sharding,
        halo_size=_to_static_tuple(item["halo_size"], int),
        nside=int(item["nside"]) if "nside" in item and item["nside"] is not None else None,
        flatsky_npix=_to_static_tuple(item["flatsky_npix"], int)
        if "flatsky_npix" in item and item["flatsky_npix"] is not None
        else None,
        field_size=_to_static_tuple(item["field_size"], float)
        if "field_size" in item and item["field_size"] is not None
        else None,
        z_sources=z_sources,
        scale_factors=_read_dynamic("scale_factors") if "scale_factors" in item else None,
        comoving_centers=_read_dynamic("comoving_centers") if "comoving_centers" in item else None,
        density_width=_read_dynamic("density_width") if "density_width" in item else None,
        status=FieldStatus[item["status"]],
        unit=unit,
        name=name,
    )
    field = field.apply_sharding()

    cosmology = jc.Cosmology(
        Omega_c=item["Omega_c"],
        Omega_b=item["Omega_b"],
        h=item["h"],
        n_s=item["n_s"],
        sigma8=item["sigma8"],
        w0=item["w0"],
        wa=item["wa"],
        Omega_k=item["Omega_k"],
        Omega_nu=item["Omega_nu"],
    )

    version = item["version"]
    if isinstance(version, Array):
        version = int(version.squeeze())

    return field, cosmology, version
