"""Catalog serialization for the higher-order map statistics (HuggingFace parquet backend).

Handles the new summary-statistic result objects exactly like
:mod:`jax_fli._src.io._power_spec_catalog` handles ``PowerSpectrum`` — as a dedicated, self-contained
backend, so the tested field/PowerSpectrum serializers are left untouched:

- ``PDF`` / ``PeakCounts`` (``BinnedStatistic``): a 1-D ``bins`` grid + ``array``.
- ``StarletCoefficients`` (a ``SphericalDensity``): per-scale maps ``(nscales, npix)`` + ``tab_norm``.

One row per entry. ``entry_type`` (the class name) drives reconstruction; ``nscales`` is recovered
from ``array.shape[0]`` and is not stored. Field metadata is stored with ``has_*`` flags (like the
PowerSpectrum backend) so ``None`` / scalar / per-shell values all round-trip without the rigid
batch-length checks of the field backend.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
from jax.experimental.multihost_utils import process_allgather

from ...summary_statistics.binned import BinnedStatistic
from ..base._enums import ConvergenceUnit, DensityUnit, FieldStatus, PhysicalUnit, PositionUnit, SpectralUnit

# Matches CATALOG_VERSION / PS_CATALOG_VERSION (= 2, the v2 catalog format) so a default-constructed
# Catalog round-trips its static `version` field losslessly.
SUMMARY_CATALOG_VERSION = 2

# entry_type -> class. Imported lazily in the reconstruction map to avoid import cost at module load.
_ENTRY_TYPES = ("PDF", "PeakCounts", "StarletCoefficients")


def is_summary_stat(obj) -> bool:
    """True for objects handled by this backend (BinnedStatistic or StarletCoefficients)."""
    from ...fields import StarletCoefficients

    return isinstance(obj, BinnedStatistic | StarletCoefficients)


def _summary_classes() -> dict:
    from ...fields import StarletCoefficients
    from ...summary_statistics.pdf import PDF
    from ...summary_statistics.peak_counts import PeakCounts

    return {"PDF": PDF, "PeakCounts": PeakCounts, "StarletCoefficients": StarletCoefficients}


def _resolve_unit(name: str):
    """Resolve a stored unit name across every unit enum (units differ by source field)."""
    for enum in (DensityUnit, ConvergenceUnit, SpectralUnit, PositionUnit, PhysicalUnit):
        try:
            return enum[name]
        except (KeyError, ValueError):
            continue
    return PhysicalUnit.INVALID_UNIT


def build_summary_features(obj):
    """HuggingFace Features schema for a summary-statistic catalog row."""
    from datasets import Features, Sequence, Value

    array = np.asarray(obj.array)
    dtype_str = np.dtype(array.dtype).name

    feature_dict = {
        # Stored flat (+ shape) as a 1-D Sequence: the datasets NumpyFormatter silently downcasts a
        # multi-dim Array2D float64 to float32 on read, whereas a 1-D Sequence preserves the dtype.
        "array": Sequence(Value(dtype_str)),
        "array_shape": Sequence(Value("int64")),
        "entry_type": Value("string"),
        "name": Value("string"),
        # type-specific 1-D vectors (one is populated, the other is empty)
        "bins": Sequence(Value("float64")),
        "tab_norm": Sequence(Value("float64")),
        # AbstractField geometry / metadata (has_* flags like the PowerSpectrum backend)
        "mesh_size_0": Value("int32"),
        "mesh_size_1": Value("int32"),
        "mesh_size_2": Value("int32"),
        "box_size_0": Value("float64"),
        "box_size_1": Value("float64"),
        "box_size_2": Value("float64"),
        "nside": Value("int32"),
        "flatsky_npix_0": Value("int32"),
        "flatsky_npix_1": Value("int32"),
        "field_size": Value("float64"),
        "has_scale_factors": Value("bool"),
        "scale_factors": Sequence(Value("float64")),
        "has_comoving_centers": Value("bool"),
        "comoving_centers": Sequence(Value("float64")),
        "has_density_width": Value("bool"),
        "density_width": Sequence(Value("float64")),
        "has_z_sources": Value("bool"),
        "z_sources": Sequence(Value("float64")),
        "field_status": Value("string"),
        "unit": Value("string"),
        # Cosmology
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
    }
    return Features(feature_dict)


def summary_to_row(obj, cosmology: jc.Cosmology, version: int) -> dict | None:
    """Convert one summary-statistic object + cosmology into a 1-row column dict."""
    array = np.asarray(process_allgather(np.asarray(obj.array), tiled=True))
    bins_arr = getattr(obj, "bins", None)
    tab_norm_arr = getattr(obj, "tab_norm", None)

    if jax.process_index() != 0:
        return None

    entry_type = type(obj).__name__

    def _flat_list(v):
        return [] if v is None else np.asarray(v).ravel().tolist()

    mesh = obj.mesh_size if obj.mesh_size is not None else (0, 0, 0)
    box = obj.box_size if obj.box_size is not None else (0.0, 0.0, 0.0)
    fp = obj.flatsky_npix

    data = {
        "array": [np.asarray(array).ravel()],
        "array_shape": [list(np.asarray(array).shape)],
        "entry_type": [entry_type],
        "name": [obj.name if obj.name is not None else ""],
        "bins": [_flat_list(bins_arr)],
        "tab_norm": [_flat_list(tab_norm_arr)],
        "mesh_size_0": [int(mesh[0])],
        "mesh_size_1": [int(mesh[1])],
        "mesh_size_2": [int(mesh[2])],
        "box_size_0": [float(box[0])],
        "box_size_1": [float(box[1])],
        "box_size_2": [float(box[2])],
        "nside": [int(obj.nside) if obj.nside is not None else -1],
        "flatsky_npix_0": [int(fp[0]) if fp is not None else -1],
        "flatsky_npix_1": [int(fp[1]) if fp is not None else -1],
        "field_size": [float(obj.field_size) if obj.field_size is not None else -1.0],
        "has_scale_factors": [obj.scale_factors is not None],
        "scale_factors": [_flat_list(obj.scale_factors)],
        "has_comoving_centers": [obj.comoving_centers is not None],
        "comoving_centers": [_flat_list(obj.comoving_centers)],
        "has_density_width": [obj.density_width is not None],
        "density_width": [_flat_list(obj.density_width)],
        "has_z_sources": [obj.z_sources is not None],
        "z_sources": [_flat_list(obj.z_sources)],
        "field_status": [obj.status.name if obj.status is not None else FieldStatus.SPECTRA.name],
        "unit": [obj.unit.name if obj.unit is not None else PhysicalUnit.INVALID_UNIT.name],
        "Omega_c": [float(cosmology.Omega_c)],
        "Omega_b": [float(cosmology.Omega_b)],
        "h": [float(cosmology.h)],
        "n_s": [float(cosmology.n_s)],
        "sigma8": [float(cosmology.sigma8)],
        "w0": [float(cosmology.w0)],
        "wa": [float(cosmology.wa)],
        "Omega_k": [float(cosmology.Omega_k)],
        "Omega_nu": [float(cosmology.Omega_nu)],
        "version": [int(version)],
    }
    return data


def row_to_summary_cosmo(item: dict, sharding=None) -> tuple:
    """Reconstruct a summary-statistic object + cosmology from a dataset row dict."""
    entry_type = str(np.asarray(item["entry_type"]).flat[0])
    cls = _summary_classes()[entry_type]

    shape = tuple(int(s) for s in np.asarray(item["array_shape"]).ravel())
    array = jnp.asarray(np.asarray(item["array"]).reshape(shape))

    def _opt_vector(has_key, key):
        if not bool(np.asarray(item.get(has_key, False)).flat[0]):
            return None
        vals = item.get(key, [])
        vals = np.asarray(vals, dtype=np.float64).ravel()
        return jnp.asarray(vals) if vals.size > 0 else None

    raw_name = item.get("name", "")
    name = str(np.asarray(raw_name).flat[0]) if raw_name is not None else ""
    name = name if name else None

    mesh_size = tuple(int(item.get(f"mesh_size_{i}", 0)) for i in range(3))
    box_size = tuple(float(item.get(f"box_size_{i}", 0.0)) for i in range(3))
    nside_raw = int(np.asarray(item.get("nside", -1)).flat[0])
    nside = None if nside_raw < 0 else nside_raw
    fp0 = int(np.asarray(item.get("flatsky_npix_0", -1)).flat[0])
    fp1 = int(np.asarray(item.get("flatsky_npix_1", -1)).flat[0])
    flatsky_npix = None if fp0 < 0 else (fp0, fp1)
    field_size_raw = float(np.asarray(item.get("field_size", -1.0)).flat[0])
    field_size = None if field_size_raw < 0 else field_size_raw

    scale_factors = _opt_vector("has_scale_factors", "scale_factors")
    comoving_centers = _opt_vector("has_comoving_centers", "comoving_centers")
    density_width = _opt_vector("has_density_width", "density_width")
    z_sources = _opt_vector("has_z_sources", "z_sources")

    try:
        status = FieldStatus[str(np.asarray(item.get("field_status", "SPECTRA")).flat[0])]
    except (KeyError, ValueError):
        status = FieldStatus.SPECTRA
    unit = _resolve_unit(str(np.asarray(item.get("unit", "INVALID_UNIT")).flat[0]))

    common = dict(
        array=array,
        name=name,
        mesh_size=mesh_size,
        box_size=box_size,
        nside=nside,
        flatsky_npix=flatsky_npix,
        field_size=field_size,
        scale_factors=scale_factors,
        comoving_centers=comoving_centers,
        density_width=density_width,
        z_sources=z_sources,
        status=status,
        unit=unit,
    )

    if entry_type == "StarletCoefficients":
        tab_norm = jnp.asarray(np.asarray(item.get("tab_norm", []), dtype=np.float64))
        obj = cls(nscales=int(array.shape[0]), tab_norm=tab_norm if tab_norm.size > 0 else None, **common)
    else:  # PDF / PeakCounts — kind is set by the subclass default
        bins = jnp.asarray(np.asarray(item.get("bins", []), dtype=np.float64))
        obj = cls(bins=bins, **common)

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

    version = item.get("version", SUMMARY_CATALOG_VERSION)
    try:
        version = int(np.asarray(version).flat[0])
    except (TypeError, ValueError):
        version = SUMMARY_CATALOG_VERSION

    return obj, cosmology, version
