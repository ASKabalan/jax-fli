"""PowerSpectrum catalog serialization (HuggingFace parquet backend)."""

from __future__ import annotations

import jax
import jax_cosmo as jc
import numpy as np
from jax.experimental.multihost_utils import process_allgather

from ...power.power_spec import PowerSpectrum

PS_CATALOG_VERSION = 1


def build_ps_features(ps: PowerSpectrum, cosmology: jc.Cosmology):
    """Build HuggingFace Features schema for a PowerSpectrum catalog row."""
    from datasets import Array2D, Features, Sequence, Value

    n_k = ps.wavenumber.shape[0]
    dtype_str = np.dtype(np.asarray(ps.array).dtype).name

    # 1D spectra are stored as a Sequence; 2D (n_spec, n_k) as Array2D
    if ps.array.ndim == 1:
        array_feature = Sequence(Value(dtype_str))
    else:
        n_spec = ps.array.shape[0]
        array_feature = Array2D(shape=(n_spec, n_k), dtype=dtype_str)

    feature_dict = {
        "wavenumber": Sequence(Value("float64")),
        "array": array_feature,
        "name": Value("string"),
        "has_scale_factors": Value("bool"),
        "scale_factors": Sequence(Value("float64")),
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
    }

    return Features(feature_dict)


def ps_to_row(ps: PowerSpectrum, cosmology: jc.Cosmology, version: int) -> dict | None:
    """Convert a single PowerSpectrum + cosmology into a 1-row column-oriented dict."""
    # Gather from all devices (no-op on single device)
    wavenumber = np.asarray(process_allgather(np.asarray(ps.wavenumber), tiled=True))
    array = np.asarray(process_allgather(np.asarray(ps.array), tiled=True))

    if jax.process_index() != 0:
        return None

    scale_factors_list: list[float] = []
    has_sf = ps.scale_factors is not None
    if has_sf:
        scale_factors_list = np.asarray(ps.scale_factors).ravel().tolist()

    data = {
        "wavenumber": [wavenumber],
        "array": [array],
        "name": [ps.name if ps.name is not None else ""],
        "has_scale_factors": [has_sf],
        "scale_factors": [scale_factors_list],
        "entry_type": ["PowerSpectrum"],
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


def row_to_ps_cosmo(item: dict, sharding=None) -> tuple[PowerSpectrum, jc.Cosmology, int]:
    """Reconstruct a PowerSpectrum + cosmology from a dataset row dict."""
    import jax.numpy as jnp

    wavenumber = jnp.asarray(np.asarray(item["wavenumber"]))
    array = jnp.asarray(np.asarray(item["array"]))

    raw_name = item.get("name", "")
    name = raw_name if raw_name else None

    has_sf = bool(item.get("has_scale_factors", False))
    scale_factors = None
    if has_sf:
        sf_list = item.get("scale_factors", [])
        if sf_list is not None and len(sf_list) > 0:
            scale_factors = jnp.asarray(np.asarray(sf_list, dtype=np.float64))

    ps = PowerSpectrum(wavenumber=wavenumber, array=array, name=name, scale_factors=scale_factors)

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
    try:
        version = int(version)
    except (TypeError, ValueError):
        version = PS_CATALOG_VERSION

    return ps, cosmology, version
