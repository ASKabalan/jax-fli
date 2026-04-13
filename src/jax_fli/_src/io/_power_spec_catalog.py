"""PowerSpectrum catalog serialization (HuggingFace parquet backend)."""

from __future__ import annotations

import jax
import jax_cosmo as jc
import numpy as np
from jax.experimental.multihost_utils import process_allgather

from ...power.power_spec import PowerSpectrum
from ..base._enums import FieldStatus, SpectralUnit

PS_CATALOG_VERSION = 2


def build_ps_features(ps: PowerSpectrum, cosmology: jc.Cosmology):
    """Build HuggingFace Features schema for a PowerSpectrum catalog row."""
    from datasets import Array2D, Features, Sequence, Value

    n_k = ps.wavenumber.shape[0]  # type: ignore[reportOptionalMemberAccess]
    dtype_str = np.dtype(np.asarray(ps.array).dtype).name

    # 1D spectra are stored as a Sequence; 2D (n_spec, n_k) as Array2D
    if ps.array.ndim == 1:
        array_feature = Sequence(Value(dtype_str))
    else:
        n_spec = ps.array.shape[0]
        array_feature = Array2D(shape=(n_spec, n_k), dtype=dtype_str)

    feature_dict = {
        # Core spectrum data
        "wavenumber": Sequence(Value("float64")),
        "array": array_feature,
        "name": Value("string"),
        "has_scale_factors": Value("bool"),
        "scale_factors": Sequence(Value("float64")),
        "entry_type": Value("string"),
        # AbstractField geometry (v2)
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
        "has_comoving_centers": Value("bool"),
        "comoving_centers": Sequence(Value("float64")),
        "has_density_width": Value("bool"),
        "density_width": Sequence(Value("float64")),
        "has_z_sources": Value("bool"),
        "z_sources": Sequence(Value("float64")),
        "field_status": Value("string"),
        "spectral_unit": Value("string"),
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

    # AbstractField geometry fields
    mesh = ps.mesh_size if ps.mesh_size is not None else (0, 0, 0)
    box = ps.box_size if ps.box_size is not None else (0.0, 0.0, 0.0)
    nside_val = ps.nside if ps.nside is not None else -1
    fp = ps.flatsky_npix
    fp0 = fp[0] if fp is not None else -1
    fp1 = fp[1] if fp is not None else -1
    field_size_val = ps.field_size if ps.field_size is not None else -1.0

    has_cc = ps.comoving_centers is not None
    cc_list: list[float] = np.asarray(ps.comoving_centers).ravel().tolist() if has_cc else []

    has_dw = ps.density_width is not None
    dw_list: list[float] = np.asarray(ps.density_width).ravel().tolist() if has_dw else []

    has_zs = ps.z_sources is not None
    zs_list: list[float] = np.asarray(ps.z_sources).ravel().tolist() if has_zs else []

    field_status_str = ps.status.name if ps.status is not None else FieldStatus.SPECTRA.name
    spectral_unit_str = ps.unit.name if ps.unit is not None else SpectralUnit.ANGULAR_CL.name

    data = {
        "wavenumber": [wavenumber],
        "array": [array],
        "name": [ps.name if ps.name is not None else ""],
        "has_scale_factors": [has_sf],
        "scale_factors": [scale_factors_list],
        "entry_type": ["PowerSpectrum"],
        # AbstractField geometry
        "mesh_size_0": [int(mesh[0])],
        "mesh_size_1": [int(mesh[1])],
        "mesh_size_2": [int(mesh[2])],
        "box_size_0": [float(box[0])],
        "box_size_1": [float(box[1])],
        "box_size_2": [float(box[2])],
        "nside": [int(nside_val)],
        "flatsky_npix_0": [int(fp0)],
        "flatsky_npix_1": [int(fp1)],
        "field_size": [float(field_size_val)],
        "has_comoving_centers": [has_cc],
        "comoving_centers": [cc_list],
        "has_density_width": [has_dw],
        "density_width": [dw_list],
        "has_z_sources": [has_zs],
        "z_sources": [zs_list],
        "field_status": [field_status_str],
        "spectral_unit": [spectral_unit_str],
        # Cosmology
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
    """Reconstruct a PowerSpectrum + cosmology from a dataset row dict.

    Backward-compatible: v1 files (without AbstractField columns) load fine
    with sentinel defaults.
    """
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

    # AbstractField geometry — backward compat: default to sentinels for v1 files
    m0 = int(item.get("mesh_size_0", 0))
    m1 = int(item.get("mesh_size_1", 0))
    m2 = int(item.get("mesh_size_2", 0))
    mesh_size = (m0, m1, m2)

    b0 = float(item.get("box_size_0", 0.0))
    b1 = float(item.get("box_size_1", 0.0))
    b2 = float(item.get("box_size_2", 0.0))
    box_size = (b0, b1, b2)

    nside_raw = int(item.get("nside", -1))
    nside = None if nside_raw < 0 else nside_raw

    fp0 = int(item.get("flatsky_npix_0", -1))
    fp1 = int(item.get("flatsky_npix_1", -1))
    flatsky_npix = None if fp0 < 0 else (fp0, fp1)

    field_size_raw = float(item.get("field_size", -1.0))
    field_size = None if field_size_raw < 0 else field_size_raw

    has_cc = bool(item.get("has_comoving_centers", False))
    comoving_centers = None
    if has_cc:
        cc_list = item.get("comoving_centers", [])
        if cc_list is not None and len(cc_list) > 0:
            comoving_centers = jnp.asarray(np.asarray(cc_list, dtype=np.float64))

    has_dw = bool(item.get("has_density_width", False))
    density_width = None
    if has_dw:
        dw_list = item.get("density_width", [])
        if dw_list is not None and len(dw_list) > 0:
            density_width = jnp.asarray(np.asarray(dw_list, dtype=np.float64))

    has_zs = bool(item.get("has_z_sources", False))
    z_sources = None
    if has_zs:
        zs_list = item.get("z_sources", [])
        if zs_list is not None and len(zs_list) > 0:
            z_sources = jnp.asarray(np.asarray(zs_list, dtype=np.float64))

    # Status
    raw_status = item.get("field_status", FieldStatus.SPECTRA.name)
    try:
        status = FieldStatus[raw_status]
    except (KeyError, ValueError):
        status = FieldStatus.SPECTRA

    # Unit
    raw_unit = item.get("spectral_unit", SpectralUnit.ANGULAR_CL.name)
    try:
        unit = SpectralUnit[raw_unit]
    except (KeyError, ValueError):
        unit = SpectralUnit.ANGULAR_CL

    ps = PowerSpectrum(
        wavenumber=wavenumber,
        array=array,
        name=name,
        scale_factors=scale_factors,
        mesh_size=mesh_size,
        box_size=box_size,
        nside=nside,
        flatsky_npix=flatsky_npix,
        field_size=field_size,
        comoving_centers=comoving_centers,
        density_width=density_width,
        z_sources=z_sources,
        status=status,
        unit=unit,
    )

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

    version = item.get("version", PS_CATALOG_VERSION)
    try:
        version = int(version)
    except (TypeError, ValueError):
        version = PS_CATALOG_VERSION

    return ps, cosmology, version
