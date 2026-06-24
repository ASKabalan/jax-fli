"""Streaming catalog analysis: per-chain statistics via Welford's algorithm."""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.multihost_utils import process_allgather

from ..fields import DensityField
from .catalog import Catalog

__all__ = ["extract_catalog", "extract_cosmo_catalog", "requires_datasets", "CatalogExtract"]

_Param = ParamSpec("_Param")
_Return = TypeVar("_Return")

# pyright: reportOptionalMemberAccess=false


def requires_datasets(func: Callable[_Param, _Return]) -> Callable[_Param, _Return]:
    """Decorator that raises ImportError when the 'datasets' package is not installed."""
    try:
        import datasets  # noqa: F401

        return func
    except ImportError:
        pass

    @wraps(func)
    def _deferred(*args: _Param.args, **kwargs: _Param.kwargs) -> _Return:
        raise ImportError("Missing optional dependency 'datasets'. Install with: pip install jax-fli[catalog]")

    return _deferred


class _RunningStats:
    """Welford's one-pass online algorithm for mean and variance.

    Works with any ``AbstractPytree`` (DensityField, PowerSpectrum, etc.) by
    operating through the object's arithmetic operators and ``apply_fn``.  All
    accumulators are kept in ``float64`` numpy arrays so no JAX memory is
    consumed during the streaming loop.
    """

    def __init__(self, template):
        self.n = 0
        self.mean = template.apply_fn(lambda x: jnp.zeros_like(jnp.asarray(x), dtype=jnp.float64))
        self.M2 = template.apply_fn(lambda x: jnp.zeros_like(jnp.asarray(x), dtype=jnp.float64))

    def update(self, obj):
        x = obj.apply_fn(lambda arr: jnp.asarray(arr, dtype=jnp.float64))
        self.n += 1
        delta = x - self.mean
        self.mean = self.mean + delta * (1.0 / self.n)
        delta2 = x - self.mean
        self.M2 = self.M2 + delta * delta2

    def get_mean(self):
        return self.mean

    def get_std(self, ddof: int = 0):
        if self.n < 2:
            return self.mean.apply_fn(lambda x: jnp.zeros_like(x))
        denom = self.n - ddof
        return self.M2.apply_fn(lambda arr: jnp.sqrt(arr / denom))


def _detect_chains(path: Path) -> list[str]:
    """Detect chain layout under *path* and return glob strings for each chain.

    Two supported layouts (both require a ``samples/`` subdirectory):

    Single-chain layout::

        path/
        └── samples/
            ├── samples_0.parquet
            └── samples_1.parquet

    Multi-chain layout::

        path/
        ├── chain_0/
        │   └── samples/
        │       └── samples_0.parquet
        └── chain_1/
            └── samples/
                └── ...

    Returns
    -------
    list[str]
        One glob string per chain, usable directly as *data_files* in
        ``datasets.load_dataset``.
    """
    samples_dir = path / "samples"
    if samples_dir.exists() and any(samples_dir.glob("*.parquet")):
        return [str(samples_dir / "*.parquet")]

    subdirs = sorted(d for d in path.iterdir() if d.is_dir())
    chains = []
    for subdir in subdirs:
        chain_samples_dir = subdir / "samples"
        if chain_samples_dir.exists() and any(chain_samples_dir.glob("*.parquet")):
            chains.append(str(chain_samples_dir / "*.parquet"))

    if not chains:
        raise FileNotFoundError(
            f"No parquet files found under '{path}'. "
            "Expected either 'path/samples/*.parquet' (single chain) "
            "or 'path/chain_N/samples/*.parquet' (multi-chain)."
        )
    return chains


def _field_to_metadata_dict(field: DensityField, prefix: str) -> dict:
    """Convert a DensityField's metadata to a flat dict with given prefix."""
    d: dict = {
        f"{prefix}mesh_size": list(field.mesh_size),
        f"{prefix}box_size": list(field.box_size),
        f"{prefix}observer_position": list(field.observer_position),
        f"{prefix}halo_size": list(field.halo_size),
        f"{prefix}status": field.status.name,
        f"{prefix}unit": field.unit.name,
    }
    for attr in ["z_sources", "scale_factors", "comoving_centers", "density_width"]:
        val = getattr(field, attr, None)
        d[f"{prefix}{attr}"] = jnp.asarray(val).flatten().tolist() if val is not None else []
    return d


def _metadata_dict_to_field_kwargs(row: dict, prefix: str) -> dict:
    """Convert a parquet row back to DensityField constructor kwargs."""
    from .._src.base._enums import DensityUnit, FieldStatus

    def _to_tuple(v, _type):
        return tuple(_type(x) for x in v)

    kwargs: dict = {
        "mesh_size": _to_tuple(row[f"{prefix}mesh_size"], int),
        "box_size": _to_tuple(row[f"{prefix}box_size"], float),
        "observer_position": _to_tuple(row[f"{prefix}observer_position"], float),
        "halo_size": _to_tuple(row[f"{prefix}halo_size"], float),
        "status": FieldStatus[str(row[f"{prefix}status"])],
        "unit": DensityUnit[str(row[f"{prefix}unit"])],
    }
    for attr in ["z_sources", "scale_factors", "comoving_centers", "density_width"]:
        val = row.get(f"{prefix}{attr}")
        if val is not None and len(val) > 0:
            kwargs[attr] = jnp.asarray(val, dtype=jnp.float64)
        else:
            kwargs[attr] = None
    return kwargs


class CatalogExtract(eqx.Module):
    """Typed container for extract_catalog results.

    All fields default to None; only populated when the corresponding
    option was passed to extract_catalog().  No fields are marked static —
    this module is a plain data container (never JIT-compiled through).
    """

    name: str = eqx.field(static=True)  # for pretty printing
    cosmo: dict  # {key: np.ndarray (n_chains, n_samples)}
    truth_cosmo: dict | None = None  # flat dict of truth cosmology params (per-run, not chain-indexed)
    true_ic: DensityField | None = None  # reference IC if provided
    mean_field: DensityField | None = None  # array shape (n_chains, X, Y, Z)
    std_field: DensityField | None = None  # array shape (n_chains, X, Y, Z)
    power_spectra: tuple | None = None  # (mean_tf, std_tf, mean_coh, std_coh)

    @property
    def n_chains(self) -> int:
        vals = list(self.cosmo.values())
        if not vals:
            return 0
        arr = jnp.asarray(vals[0])
        return arr.shape[0] if arr.ndim == 2 else 1

    @property
    def cosmo_keys(self) -> list[str]:
        return list(self.cosmo.keys())

    def __getitem__(self, idx) -> CatalogExtract:
        """Select chain(s) by index or slice.

        Integer index is normalised to a length-1 slice so cosmo arrays remain
        2-D ``(n_selected_chains, n_samples)`` and ``n_chains`` stays consistent.

        Examples
        --------
        result[0]    # first chain (n_chains=1)
        result[1:3]  # chains 1 and 2
        result[::2]  # every other chain
        """
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)

        new_cosmo = {k: jnp.asarray(v)[idx] for k, v in self.cosmo.items()}

        new_mean_field = (
            self.mean_field.replace(array=self.mean_field.array[idx]) if self.mean_field is not None else None
        )
        new_std_field = self.std_field.replace(array=self.std_field.array[idx]) if self.std_field is not None else None
        new_power_spectra = (
            tuple(ps.replace(array=ps.array[idx]) for ps in self.power_spectra)
            if self.power_spectra is not None
            else None
        )

        return CatalogExtract(
            name=self.name,
            cosmo=new_cosmo,
            truth_cosmo=self.truth_cosmo,  # per-run, not chain-indexed
            true_ic=self.true_ic,
            mean_field=new_mean_field,
            std_field=new_std_field,
            power_spectra=new_power_spectra,
        )

    @requires_datasets
    def to_dataset(self) -> Any:
        """Serialise this CatalogExtract to a HuggingFace Dataset (one row).

        Columns are added dynamically: truth cosmo, field arrays, and power spectra
        columns are omitted when the corresponding attribute is ``None``.

        In multi-process runs this method performs a collective ``process_allgather``
        on all sharded arrays and then returns ``None`` on non-zero processes.

        Returns
        -------
        datasets.Dataset or None
            Single-row dataset on process 0; ``None`` on all other processes.
        """
        from datasets import Array2D, Array3D, Array4D, Dataset, Features, Sequence, Value

        # --- Gather sharded field arrays (collective, all processes participate) ---
        gathered_ic_arr = None
        if self.true_ic is not None:
            gathered_ic_arr = process_allgather(self.true_ic.array, tiled=True)

        mean_arr = None
        std_arr = None

        if self.mean_field is not None:
            assert self.std_field is not None
            mean_arr = process_allgather(self.mean_field.array, tiled=True)
            std_arr = process_allgather(self.std_field.array, tiled=True)

        if jax.process_index() != 0:
            return None

        data: dict = {"name": [self.name], "cosmo_keys": [self.cosmo_keys]}
        feature_dict: dict = {
            "name": Value("string"),
            "cosmo_keys": Sequence(Value("string")),
        }

        # --- Truth cosmology (one column per param) ---
        if self.truth_cosmo is not None:
            for k, v in self.truth_cosmo.items():
                col = f"truth_cosmo_{k}"
                data[col] = [float(v)]
                feature_dict[col] = Value("float64")

        # --- Cosmo samples: shape (n_chains, n_samples) ---
        for key in self.cosmo_keys:
            col = f"cosmo_{key}"
            arr = np.asarray(self.cosmo[key], dtype=np.float64)
            data[col] = [arr]
            feature_dict[col] = Array2D(shape=arr.shape, dtype="float64")

        # --- True IC field ---
        if self.true_ic is not None:
            ic_arr = gathered_ic_arr
            data["ic_array"] = [ic_arr]
            feature_dict["ic_array"] = Array3D(shape=ic_arr.shape, dtype="float64")
            for k, v in _field_to_metadata_dict(self.true_ic, "ic_").items():
                data[k] = [v]
            feature_dict.update(
                {
                    "ic_mesh_size": Sequence(Value("int32"), length=3),
                    "ic_box_size": Sequence(Value("float32"), length=3),
                    "ic_observer_position": Sequence(Value("float32"), length=3),
                    "ic_halo_size": Sequence(Value("int32"), length=2),
                    "ic_status": Value("string"),
                    "ic_unit": Value("string"),
                    "ic_z_sources": Sequence(Value("float64")),
                    "ic_scale_factors": Sequence(Value("float64")),
                    "ic_comoving_centers": Sequence(Value("float64")),
                    "ic_density_width": Sequence(Value("float64")),
                }
            )

        # --- Mean / std field arrays: shape (n_chains, X, Y, Z) ---
        if self.mean_field is not None:
            assert self.std_field is not None
            data["mean_field_array"] = [mean_arr]
            data["std_field_array"] = [std_arr]
            feature_dict["mean_field_array"] = Array4D(shape=mean_arr.shape, dtype="float64")
            feature_dict["std_field_array"] = Array4D(shape=std_arr.shape, dtype="float64")
            for k, v in _field_to_metadata_dict(self.mean_field, "field_").items():
                data[k] = [v]
            feature_dict.update(
                {
                    "field_mesh_size": Sequence(Value("int32"), length=3),
                    "field_box_size": Sequence(Value("float32"), length=3),
                    "field_observer_position": Sequence(Value("float32"), length=3),
                    "field_halo_size": Sequence(Value("int32"), length=2),
                    "field_status": Value("string"),
                    "field_unit": Value("string"),
                    "field_z_sources": Sequence(Value("float64")),
                    "field_scale_factors": Sequence(Value("float64")),
                    "field_comoving_centers": Sequence(Value("float64")),
                    "field_density_width": Sequence(Value("float64")),
                }
            )

        # --- Power spectra: wavenumber (n_k,), each ps array (n_chains, n_k) ---
        if self.power_spectra is not None:
            mean_tf, std_tf, mean_coh, std_coh = self.power_spectra
            k = np.asarray(mean_tf.wavenumber, dtype=np.float64)
            data["ps_wavenumber"] = [k.tolist()]
            feature_dict["ps_wavenumber"] = Sequence(Value("float64"))
            for col, ps in [
                ("ps_mean_tf", mean_tf),
                ("ps_std_tf", std_tf),
                ("ps_mean_coh", mean_coh),
                ("ps_std_coh", std_coh),
            ]:
                arr = np.asarray(ps.array, dtype=np.float64)
                data[col] = [arr]
                feature_dict[col] = Array2D(shape=arr.shape, dtype="float64")

        return Dataset.from_dict(data, features=Features(feature_dict))

    @requires_datasets
    def to_parquet(self, path: str) -> None:
        """Save this CatalogExtract to a parquet file.

        Parameters
        ----------
        path : str
            Destination parquet file path.
        """
        ds = self.to_dataset()
        if ds is not None:
            ds.to_parquet(path)

    @classmethod
    @requires_datasets
    def from_dataset(cls, ds, sharding=None) -> CatalogExtract:
        """Reconstruct a CatalogExtract from a HuggingFace Dataset or single-row dict.

        Parameters
        ----------
        ds : datasets.Dataset or dict
            A HuggingFace Dataset (as returned by :meth:`to_dataset`) or a single-row
            dict (e.g. from iterating a streaming dataset).
        sharding : jax.sharding.Sharding, optional
            If provided, apply ``lax.with_sharding_constraint`` to loaded arrays.

        Returns
        -------
        CatalogExtract
        """
        import datasets as hf_datasets

        if isinstance(ds, (hf_datasets.Dataset | hf_datasets.IterableDataset)):
            row = ds.with_format("numpy")[0]
        elif isinstance(ds, dict):
            row = ds
        else:
            raise ValueError(f"Unsupported dataset type: {type(ds)}")

        name = str(row["name"])
        cosmo_keys = [str(k) for k in row["cosmo_keys"]]

        # --- Truth cosmology ---
        _truth_params = ["Omega_c", "Omega_b", "h", "n_s", "sigma8", "w0", "wa", "Omega_k", "Omega_nu"]
        truth_cosmo = None
        if any(f"truth_cosmo_{k}" in row for k in _truth_params):
            truth_cosmo = {k: float(row[f"truth_cosmo_{k}"]) for k in _truth_params if f"truth_cosmo_{k}" in row}

        # --- Cosmo samples ---
        cosmo = {key: jnp.asarray(row[f"cosmo_{key}"], dtype=np.float64) for key in cosmo_keys}

        # --- True IC ---
        true_ic = None
        if "ic_array" in row and row.get("ic_array") is not None:
            ic_arr = np.asarray(row["ic_array"])
            kwargs = _metadata_dict_to_field_kwargs(row, "ic_")
            true_ic = DensityField(array=ic_arr, field_sharding=sharding, **kwargs)
            true_ic = true_ic.apply_sharding()
        # --- Mean / std fields ---
        mean_field = None
        std_field = None
        if "mean_field_array" in row and row.get("mean_field_array") is not None:
            mean_arr = np.asarray(row["mean_field_array"])
            std_arr = np.asarray(row["std_field_array"])
            kwargs = _metadata_dict_to_field_kwargs(row, "field_")
            mean_field = DensityField(array=mean_arr, field_sharding=sharding, **kwargs)
            std_field = DensityField(array=std_arr, field_sharding=sharding, **kwargs)
            mean_field = mean_field.apply_sharding()
            std_field = std_field.apply_sharding()

        # --- Power spectra ---
        power_spectra = None
        if "ps_wavenumber" in row and row.get("ps_wavenumber") is not None:
            from ..summary_statistics.power_spec import PowerSpectrum

            k = jnp.asarray(row["ps_wavenumber"], dtype=jnp.float64)
            mean_tf = PowerSpectrum(wavenumber=k, array=jnp.asarray(row["ps_mean_tf"], dtype=jnp.float64))
            std_tf = PowerSpectrum(wavenumber=k, array=jnp.asarray(row["ps_std_tf"], dtype=jnp.float64))
            mean_coh = PowerSpectrum(wavenumber=k, array=jnp.asarray(row["ps_mean_coh"], dtype=jnp.float64))
            std_coh = PowerSpectrum(wavenumber=k, array=jnp.asarray(row["ps_std_coh"], dtype=jnp.float64))
            power_spectra = (mean_tf, std_tf, mean_coh, std_coh)

        return cls(
            name=name,
            cosmo=cosmo,
            truth_cosmo=truth_cosmo,
            true_ic=true_ic,
            mean_field=mean_field,
            std_field=std_field,
            power_spectra=power_spectra,
        )

    @classmethod
    @requires_datasets
    def from_parquet(cls, path: str, sharding=None) -> CatalogExtract:
        """Load a CatalogExtract from a parquet file.

        Parameters
        ----------
        path : str
            Path to the parquet file saved by :meth:`to_parquet`.
        sharding : jax.sharding.Sharding, optional
            If provided, apply ``lax.with_sharding_constraint`` to loaded arrays.

        Returns
        -------
        CatalogExtract
        """
        from datasets import load_dataset

        ds = load_dataset("parquet", data_files=path, split="train").with_format("numpy")
        return cls.from_dataset(ds, sharding=sharding)


@requires_datasets
def extract_catalog(
    set_name: str,
    cosmo_keys: list[str] | tuple[str, ...],
    patterns: list[str] | tuple[str, ...] | None = None,
    repo: str | None = None,
    truth: Catalog | None = None,
    field_statistic: bool = False,
    power_statistic: bool = False,
    ddof: int = 0,
    sharding=None,
) -> CatalogExtract:
    """Stream MCMC catalog parquet files and compute per-chain statistics.

    Parquet files are read one row at a time (streaming mode) so the full
    sample set never needs to fit in memory.  Statistics are accumulated with
    Welford's online algorithm in ``float64`` precision.

    Parameters
    ----------
    cosmo_keys : list or tuple of str
        Cosmological parameter names to collect (e.g. ``["Omega_c", "sigma8"]``).
    set_name : str
        Name for the returned :class:`CatalogExtract`.
    patterns : list of str
        Per-chain parquet sources — **one pattern == one MCMC chain**, each streamed
        and accumulated independently (a single pooled glob would mix chains and
        compute a different, wrong statistic). A local pattern is a parquet glob or a
        root directory holding ``samples/`` (single chain) or ``chain_N/samples/``
        subdirectories (auto-expanded via :func:`_detect_chains`). With ``repo``, each
        pattern is a glob of parquet files inside the HF dataset repo.
    repo : str, optional
        HuggingFace Hub dataset repository ID (e.g. ``"user/repo"``). When set, each
        entry of ``patterns`` is resolved as ``data_files`` inside this repo; when
        ``None`` (default) the patterns are local globs/directories.
    truth : Catalog, optional
        Truth Catalog. ``truth.field[0]`` is used as the reference IC for
        transfer/coherence spectra; ``truth.cosmology[0]`` is stored in
        ``CatalogExtract.truth_cosmo``.  Required when ``power_statistic=True``.
    field_statistic : bool
        If *True*, compute per-chain mean and std of the density fields.
    power_statistic : bool
        If *True*, compute per-chain mean and std of the transfer function and
        coherence spectrum relative to the truth IC.
    ddof : int
        Delta degrees of freedom for std computation (0 = population, 1 = sample).
    sharding : jax.sharding.Sharding, optional
        Device sharding forwarded to :meth:`Catalog.from_dataset` during streaming.

    Returns
    -------
    CatalogExtract
        Typed container with attributes ``cosmo``, ``truth_cosmo``, ``true_ic``,
        ``mean_field``, ``std_field``, and ``power_spectra``.
    """
    if not patterns:
        raise ValueError("'patterns' must be a non-empty list of per-chain parquet sources.")
    if power_statistic and truth is None:
        raise ValueError("power_statistic=True requires 'truth' to be provided.")

    import datasets as hf_datasets

    # --- Resolve truth ---
    true_ic_field = None
    truth_cosmo = None
    if truth is not None:
        true_ic_field = truth.field[0]
        _cosmo_param_keys = ["Omega_c", "Omega_b", "h", "n_s", "sigma8", "w0", "wa", "Omega_k", "Omega_nu"]
        truth_cosmo = {k: float(getattr(truth.cosmology[0], k)) for k in _cosmo_param_keys}

    # --- Build per-chain streaming datasets: ONE pattern == ONE chain ---
    # Each chain is streamed and accumulated independently; patterns must never be pooled into a
    # single load_dataset call (that would mix chains and corrupt the per-chain statistics).
    if repo is not None:
        # Resolve the pre-warmed HF cache offline (never hits the network; raises if the cache is
        # cold), then read each chain as local parquet so it works with no internet.
        from huggingface_hub import snapshot_download

        root = snapshot_download(repo, repo_type="dataset", local_files_only=True)
        chain_globs = [f"{root}/{glob}" for glob in patterns]
        chain_streams = [
            hf_datasets.load_dataset("parquet", data_files=glob, split="train", streaming=True).with_format("numpy")
            for glob in chain_globs
        ]
    else:
        # Local: a pattern is a parquet glob, or a directory expanded to per-chain globs.
        chain_globs = []
        for pat in patterns:
            if os.path.isdir(pat):
                chain_globs.extend(_detect_chains(Path(pat)))
            else:
                chain_globs.append(pat)
        chain_streams = [
            hf_datasets.load_dataset("parquet", data_files=glob, split="train", streaming=True).with_format("numpy")
            for glob in chain_globs
        ]
    n_chains = len(chain_streams)

    cosmo_lists: dict[str, list[list[float]]] = {key: [[] for _ in range(n_chains)] for key in cosmo_keys}
    chain_field_stats: list[_RunningStats | None] = [None] * n_chains
    chain_transfer_stats: list[_RunningStats | None] = [None] * n_chains
    chain_coherence_stats: list[_RunningStats | None] = [None] * n_chains

    for chain_idx, streaming_ds in enumerate(chain_streams):
        field_stats: _RunningStats | None = None
        transfer_stats: _RunningStats | None = None
        coherence_stats: _RunningStats | None = None

        for sample in streaming_ds:
            catalog = Catalog.from_dataset(sample, sharding=sharding)
            cosmo = catalog.cosmology[0]

            # --- cosmological parameters ---
            for key in cosmo_keys:
                cosmo_lists[key][chain_idx].append(float(getattr(cosmo, key)))

            # --- density field (cast to float64 for stable accumulation) ---
            # Welfords online mean calculating is iterative and relies on the arithmetic operators of the object, so we convert to numpy arrays here to ensure all intermediate calculations are done in float64 precision.
            field = catalog.field[0].apply_fn(lambda x: jnp.asarray(x, dtype=jnp.float64))

            # --- field statistics ---
            if field_statistic:
                if field_stats is None:
                    field_stats = _RunningStats(field)
                field_stats.update(field)

            # --- power spectra statistics ---
            if power_statistic:
                transfer_ps = field.transfer(true_ic_field).apply_fn(lambda x: jnp.asarray(x, dtype=jnp.float64))
                coherence_ps = field.coherence(true_ic_field).apply_fn(lambda x: jnp.asarray(x, dtype=jnp.float64))
                if transfer_stats is None:
                    transfer_stats = _RunningStats(transfer_ps)
                    coherence_stats = _RunningStats(coherence_ps)
                assert transfer_stats is not None and coherence_stats is not None  # type narrowing for checker
                transfer_stats.update(transfer_ps)
                coherence_stats.update(coherence_ps)

        chain_field_stats[chain_idx] = field_stats
        chain_transfer_stats[chain_idx] = transfer_stats
        chain_coherence_stats[chain_idx] = coherence_stats

    # --- Build cosmo dict: shape (n_chains, n_samples) ---
    cosmo_dict = {key: jnp.array(cosmo_lists[key]) for key in cosmo_keys}

    result_mean_field = None
    result_std_field = None
    result_power_spectra = None

    # --- Stack field statistics across chains ---
    if field_statistic:
        chain_means = [chain_field_stats[c].get_mean() for c in range(n_chains)]
        chain_stds = [chain_field_stats[c].get_std(ddof) for c in range(n_chains)]
        stacked_mean = jnp.stack([f.array for f in chain_means], axis=0)
        stacked_std = jnp.stack([f.array for f in chain_stds], axis=0)
        result_mean_field = chain_means[0].replace(array=jnp.array(stacked_mean))
        result_std_field = chain_stds[0].replace(array=jnp.array(stacked_std))

    # --- Stack power spectra across chains ---
    if power_statistic:

        def _stack_ps(ps_list):
            stacked = jnp.stack([ps.array for ps in ps_list], axis=0)
            return ps_list[0].replace(array=stacked)

        mean_transfer = _stack_ps([chain_transfer_stats[c].get_mean() for c in range(n_chains)])
        std_transfer = _stack_ps([chain_transfer_stats[c].get_std(ddof) for c in range(n_chains)])
        mean_coherence = _stack_ps([chain_coherence_stats[c].get_mean() for c in range(n_chains)])
        std_coherence = _stack_ps([chain_coherence_stats[c].get_std(ddof) for c in range(n_chains)])
        result_power_spectra = (mean_transfer, std_transfer, mean_coherence, std_coherence)

    return CatalogExtract(
        name=set_name,
        cosmo=cosmo_dict,
        truth_cosmo=truth_cosmo,
        true_ic=true_ic_field,
        mean_field=result_mean_field,
        std_field=result_std_field,
        power_spectra=result_power_spectra,
    )


def extract_cosmo_catalog(
    set_name: str,
    cosmo_keys: list[str] | tuple[str, ...],
    patterns: list[str] | tuple[str, ...],
    truth: Any = None,
) -> CatalogExtract:
    """Build a :class:`CatalogExtract` from cosmo-only ``npz`` outputs (power-spectrum / 2PCF runs).

    The power-spectrum model has no initial-condition field, so :func:`sample2catalog` saves the
    sampled cosmology as ``cosmo_{batch}.npz`` rather than a parquet Catalog (which
    :func:`extract_catalog` reads). This loader collects those files into the
    ``(n_chains, n_samples)`` cosmo arrays consumed by :func:`~jax_fli.infer.analyze` and
    :func:`~jax_fli.infer.plot_posterior`.

    Parameters
    ----------
    set_name : str
        Name for the returned :class:`CatalogExtract`.
    cosmo_keys : list or tuple of str
        Cosmological parameter names to collect (e.g. ``["Omega_c", "sigma8"]``).
    patterns : list or tuple of str
        Per-chain sources — **one entry == one chain**. Each entry is a directory searched
        recursively for ``cosmo_*.npz`` (batches concatenated in index order), or a direct
        glob of such files. Chains must have equal sample counts.
    truth : jax_cosmo.Cosmology, optional
        Truth cosmology; its standard parameters are stored (flattened) in
        ``CatalogExtract.truth_cosmo`` for posterior truth markers.

    Returns
    -------
    CatalogExtract
        With ``cosmo[key]`` of shape ``(n_chains, n_samples)``; field/power attributes are ``None``.
    """
    import glob as _glob

    if not patterns:
        raise ValueError("'patterns' must be a non-empty list of per-chain cosmo-npz sources.")

    def _batch_index(p: str):
        stem = os.path.splitext(os.path.basename(p))[0]
        digits = stem.split("_")[-1]
        return int(digits) if digits.isdigit() else stem

    cosmo_lists: dict[str, list[list[float]]] = {key: [] for key in cosmo_keys}
    for pat in patterns:
        files = (
            _glob.glob(os.path.join(pat, "**", "cosmo_*.npz"), recursive=True)
            if os.path.isdir(pat)
            else _glob.glob(pat)
        )
        if not files:
            raise FileNotFoundError(f"No cosmo_*.npz files found for chain pattern '{pat}'.")
        files = sorted(files, key=_batch_index)
        per_key: dict[str, list] = {key: [] for key in cosmo_keys}
        for f in files:
            with np.load(f) as data:
                for key in cosmo_keys:
                    if key not in data:
                        raise KeyError(f"'{key}' not found in {f}; available keys: {list(data.keys())}")
                    per_key[key].append(np.atleast_1d(np.asarray(data[key], dtype=np.float64)))
        for key in cosmo_keys:
            cosmo_lists[key].append(np.concatenate(per_key[key]).tolist())

    cosmo_dict = {key: jnp.asarray(cosmo_lists[key]) for key in cosmo_keys}

    truth_cosmo = None
    if truth is not None:
        _cosmo_param_keys = ["Omega_c", "Omega_b", "h", "n_s", "sigma8", "w0", "wa", "Omega_k", "Omega_nu"]
        truth_cosmo = {k: float(getattr(truth, k)) for k in _cosmo_param_keys if hasattr(truth, k)}

    return CatalogExtract(name=set_name, cosmo=cosmo_dict, truth_cosmo=truth_cosmo)
