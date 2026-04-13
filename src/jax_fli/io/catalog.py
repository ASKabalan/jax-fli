"""Catalog container for field data or power spectra with HuggingFace serialization."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

import equinox as eqx
import jax
import jax_cosmo as jc
import numpy as np
from jaxtyping import Array

from .._src.base._core import AbstractField, AbstractPytree
from .._src.io._field_catalog import (
    CATALOG_VERSION,
    build_features,
    catalog_to_row,
    row_to_field_cosmo,
)
from .._src.io._power_spec_catalog import (
    PS_CATALOG_VERSION,
    build_ps_features,
    ps_to_row,
    row_to_ps_cosmo,
)
from ..power.power_spec import PowerSpectrum

try:
    import datasets
except ImportError:
    pass

__all__ = ["Catalog", "CATALOG_VERSION"]

Param = ParamSpec("Param")
ReturnType = TypeVar("ReturnType")


def requires_datasets(func: Callable[Param, ReturnType]) -> Callable[Param, ReturnType]:
    """Decorator that raises ImportError if datasets library is not available."""
    try:
        import datasets  # noqa: F401

        return func
    except ImportError:
        pass

    @wraps(func)
    def deferred_func(*args: Param.args, **kwargs: Param.kwargs) -> ReturnType:
        msg = "Missing optional library 'datasets'. Install with: pip install jax-fli[catalog]"
        raise ImportError(msg)

    return deferred_func


def _detect_backend(entries: list) -> str:
    """Return 'field' or 'power_spec' based on the first entry type.

    PowerSpectrum is checked first because it now inherits AbstractField.
    """
    if not entries:
        raise ValueError("Cannot determine backend from an empty list.")
    first = entries[0]
    if isinstance(first, PowerSpectrum):  # must precede AbstractField check
        return "power_spec"
    elif isinstance(first, AbstractField):
        return "field"
    else:
        raise TypeError(
            f"Catalog entries must be AbstractField or PowerSpectrum instances, got {type(first).__name__}."
        )


def _validate_homogeneous(entries: list, backend: str) -> None:
    """Raise TypeError if entries are not all the same backend type."""
    if backend == "field":
        for i, entry in enumerate(entries):
            # Explicitly reject PowerSpectrum since it is now a subclass of AbstractField
            if isinstance(entry, PowerSpectrum) or not isinstance(entry, AbstractField):
                raise TypeError(
                    f"Catalog must be homogeneous: entry 0 is a field, "
                    f"but entry {i} is {type(entry).__name__}. Mixed catalogs are not supported."
                )
    else:
        for i, entry in enumerate(entries):
            if not isinstance(entry, PowerSpectrum):
                raise TypeError(
                    f"Catalog must be homogeneous: entry 0 is a power_spec, "
                    f"but entry {i} is {type(entry).__name__}. Mixed catalogs are not supported."
                )


class Catalog(eqx.Module):
    """Container for field data or power spectra, with HuggingFace parquet serialization.

    Stores **lists** of entries (AbstractField or PowerSpectrum) and cosmologies.
    A single entry/cosmology is auto-wrapped into a length-1 list.
    All entries must be the same type (homogeneous).

    Attributes
    ----------
    field : list[AbstractPytree]
        The data entries (all AbstractField or all PowerSpectrum).
    cosmology : list[jc.Cosmology]
        The cosmology associated with each entry.
    version : int
        Catalog version for compatibility tracking.
    """

    field: list[AbstractPytree]
    cosmology: list[jc.Cosmology]
    version: int = eqx.field(static=True, default=CATALOG_VERSION)

    def __init__(self, field, cosmology, version=CATALOG_VERSION):
        # Normalise single-entry inputs.
        # PowerSpectrum must be checked before AbstractField since it now subclasses it.
        if isinstance(field, PowerSpectrum) and isinstance(cosmology, jc.Cosmology):
            field = [field]
            cosmology = [cosmology]
        elif isinstance(field, AbstractField) and isinstance(cosmology, jc.Cosmology):
            cosmo_n = int(np.asarray(cosmology.Omega_c).size)
            cosmo_is_batched = int(np.asarray(cosmology.Omega_c).ndim) > 0
            if cosmo_is_batched:
                field_n = field.array.shape[0] if field.is_batched() else 1
                if field_n != cosmo_n:
                    raise ValueError(f"Cosmology batch size {cosmo_n} != field leading dimension {field_n}.")
                field = [field[i] for i in range(cosmo_n)]
                cosmology = [jax.tree.map(lambda p, i=i: p[i], cosmology) for i in range(cosmo_n)]
            elif field.is_multi_batched() and field.array.shape[0] > 1:
                raise ValueError(
                    f"Scalar cosmology cannot pair with a multi-batched field "
                    f"(shape {field.array.shape}, N={field.array.shape[0]} > 1). "
                    "Pass a batched cosmology of size N, or split the field first."
                )
            else:
                field = [field]
                cosmology = [cosmology]
        else:
            if isinstance(field, AbstractField):
                field = [field]
            if isinstance(cosmology, jc.Cosmology):
                cosmology = [cosmology]

        self.field = list(field)
        self.cosmology = list(cosmology)
        self.version = int(version)

    def __check_init__(self):
        if len(self.field) != len(self.cosmology):
            raise ValueError(
                f"field and cosmology must have the same length, got {len(self.field)} and {len(self.cosmology)}."
            )
        if not self.field:
            raise ValueError("Catalog must contain at least one entry.")
        backend = _detect_backend(self.field)
        _validate_homogeneous(self.field, backend)

    @property
    def backend(self) -> str:
        """Return 'field' or 'power_spec' based on entry types."""
        return _detect_backend(self.field)

    def __len__(self) -> int:
        return len(self.field)

    def __getitem__(self, key) -> Catalog:
        if isinstance(key, int):
            return Catalog(field=[self.field[key]], cosmology=[self.cosmology[key]], version=self.version)
        return Catalog(field=self.field[key], cosmology=self.cosmology[key], version=self.version)

    def __repr__(self) -> str:
        n = len(self.field)
        entry_types = {type(f).__name__ for f in self.field}
        return f"Catalog(n_entries={n}, entry_types={entry_types}, backend={self.backend!r}, version={self.version})"

    # ------------------------------------------------------------------ Serialization

    @requires_datasets
    def to_dataset(self) -> datasets.Dataset | None:
        """Convert this Catalog to a HuggingFace Dataset (one row per entry).

        Returns None on non-root processes.
        """
        backend = self.backend

        if backend == "field":
            for f in self.field:
                if not jax.core.is_concrete(f.array):
                    raise ValueError("Cannot convert to dataset inside a jit context (arrays are tracers).")
            ds_list = []
            for f, c in zip(self.field, self.cosmology):
                from datasets import Dataset

                data = catalog_to_row(f, c, self.version)
                if data is not None:
                    features = build_features(f)
                    ds_list.append(Dataset.from_dict(data, features=features))
        else:
            # power_spec backend
            version = PS_CATALOG_VERSION
            ds_list = []
            for ps, c in zip(self.field, self.cosmology):
                from datasets import Dataset

                data = ps_to_row(ps, c, version)
                if data is not None:
                    features = build_ps_features(ps, c)
                    ds_list.append(Dataset.from_dict(data, features=features))

        if not ds_list:
            return None

        return datasets.concatenate_datasets(ds_list)

    @requires_datasets
    def to_parquet(self, path: str) -> None:
        """Save Catalog to parquet file."""
        ds = self.to_dataset()
        if ds is not None:
            ds.to_parquet(path)

    @classmethod
    @requires_datasets
    def from_parquet(cls, path: str, sharding=None) -> Catalog:
        """Load a Catalog from parquet file.

        Automatically detects whether the file contains fields or power spectra
        from the ``entry_type`` column.
        """
        from datasets import load_dataset

        ds = load_dataset("parquet", data_files=path, split="train").with_format("numpy")
        return cls.from_dataset(ds, sharding=sharding)

    @classmethod
    @requires_datasets
    def from_dataset(cls, ds: datasets.Dataset, sharding=None) -> Catalog:
        """Reconstruct a Catalog from a HuggingFace Dataset."""
        if len(ds) == 0:
            raise ValueError("Cannot reconstruct Catalog from an empty dataset.")

        if isinstance(ds, dict):
            # Single-row dict
            entry_type = ds.get("entry_type", ["field"])
            entry_type = entry_type[0] if isinstance(entry_type, list) else entry_type
            if entry_type == "PowerSpectrum":
                ps, c, v = row_to_ps_cosmo(ds, sharding=sharding)
                return cls(field=[ps], cosmology=[c], version=v)
            else:
                f, c, v = row_to_field_cosmo(ds, sharding=sharding)
                return cls(field=[f], cosmology=[c], version=v)

        elif isinstance(ds, datasets.Dataset | datasets.IterableDataset):
            ds_jax = ds.with_format("numpy")

            # Detect backend from the first row
            first_row = ds_jax[0]
            raw_entry_type = first_row.get("entry_type", "field")
            if isinstance(raw_entry_type, (list | Array)):
                raw_entry_type = raw_entry_type[0]
            is_ps = str(raw_entry_type) == "PowerSpectrum"

            entries = []
            cosmologies = []
            version = CATALOG_VERSION
            for i in range(len(ds_jax)):
                row = ds_jax[i]
                if is_ps:
                    e, c, v = row_to_ps_cosmo(row, sharding=sharding)
                else:
                    e, c, v = row_to_field_cosmo(row, sharding=sharding)
                entries.append(e)
                cosmologies.append(c)
                version = v

            return cls(field=entries, cosmology=cosmologies, version=version)
        else:
            raise ValueError(f"Unsupported dataset type: {type(ds)}")
