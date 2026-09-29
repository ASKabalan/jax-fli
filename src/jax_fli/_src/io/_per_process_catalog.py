"""Per-process field catalogs: one parquet file per process, no gather.

``Catalog.to_parquet(path, per_process=True)`` writes ``path`` as a folder::

    path/
      part_{entry:03d}_{process:06d}.parquet   # the blocks of catalog entry ``entry`` held by ``process``
      _header.json                             # written last by process 0; its presence marks a complete write

Each row is an ordinary v2 catalog row (see ``_field_catalog``) holding one block of the entry's array,
plus ``_entry``, ``_global_shape``, ``_block_start`` and ``_block_stop`` locating the block in the global
array (batch dimension included). A process writes only the shards it holds with ``replica_id == 0``, so
nothing is gathered; an entry that is fully addressable or fully replicated is written by process 0 as
one block. ``Catalog.from_parquet`` detects the folder and reassembles each entry either as one host
array (``sharding=None``) or directly into the requested sharding, reading only the blocks each process
needs.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil

import jax
import numpy as np
from jax.experimental.multihost_utils import sync_global_devices

from ._field_catalog import (
    CATALOG_VERSION,
    _ensure_batch_dim,
    build_features,
    decode_row_array,
    row_from_host_array,
    row_to_field_cosmo,
)

FORMAT = "jax-fli-per-process-catalog"
FORMAT_VERSION = 1
HEADER = "_header.json"
_BLOCK_COLUMNS = ("_entry", "_global_shape", "_block_start", "_block_stop")
_PART_RE = re.compile(r"part_(\d{3})_(\d{6})\.parquet$")


def is_per_process_folder(path) -> bool:
    """True when ``path`` is a folder written by ``write_per_process``."""
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, HEADER))


def _is_process_sharded(array) -> bool:
    return isinstance(array, jax.Array) and not array.is_fully_addressable and not array.is_fully_replicated


def _bounds(index, shape) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(start, stop) per axis of a tuple of slices over ``shape``."""
    ranges = [s.indices(n)[:2] for s, n in zip(index, shape)]
    return tuple(r[0] for r in ranges), tuple(r[1] for r in ranges)


def _local_blocks(array):
    """Yield ``((start, stop), host_block)`` for the blocks this process writes."""
    shape = array.shape
    if _is_process_sharded(array):
        for shard in array.addressable_shards:
            if shard.replica_id == 0:
                yield _bounds(shard.index, shape), np.asarray(shard.data)
    elif jax.process_index() == 0:
        host = np.asarray(array.addressable_data(0) if isinstance(array, jax.Array) else array)
        yield ((0,) * len(shape), tuple(shape)), host


def _n_blocks(array) -> int:
    """Number of distinct blocks the whole array is written as (the completeness check on read)."""
    if not _is_process_sharded(array):
        return 1
    indices = array.sharding.devices_indices_map(array.shape).values()
    return len({_bounds(index, array.shape) for index in indices})


def write_per_process(fields, cosmologies, version, path) -> None:
    """Collective: every process writes its own blocks of every entry; no process gathers anything."""
    from datasets import Dataset, Sequence, Value, concatenate_datasets

    path = str(path)
    if jax.process_index() == 0:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)
        os.makedirs(path)
    sync_global_devices(f"per-process-catalog-clear:{path}")

    entries = []
    for entry, (field, cosmology) in enumerate(zip(fields, cosmologies)):
        field_type = type(field).__name__
        array = _ensure_batch_dim(field.array, field_type)
        rows = []
        for (start, stop), block in _local_blocks(array):
            row = row_from_host_array(field, cosmology, version, block)
            row["_entry"] = [entry]
            row["_global_shape"] = [list(array.shape)]
            row["_block_start"] = [list(start)]
            row["_block_stop"] = [list(stop)]
            features = build_features(field, array=block)
            features["_entry"] = Value("int32")
            for key in _BLOCK_COLUMNS[1:]:
                features[key] = Sequence(Value("int64"), length=array.ndim)
            rows.append(Dataset.from_dict(row, features=features))
        if rows:
            concatenate_datasets(rows).to_parquet(
                os.path.join(path, f"part_{entry:03d}_{jax.process_index():06d}.parquet")
            )
        entries.append(
            {
                "field_type": field_type,
                "global_shape": list(array.shape),
                "dtype": np.dtype(array.dtype).name,
                "n_blocks": _n_blocks(array),
                "sharding": repr(getattr(array, "sharding", None)),
            }
        )

    sync_global_devices(f"per-process-catalog-written:{path}")
    if jax.process_index() == 0:
        header = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "catalog_version": int(version) if version is not None else CATALOG_VERSION,
            "n_processes": jax.process_count(),
            "entries": entries,
        }
        with open(os.path.join(path, HEADER), "w") as f:
            json.dump(header, f, indent=2)
    sync_global_devices(f"per-process-catalog-header:{path}")


def _read_part(file):
    """Load one part file as numpy-formatted rows without going through the HuggingFace cache."""
    import pyarrow.parquet as pq
    from datasets import Dataset
    from datasets.table import InMemoryTable

    return Dataset(InMemoryTable(pq.read_table(file))).with_format("numpy", dtype=None)  # keep float64 (see catalog.py)


class _EntryBlocks:
    """The blocks of one entry: their extents up front, their arrays read lazily and cached per file."""

    def __init__(self, files):
        import pyarrow.parquet as pq

        self.files = files
        self.extents = []  # (file, row, start, stop)
        for file in files:
            table = pq.read_table(file, columns=["_block_start", "_block_stop"])
            for row, (start, stop) in enumerate(
                zip(table["_block_start"].to_pylist(), table["_block_stop"].to_pylist())
            ):
                self.extents.append((file, row, tuple(start), tuple(stop)))
        self._cache = {}

    def meta_row(self):
        return self._rows(self.files[0])[0]

    def _rows(self, file):
        if file not in self._cache:
            self._cache[file] = _read_part(file)
        return self._cache[file]

    def block(self, file, row):
        item = self._rows(file)[row]
        array = decode_row_array(item)
        stored_dtype = item.get("array_dtype", None)
        if stored_dtype is not None and array.dtype != np.dtype(str(stored_dtype)):
            array = array.astype(np.dtype(str(stored_dtype)))
        return array

    def fill(self, start, stop, dtype):
        """Assemble the region ``[start, stop)`` of the global array from every overlapping block."""
        out = np.empty(tuple(b - a for a, b in zip(start, stop)), dtype=dtype)
        covered = 0
        for file, row, b_start, b_stop in self.extents:
            lo = tuple(max(a, b) for a, b in zip(start, b_start))
            hi = tuple(min(a, b) for a, b in zip(stop, b_stop))
            if any(h <= lo_ for lo_, h in zip(lo, hi)):
                continue
            block = self.block(file, row)
            src = tuple(slice(lo_ - b, h - b) for lo_, h, b in zip(lo, hi, b_start))
            dst = tuple(slice(lo_ - a, h - a) for lo_, h, a in zip(lo, hi, start))
            out[dst] = block[src]
            covered += int(np.prod([h - lo_ for lo_, h in zip(lo, hi)]))
        if covered != out.size:
            raise ValueError(f"Blocks cover {covered} of {out.size} elements of region {start}:{stop}.")
        return out


def read_per_process(path, sharding=None):
    """Read a per-process folder into ``(fields, cosmologies, version)``.

    ``sharding=None`` assembles each entry as one host array (on every process that calls it). With a
    ``sharding`` the entry is built by ``jax.make_array_from_callback`` directly in the layout the field's
    own ``apply_sharding`` gives it, so each process reads only the blocks overlapping its shards.
    """
    path = str(path)
    with open(os.path.join(path, HEADER)) as f:
        header = json.load(f)
    if header.get("format") != FORMAT:
        raise ValueError(f"{path}/{HEADER} is not a {FORMAT} header.")
    if header.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported {FORMAT} version {header.get('format_version')} in {path}.")

    by_entry: dict[int, list[str]] = {}
    for file in sorted(glob.glob(os.path.join(path, "part_*.parquet"))):
        match = _PART_RE.search(os.path.basename(file))
        if match:
            by_entry.setdefault(int(match.group(1)), []).append(file)

    fields, cosmologies, version = [], [], header.get("catalog_version", CATALOG_VERSION)
    for entry, spec in enumerate(header["entries"]):
        blocks = _EntryBlocks(by_entry.get(entry, []))
        if len(blocks.extents) != spec["n_blocks"]:
            raise ValueError(
                f"Entry {entry} of {path}: found {len(blocks.extents)} blocks, header expects {spec['n_blocks']}."
            )
        shape, dtype = tuple(spec["global_shape"]), np.dtype(spec["dtype"])
        meta = blocks.meta_row()

        if sharding is None:
            array = blocks.fill((0,) * len(shape), shape, dtype)
        else:
            # The layout apply_sharding gives this field (subclasses may transpose), found without allocating.
            def _final_array(a):
                return row_to_field_cosmo(meta, sharding=sharding, array=a)[0].array

            abstract = jax.ShapeDtypeStruct(shape, dtype)
            final = jax.eval_shape(_final_array, abstract)
            target = jax.jit(_final_array).lower(abstract).compile().output_shardings
            unbatched = len(final.shape) == len(shape) - 1

            def _callback(index, final_shape=final.shape, unbatched=unbatched, blocks=blocks, dtype=dtype):
                start, stop = _bounds(index, final_shape)
                if unbatched:
                    return blocks.fill((0, *start), (1, *stop), dtype)[0]
                return blocks.fill(start, stop, dtype)

            final_array = jax.make_array_from_callback(final.shape, target, _callback)
            array = final_array[None] if unbatched else final_array

        field, cosmology, version = row_to_field_cosmo(meta, sharding=sharding, array=array)
        fields.append(field)
        cosmologies.append(cosmology)
    return fields, cosmologies, version
