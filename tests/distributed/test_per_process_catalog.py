"""``Catalog.to_parquet(per_process=True)`` across real processes (MPI), then read back in one process.

Runs ``mpi_per_process_catalog.py`` under ``mpirun -n 4`` (4 processes x 2 CPU devices), where the
arrays are genuinely not fully addressable, then checks from this single process that the folders it
wrote load back exactly and that the default single-file path is unchanged.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mpi4py")
datasets = pytest.importorskip("datasets")

import jax_cosmo as jc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import jax_fli as jfli  # noqa: E402
from jax_fli.io.catalog import Catalog  # noqa: E402

SCRIPT = Path(__file__).with_name("mpi_per_process_catalog.py")
N_PROCESSES = 4

pytestmark = [
    pytest.mark.distributed,
    pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun not found"),
]


@pytest.fixture(scope="module")
def mpi_output(tmp_path_factory):
    out = tmp_path_factory.mktemp("per_process")
    cmd = ["mpirun", "-n", str(N_PROCESSES)]
    if "Open MPI" in subprocess.run(["mpirun", "--version"], capture_output=True, text=True).stdout:
        cmd.append("--oversubscribe")
    env = {k: v for k, v in os.environ.items() if k not in ("XLA_FLAGS", "JAX_PLATFORM_NAME")}
    env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [*cmd, sys.executable, str(SCRIPT), str(out)], env=env, capture_output=True, text=True, timeout=900
    )
    assert result.returncode == 0, f"mpirun failed:\n{result.stdout[-5000:]}\n{result.stderr[-5000:]}"
    assert result.stdout.count(": OK") == N_PROCESSES, result.stdout[-5000:]
    return out


def test_folders_load_in_one_process(mpi_output):
    ic = Catalog.from_parquet(str(mpi_output / "ic.parquet")).field[0]
    np.testing.assert_array_equal(np.asarray(ic.array), np.load(mpi_output / "ref_ic.npy"))

    fields = Catalog.from_parquet(str(mpi_output / "multi.parquet")).field
    for field, name in zip(fields, ("lc", "kappa", "kappa_rep")):
        ref = np.load(mpi_output / f"ref_{name}.npy")
        assert field.array.dtype == ref.dtype
        np.testing.assert_array_equal(np.asarray(field.array), ref)


def test_default_path_unchanged(mpi_output, tmp_path):
    """The gathered multi-process file equals a single-process write of the same field."""
    ref = np.load(mpi_output / "ref_ic.npy")
    field = jfli.DensityField(
        array=ref,
        mesh_size=ref.shape,
        box_size=(100.0, 100.0, 100.0),
        observer_position=(0.5, 0.5, 0.5),
        halo_size=(0, 0),
        z_sources=np.float64(0.0),
        scale_factors=np.float64(0.1),
        comoving_centers=np.float64(0.0),
        density_width=np.float64(0.0),
        status=jfli.FieldStatus.INITIAL_FIELD,
        unit=jfli.DensityUnit.DENSITY,
        name="ic",
    )
    Catalog(field=[field], cosmology=[jc.Planck18()]).to_parquet(str(tmp_path / "single.parquet"))
    assert pq.read_table(mpi_output / "ic_single.parquet").equals(pq.read_table(tmp_path / "single.parquet"))


def test_missing_part_is_detected(mpi_output, tmp_path):
    broken = tmp_path / "ic.parquet"
    shutil.copytree(mpi_output / "ic.parquet", broken)
    os.remove(broken / "part_000_000002.parquet")
    with pytest.raises(ValueError, match="blocks"):
        Catalog.from_parquet(str(broken))
