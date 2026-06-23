"""Tests for the per-shell streaming save (``--shells-per-file``).

A batched (multi-shell) lightcone written N shells per parquet round-trips — array AND per-shell
metadata — back to the original, the legacy single-file path is preserved when the flag is unset,
and the per-shell CLI output equals the single-file output.
"""

from __future__ import annotations

from argparse import Namespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

datasets = pytest.importorskip("datasets")

import jax_cosmo as jc

import jax_fli as jfli
from jax_fli._src.base._enums import DensityUnit, FieldStatus
from jax_fli.io.catalog import Catalog
from jax_fli.scripts.entry.fli_simulate import _save_result, parser
from tests.scripts.conftest import _BASE_CLI, run_sim

jax.config.update("jax_enable_x64", True)

_N_SHELLS = 5
_NSIDE = 4  # npix = 12 * 4**2 = 192


@pytest.fixture
def batched_lightcone():
    """A batched SphericalDensity of ``_N_SHELLS`` shells (small nside) + a scalar cosmology."""
    rng = np.random.RandomState(0)
    npix = 12 * _NSIDE**2
    field = jfli.SphericalDensity(
        array=jnp.asarray(rng.normal(size=(_N_SHELLS, npix)), dtype=jnp.float64),
        mesh_size=(8, 8, 8),
        box_size=(100.0, 100.0, 100.0),
        observer_position=(0.5, 0.5, 0.5),
        halo_size=(0, 0),
        nside=_NSIDE,
        z_sources=jnp.asarray(rng.uniform(0.1, 2.0, size=_N_SHELLS)),
        scale_factors=jnp.asarray(rng.uniform(0.4, 1.0, size=_N_SHELLS)),
        comoving_centers=jnp.asarray(rng.uniform(100.0, 800.0, size=_N_SHELLS)),
        density_width=jnp.asarray(rng.uniform(10.0, 60.0, size=_N_SHELLS)),
        status=FieldStatus.LIGHTCONE,
        unit=DensityUnit.OVERDENSITY,
    )
    cosmo = jc.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.96, sigma8=0.8, w0=-1.0, wa=0.0, Omega_k=0.0, Omega_nu=0.0
    )
    return field, cosmo


def _concat_meta(chunks, attr):
    """Concatenate a per-shell metadata attribute across reloaded chunks (in shell order)."""
    return np.concatenate([np.atleast_1d(np.asarray(getattr(c, attr))) for c in chunks], axis=0)


@pytest.mark.parametrize("shells_per_file", [1, 2, 5])
def test_save_result_per_shell_matches_single_file(tmp_path, batched_lightcone, shells_per_file):
    """Chunked per-shell save reloads to the SAME data as the single-file save.

    Both paths go through the identical HF parquet round-trip (which downcasts float64->float32 on
    read), so comparing the two reloads is exact and isolates *chunking* correctness — array values
    in shell order, per-shell metadata, and the ceil(N/k) file count — from float precision.
    """
    field, cosmo = batched_lightcone

    # Reference: the established single-file path (validated elsewhere against the API).
    single = tmp_path / "single.parquet"
    _save_result(field, cosmo, Namespace(output=str(single), name=None, shells_per_file=0))
    ref = Catalog.from_parquet(str(single)).field[0]
    ref_arr = np.asarray(ref.array)  # (N, npix)

    # Per-shell directory.
    out_dir = tmp_path / "lc"
    _save_result(field, cosmo, Namespace(output=str(out_dir), name=None, shells_per_file=shells_per_file))
    files = sorted(out_dir.glob("shell_*.parquet"))
    assert len(files) == -(-_N_SHELLS // shells_per_file)  # ceil(N / k)

    chunks = [Catalog.from_parquet(str(f)).field[0] for f in files]
    # A 1-shell chunk reloads collapsed to (npix,); a multi-shell chunk stays (k, npix).
    arrays = [c.array if c.array.ndim == 2 else c.array[None] for c in chunks]
    got_arr = np.concatenate([np.asarray(a) for a in arrays], axis=0)
    assert got_arr.shape == ref_arr.shape
    np.testing.assert_array_equal(got_arr, ref_arr)  # identical HF round-trip -> bit-exact

    for attr in ("z_sources", "scale_factors", "comoving_centers", "density_width"):
        np.testing.assert_array_equal(
            _concat_meta(chunks, attr),
            np.atleast_1d(np.asarray(getattr(ref, attr))),
            err_msg=f"{attr} mismatch",
        )


def test_save_result_single_file_when_flag_unset(tmp_path, batched_lightcone):
    """``shells_per_file=0`` keeps the legacy single-file behavior (one parquet, one entry)."""
    field, cosmo = batched_lightcone
    out_file = tmp_path / "single.parquet"
    args = Namespace(output=str(out_file), name=None, shells_per_file=0)

    _save_result(field, cosmo, args)

    assert out_file.is_file()  # a file, not a directory
    reloaded = Catalog.from_parquet(str(out_file)).field[0]
    assert reloaded.array.shape == field.array.shape
    # float32 tolerance: HF's NumpyFormatter downcasts float64->float32 on read.
    np.testing.assert_allclose(np.asarray(reloaded.array), np.asarray(field.array), rtol=1e-5, atol=1e-6)


def test_shells_per_file_argparse_wiring():
    """``--shells-per-file`` parses to ``args.shells_per_file`` (default 0)."""
    args = parser().parse_args(["--sim-mode", "pm", "--output", "out", "--shells-per-file", "3"])
    assert args.shells_per_file == 3
    default = parser().parse_args(["--sim-mode", "pm", "--output", "out"])
    assert default.shells_per_file == 0


def test_per_shell_gathers_one_chunk_at_a_time(tmp_path, batched_lightcone, monkeypatch):
    """The memory-property that fixes the OOM: ``process_allgather`` only ever sees ONE chunk.

    Output-equality tests alone would still pass if the whole lightcone were gathered once and sliced
    host-side (the exact OOM). So spy on the gather and assert its leading dimension per call: with
    ``--shells-per-file 1`` it is gathered N times with leading dim 1 (peak host RAM ~= one shell);
    with the flag unset, once with leading dim N (the legacy whole-lightcone gather).
    """
    import jax_fli._src.io._field_catalog as fc

    field, cosmo = batched_lightcone
    original = fc.all_gather
    gathered_leading_dims: list[int] = []

    def spy(arr, *a, **k):
        gathered_leading_dims.append(int(arr.shape[0]))
        return original(arr, *a, **k)

    monkeypatch.setattr(fc, "all_gather", spy)

    _save_result(field, cosmo, Namespace(output=str(tmp_path / "lc"), name=None, shells_per_file=1))
    assert gathered_leading_dims == [1] * _N_SHELLS  # one shell gathered at a time

    gathered_leading_dims.clear()
    _save_result(field, cosmo, Namespace(output=str(tmp_path / "single.parquet"), name=None, shells_per_file=0))
    assert gathered_leading_dims == [_N_SHELLS]  # legacy path gathers the whole stack at once


@pytest.mark.scripts
def test_nbody_shells_per_file_cli_matches_single_file(tmp_path):
    """A 3-shell spherical run saved per-shell reloads+stacks to the same array as the single-file run."""
    ts = ["0.5", "0.75", "1.0"]
    base = (
        ["fli-simulate", "--sim-mode", "pm", "--nside", "16", "--ts", *ts]
        + _BASE_CLI
        + [
            "--t0",
            "0.01",
            "--t1",
            "1.0",
            "--nb-steps",
            "10",
            "--interp",
            "none",
            "--solver",
            "bf",
            "--time-stepping",
            "D",
            "--lpt-order",
            "1",
            "--gradient-order",
            "1",
        ]
    )

    single = str(tmp_path / "single.parquet")
    run_sim(base + ["--output", single])
    ref = np.asarray(jfli.io.Catalog.from_parquet(single).field[0].array)  # (3, npix)

    out_dir = tmp_path / "pershell"
    run_sim(base + ["--output", str(out_dir), "--shells-per-file", "1"])
    files = sorted(out_dir.glob("shell_*.parquet"))
    assert len(files) == 3
    stacked = np.stack([np.asarray(jfli.io.Catalog.from_parquet(str(f)).field[0].array) for f in files], axis=0)
    np.testing.assert_allclose(stacked, ref, rtol=1e-6, atol=1e-10)
