#!/usr/bin/env python
# ruff: noqa: E402
"""Experiment 00 — re-publish the CosmoGrid density reference at native nside 2048, uint16.

The HuggingFace config ``00-cosmogrid-density`` was previously a float32 nside-512 downsample. The
native CosmoGrid lightcone is **nside 2048, uint16 particle counts, 69 shells** (z < 3.5), stored in
``compressed_shells.npz`` (confirmed against cosmogrid.ai/data_docs). This script rebuilds the field
straight from that npz at full resolution, **preserving the raw uint16 precision** (≈ 7 GB, lossless —
CosmoGrid stores counts as uint16, so nothing overflows), and overwrites the HuggingFace config.

The jax-fli catalog serializer is dtype-preserving (``build_features`` reads ``array.dtype``;
``row_to_field_cosmo`` restores it from the ``array_dtype`` column), so a uint16 ``SphericalDensity``
round-trips as uint16. A 6.9 GB array exceeds ``INT32_MAX`` and therefore takes the parquet *split*
path (``array_0..N`` + ``_n_splits`` + ``_original_n0``) that the old 512 file never exercised — the
``--self-test`` below validates that path on a tiny array before the multi-GB build.

Run on CPU (pure I/O + serialization; avoids GPU OOM on the 7 GB array). ~16 GB host RAM recommended:

    python publish_density_2048.py --self-test     # fast: validate the split round-trip, then exit
    python publish_density_2048.py --check          # inspect the PUBLISHED 00-cosmogrid-density config on HF + its attrs
    python publish_density_2048.py                 # build + save locally (no upload)
    python publish_density_2048.py --publish       # build + OVERWRITE 00-cosmogrid-density on HF

``HF_TOKEN`` must be set in the environment for ``--publish``.
"""

from __future__ import annotations

import os

# Pure host-side I/O + arrow serialization — pin to CPU so jnp.asarray of the 7 GB array does not
# land on a GPU and OOM. Must be set before jax / jax_fli import.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import tarfile
from pathlib import Path

import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np

import jax_fli as jfli

REPO = "ASKabalan/jax-fli-experiments"
DENSITY_CONFIG = "00-cosmogrid-density"
HERE = Path(__file__).resolve().parent

# Provenance of the current Experiment 0 reference (cosmo_000001/run_0).
DEFAULT_SIM_ROOT = Path("/home/wassim/Projects/NBody/Simulations")
DEFAULT_RUN = "CosmoGrid/raw/cosmo_000001/run_0"


# --------------------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------------------
def _parse_cosmo_and_geometry(run_dir: Path, shell_info: np.ndarray):
    """Mirror ``load_cosmogrid_lc``'s metadata parsing (cosmology + box/grid + shell geometry).

    Reads only the small text/struct metadata — NOT the 7 GB ``shells`` array.
    """
    # Cosmology from params.yml
    params: dict = {}
    with open(run_dir / "params.yml") as f:
        for line in f:
            if ":" in line:
                key, val = line.strip().split(":", 1)
                try:
                    params[key.strip()] = float(val.strip())
                except ValueError:
                    params[key.strip()] = val.strip()
    cosmo = jc.Cosmology(
        Omega_c=float(params["O_cdm"]),
        Omega_b=float(params["Ob"]),
        h=float(params["H0"]) / 100.0,
        n_s=float(params["ns"]),
        sigma8=float(params["s8"]),
        w0=float(params["w0"]),
        wa=float(params["wa"]),
        Omega_k=0.0,
        Omega_nu=float(params["O_nu"]),
    )

    # Box / grid from cosmology.par (flat or inside param_files.tar.gz)
    sim_specs: dict = {}
    flat = run_dir / "param_files" / "cosmology.par"
    tar = run_dir / "param_files.tar.gz"
    if flat.exists():
        for line in open(flat):
            line = line.split("#")[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                sim_specs[k.strip()] = v.strip()
    elif tar.exists():
        with tarfile.open(tar, "r:gz") as t:
            try:
                fobj = t.extractfile(t.getmember("cosmology.par"))
                if fobj is not None:
                    for line in fobj.read().decode("utf-8").splitlines():
                        line = line.split("#")[0].strip()
                        if "=" in line:
                            k, v = line.split("=", 1)
                            sim_specs[k.strip()] = v.strip()
            except KeyError:
                pass

    upper_com = shell_info["upper_com"]
    box = float(sim_specs.get("dBoxSize", np.max(upper_com)))
    n_grid = int(sim_specs.get("nGrid", 0))
    mesh_size = (n_grid, n_grid, n_grid) if n_grid > 0 else (int(box), int(box), int(box))
    box_size = (box, box, box)

    lower_z, upper_z = shell_info["lower_z"], shell_info["upper_z"]
    lower_com = shell_info["lower_com"]
    z_centers = 0.5 * (lower_z + upper_z)
    comoving_centers = 0.5 * (lower_com + upper_com)
    shell_widths = np.abs(upper_com - lower_com)
    scale_factors = 1.0 / (1.0 + z_centers)
    return cosmo, mesh_size, box_size, z_centers, scale_factors, comoving_centers, shell_widths


def build_density_field(shells, nside, z_centers, scale_factors, comoving_centers, shell_widths, mesh_size, box_size):
    """One stacked ``SphericalDensity`` (n_shells, npix) keeping the raw dtype of ``shells``.

    No ``ud_sample`` (native→native is a no-op that would force a float pass). Mirrors the stacked
    layout ``load_cosmogrid_lc`` produces: dynamic metadata is 1-D, length n_shells.
    """
    return jfli.SphericalDensity(
        array=jnp.asarray(shells),  # keeps uint16 (x64-off does not affect <64-bit dtypes)
        mesh_size=tuple(int(x) for x in mesh_size),
        box_size=tuple(float(x) for x in box_size),
        observer_position=(0.5, 0.5, 0.5),
        field_sharding=None,
        halo_size=(0, 0),
        nside=int(nside),
        z_sources=jnp.asarray(np.asarray(z_centers, dtype=np.float32)),
        scale_factors=jnp.asarray(np.asarray(scale_factors, dtype=np.float32)),
        comoving_centers=jnp.asarray(np.asarray(comoving_centers, dtype=np.float32)),
        density_width=jnp.asarray(np.asarray(shell_widths, dtype=np.float32)),
        status=jfli.FieldStatus.LIGHTCONE,
        unit=jfli.DensityUnit.COUNTS,
    )


# --------------------------------------------------------------------------------------------------
# Pre-flight: the >2 GB split path is untested in the repo — validate it on a tiny array.
# --------------------------------------------------------------------------------------------------
def self_test() -> None:
    """Force the parquet split path on a tiny uint16 field; assert dtype/shape/values survive."""
    import tempfile

    import pyarrow.parquet as pq

    import jax_fli._src.io._field_catalog as fc

    saved = fc._INT32_MAX
    fc._INT32_MAX = 1000  # tiny threshold → even a small array takes the split branch
    try:
        nside, nsh = 8, 4
        npix = 12 * nside * nside
        shells = np.random.default_rng(0).integers(0, 65535, size=(nsh, npix)).astype(np.uint16)
        z = np.linspace(0.1, 1.0, nsh).astype(np.float32)
        cosmo = jc.Cosmology(
            Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.97, sigma8=0.8, w0=-1.0, wa=0.0, Omega_k=0.0, Omega_nu=0.0
        )
        field = build_density_field(
            shells,
            nside,
            z,
            1.0 / (1.0 + z),
            np.linspace(100, 500, nsh),
            np.full(nsh, 50.0),
            (128, 128, 128),
            (400.0, 400.0, 400.0),
        )
        cat = jfli.io.Catalog(field=field, cosmology=cosmo)
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "t.parquet")
            cat.to_parquet(p)
            cols = pq.ParquetFile(p).schema_arrow.names
            assert "_n_splits" in cols and "array_0" in cols, f"split path not taken — columns: {cols}"
            back = jfli.io.Catalog.from_parquet(p).field[0]
        assert back.array.dtype == np.uint16, f"dtype changed: {back.array.dtype}"
        assert tuple(back.array.shape) == (nsh, npix), f"shape changed: {back.array.shape}"
        assert np.array_equal(np.asarray(back.array), shells), "values changed across the split round-trip"
        # Metadata/status that jfli.raytrace dereferences must survive the round-trip too — otherwise
        # a bad direct build passes the array check but crashes raytrace_kappa.py after a 7 GB upload.
        assert back.status == jfli.FieldStatus.LIGHTCONE, f"status changed: {back.status}"
        assert back.unit == jfli.DensityUnit.COUNTS, f"unit changed: {back.unit}"
        assert back.density_width is not None, "density_width lost (raytrace requires it)"
        assert int(back.nside) == nside, f"nside changed: {back.nside}"
        assert np.asarray(back.scale_factors).shape[0] == nsh, "scale_factors length != n_shells"
        assert np.asarray(back.comoving_centers).shape[0] == nsh, "comoving_centers length != n_shells"
        print(
            "[self-test] PASS — split-path uint16 round-trip preserved array (dtype/shape/values) "
            "and the LIGHTCONE status + per-shell metadata raytrace needs."
        )
    finally:
        fc._INT32_MAX = saved


# --------------------------------------------------------------------------------------------------
# Check — does the published HF config exist, and what is in it?
# --------------------------------------------------------------------------------------------------
def _fmt_array(x) -> str:
    if x is None:
        return "None"
    a = np.asarray(x).ravel()
    if a.size > 8:
        return f"len={a.size} min={float(a.min()):.4g} max={float(a.max()):.4g}"
    return np.array2string(a, precision=4, separator=", ")


def check(config: str) -> None:
    """Report whether the published HuggingFace config exists and print its field + cosmology attributes.

    Downloads the config's parquet from the Hub and loads via ``Catalog.from_parquet``, which
    materializes the full array — so once the 2048 reference is published this pulls/reads the whole
    ≈ 7 GB density (cached under ``HF_HOME``; run on the CPU box).
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    try:
        target = _hf_data_files_path(api, config)
    except RuntimeError as e:
        print(f"[check] config {config} is not in the dataset card on {REPO}: {e}")
        return
    if not api.file_exists(REPO, target, repo_type="dataset"):
        print(f"[check] config {config} points at {target}, which does NOT exist on {REPO}.")
        return
    print(f"[check] {REPO}:{config} → {target} exists. Downloading + loading …")
    local = hf_hub_download(REPO, target, repo_type="dataset")
    cat = jfli.io.Catalog.from_parquet(local)
    f = cat.field[0]
    c = cat.cosmology[0]
    print(f"  source     : {local} ({os.path.getsize(local) / 1e9:.2f} GB)")
    print(f"  field      : {type(f).__name__}  status={f.status.name}  unit={f.unit.name}")
    print(f"  array      : shape={tuple(f.array.shape)}  dtype={f.array.dtype}  nside={f.nside}")
    print(f"  geometry   : mesh_size={f.mesh_size}  box_size={f.box_size}  observer={f.observer_position}")
    print(f"  z_sources  : {_fmt_array(f.z_sources)}")
    print(f"  scale_facs : {_fmt_array(f.scale_factors)}")
    print(f"  comoving   : {_fmt_array(f.comoving_centers)}")
    print(f"  width      : {_fmt_array(f.density_width)}")
    print(
        f"  cosmo      : Oc={float(c.Omega_c):.4f} Ob={float(c.Omega_b):.4f} h={float(c.h):.4f} "
        f"s8={float(c.sigma8):.4f} ns={float(c.n_s):.4f} w0={float(c.w0):.4f} wa={float(c.wa):.4f} "
        f"Onu={float(c.Omega_nu):.5f}"
    )


# --------------------------------------------------------------------------------------------------
# Publish
# --------------------------------------------------------------------------------------------------
def _hf_data_files_path(api, config_name: str) -> str:
    """Resolve the concrete parquet path the given dataset config points at (from the live card)."""
    import fnmatch

    import yaml
    from huggingface_hub import hf_hub_download

    fm = open(hf_hub_download(REPO, "README.md", repo_type="dataset")).read().split("---")[1]
    target = None
    for cfg in yaml.safe_load(fm).get("configs") or []:
        if cfg.get("config_name") == config_name:
            df = cfg["data_files"]
            target = df[0] if isinstance(df, list) else df
            break
    if target is None:
        raise RuntimeError(f"config '{config_name}' not found in the dataset card on {REPO}")
    if "*" in target:
        matches = [f for f in api.list_repo_files(REPO, repo_type="dataset") if fnmatch.fnmatch(f, target)]
        if len(matches) != 1:
            raise RuntimeError(f"data_files glob '{target}' resolves ambiguously: {matches}")
        target = matches[0]
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--check",
        action="store_true",
        help=f"only check the published {DENSITY_CONFIG} config on HuggingFace and print its field attributes, then exit",
    )
    ap.add_argument("--self-test", action="store_true", help="run only the split-path round-trip test, then exit")
    ap.add_argument("--skip-self-test", action="store_true", help="skip the pre-flight test before the big build")
    ap.add_argument("--sim-root", default=str(DEFAULT_SIM_ROOT))
    ap.add_argument("--run", default=DEFAULT_RUN, help="run dir relative to --sim-root")
    ap.add_argument("--out", default=None, help="local parquet path (default: next to this script)")
    ap.add_argument("--publish", action="store_true", help="OVERWRITE 00-cosmogrid-density on HuggingFace")
    args = ap.parse_args()

    if args.check:
        check(DENSITY_CONFIG)
        return

    out = Path(args.out) if args.out else HERE / "cosmogrid_density_nside2048.parquet"

    if args.self_test:
        self_test()
        return
    if not args.skip_self_test:
        self_test()

    run_dir = Path(args.sim_root) / args.run
    npz = np.load(run_dir / "compressed_shells.npz")  # lazy NpzFile
    shell_info = npz["shell_info"]  # small struct array
    cosmo, mesh_size, box_size, z, a, com, dw = _parse_cosmo_and_geometry(run_dir, shell_info)

    print(f"Loading raw uint16 shells from {run_dir / 'compressed_shells.npz'} …")
    shells = npz["shells"]  # (69, 50331648) uint16 ≈ 7 GB
    nside = int(round((shells.shape[-1] / 12) ** 0.5))
    assert shells.dtype == np.uint16, f"expected uint16 raw counts, got {shells.dtype}"
    assert 12 * nside * nside == shells.shape[-1], "npix is not 12·nside²"

    field = build_density_field(shells, nside, z, a, com, dw, mesh_size, box_size)
    cat = jfli.io.Catalog(field=field, cosmology=cosmo)
    del shells, npz  # free the numpy copy; the field holds its own array

    print("Built SphericalDensity:")
    print(f"   shape={tuple(field.array.shape)} dtype={field.array.dtype} nside={nside}  unit={field.unit.name}")
    print(f"   shells={field.array.shape[0]}  z=[{float(z.min()):.3f}, {float(z.max()):.3f}]")
    print(
        f"   cosmo: Oc={float(cosmo.Omega_c):.4f} Ob={float(cosmo.Omega_b):.4f} h={float(cosmo.h):.4f} "
        f"s8={float(cosmo.sigma8):.4f} ns={float(cosmo.n_s):.4f} w0={float(cosmo.w0):.4f}"
    )

    print(f"Writing parquet → {out} (this materializes ~7 GB through arrow) …")
    cat.to_parquet(str(out))
    print(f"   wrote {out}  ({out.stat().st_size / 1e9:.2f} GB)")

    if not args.publish:
        print("Not publishing (pass --publish to overwrite the HuggingFace config).")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    target = _hf_data_files_path(api, DENSITY_CONFIG)
    print(f"\nOVERWRITING {REPO}:{target}")
    print(f"   replacing the old nside-512 float32 density with nside-{nside} uint16 ({field.array.shape[0]} shells).")
    api.upload_file(path_or_fileobj=str(out), path_in_repo=target, repo_id=REPO, repo_type="dataset")
    print("   uploaded. 00-cosmogrid-density now serves the 2048 uint16 reference.")


if __name__ == "__main__":
    main()
