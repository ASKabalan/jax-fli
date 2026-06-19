#!/usr/bin/env python
# ruff: noqa: E402
"""Experiment 00 — re-publish the CosmoGrid density reference at native nside 2048.

The HuggingFace config ``00-cosmogrid-density`` was previously a float32 nside-512 downsample. The
native CosmoGrid lightcone is **nside 2048, uint16 particle counts, 69 shells** (z < 3.5), stored in
``compressed_shells.npz`` (confirmed against cosmogrid.ai/data_docs). This script rebuilds the field
straight from that npz at full resolution and publishes it as **density** (ρ = N / V_shell, float32),
split into a few per-redshift **set configs** ``00-cosmogrid-density-NN`` (see "Why sets + DENSITY").

**Why sets + DENSITY.** The full nside-2048 lightcone array (~67 shells) can't be one dataset config:
serializing it OOMs (``datasets.Dataset.from_dict`` balloons ~10× → > 62 GB RAM), and even if it fit,
``load_dataset`` combining ~67·npix ≈ 3.4 G elements overflows arrow's INT32 list offset. So we split
the **shell axis into ``--n-sets`` sets** (default 4), each published as its **own config**
``00-cosmogrid-density-NN`` — small enough to write under RAM and to load independently with plain
``load_dataset`` (each set's combine stays under INT32). We save as **DENSITY** (ρ = N / V_shell,
float32) rather than raw uint16 counts: that is the physical quantity AND makes each cell genuinely
4-byte, so the serializer's auto-split (sized by ``itemsize``) matches the parquet data-page size
(uint16 is upcast to int32 on disk, so a uint16 cell mis-splits and overflows the page).

By default we keep only the leading shells covering the source n(z) of **both** shears in ``jfli.data``
(Stage-3 reaches z≈1.7, DES Y3 z≈3.0 → ~67/69 shells); ``--all-shells`` keeps all 69. Reassemble the
full lightcone by loading each ``-NN`` config and concatenating along the shell axis (see the README).

Run on CPU (pure I/O + serialization; no GPU). ~16 GB host RAM is enough per set:

    python publish_density_2048.py --self-test     # fast: validate the serializer split round-trip, then exit
    python publish_density_2048.py --check          # inspect the PUBLISHED 00-cosmogrid-density-NN configs on HF
    python publish_density_2048.py --out /scratch/d.parquet   # build the set parquets locally (no upload) to verify
    python publish_density_2048.py --publish       # build sets + upload as 00-cosmogrid-density-NN configs

``HF_TOKEN`` must be set in the environment for ``--publish``.
"""

from __future__ import annotations

import os

# Pure host-side I/O + arrow serialization — pin to CPU so jnp.asarray of the 7 GB array does not
# land on a GPU and OOM. Must be set before jax / jax_fli import.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import gc
import re
import tarfile
from pathlib import Path

import jax
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


def _n_shells_for_sources(shell_info, frac: float = 1e-3) -> tuple[int, float]:
    """Leading shells needed so the lightcone covers the source n(z) of BOTH shears in ``jfli.data``.

    A shell is a lens plane for a source at z_s when its near edge z_lower < z_s, so we keep every
    shell whose ``lower_z`` is below the max source redshift across Stage-3 + DES Y3 (Stage-3 reaches
    z≈1.7, DES Y3 z≈3.0). Shells entirely beyond all sources contribute nothing and are dropped.
    """
    from jax_fli.data import get_des_y3_nz_shear, get_stage3_nz_shear

    z_src_max = 0.0
    for nz in list(get_stage3_nz_shear()) + list(get_des_y3_nz_shear()):
        zc = np.asarray(nz.params[0])
        w = np.asarray(nz.params[1])
        zsup = zc[w > frac * w.max()]
        z_src_max = max(z_src_max, float(zsup.max()))
    lower_z = np.asarray(shell_info["lower_z"])
    return int(np.sum(lower_z < z_src_max)), z_src_max


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
# Check — what do the published HF density entries (00-cosmogrid-density-NN) expose?
# --------------------------------------------------------------------------------------------------
def _fmt_array(x) -> str:
    if x is None:
        return "None"
    a = np.asarray(x).ravel()
    if a.size > 8:
        return f"len={a.size} min={float(a.min()):.4g} max={float(a.max()):.4g}"
    return np.array2string(a, precision=4, separator=", ")


def _density_config_names(meta, prefix: str) -> list[str]:
    """All set configs ``<prefix>-NN`` in the card (sorted); falls back to an exact ``prefix`` config."""
    names = [c.get("config_name") for c in meta.get("configs") or []]
    sets = sorted(n for n in names if n and re.fullmatch(re.escape(prefix) + r"-\d+", n))
    if not sets and prefix in names:
        sets = [prefix]
    return sets


def check(prefix: str) -> None:
    """Report the published density entries: load each ``<prefix>-NN`` config, concatenate, summarize.

    Each set config loads independently with plain ``load_dataset`` (no combine overflow); the sets are
    concatenated along the shell axis into the full lightcone. Downloads the parquet(s) into ``HF_HOME``.
    """
    from datasets import load_dataset
    from huggingface_hub import HfApi

    api = HfApi()
    meta, _ = _load_card(api)
    sets = _density_config_names(meta, prefix)
    if not sets:
        print(f"[check] no '{prefix}' or '{prefix}-NN' configs in the dataset card on {REPO}.")
        return
    print(f"[check] {REPO}: {len(sets)} density entr(y/ies) {sets}. Loading each + concatenating …")
    fields, c = [], None
    for n in sets:
        cat = jfli.io.Catalog.from_dataset(load_dataset(REPO, n, split="train").with_format("numpy"))
        fields += cat.field
        c = cat.cosmology[0]
        f = cat.field[0]
        print(f"   {n}: {tuple(f.array.shape)} {f.array.dtype} nside={f.nside} {f.status.name}/{f.unit.name}")
    full = jax.tree.map(lambda *a: jnp.concatenate(a, axis=0), *fields) if len(fields) > 1 else fields[0]
    z = np.asarray(full.z_sources)
    print(
        f"  joined     : {tuple(full.array.shape)} {full.array.dtype} nside={full.nside} {full.status.name}/{full.unit.name}"
    )
    print(
        f"  shells={z.size}  z=[{float(z.min()):.3f}, {float(z.max()):.3f}]  density_width={_fmt_array(full.density_width)}"
    )
    print(
        f"  cosmo      : Oc={float(c.Omega_c):.4f} Ob={float(c.Omega_b):.4f} h={float(c.h):.4f} "
        f"s8={float(c.sigma8):.4f} ns={float(c.n_s):.4f} w0={float(c.w0):.4f} wa={float(c.wa):.4f} "
        f"Onu={float(c.Omega_nu):.5f}"
    )


# --------------------------------------------------------------------------------------------------
# Publish — dataset-card helpers (the chunks register a glob ``data_files`` for the config)
# --------------------------------------------------------------------------------------------------
def _load_card(api):
    """Return (meta, body): the dataset card's parsed YAML front-matter and the markdown body."""
    import yaml
    from huggingface_hub import hf_hub_download

    text = open(hf_hub_download(REPO, "README.md", repo_type="dataset")).read()
    parts = text.split("---")
    if len(parts) >= 3:
        return (yaml.safe_load(parts[1]) or {}), "---".join(parts[2:])
    return {}, text


def _config_path(meta, name):
    for cfg in meta.get("configs") or []:
        if cfg.get("config_name") == name:
            df = cfg["data_files"]
            return df[0] if isinstance(df, list) else df
    return None


def _ensure_config(meta, name, path):
    cfgs = meta.setdefault("configs", [])
    for cfg in cfgs:
        if cfg.get("config_name") == name:
            cfg["data_files"] = path
            return
    cfgs.append({"config_name": name, "data_files": path})


def _save_card(api, meta, body):
    import yaml

    text = "---\n" + yaml.safe_dump(meta, sort_keys=False) + "---" + body
    api.upload_file(path_or_fileobj=text.encode(), path_in_repo="README.md", repo_id=REPO, repo_type="dataset")


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
    ap.add_argument(
        "--out", default=None, help="local parquet path stem (sets append _setNNofMM; default: next to this script)"
    )
    ap.add_argument(
        "--n-sets",
        type=int,
        default=4,
        help="number of shell-sets / configs to split the lightcone into (each set keeps to_parquet under RAM)",
    )
    ap.add_argument(
        "--all-shells",
        action="store_true",
        help="publish all 69 shells (default: only those covering the jfli.data source n(z), ~67)",
    )
    ap.add_argument(
        "--publish",
        action="store_true",
        help="upload the sets to HuggingFace as 00-cosmogrid-density-NN configs (replacing the old single config)",
    )
    args = ap.parse_args()

    if args.check:
        check(DENSITY_CONFIG)
        return

    if args.self_test:
        self_test()
        return
    if not args.skip_self_test:
        self_test()

    run_dir = Path(args.sim_root) / args.run
    npz = np.load(run_dir / "compressed_shells.npz")  # lazy NpzFile
    shell_info = npz["shell_info"]  # small struct array
    cosmo, mesh_size, box_size, z, a, com, dw = _parse_cosmo_and_geometry(run_dir, shell_info)

    # Extent: keep enough leading shells to cover the source n(z) of BOTH shears in jfli.data
    # (Stage-3 reaches z≈1.7, DES Y3 z≈3.0), unless --all-shells is given.
    if args.all_shells:
        n_keep, z_src_max = len(z), float(z.max())
    else:
        n_keep, z_src_max = _n_shells_for_sources(shell_info)
    print(f"Source n(z) (Stage-3 + DES Y3) reach z≈{z_src_max:.3f} → keeping {n_keep}/{len(z)} shells.")

    print(f"Loading raw uint16 shells from {run_dir / 'compressed_shells.npz'} …")
    shells = npz["shells"]  # (69, 50331648) uint16 ≈ 7 GB
    nside = int(round((shells.shape[-1] / 12) ** 0.5))
    assert shells.dtype == np.uint16, f"expected uint16 raw counts, got {shells.dtype}"
    assert 12 * nside * nside == shells.shape[-1], "npix is not 12·nside²"
    print(
        f"   cosmo: Oc={float(cosmo.Omega_c):.4f} Ob={float(cosmo.Omega_b):.4f} h={float(cosmo.h):.4f} "
        f"s8={float(cosmo.sigma8):.4f} ns={float(cosmo.n_s):.4f} w0={float(cosmo.w0):.4f}"
    )

    # Split the shell axis into N sets, each saved as its OWN config ("entry"). A single config
    # spanning all shells would overflow arrow's INT32 element offset when load_dataset combines it
    # (67·npix ≈ 3.4 G > 2.1 G), so per-set configs let plain load_dataset read them one by one.
    # Save as DENSITY (ρ = N / V_shell, float32): this is the physical quantity AND makes each cell
    # genuinely 4-byte, so the serializer's auto-split (which sizes by itemsize) matches the parquet
    # data page (raw uint16 is upcast to int32 on disk → mis-split → page overflow).
    n_sets = args.n_sets
    bounds = np.linspace(0, n_keep, n_sets + 1).astype(int)
    ranges = [(int(bounds[i]), int(bounds[i + 1] - bounds[i])) for i in range(n_sets) if bounds[i + 1] > bounds[i]]
    n_sets = len(ranges)
    base = Path(args.out).parent if args.out else HERE
    stem = Path(args.out).stem if args.out else "cosmogrid_density_nside2048"
    print(
        f"Splitting {n_keep} shells → {n_sets} sets of {[sz for _, sz in ranges]} shells, saved as DENSITY (float32):"
    )

    set_paths = []
    for ci, (s, sz) in enumerate(ranges):
        e = s + sz
        fld = build_density_field(shells[s:e], nside, z[s:e], a[s:e], com[s:e], dw[s:e], mesh_size, box_size)
        fld = fld.to(jfli.DensityUnit.DENSITY)  # COUNTS → ρ = N / V_shell (float32)
        outp = base / f"{stem}_set{ci:02d}of{n_sets:02d}.parquet"
        print(
            f"   [{ci + 1}/{n_sets}] shells {s}:{e}  {tuple(fld.array.shape)} {fld.array.dtype} ({fld.array.nbytes / 1e9:.2f} GB) → {outp.name} …"
        )
        jfli.io.Catalog(field=fld, cosmology=cosmo).to_parquet(str(outp))
        print(f"        wrote {outp} ({outp.stat().st_size / 1e9:.2f} GB)")
        set_paths.append(outp)
        del fld
        gc.collect()
    del shells, npz

    if not args.publish:
        print(f"Not publishing (pass --publish). Wrote {n_sets} local set parquet(s) under {base}.")
        print(f"   Load: concatenate the {n_sets} '{DENSITY_CONFIG}-NN' configs along the shell axis (see README).")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    meta, body = _load_card(api)
    existing = _config_path(meta, DENSITY_CONFIG)
    catalogs_dir = str(Path(existing).parent) if existing else "00-cosmogrid/catalogs"
    print(f"\nPublishing {n_sets} sets to {REPO} as configs {DENSITY_CONFIG}-00..{n_sets - 1:02d}:")
    for ci, outp in enumerate(set_paths):
        repo_path = f"{catalogs_dir}/{outp.name}"
        config_name = f"{DENSITY_CONFIG}-{ci:02d}"
        print(f"   uploading [{ci + 1}/{n_sets}] → {repo_path}  (config {config_name}) …")
        api.upload_file(path_or_fileobj=str(outp), path_in_repo=repo_path, repo_id=REPO, repo_type="dataset")
        _ensure_config(meta, config_name, repo_path)
    # Drop the stale single-file 00-cosmogrid-density config (old nside-512) — only the set entries remain.
    meta["configs"] = [c for c in meta.get("configs", []) if c.get("config_name") != DENSITY_CONFIG]
    _save_card(api, meta, body)
    print(
        f"   done. nside-{nside} DENSITY published as {n_sets} configs {DENSITY_CONFIG}-00..{n_sets - 1:02d} ({n_keep} shells)."
    )
    print("   Load each config, then concatenate along the shell axis (see README).")
    if existing:
        print(f"   NOTE: old parquet {existing} is now orphaned (config removed); delete it on the Hub if you want.")


if __name__ == "__main__":
    main()
