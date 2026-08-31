#!/usr/bin/env python
# ruff: noqa: E402
"""Experiment 00 — re-publish the CosmoGrid density reference at native nside 2048.

Streams the native CosmoGrid lightcone to disk with the shared loader
``jfli.io.load_cosmogrid_lc(..., output=folder)`` (which parses the cosmology + shell geometry,
**selects shells by redshift**, and writes **one parquet per shell** — ``shell_NNN.parquet`` — in its
native **COUNTS** unit, float32), then publishes those files as **one config** ``00-cosmogrid-<id>-density``
with **one row per shell** on HuggingFace. (COUNTS is what every other ``load_cosmogrid_lc`` consumer
gets; ``jfli.born``/``jfli.raytrace`` convert to overdensity internally.)

**Extent (z ≤ ``--max-z``, default 1.6).** Depth is set by the **DES Y3** weak-lensing sources, whose
deepest bin ends at z≈1.45 (10%-of-peak; see ``docs/5-experiments/06-cosmogrid-shells``). So we keep
shells with ``lower_z <= max_z`` (~56 shells at 1.6) — the small Stage-3 bin-4 tail beyond 1.6 is
negligible for the lensing kernel, so this one reference serves both the DES and Stage-3 born/raytrace
paths. (``load_cosmogrid_lc(max_redshift=…)`` does the masking.)

**One config, one parquet per shell.** The full nside-2048 lightcone can't be one parquet: stacking it
to write OOMs, and ``load_dataset`` combining ~n_shells·npix > INT32 elements overflows arrow's list
offset. So the loader writes **one ``(npix,)`` parquet per shell** (~200 MB), and we register them all
as separate **rows** under **one config** ``00-cosmogrid-<id>-density`` (a glob ``data_files``). Writing one
shell at a time keeps peak RAM ~one shell. Reassemble by **streaming** the config and stacking:

    ds = load_dataset(REPO, "00-cosmogrid-000001-density", split="train", streaming=True)
    field = SphericalDensity.stack([Catalog.from_dataset(r).field[0] for r in ds])  # (n_shells, npix)

Each row is one ``(npix,)`` shell; ``stack`` adds the leading shell axis. Streaming is **required**: a
plain (non-streaming) ``load_dataset`` would combine all shell rows into one arrow table and overflow
the INT32 offset.

Run on CPU (pure I/O + serialization; no GPU); the per-shell stream keeps peak RAM low:

    python publish_density_2048.py --self-test     # fast: validate the serializer split round-trip, then exit
    python publish_density_2048.py --check          # STREAM the PUBLISHED 00-cosmogrid-<id>-density config on HF
    python publish_density_2048.py --out /scratch/density   # write the per-shell parquets to a folder (no upload)
    python publish_density_2048.py --publish       # write shells + upload as one config 00-cosmogrid-<id>-density

A second cosmology goes to its own folder and its own config — ``--run`` and ``--cosmo`` must agree::

    python publish_density_2048.py --run CosmoGrid/raw/cosmo_172798/run_0 --cosmo cosmo_172798 --publish

``HF_TOKEN`` must be set in the environment for ``--publish``.
"""

from __future__ import annotations

import os

# Pure host-side I/O + arrow serialization — pin to CPU so jnp.asarray of the 7 GB array does not
# land on a GPU and OOM. Must be set before jax / jax_fli import.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import re
from pathlib import Path

import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np

import jax_fli as jfli

REPO = "ASKabalan/jax-fli-experiments"
HERE = Path(__file__).resolve().parent

# Provenance of the first Experiment 0 reference (cosmo_000001/run_0). --run and --cosmo travel
# together: the raw run supplies the data, the cosmology names both the folder it lands in and the
# config that exposes it. They are cross-checked below, because publishing one cosmology under
# another's name is silent and only shows up as a wrong reference curve months later.
DEFAULT_SIM_ROOT = Path("/home/wassim/Projects/NBody/Simulations")
DEFAULT_RUN = "CosmoGrid/raw/cosmo_000001/run_0"
DEFAULT_COSMO = "cosmo_000001"


# --------------------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------------------
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
# Pre-flight: validate the serializer split round-trip on a tiny array.
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
# Check — what does the published HF config 00-cosmogrid-<id>-density expose? (one config, many shell rows)
# --------------------------------------------------------------------------------------------------
def _fmt_array(x) -> str:
    if x is None:
        return "None"
    a = np.asarray(x).ravel()
    if a.size > 8:
        return f"len={a.size} min={float(a.min()):.4g} max={float(a.max()):.4g}"
    return np.array2string(a, precision=4, separator=", ")


def check(config: str) -> None:
    """STREAM the published config's per-shell rows, stack along the shell axis, and summarize.

    The config holds one ``(npix,)`` COUNTS row per shell. We must STREAM it (a non-streaming
    ``load_dataset`` would combine ~n_shells·npix > INT32 elements and overflow arrow's list offset),
    then ``SphericalDensity.stack`` the rows into a ``(n_shells, npix)`` lightcone.
    """
    from datasets import load_dataset
    from huggingface_hub import HfApi

    api = HfApi()
    meta, _ = _load_card(api)
    if _config_path(meta, config) is None:
        print(f"[check] config {config} is not in the dataset card on {REPO}.")
        return
    print(f"[check] {REPO}:{config} — streaming per-shell rows + concatenating …")
    fields, c = [], None
    for row in load_dataset(REPO, config, split="train", streaming=True):
        cat = jfli.io.Catalog.from_dataset(row)
        fields += cat.field
        c = cat.cosmology[0]
    if not fields:
        print(f"[check] config {config} has no rows.")
        return
    # Each row is one (npix,) shell; stack adds the leading shell axis → (n_shells, npix).
    full = jfli.SphericalDensity.stack(fields)
    f0 = fields[0]
    z = np.asarray(full.z_sources)
    print(
        f"  rows={len(fields)} (per-shell)  each {tuple(f0.array.shape)} {f0.array.dtype} {f0.status.name}/{f0.unit.name}"
    )
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
        help="only check the published 00-cosmogrid-<id>-density config on HuggingFace and print its field attributes, then exit",
    )
    ap.add_argument("--self-test", action="store_true", help="run only the split-path round-trip test, then exit")
    ap.add_argument("--skip-self-test", action="store_true", help="skip the pre-flight test before the big build")
    ap.add_argument("--sim-root", default=str(DEFAULT_SIM_ROOT))
    ap.add_argument("--run", default=DEFAULT_RUN, help="run dir relative to --sim-root")
    ap.add_argument(
        "--cosmo",
        default=DEFAULT_COSMO,
        help=f"cosmology this run belongs to; sets the destination 00-cosmogrid/COSMO/density and the config 00-cosmogrid-<id>-density. Must appear in --run. Default {DEFAULT_COSMO}.",
    )
    ap.add_argument("--out", default=None, help="local folder for the per-shell parquets (default: a temp dir)")
    ap.add_argument(
        "--max-z",
        type=float,
        default=1.6,
        dest="max_z",
        help="keep shells with lower_z <= MAX_Z (DES Y3 source depth ~1.45, see exp 06; passed to load_cosmogrid_lc(max_redshift=...)). Default 1.6.",
    )
    ap.add_argument(
        "--publish",
        action="store_true",
        help="upload per-shell parquet rows to HuggingFace under one config 00-cosmogrid-<id>-density",
    )
    args = ap.parse_args()

    if not re.fullmatch(r"cosmo_\d{6}", args.cosmo):
        raise SystemExit(f"--cosmo must look like cosmo_000001, got {args.cosmo!r}")
    if args.cosmo not in args.run:
        raise SystemExit(f"--cosmo {args.cosmo} does not appear in --run {args.run} — refusing to mislabel the upload")
    density_dir = f"00-cosmogrid/{args.cosmo}/density"
    density_config = f"00-cosmogrid-{args.cosmo.removeprefix('cosmo_')}-density"

    if args.check:
        check(density_config)
        return

    if args.self_test:
        self_test()
        return
    if not args.skip_self_test:
        self_test()

    import tempfile

    run_dir = Path(args.sim_root) / args.run
    # The loader streams ONE parquet per shell to a folder (shell_NNN.parquet) — it never stacks the
    # whole lightcone, so peak RAM stays ~one shell (a full-stack load of the nside-2048 run OOMs).
    # Native COUNTS float32 (born/raytrace convert to overdensity internally; this matches every other
    # load_cosmogrid_lc consumer). Each file is one HF ROW; all register under ONE config via a glob.
    out_dir = Path(args.out).resolve() if args.out else Path(tempfile.mkdtemp(prefix="cosmogrid_density_"))
    print(f"Streaming CosmoGrid density shells (z ≤ {args.max_z}) → {out_dir} via load_cosmogrid_lc …")
    jfli.io.load_cosmogrid_lc(run_dir, max_redshift=args.max_z, output=out_dir)
    shell_paths = sorted(out_dir.glob("shell_*.parquet"))
    if not shell_paths:
        raise RuntimeError(f"no shell parquets written to {out_dir}")
    s0 = jfli.io.Catalog.from_parquet(str(shell_paths[0]))
    f0, cosmo = s0.field[0], s0.cosmology[0]
    print(
        f"   wrote {len(shell_paths)} shell parquet(s) (~{shell_paths[0].stat().st_size / 1e6:.0f} MB each); "
        f"{type(f0).__name__} nside={f0.nside} {f0.array.dtype} {f0.unit.name}; "
        f"cosmo Oc={float(cosmo.Omega_c):.4f} h={float(cosmo.h):.4f} s8={float(cosmo.sigma8):.4f}"
    )

    if not args.publish:
        print(f"Not publishing (pass --publish). Wrote {len(shell_paths)} per-shell parquet(s) under {out_dir}.")
        print(f"   Load: load_dataset(REPO, '{density_config}', split='train', streaming=True) → concatenate rows.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    meta, body = _load_card(api)
    prefix = "cosmogrid_density_nside2048"
    glob = f"{density_dir}/{prefix}_shell*.parquet"
    print(f"\nUploading {len(shell_paths)} shell parquet(s) to {REPO} under ONE config {density_config} …")
    for i, p in enumerate(shell_paths):
        repo_path = f"{density_dir}/{prefix}_{p.name}"  # cosmogrid_density_nside2048_shell_NNN.parquet
        print(f"   [{i + 1}/{len(shell_paths)}] → {repo_path}")
        api.upload_file(path_or_fileobj=str(p), path_in_repo=repo_path, repo_id=REPO, repo_type="dataset")
    # ONE config pointing at all shell files via a glob; drop any stale per-set -NN configs.
    meta["configs"] = [
        c
        for c in meta.get("configs", [])
        if not re.fullmatch(re.escape(density_config) + r"-\d+", c.get("config_name", ""))
    ]
    _ensure_config(meta, density_config, glob)
    _save_card(api, meta, body)
    print(f"   done. {len(shell_paths)} shells published as ONE config {density_config} (glob {glob}).")
    print(
        f"   Load: stream + concatenate — load_dataset(REPO, '{density_config}', split='train', streaming=True) (see README)."
    )


if __name__ == "__main__":
    main()
