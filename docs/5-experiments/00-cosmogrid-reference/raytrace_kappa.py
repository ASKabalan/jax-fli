#!/usr/bin/env python
# ruff: noqa: E402
"""Experiment 00 — ray-trace a convergence (κ) reference from the CosmoGrid density (dorian).

Loads the ``00-cosmogrid-density`` lightcone from HuggingFace, ray-traces it through **dorian**
(``jax_fli.raytrace``), and **saves a parquet** holding the resulting ``SphericalKappaField``. It
does NOT publish — run it on the cluster, then publish the parquet locally with ``publish_local.py``.

This is a single-process, host-side (numpy) computation: one process holds the full lightcone in RAM
and dorian ray-traces every source plane sequentially. Run it:

    python raytrace_kappa.py --nz s3 --out kappa_raytrace.parquet

    # tiny synthetic lightcone (no HuggingFace) — proves the plumbing works:
    python raytrace_kappa.py --smoke-test

Offline cluster (Jean Zay): pre-cache the density on a login node, then export
``HF_HOME=$WORK/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`` on compute nodes (see README.md).
"""

from __future__ import annotations

import os

# dorian ray-tracing runs on the host (numpy). Pin JAX to CPU so loading the lightcone does not
# stage the maps onto a GPU. Set before jax / jax_fli import.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Run fully offline from the warm HuggingFace cache (pre-populated by download.py): the per-shell
# snapshot_download + streaming load below must NEVER touch the network. The cache LOCATION is still
# $HF_HOME (site-specific) — set that separately on offline compute nodes. setdefault, so an explicit
# HF_HUB_OFFLINE=0 can still re-warm the cache if ever needed.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import argparse
from pathlib import Path

import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np

import jax_fli as jfli
from jax_fli.data import get_des_y3_nz_shear, get_stage3_nz_shear
from jax_fli.io import Catalog

REPO = "ASKabalan/jax-fli-experiments"
DENSITY_CONFIG = "00-cosmogrid-density"
DENSITY_GLOB = "00-cosmogrid/catalogs/cosmogrid_density_nside2048_shell*.parquet"
HERE = Path(__file__).resolve().parent

# n(z) selector: --nz s3 -> Stage-3 forecast (4 bins, matches the current κ consumers like
# 08-masked-shear); --nz des -> DES Y3 (the README's stated reference distribution). Both 4 bins.
NZ_CHOICES = {"s3": get_stage3_nz_shear, "des": get_des_y3_nz_shear}


def _synthetic_lightcone(nside: int = 32, n_shells: int = 4):
    """A tiny, physically-plausible HEALPix lightcone for the smoke test."""
    npix = 12 * nside * nside
    z = np.linspace(0.2, 1.0, n_shells).astype(np.float32)
    a = 1.0 / (1.0 + z)
    cosmo = jc.Planck18()
    com = np.asarray(jc.background.radial_comoving_distance(cosmo, a), dtype=np.float32)
    dw = np.full(n_shells, float(np.mean(np.abs(np.diff(com))) or 200.0), dtype=np.float32)
    counts = np.random.default_rng(0).integers(0, 50, size=(n_shells, npix)).astype(np.float32)
    field = jfli.SphericalDensity(
        array=jnp.asarray(counts),
        mesh_size=(128, 128, 128),
        box_size=(2560.0, 2560.0, 2560.0),
        observer_position=(0.5, 0.5, 0.5),
        halo_size=(0, 0),
        nside=nside,
        z_sources=jnp.asarray(z),
        scale_factors=jnp.asarray(a),
        comoving_centers=jnp.asarray(com),
        density_width=jnp.asarray(dw),
        status=jfli.FieldStatus.LIGHTCONE,
        unit=jfli.DensityUnit.COUNTS,
    )
    return cosmo, field


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nz", choices=list(NZ_CHOICES), default="s3", help="source n(z): s3 (Stage-3) or des (DES Y3)")
    ap.add_argument("--nside", type=int, default=None, help="downsample density to this nside (default: native)")
    ap.add_argument("--out", default=None, help="output parquet (default: kappa_raytrace.parquet beside this script)")
    ap.add_argument("--min-z", type=float, default=0.01)
    ap.add_argument("--max-z", type=float, default=3.0, help="integration ceiling (Stage-3/DES sources + z<3.5 shells)")
    ap.add_argument("--n-integrate", type=int, default=32)
    ap.add_argument("--interp", choices=["bilinear", "ngp", "nufft"], default="bilinear")
    ap.add_argument("--no-parallel-transport", action="store_true")
    ap.add_argument("--smoke-test", action="store_true", help="smoke test on a tiny synthetic lightcone (no HF)")
    args = ap.parse_args()

    if args.smoke_test:
        cosmo, field = _synthetic_lightcone()
        nz_shear = [0.5, 1.0]  # two scalar sources — exercises the multi-source path cheaply
        print(f"[smoke-test] synthetic lightcone {tuple(field.array.shape)} nside={field.nside}")
    else:
        nz_shear = NZ_CHOICES[args.nz]()
        # Stream the per-shell density and downgrade each shell on load. The density is ONE config
        # 00-cosmogrid-density with a (npix,) row per shell; it MUST be streamed (a plain load_dataset
        # would combine rows past arrow's INT32 list-offset across the 56 shells).
        from datasets import load_dataset
        from huggingface_hub import snapshot_download

        # Cache the per-shell parquets + README once (idempotent; offline-capable under
        # HF_HUB_OFFLINE=1), then STREAM from the local dir.
        local = snapshot_download(REPO, repo_type="dataset", allow_patterns=[DENSITY_GLOB, "README.md"])
        print(f"Streaming {DENSITY_CONFIG} per-shell rows from {local} …")
        fields = []
        cosmo = None
        for row in load_dataset(local, DENSITY_CONFIG, split="train", streaming=True):
            cat = Catalog.from_dataset(row)
            fld = cat.field[0]
            if args.nside is not None and int(args.nside) != int(fld.nside):
                fld = fld.ud_sample(int(args.nside))
            fields.append(fld)
            cosmo = cat.cosmology[0]
        # Each row is one (npix,) shell; stack adds the leading shell axis → (n_shells, npix).
        field = jfli.SphericalDensity.stack(fields)
        print(
            f"   density: {type(field).__name__} shape={tuple(field.array.shape)} "
            f"nside={field.nside} dtype={field.array.dtype} | n(z)={args.nz}, interp={args.interp}"
        )

    # The synthetic smoke-test geometry produces degenerate parallel-transport angles (dorian raises
    # "THETA out of range"); PT is orthogonal to what the smoke test exercises, so force it off there.
    # Real CosmoGrid geometry keeps PT on by default.
    parallel_transport = (not args.no_parallel_transport) and (not args.smoke_test)

    kappa_rt, kappa_born = jfli.raytrace(
        cosmo,
        field,
        nz_shear,
        min_z=args.min_z,
        max_z=args.max_z,
        n_integrate=args.n_integrate,
        interp=args.interp,
        parallel_transport=parallel_transport,
        born=True,
        raytrace=True,
    )

    print(
        f"   kappa: {type(kappa_rt).__name__} shape={tuple(kappa_rt.array.shape)} "
        f"nside={kappa_rt.nside} dtype={kappa_rt.array.dtype}"
    )
    print(
        f"   kappa_born: {type(kappa_born).__name__} shape={tuple(kappa_born.array.shape)} "
        f"nside={kappa_born.nside} dtype={kappa_born.array.dtype}"
    )
    if args.smoke_test:
        assert kappa_rt.array.shape[-1] == 12 * field.nside * field.nside
        print("[smoke-test] PASS — dorian ray-tracing ran and returned a κ map.")
        return

    print("ray-tracing complete, writing parquet …")
    # ray-traced κ → --out (default kappa_raytrace.parquet). The Born byproduct from the SAME dorian
    # shells goes to a SEPARATE path so it never overwrites the ray-traced map: with --out set, insert
    # "_born" before the suffix (else the default kappa_born.parquet). publish_local.py reads
    # kappa_raytrace.parquet (this rt) and kappa_born.parquet (from born_kappa.py) — unaffected.
    out_rt = Path(args.out) if args.out else HERE / "kappa_raytrace.parquet"
    Catalog(field=kappa_rt, cosmology=cosmo).to_parquet(str(out_rt))
    print(f"   wrote {out_rt}  ({out_rt.stat().st_size / 1e9:.2f} GB)  [ray-traced]. Publish: python publish_local.py")
    out_born = out_rt.with_name(f"{out_rt.stem}_born{out_rt.suffix}") if args.out else HERE / "kappa_born.parquet"
    Catalog(field=kappa_born, cosmology=cosmo).to_parquet(str(out_born))
    print(f"   wrote {out_born}  ({out_born.stat().st_size / 1e9:.2f} GB)  [Born byproduct]")


if __name__ == "__main__":
    main()
