#!/usr/bin/env python
# ruff: noqa: E402
"""Experiment 00 — publish the CosmoGrid density reference at nside 512, as one stacked parquet.

The companion of ``publish_density_2048.py``, which had no recorded generator until this script.
That one streams the native nside-2048 lightcone as 56 separate files, because a stacked write OOMs
and a stacked read overflows arrow's INT32 list offset; at nside 512 the whole lightcone is
56 · 3145728 · 4 B ≈ 700 MB, so it fits in one parquet with one row and is loaded by plain path,
not through a dataset config.

**This is a down-grade of the nside-2048 shells, and deliberately not CosmoGrid's own 512 archive.**
CosmoGrid ships ``shells_nside=512.npz`` beside ``compressed_shells.npz``, and
``load_cosmogrid_lc(..., ud_nside=512)`` reads *that* archive — but the published reference was not
made that way. It is ``ud_grade(nside 2048 -> 512)``, which **averages** the 16 sub-pixels, so it
carries counts per nside-2048 pixel and sits a factor 16 below CosmoGrid's own 512 counts. Verified
against the live artefact: shell 30 of ``ud_grade`` on the published
``cosmogrid_density_nside2048_shell_030.parquet`` is bit-identical to shell 30 of the published
``cosmogrid_density_nside512.parquet``, while CosmoGrid's own 512 shells differ from it by exactly
that factor of 16 in every shell. Reproducing it the other way would give a similar but different
field, and the two cosmologies would no longer be comparable. (The factor is harmless downstream:
every consumer converts to overdensity first, which divides it out.)

Input is the folder of per-shell nside-2048 parquets that ``publish_density_2048.py --out`` writes,
so both resolutions are guaranteed to cover the same shell selection. One shell is held at a time.

Consumed by ``06-cosmogrid-shells/build.py`` and the thesis figures ``cosmogrid_map.py`` /
``cosmogrid_starlet.py``, all by literal path::

    00-cosmogrid/<cosmo>/density/cosmogrid_density_nside512.parquet

Run on CPU (pure I/O + serialization; no GPU)::

    python publish_density_2048.py --run … --cosmo … --out /scratch/shells --publish
    python publish_density_512.py --shells /scratch/shells --cosmo cosmo_172798 --publish

``HF_TOKEN`` must be set in the environment for ``--publish``.
"""

from __future__ import annotations

import os

# Pure host-side I/O + arrow serialization — pin to CPU so the stacked lightcone does not land on a
# GPU. Must be set before jax / jax_fli import.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import re
import tempfile
from pathlib import Path

import numpy as np

import jax_fli as jfli

REPO = "ASKabalan/jax-fli-experiments"
DEFAULT_COSMO = "cosmo_000001"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--shells",
        required=True,
        help="folder of per-shell nside-2048 parquets, i.e. the --out of publish_density_2048.py",
    )
    ap.add_argument(
        "--cosmo",
        default=DEFAULT_COSMO,
        help=f"cosmology these shells belong to; sets the destination 00-cosmogrid/COSMO/density/. Default {DEFAULT_COSMO}.",
    )
    ap.add_argument("--out", default=None, help="local folder for the parquet (default: a temp dir)")
    ap.add_argument("--publish", action="store_true", help="upload the parquet to HuggingFace")
    args = ap.parse_args()

    if not re.fullmatch(r"cosmo_\d{6}", args.cosmo):
        raise SystemExit(f"--cosmo must look like cosmo_000001, got {args.cosmo!r}")

    shell_paths = sorted(Path(args.shells).glob("*shell_*.parquet"))
    if not shell_paths:
        raise SystemExit(f"no shell parquets under {args.shells}")

    out_dir = Path(args.out).resolve() if args.out else Path(tempfile.mkdtemp(prefix="cosmogrid_density_512_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cosmogrid_density_nside512.parquet"

    print(f"Down-grading {len(shell_paths)} nside-2048 shells to 512, one at a time …")
    shells, cosmo = [], None
    for i, p in enumerate(shell_paths):
        cat = jfli.io.Catalog.from_parquet(str(p))
        if cosmo is None:
            cosmo = cat.cosmology[0]
        # `name` is static metadata, so shells relabelled per-shell ("… shell 0", "… shell 1", …)
        # make `stack` fail on a pytree-metadata mismatch. The stacked lightcone is one field with
        # one label anyway, which legend_names.py sets afterwards.
        shells.append(cat.field[0].ud_sample(512).replace(name=None))
        if i % 10 == 0 or i == len(shell_paths) - 1:
            print(f"   [{i + 1}/{len(shell_paths)}] {p.name}")

    field = jfli.SphericalDensity.stack(shells)
    if int(field.nside) != 512:
        raise RuntimeError(f"expected nside 512, got {field.nside}")
    z = np.asarray(field.z_sources)
    print(
        f"   {type(field).__name__} {tuple(field.array.shape)} {field.array.dtype} nside={field.nside} "
        f"{field.status.name}/{field.unit.name}; shells={z.size} z=[{float(z.min()):.3f}, {float(z.max()):.3f}]; "
        f"cosmo Oc={float(cosmo.Omega_c):.4f} Ob={float(cosmo.Omega_b):.4f} h={float(cosmo.h):.4f} "
        f"s8={float(cosmo.sigma8):.4f} ns={float(cosmo.n_s):.4f} w0={float(cosmo.w0):.4f}"
    )

    jfli.io.Catalog(field=field, cosmology=cosmo).to_parquet(str(out_path))
    print(f"   wrote {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")

    if not args.publish:
        print("Not publishing (pass --publish).")
        return

    from huggingface_hub import HfApi

    repo_path = f"00-cosmogrid/{args.cosmo}/density/cosmogrid_density_nside512.parquet"
    print(f"\nUploading → {REPO}:{repo_path} …")
    HfApi().upload_file(path_or_fileobj=str(out_path), path_in_repo=repo_path, repo_id=REPO, repo_type="dataset")
    # No dataset-card entry: the nside-2048 config's glob is ...nside2048_shell*.parquet, which does
    # not match this file, and every consumer reads it by literal path through snapshot_download.
    print("   done.")


if __name__ == "__main__":
    main()
