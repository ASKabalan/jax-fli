#!/usr/bin/env python
# ruff: noqa: E402
"""Experiment 00 — (re)publish CosmoGrid's own Stage-3 forecast convergence (κ) at native nside 512.

This is the κ companion to ``publish_density_2048.py``: where that script rebuilds the density
reference and overwrites the HuggingFace config ``00-cosmogrid-density``, this one rebuilds CosmoGrid's
**own** lensed convergence — the Stage-3 *forecast* κ maps — and overwrites ``00-cosmogrid-kappa``.

The forecast maps are stored per-probe in ``projected_probes_maps_nobaryons512.h5`` (4 tomographic
bins under ``kg/stage3_lensing{1..4}``), natively **nside 512, float32** (≈ 50 MB for 4 bins), so —
unlike the 7 GB density — the parquet stays well below ``INT32_MAX`` and never takes the split path.

We do **not** re-read the h5 by hand: ``jfli.io.load_cosmogrid_kappa`` (the exact call used in
``docs/2-advanced-usage/10-External-Catalog.ipynb``) already resolves the cosmology from
``CosmoGridV1_metainfo.h5``, computes the per-bin Stage-3 n(z) geometry, and builds a stacked
``SphericalKappaField`` (status ``KAPPA``, unit ``DIMENSIONLESS``). The default run is
``cosmo_000001/perm_0000`` — the **first** perm of ``cosmo_000001`` — so this κ shares the **same
cosmology** as the density reference (``raw/cosmo_000001``); that match is enforced as a hard gate, not
left to inspection.

Run on CPU (pure I/O + serialization; no GPU needed):

    python publish_kappa_512.py --self-test     # fast: validate the κ round-trip, then exit
    python publish_kappa_512.py --check          # report whether the local parquet exists + its attrs
    python publish_kappa_512.py                  # build + save locally (no upload)
    python publish_kappa_512.py --publish        # build + OVERWRITE 00-cosmogrid-kappa on HF

``HF_TOKEN`` must be set in the environment for ``--publish``.
"""

from __future__ import annotations

import os

# Pure host-side I/O + arrow serialization — pin to CPU so the loader does not land on a GPU. Must be
# set before jax / jax_fli import.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import re
from pathlib import Path

import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np

import jax_fli as jfli

REPO = "ASKabalan/jax-fli-experiments"
KAPPA_CONFIG = "00-cosmogrid-kappa"
HERE = Path(__file__).resolve().parent

# Default forecast realization — the first perm of cosmo_000001, matching the density reference's
# cosmology (raw/cosmo_000001/run_0).
DEFAULT_SIM_ROOT = Path("/home/wassim/Projects/NBody/Simulations")
DEFAULT_RUN = "CosmoGrid/stage3_forecast/cosmo_000001/perm_0000"

# Cosmology parameters compared by the gate (jc.Cosmology attribute names).
_COSMO_FIELDS = ("Omega_c", "Omega_b", "h", "n_s", "sigma8", "w0", "wa", "Omega_nu")


# --------------------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------------------
def build_kappa_catalog(run_dir, *, probe="kg", bins=None, baryonified=False, ud_nside=None):
    """Stacked ``SphericalKappaField`` (n_bins, npix) + cosmology, via the shared CosmoGrid loader."""
    return jfli.io.load_cosmogrid_kappa(run_dir, probe=probe, bins=bins, baryonified=baryonified, ud_nside=ud_nside)


# --------------------------------------------------------------------------------------------------
# Cosmology gate — the user's one hard requirement: this κ must share the density's cosmology.
# --------------------------------------------------------------------------------------------------
def _raw_cosmology(run_dir: Path) -> jc.Cosmology:
    """Parse a CosmoGrid raw run's ``params.yml`` into a ``jc.Cosmology`` (mirrors ``load_cosmogrid_lc``)."""
    params: dict = {}
    with open(run_dir / "params.yml") as f:
        for line in f:
            if ":" in line:
                key, val = line.strip().split(":", 1)
                try:
                    params[key.strip()] = float(val.strip())
                except ValueError:
                    params[key.strip()] = val.strip()
    return jc.Cosmology(
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


def _raw_run_for(run: str, sim_root: Path) -> Path | None:
    """Map a forecast run (…/cosmo_NNNNNN/perm_YYYY) to its matching raw run (…/cosmo_NNNNNN/run_0)."""
    m = re.search(r"cosmo_(\d+)", str(run))
    if m is None:
        return None
    raw = sim_root / "CosmoGrid" / "raw" / f"cosmo_{m.group(1)}" / "run_0"
    return raw if (raw / "params.yml").exists() else None


def cosmo_gate(loaded: jc.Cosmology, raw_run: Path | None) -> None:
    """Hard-assert the κ cosmology equals the matching raw density run's cosmology."""
    if raw_run is None:
        print("[cosmo-gate] SKIP — no matching raw run/params.yml found; cannot cross-check cosmology.")
        return
    ref = _raw_cosmology(raw_run)
    bad = []
    for name in _COSMO_FIELDS:
        a, b = float(getattr(loaded, name)), float(getattr(ref, name))
        if not np.isclose(a, b, rtol=1e-4, atol=1e-6):
            bad.append(f"{name}: kappa={a:.6g} != density={b:.6g}")
    assert not bad, "[cosmo-gate] FAIL — κ cosmology != density cosmology:\n  " + "\n  ".join(bad)
    print(f"[cosmo-gate] PASS — κ cosmology matches {raw_run} on {', '.join(_COSMO_FIELDS)}.")


# --------------------------------------------------------------------------------------------------
# Pre-flight round-trip — validate the κ field (and its KAPPA status/unit/metadata) survives parquet.
# --------------------------------------------------------------------------------------------------
def self_test() -> None:
    """Round-trip a tiny synthetic ``SphericalKappaField``; assert array + KAPPA status/unit/metadata survive."""
    import tempfile

    from jax_fli.fields.units import ConvergenceUnit

    nside, nbins = 8, 3
    npix = 12 * nside * nside
    arr = np.random.default_rng(0).standard_normal((nbins, npix)).astype(np.float32)
    z = np.linspace(0.3, 1.2, nbins).astype(np.float32)
    field = jfli.SphericalKappaField(
        array=jnp.asarray(arr),
        mesh_size=(1, 1, 1),
        box_size=(1.0, 1.0, 1.0),
        observer_position=(0.5, 0.5, 0.5),
        field_sharding=None,
        halo_size=(0, 0),
        nside=nside,
        z_sources=jnp.asarray(z),
        scale_factors=jnp.asarray(1.0 / (1.0 + z)),
        comoving_centers=jnp.asarray(np.linspace(800.0, 2500.0, nbins, dtype=np.float32)),
        density_width=jnp.asarray(np.full(nbins, 400.0, dtype=np.float32)),
        status=jfli.FieldStatus.KAPPA,
        unit=ConvergenceUnit.DIMENSIONLESS,
    )
    cosmo = jc.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.97, sigma8=0.8, w0=-1.0, wa=0.0, Omega_k=0.0, Omega_nu=0.0
    )
    cat = jfli.io.Catalog(field=field, cosmology=cosmo)
    with tempfile.TemporaryDirectory() as d:
        p = str(Path(d) / "t.parquet")
        cat.to_parquet(p)
        back = jfli.io.Catalog.from_parquet(p).field[0]
    assert back.array.dtype == np.float32, f"dtype changed: {back.array.dtype}"
    assert tuple(back.array.shape) == (nbins, npix), f"shape changed: {back.array.shape}"
    assert np.allclose(np.asarray(back.array), arr), "values changed across the round-trip"
    # Metadata/status that the downstream lensing code dereferences must survive too.
    assert back.status == jfli.FieldStatus.KAPPA, f"status changed: {back.status}"
    assert back.unit == ConvergenceUnit.DIMENSIONLESS, f"unit changed: {back.unit}"
    assert int(back.nside) == nside, f"nside changed: {back.nside}"
    assert back.density_width is not None, "density_width lost"
    assert np.asarray(back.z_sources).shape[0] == nbins, "z_sources length != n_bins"
    assert np.asarray(back.scale_factors).shape[0] == nbins, "scale_factors length != n_bins"
    assert np.asarray(back.comoving_centers).shape[0] == nbins, "comoving_centers length != n_bins"
    print(
        "[self-test] PASS — κ round-trip preserved array (dtype/shape/values) and the KAPPA "
        "status + DIMENSIONLESS unit + per-bin metadata downstream lensing needs."
    )


# --------------------------------------------------------------------------------------------------
# Check — does the local parquet exist, and what is in it?
# --------------------------------------------------------------------------------------------------
def _fmt_array(x) -> str:
    if x is None:
        return "None"
    a = np.asarray(x).ravel()
    if a.size > 8:
        return f"len={a.size} min={float(a.min()):.4g} max={float(a.max()):.4g}"
    return np.array2string(a, precision=4, separator=", ")


def check(config: str) -> None:
    """Report whether the published HuggingFace config exists and print its field + cosmology attributes."""
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
    print(f"  source     : {local} ({os.path.getsize(local) / 1e6:.1f} MB)")
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
        help=f"only check the published {KAPPA_CONFIG} config on HuggingFace and print its field attributes, then exit",
    )
    ap.add_argument("--self-test", action="store_true", help="run only the κ round-trip test, then exit")
    ap.add_argument(
        "--skip-self-test", action="store_true", help="skip the pre-flight round-trip test before the build"
    )
    ap.add_argument("--sim-root", default=str(DEFAULT_SIM_ROOT))
    ap.add_argument("--run", default=DEFAULT_RUN, help="forecast run dir relative to --sim-root")
    ap.add_argument("--probe", default="kg", help="HDF5 probe group: kg (kappa), ia, dg")
    ap.add_argument("--bins", type=int, nargs="+", default=None, help="tomographic bins 1-4 (default: all four)")
    ap.add_argument("--baryonified", action="store_true", help="load the baryonified HDF5 variant")
    ap.add_argument("--ud-nside", type=int, default=None, help="resample to this nside (default: native 512)")
    ap.add_argument("--out", default=None, help="local parquet path (default: next to this script)")
    ap.add_argument("--publish", action="store_true", help="OVERWRITE 00-cosmogrid-kappa on HuggingFace")
    args = ap.parse_args()

    if args.check:
        check(KAPPA_CONFIG)
        return

    out = Path(args.out) if args.out else HERE / "cosmogrid_kappa_nside512.parquet"

    if args.self_test:
        self_test()
        return
    if not args.skip_self_test:
        self_test()

    run_dir = Path(args.sim_root) / args.run
    bins = tuple(args.bins) if args.bins else None
    print(
        f"Loading CosmoGrid forecast κ from {run_dir} "
        f"(probe={args.probe}, bins={bins or 'all'}, baryonified={args.baryonified}) …"
    )
    cat = build_kappa_catalog(
        run_dir, probe=args.probe, bins=bins, baryonified=args.baryonified, ud_nside=args.ud_nside
    )
    field = cat.field[0]
    cosmo = cat.cosmology[0]

    # The user's one hard requirement: this κ must share the density reference's cosmology.
    cosmo_gate(cosmo, _raw_run_for(args.run, Path(args.sim_root)))

    print("Built SphericalKappaField:")
    print(f"   shape={tuple(field.array.shape)} dtype={field.array.dtype} nside={field.nside}  unit={field.unit.name}")
    print(f"   bins={field.array.shape[0]}  z_eff={_fmt_array(field.z_sources)}")
    print(
        f"   cosmo: Oc={float(cosmo.Omega_c):.4f} Ob={float(cosmo.Omega_b):.4f} h={float(cosmo.h):.4f} "
        f"s8={float(cosmo.sigma8):.4f} ns={float(cosmo.n_s):.4f} w0={float(cosmo.w0):.4f}"
    )

    print(f"Writing parquet → {out} …")
    cat.to_parquet(str(out))
    print(f"   wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")

    if not args.publish:
        print("Not publishing (pass --publish to overwrite the HuggingFace config).")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    target = _hf_data_files_path(api, KAPPA_CONFIG)
    print(f"\nOVERWRITING {REPO}:{target}")
    print(
        f"   replacing the {KAPPA_CONFIG} parquet with the rebuilt nside-{field.nside} forecast κ ({field.array.shape[0]} bins)."
    )
    api.upload_file(path_or_fileobj=str(out), path_in_repo=target, repo_id=REPO, repo_type="dataset")
    print(f"   uploaded. {KAPPA_CONFIG} now serves the rebuilt forecast κ.")


if __name__ == "__main__":
    main()
