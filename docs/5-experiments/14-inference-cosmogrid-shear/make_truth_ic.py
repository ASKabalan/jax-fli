#!/usr/bin/env python
# ruff: noqa: E402
"""Experiment 14 — the truth INPUT of CosmoGrid ``cosmo_000001/run_0``: the white IC field + its cosmology.

CosmoGridV1 never stored its initial conditions — PKDGRAV3 generates them internally from the scalar
``iSeed``. ``ic_resample`` recreates the primordial **white-noise field** from that seed as a standalone
NumPy generator, verified bit-for-bit against PKDGRAV3 at the production size (0/692,224 desynced
pencils, 0/288,657,407 modes with r(k) < 0.999999 at nGrid=832; see ``ic_resample/CONVENTION.md``).

This pairs that recreated input with the published output: the density shells of this same run are
already on HuggingFace as ``00-cosmogrid/cosmo_000001/density`` (``00-cosmogrid-reference/publish_density_2048.py``
has the same ``DEFAULT_RUN``). Together they are a real external simulation with a known latent — the
mock that field-level inference needs but cannot generate itself.

**What is stored: the WHITE field, before coloring.** ``jfli.interpolate_initial_conditions`` is the
white -> colored boundary (``gaussian_initial_conditions`` is just ``normal_field`` + that call), and
PKDGRAV3's noise is exactly its input: real-space float32 ``[x, y, z]``, mean 0, variance 1. We keep it
there. Coloring would bake in jax_cosmo's Eisenstein-Hu P(k) normalized to sigma8, whereas CosmoGrid ran
CLASS with As = 2.02e-9 and three 0.02 eV neutrinos — a convention the source simulation never used. The
white field is also the actual latent of the forward model (``full_field_model.py`` samples
``initial_conditions`` white and colors inline with the *sampled* cosmology), and it is the object that
is bit-verified.

WARNING — do not feed this file to ``fli-infer --ic-input`` unmodified. That path assumes the catalog
holds the COLORED delta and de-colors it (``scripts/entry/fli_infer.py:252-267``); it would divide by
sqrt(P(k)) a second time. Nothing in the schema distinguishes white from colored:
``DistributedNormal.sample`` (``infer/dist.py:326-338``) stamps a white field with the same
INITIAL_FIELD/DENSITY as ``interpolate_initial_conditions`` stamps a colored one. Color it first with
``interpolate_initial_conditions(field.array, mesh_size, box_size, cosmo=cosmo)``, or bypass that path.

WARNING — the axis orientation is NOT verified. The white *noise* is bit-verified, but nothing here
checks that pkdgrav3's ``[x, y, z]`` drops into jax-fli's mesh the right way round. CONVENTION.md warns
that getting the FFT axes wrong "transposes the box silently", and no statistic can catch it — coloring
and P(k) are isotropic, so a transpose survives into the artifact looking perfectly healthy. The only
real check is running this IC forward and cross-correlating against ``00-cosmogrid-000001-density`` at matched
redshift. That is this experiment's job, not this script's.

Run on CPU (NumPy generation + arrow serialization; no GPU):

    python make_truth_ic.py --self-test   # fast: generator regression + split round-trip, then exit
    python make_truth_ic.py               # generate 832^3 and write the truth parquet (~1 min, ~25 GB RAM)
    python make_truth_ic.py --wn-npz /path/to/white_noise_cosmo_000001_run_0.npz   # reuse a saved field
    python make_truth_ic.py --check       # reload the written parquet and summarize
"""

from __future__ import annotations

import os

# Pure host-side generation + arrow serialization — pin to CPU so jnp.asarray of the 2.3 GB array does
# not land on a GPU and OOM. Must be set before jax / jax_fli import.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np

import jax_fli as jfli

# ic_resample is the NumPy-only PKDGRAV3 white-noise generator. It is a directory of scripts, not an
# installed package, so it joins sys.path rather than being declared as a dependency.
IC_RESAMPLE = Path("/home/wassim/Projects/NBody/ic_resample")
sys.path.insert(0, str(IC_RESAMPLE))

import pkd_whitenoise as pw
from check_against_pkdgrav3 import compare
from cosmogrid_params import read_run

# Provenance: the same run Experiment 0 published as 00-cosmogrid/cosmo_000001/density.
DEFAULT_SIM_ROOT = Path("/home/wassim/Projects/NBody/Simulations")
DEFAULT_RUN = "CosmoGrid/raw/cosmo_000001/run_0"
DEFAULT_OUT = Path("/home/wassim/Projects/NBody/jax-fli-experiments/14-inference-cosmogrid/truth/input_cg.parquet")


# --------------------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------------------
def read_cosmology(run_dir: Path) -> jc.Cosmology:
    """The run's cosmology from ``params.yml``.

    Mirrors the parameter mapping in ``jax_fli.io.cosmogrid.load_cosmogrid_lc`` (:106-118, :199-209),
    duplicated rather than imported because that loader is a *shell* loader — it needs
    ``compressed_shells.npz``, which this script never touches. Note ``As`` is dropped in favour of
    ``s8``, exactly as the loader does, and ``Om``/``Ol`` are ignored (``Ol: -1.0`` is a sentinel, and
    the file's ``Om`` disagrees with ``O_cdm + Ob + O_nu``).
    """
    params = {}
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


def build_white_field(wn: np.ndarray, box_size: float) -> jfli.DensityField:
    """Wrap the white noise in a ``DensityField``, as ``DistributedNormal.sample`` does.

    ``infer/dist.py:326-338`` is the repo's own precedent for putting a *white* field in a container:
    ``status=INITIAL_FIELD``, ``unit=DENSITY``. The four dynamic metadata slots are 0.0 sentinels — the
    serializer rejects ``None`` (``_field_catalog.py:_ensure_1d_metadata``), and a white field has no
    redshift, no comoving centre and no shell width. ``scale_factors=0.0`` matches what
    ``interpolate_initial_conditions`` stamps on an initial field.
    """
    n = int(wn.shape[0])
    return jfli.DensityField(
        array=jnp.asarray(wn),
        mesh_size=(n, n, n),
        box_size=(float(box_size),) * 3,
        observer_position=(0.5, 0.5, 0.5),
        field_sharding=None,
        halo_size=(0, 0),
        z_sources=0.0,
        scale_factors=0.0,
        comoving_centers=0.0,
        density_width=0.0,
        status=jfli.FieldStatus.INITIAL_FIELD,
        unit=jfli.DensityUnit.DENSITY,
    )


# --------------------------------------------------------------------------------------------------
# Pre-flight: the generator still matches PKDGRAV3, and the catalog round-trip is bit-exact.
# --------------------------------------------------------------------------------------------------
def self_test() -> None:
    """Generator regression vs a shipped PKDGRAV3 dump, then a forced-split bit-exact round-trip."""
    import tempfile

    import pyarrow.parquet as pq

    import jax_fli._src.io._field_catalog as fc

    # 1. The generator still reproduces PKDGRAV3 itself. This ties the artifact to the C code rather
    #    than to whatever the NumPy port happens to emit today. The bar is PHASE agreement (min_r);
    #    max|d| of a few float32 ulp is the known np.log-vs-glibc-logf difference, amplitude only.
    ref = IC_RESAMPLE / "refs" / "seed111115_n32_fixed0_phase0.bin"
    res = compare(str(ref), 32, 111115)
    assert res["pass"], f"generator no longer matches {ref.name}: {res}"
    print(
        f"[self-test] generator vs PKDGRAV3 {ref.name}: struct={res['structural']} "
        f"desync={res['desynced']}/{res['npencil']} min_r={res['min_r']:.9f} ulp={res['ulp']:.2f} PASS"
    )

    # 2. Catalog round-trip through the split path. No coloring means this must be EXACT, not close:
    #    float32 in, float32 out, no arithmetic in between.
    wn = pw.generate(32, 111115)
    cosmo = read_cosmology(DEFAULT_SIM_ROOT / DEFAULT_RUN)
    field = build_white_field(wn, 900.0)

    # (1, 32, 32, 32) float32 = 131072 B; a plane along axis 1 is 4096 B. A threshold of 70000 gives
    # max_planes_per_split=17 -> 2 splits of 16, the same 2-way split the real 832^3 array takes.
    saved = fc._INT32_MAX
    fc._INT32_MAX = 70000
    try:
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "t.parquet")
            jfli.io.Catalog(field=field, cosmology=cosmo).to_parquet(p)
            cols = pq.ParquetFile(p).schema_arrow.names
            assert "_n_splits" in cols and "array_0" in cols, f"split path not taken — columns: {cols}"
            back = jfli.io.Catalog.from_parquet(p)
    finally:
        fc._INT32_MAX = saved

    f, c = back.field[0], back.cosmology[0]
    assert np.array_equal(np.asarray(f.array), wn), "white noise changed across the split round-trip"
    assert f.array.dtype == np.float32, f"dtype changed: {f.array.dtype}"
    assert tuple(f.array.shape) == (32, 32, 32), f"shape changed: {f.array.shape}"
    assert f.status == jfli.FieldStatus.INITIAL_FIELD, f"status changed: {f.status}"
    assert f.unit == jfli.DensityUnit.DENSITY, f"unit changed: {f.unit}"
    assert tuple(f.mesh_size) == (32, 32, 32), f"mesh_size changed: {f.mesh_size}"
    assert tuple(f.box_size) == (900.0,) * 3, f"box_size changed: {f.box_size}"
    # The cosmology round-trips through float32 columns, so compare at float32 tolerance.
    for k in ("Omega_c", "Omega_b", "h", "n_s", "sigma8", "w0", "wa", "Omega_k", "Omega_nu"):
        a, b = float(getattr(cosmo, k)), float(getattr(c, k))
        assert np.isclose(a, b, rtol=1e-6), f"cosmology {k} changed: {a} -> {b}"
    print("[self-test] PASS — split-path round-trip is bit-exact and preserved INITIAL_FIELD/DENSITY + cosmology.")


# --------------------------------------------------------------------------------------------------
# Check — what is actually in the written parquet?
# --------------------------------------------------------------------------------------------------
def check(path: Path, run_dir: Path) -> None:
    """Reload the written truth catalog, summarize it, and assert it still says what it should."""
    if not path.is_file():
        print(f"[check] {path} does not exist — run without --check first.")
        return
    print(f"[check] {path} ({path.stat().st_size / 2**30:.2f} GB) — loading …")
    cat = jfli.io.Catalog.from_parquet(str(path))
    f, c = cat.field[0], cat.cosmology[0]
    arr = np.asarray(f.array)
    mean, var = float(arr.mean()), float(arr.var())
    print(f"  entries    : {len(cat)}  {type(f).__name__}  {f.status.name}/{f.unit.name}")
    print(f"  array      : {arr.shape} {arr.dtype}  mean={mean:.3e}  var={var:.8f}  (white: mean 0, var 1)")
    print(f"  geometry   : mesh={tuple(f.mesh_size)}  box={tuple(f.box_size)} Mpc/h  observer={f.observer_position}")
    print(
        f"  cosmo      : Oc={float(c.Omega_c):.6f} Ob={float(c.Omega_b):.4f} h={float(c.h):.4f} "
        f"s8={float(c.sigma8):.4f} ns={float(c.n_s):.4f} w0={float(c.w0):.4f} wa={float(c.wa):.4f} "
        f"Onu={float(c.Omega_nu):.7f}"
    )
    assert abs(mean) < 1e-6, f"not white: mean={mean}"
    assert abs(var - 1.0) < 1e-2, f"not white: var={var}"

    truth = read_cosmology(run_dir)
    for k in ("Omega_c", "Omega_b", "h", "n_s", "sigma8", "w0", "wa", "Omega_k", "Omega_nu"):
        a, b = float(getattr(truth, k)), float(getattr(c, k))
        assert np.isclose(a, b, rtol=1e-6), f"cosmology {k} does not match params.yml: {a} != {b}"
    print("  PASS — field is white and the cosmology matches params.yml (to float32).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run only the pre-flight checks, then exit")
    ap.add_argument("--skip-self-test", action="store_true", help="skip the pre-flight before the big build")
    ap.add_argument("--check", action="store_true", help="only reload and summarize the written parquet, then exit")
    ap.add_argument("--sim-root", default=str(DEFAULT_SIM_ROOT))
    ap.add_argument("--run", default=DEFAULT_RUN, help="run dir relative to --sim-root")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="path of the truth parquet to write")
    ap.add_argument(
        "--wn-npz",
        default=None,
        help="reuse a make_ic.py white_noise_*.npz instead of regenerating (skips ~1 min and ~25 GB peak RAM)",
    )
    args = ap.parse_args()

    run_dir = Path(args.sim_root) / args.run
    out = Path(args.out)

    if args.check:
        check(out, run_dir)
        return
    if args.self_test:
        self_test()
        return
    if not args.skip_self_test:
        self_test()

    # 1. Parameters. read_run reads cosmology.par (the four that reach the noise, plus metadata);
    #    read_cosmology reads params.yml (the cosmology, which never reaches the noise at all).
    p = read_run(run_dir)
    cosmo = read_cosmology(run_dir)
    print(f"\nrun          : {p['simulation']}  ({p['source']})")
    print(
        f"  iSeed={p['iSeed']}  nGrid={p['nGrid']}  bFixedAmpIC={p['bFixedAmpIC']}  dFixedAmpPhasePI={p['dFixedAmpPhasePI']}"
    )
    print(f"  dBoxSize={p['dBoxSize']}  iLPT={p['iLPT']}  dRedFrom={p['dRedFrom']}  pkdgrav3={p['pkdgrav3_hash']}")
    print(
        f"  cosmo      : Oc={float(cosmo.Omega_c):.6f} Ob={float(cosmo.Omega_b):.4f} h={float(cosmo.h):.4f} "
        f"s8={float(cosmo.sigma8):.4f} ns={float(cosmo.n_s):.4f} w0={float(cosmo.w0):.4f} "
        f"wa={float(cosmo.wa):.4f} Onu={float(cosmo.Omega_nu):.7f}"
    )

    # 2. The white noise. NOT mean-subtracted: the artifact stays bit-identical to what the verified
    #    generator emits. The DC mode is exactly 0 in k-space by construction (pkd_whitenoise.py:238);
    #    the ~1e-9 real-space residual is float32 storage re-summing (CONVENTION.md:199-203), some 7
    #    orders below any mode amplitude.
    if args.wn_npz:
        print(f"\nloading white noise from {args.wn_npz} …", flush=True)
        d = np.load(args.wn_npz)
        wn = d["white_noise"]
        if int(d["iSeed"]) != p["iSeed"] or int(d["nGrid"]) != p["nGrid"]:
            raise SystemExit(
                f"{args.wn_npz} is iSeed={int(d['iSeed'])} nGrid={int(d['nGrid'])}, "
                f"but {p['simulation']} is iSeed={p['iSeed']} nGrid={p['nGrid']}"
            )
    else:
        print(f"\ngenerating nGrid={p['nGrid']} seed={p['iSeed']} white noise (~1 min, ~25 GB RAM) …", flush=True)
        wn = pw.generate(
            p["nGrid"], p["iSeed"], b_fixed=bool(p["bFixedAmpIC"]), f_phase=p["dFixedAmpPhasePI"] * np.pi, verbose=True
        )

    mean, var = float(wn.mean()), float(wn.var())
    n = p["nGrid"]
    print(f"  shape={wn.shape} dtype={wn.dtype} C-contig={wn.flags['C_CONTIGUOUS']} mean={mean:.6e} var={var:.10f}")
    assert wn.dtype == np.float32 and wn.shape == (n,) * 3 and wn.flags["C_CONTIGUOUS"]
    assert abs(mean) < 1e-6, mean
    assert abs(var - 1.0) < 1e-2, var

    # 3 + 4. Wrap and write. > INT32_MAX bytes at 832^3, so the serializer splits the array across two
    #        columns and reassembles it transparently on read.
    field = build_white_field(wn, p["dBoxSize"])
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nwriting {out} …", flush=True)
    jfli.io.Catalog(field=field, cosmology=cosmo).to_parquet(str(out))
    print(f"done -> {out}  ({out.stat().st_size / 2**30:.2f} GB)")
    print("   This is the WHITE field, before interpolate_initial_conditions. See the module docstring")
    print("   before feeding it to fli-infer --ic-input.")


if __name__ == "__main__":
    main()
