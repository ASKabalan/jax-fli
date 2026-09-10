"""Fiducial CosmoGrid κ spectra — the cosmic-variance band for the ratio figures.

The 05b/05c ratio panels and the thesis `lensing_vs_cosmogrid` figure divide our spectra by a
CosmoGrid reference, so the plotted residual carries the cosmic variance of that reference. This
script quantifies it from the fiducial CosmoGrid permutations themselves: for each of the 200
`perm_XXXX` directories under `stage3_forecast/fiducial/cosmo_fiducial` it loads the four
Stage-3 forecast κ maps (nside 512, full sky, no mask), measures the full-sky C_ℓ up to
ℓ = 1500 with healpy's anafast — the same estimator and the same banding (linear bins of 32
multipoles, (2ℓ+1)-weighted, `lmin=2`) as every figure — and stores

  00-cosmogrid/fiducial_kappa_spectra/cosmo_fiducial_part{0..3}.parquet
      one Catalog per 50 permutations, one row per permutation, array (4 bins, 1501), so the
      perm scatter (the empirical CV) is reconstructible from the parquets alone;
  00-cosmogrid/fiducial_kappa_spectra/cosmo_fiducial_stats.parquet
      the ensemble summary as a two-row spectra file in the same format: the per-multipole
      mean (row 0) and std (row 1, ddof=1) across the 200 permutations. It is a summary, not
      the band source: the empirical CV is the std across perms of the (2l+1)-weighted
      bandpowers, which the pointwise std row does not reproduce (the in-band C_ell variation
      biases bin(std row)/bin(mean row) high by ~0.2 in the first band).

The three CV estimates (empirical std/mean across perms, the jax_cosmo Gaussian full-sky band
via `gaussian_cl_covariance` propagated exactly to bandpowers, and the closed form
sqrt(2 / sum(2l+1))) are recomputed on load; loading also cross-checks the stats rows against
the parts. These fractions are cosmology-independent (pure mode counting), so the fiducial
cosmology differing from cosmo_172798 / cosmo_000001 does not matter for the band.

Two further checks gate the product: perm-mean vs Limber theory at the fiducial cosmology
(pixel-window matched) inside the analytic band for l <~ 550, and jax_cosmo vs the closed
form. The figure against the data is fig09.

The cosmology is resolved with a temporary in-script patch: `load_cosmogrid_kappa` extracts a
`cosmo_XXXXXX` identifier and reads `parameters/grid`, which cannot represent the fiducial tree
(`cosmo_fiducial`, parameters under `parameters/fiducial`). The patch handles the fiducial rows
and defers everything else to the library function.

    JAX_PLATFORM_NAME=cpu uv run python docs/5-experiments/00-cosmogrid-reference/2-build.py
    FORCE_REGEN=1 ...   # recompute even if the products exist
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import sys
from pathlib import Path

import healpy as hp
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np

from jax_fli import compute_theory_cl
from jax_fli.io import Catalog, get_stage3_nz_shear, load_cosmogrid_kappa
from jax_fli.summary_statistics.binning import bin_bandpowers, linear_edges

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # -> docs/5-experiments
import matplotlib.pyplot as plt
from _exputils import savefig, set_style

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

CG_FIDUCIAL = Path("/home/wassim/Projects/NBody/Simulations/CosmoGrid/stage3_forecast/fiducial/cosmo_fiducial")
OUT_DIR = Path("/home/wassim/Projects/NBody/jax-fli-experiments/00-cosmogrid/fiducial_kappa_spectra")

LMAX = 1500
NLB = 32  # multipoles per bandpower bin, as in every figure
N_PER_PART = 50  # permutations per parquet part
NAME = "Cosmogrid κ Stage3 NZ (fiducial)"

FORCE_REGEN = os.getenv("FORCE_REGEN", "False").lower() in ("true", "1", "t")

PARTS = [OUT_DIR / f"cosmo_fiducial_part{p}.parquet" for p in range(4)]
STATS = OUT_DIR / "cosmo_fiducial_stats.parquet"


# -----------------------------------------------------------------------------
# Temporary cosmology resolution for the fiducial tree (no src/ change).
# -----------------------------------------------------------------------------
def _resolve_fiducial_cosmology(path: Path) -> jc.Cosmology:
    """Fiducial cosmology from CosmoGridV1_metainfo.h5 (parameters/fiducial)."""
    import h5py

    metainfo_path = None
    for parent in path.parents:
        candidate = parent / "CosmoGridV1_metainfo.h5"
        if candidate.exists():
            metainfo_path = candidate
            break
    if metainfo_path is None:
        raise FileNotFoundError(f"CosmoGridV1_metainfo.h5 not found in any parent of {path}")

    with h5py.File(metainfo_path, "r") as f:
        rows = f["parameters/fiducial"][:]
    row = next(r for r in rows if r["delta"] == b"fiducial")
    return jc.Cosmology(
        Omega_c=float(row["O_cdm"]),
        Omega_b=float(row["Ob"]),
        h=float(row["H0"]) / 100.0,
        n_s=float(row["ns"]),
        sigma8=float(row["s8"]),
        w0=float(row["w0"]),
        wa=float(row["wa"]),
        Omega_k=0.0,
        Omega_nu=float(row["O_nu"]),
    )


import jax_fli.io.cosmogrid as _cg

_orig_resolve = _cg._resolve_cosmogrid_cosmology


def _resolve_patch(path):
    if "cosmo_fiducial" in str(path):
        return _resolve_fiducial_cosmology(path)
    return _orig_resolve(path)


_cg._resolve_cosmogrid_cosmology = _resolve_patch


# -----------------------------------------------------------------------------
# Compute: per-perm full-sky C_ell, four parquet parts, two-row stats parquet.
# -----------------------------------------------------------------------------
def write_stats_parquet(raw: np.ndarray, cosmo: jc.Cosmology) -> None:
    """The ensemble summary as a two-row spectra file: row 0 = per-multipole mean across the
    perms, row 1 = std (ddof=1). A summary, not the band source (see the module docstring)."""
    template = Catalog.from_parquet(str(PARTS[0])).field[0]
    rows = [
        template.replace(array=raw.mean(axis=0).astype(jnp.float64)),
        template.replace(array=raw.std(axis=0, ddof=1).astype(jnp.float64)),
    ]
    Catalog(field=rows, cosmology=[cosmo] * len(rows)).to_parquet(str(STATS))
    print(f"wrote {STATS.name} (2 rows: perm-mean, perm-std)")


def _cv_curves(
    cosmo: jc.Cosmology, ell: np.ndarray, edges: np.ndarray, nmodes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pixwin-matched Limber theory at `cosmo` plus the two analytic CV curves (the jax_cosmo
    Gaussian full-sky band propagated exactly to bandpowers, and the closed form) — shared by
    the compute and load paths."""
    nb = len(edges) - 1
    nz = get_stage3_nz_shear()
    ell_j = jnp.arange(LMAX + 1)
    theory = np.asarray(compute_theory_cl(cosmo, ell_j, nz))  # (4, LMAX+1)
    pw2 = hp.pixwin(512, lmax=LMAX) ** 2
    w = 2.0 * ell + 1.0
    th_b = np.zeros((4, nb))
    for b in range(4):
        for q, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            m = (ell >= lo) & (ell < hi)
            # .bin convention: (2l+1) weights only — the pw^2 rides inside the spectrum values.
            th_b[b, q] = (theory[b][m] * pw2[m] * w[m]).sum() / w[m].sum()

    jc_frac = np.zeros((4, nb))
    for b in range(4):
        probe = [jc.probes.WeakLensing([nz[b]])]
        cl = jnp.asarray(theory[b : b + 1])  # (1, LMAX+1), noise-free
        cov = jc.angular_cl.gaussian_cl_covariance(ell_j, probe, cl, jnp.zeros_like(cl), f_sky=1.0, sparse=True)
        var_ell = np.asarray(cov)[0, 0]  # Var(C_ell) per ell
        for q, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            m = (ell >= lo) & (ell < hi)
            wsum = w[m].sum()
            var_band = (w[m] ** 2 * var_ell[m]).sum() / wsum**2  # exact propagation to the weighted mean
            c_band = (theory[b][m] * w[m]).sum() / wsum
            jc_frac[b, q] = np.sqrt(var_band) / c_band

    closed_frac = np.sqrt(2.0 / nmodes)[None, :]  # cosmology- and bin-independent
    return theory, th_b, jc_frac, closed_frac


def compute() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    perm_dirs = sorted(CG_FIDUCIAL.glob("perm_*"))
    print(f"computing C_ell for {len(perm_dirs)} perms (lmax={LMAX})")

    spectra = []
    cosmo = None
    for i, perm in enumerate(perm_dirs):
        cat = load_cosmogrid_kappa(perm)
        if cosmo is None:
            cosmo = cat.cosmology[0]
        field = cat.field[0].replace(array=cat.field[0].array.astype(jnp.float64))
        ps = field.angular_cl(lmax=LMAX, method="healpy").replace(name=NAME)
        spectra.append(ps)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(perm_dirs)}")

    for p, part in enumerate(PARTS):
        chunk = spectra[p * N_PER_PART : (p + 1) * N_PER_PART]
        Catalog(field=chunk, cosmology=[cosmo] * len(chunk)).to_parquet(str(part))
        print(f"wrote {part.name} ({len(chunk)} rows)")

    ell = np.asarray(spectra[0].wavenumber)
    raw = np.stack([np.asarray(ps.array) for ps in spectra])  # (N, 4, LMAX+1)
    write_stats_parquet(raw, cosmo)

    edges = linear_edges(NLB, 2, LMAX)
    leff, binned, nmodes = bin_bandpowers(ell, raw, edges=edges, weight="modes")
    leff, binned, nmodes = np.asarray(leff), np.asarray(binned), np.asarray(nmodes)
    print(f"binned into {binned.shape[2]} bands of {NLB} multipoles (partial band dropped)")

    theory, th_b, jc_frac, closed_frac = _cv_curves(cosmo, ell, edges, nmodes)
    mean = binned.mean(axis=0)  # (4, nb)
    std = binned.std(axis=0, ddof=1)

    return {
        "cosmo": cosmo,
        "ell": ell,
        "edges": edges,
        "leff": leff,
        "nmodes": nmodes,
        "binned": binned,
        "mean": mean,
        "std": std,
        "emp_frac": std / mean,
        "jc_frac": jc_frac,
        "closed_frac": closed_frac,
        "theory": theory,
        "th_b": th_b,
        "z_eff": np.asarray(spectra[0].z_sources),
    }


def _parts_stack() -> tuple[np.ndarray, jc.Cosmology]:
    """The (200, 4, LMAX+1) perm stack and the fiducial cosmology, from the four parts."""
    cats = [Catalog.from_parquet(str(part)) for part in PARTS]
    raw = np.stack([np.asarray(f.array) for cat in cats for f in cat.field])
    return raw, cats[0].cosmology[0]


def load_products() -> dict:
    """Rebuild the product dict from the published parquets alone: bin the 200 perm rows from
    the parts (the empirical CV lives there — the two-row stats file is a summary, not the
    band source), cross-check the stats rows against the parts, and recompute the
    theory-matched curves at the fiducial cosmology."""
    per_ell, cosmo = _parts_stack()
    stats = Catalog.from_parquet(str(STATS))
    ell = np.asarray(stats.field[0].wavenumber)
    assert np.allclose(np.asarray(stats.field[0].array), per_ell.mean(axis=0), rtol=1e-9, atol=1e-15)
    assert np.allclose(np.asarray(stats.field[1].array), per_ell.std(axis=0, ddof=1), rtol=1e-9, atol=1e-15)
    print("stats rows match the parts (mean, std)")

    edges = linear_edges(NLB, 2, LMAX)
    leff, binned, nmodes = bin_bandpowers(ell, per_ell, edges=edges, weight="modes")
    leff, binned, nmodes = np.asarray(leff), np.asarray(binned), np.asarray(nmodes)

    theory, th_b, jc_frac, closed_frac = _cv_curves(cosmo, ell, edges, nmodes)
    mean = binned.mean(axis=0)
    std = binned.std(axis=0, ddof=1)

    return {
        "cosmo": cosmo,
        "ell": ell,
        "edges": edges,
        "leff": leff,
        "nmodes": nmodes,
        "binned": binned,
        "mean": mean,
        "std": std,
        "emp_frac": std / mean,
        "jc_frac": jc_frac,
        "closed_frac": closed_frac,
        "theory": theory,
        "th_b": th_b,
        "z_eff": np.asarray(stats.field[0].z_sources),
    }


# -----------------------------------------------------------------------------
# Gates — printed checks that define the product as good.
# -----------------------------------------------------------------------------
def run_gates(P: dict) -> None:
    print("\n=== gates ===")

    # 1. roundtrip: the four parts reload into a 200-row Catalog with (4, 1501) arrays.
    n = 0
    shapes = set()
    for part in PARTS:
        cat = Catalog.from_parquet(str(part))
        n += len(cat.field)
        shapes |= {tuple(np.asarray(f.array).shape) for f in cat.field}
        c = cat.cosmology[0]
        assert np.allclose(
            [float(c.Omega_c), float(c.Omega_b), float(c.h), float(c.n_s), float(c.sigma8), float(c.w0)],
            [
                float(P["cosmo"].Omega_c),
                float(P["cosmo"].Omega_b),
                float(P["cosmo"].h),
                float(P["cosmo"].n_s),
                float(P["cosmo"].sigma8),
                float(P["cosmo"].w0),
            ],
        )
    assert n == 200 and shapes == {(4, LMAX + 1)}, (n, shapes)
    print(f"1. roundtrip: {n} rows over {len(PARTS)} parts, shapes {shapes}, cosmology OK")

    # 2. perm-mean vs theory (pixwin-matched) inside the analytic band up to leff ~ 550.
    ratio = P["mean"] / P["th_b"] - 1.0
    sel = P["leff"] <= 550
    worst = [np.max(np.abs(ratio[b, sel]) / P["jc_frac"][b, sel]) for b in range(4)]
    ok = all(wv < 3.0 for wv in worst)
    print(f"2. perm-mean/theory - 1 within 3x analytic band for leff<=550: {ok}")
    print("   worst |ratio|/band per bin: " + " ".join(f"{wv:.2f}x" for wv in worst))

    # 3. jax_cosmo band vs the closed form. Below leff ~ 100 the C_ell variation inside a band
    #    inflates the propagated variance above the constant-C closed form — expected physics,
    #    so the gate applies where the two must agree.
    rel = np.abs(P["jc_frac"] / P["closed_frac"] - 1.0)
    sel3 = P["leff"] >= 100
    print(
        f"3. jax_cosmo vs closed form: max |rel diff| for leff>=100 = {rel[:, sel3].max():.2e}"
        f" (band 0, leff={P['leff'][0]:.0f}: {rel[0, 0]:.2e}, in-band C_ell variation)"
    )

    # 4. empirical stability, first 50 perms vs all.
    e50 = P["binned"][:50].std(axis=0, ddof=1) / P["binned"][:50].mean(axis=0)
    diff = np.abs(e50 - P["emp_frac"]).max()
    print(f"4. empirical frac, N=50 vs N=200: max abs diff {diff:.4f} (sigma-error ~{1 / np.sqrt(2 * 49):.2%} of frac)")


# -----------------------------------------------------------------------------
# fig09 — empirical vs jax_cosmo vs closed form, per bin, plus the theory check.
# -----------------------------------------------------------------------------
def fig09(P: dict) -> None:
    set_style(width_in=4.2 * 4)  # fonts scale with the figure width
    bincol = {0: "#4C72B0", 1: "#DD8452", 2: "#C44E52", 3: "#55A868"}
    leff, emp, jcf, closed = P["leff"], P["emp_frac"], P["jc_frac"], P["closed_frac"]
    z = P["z_eff"]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(4.2 * 4, 5.8),
        sharex="col",
        gridspec_kw={"height_ratios": [1.1, 1]},
    )
    for b in range(4):
        ax_frac, ax_res = axes[0, b], axes[1, b]

        ax_frac.loglog(leff, emp[b], color=bincol[b], ls="-", lw=1.6)
        ax_frac.loglog(leff, jcf[b], color="0.25", ls=":", lw=1.3)
        ax_frac.loglog(leff, closed[0], color="0.6", ls="--", lw=1.2)
        ax_frac.set_title(rf"Bin {b + 1}  ($z\approx{z[b]:.2f}$)")
        ax_frac.grid(True, which="both", ls=":", alpha=0.4)
        if b == 0:
            ax_frac.set_ylabel(r"fractional CV  $\sigma(C_\ell)/C_\ell$")

        ax_res.fill_between(leff, -jcf[b], jcf[b], color=bincol[b], alpha=0.15, lw=0)
        ax_res.semilogx(leff, P["mean"][b] / P["th_b"][b] - 1.0, color=bincol[b], ls="-", lw=1.6)
        ax_res.axhline(0.0, color="0.4", ls="--", lw=0.9)
        ax_res.grid(True, which="both", ls=":", alpha=0.4)
        ax_res.set_ylim(-0.08, 0.08)
        # The axis starts at the first bandpower: nothing is binned below it.
        ax_res.set_xlim(leff[0] * 0.92, leff[-1] * 1.02)
        if b == 0:
            ax_res.set_ylabel(r"perm mean $/\,C_\ell^{\rm th}-1$")
        if b == 3:
            ax_frac.legend(
                handles=[
                    plt.Line2D([], [], color=bincol[0], ls="-", lw=1.6, label="empirical (200 perms)"),
                    plt.Line2D([], [], color="0.25", ls=":", lw=1.3, label="jax_cosmo full-sky Gaussian"),
                    plt.Line2D([], [], color="0.6", ls="--", lw=1.2, label=r"$\sqrt{2/\sum(2\ell+1)}$"),
                ],
                frameon=False,
                loc="lower left",
            )
            ax_res.legend(
                handles=[
                    plt.Line2D([], [], color=bincol[0], ls="-", lw=1.6, label="perm mean / theory $-1$"),
                    plt.Line2D([], [], color=bincol[0], alpha=0.15, lw=6, label=r"$\pm 1\sigma$ CV (jax_cosmo)"),
                ],
                frameon=False,
                loc="lower right",
            )
    for ax in axes[1]:
        ax.set_xlabel(r"multipole $\ell$")
    fig.tight_layout()
    savefig(ASSETS / "fig09-cosmic-variance", fig)


def main():
    if FORCE_REGEN or not all(p.exists() for p in PARTS):
        P = compute()
    else:
        if not STATS.exists():
            print("parts exist but the stats parquet is missing; writing it from the parts")
            write_stats_parquet(*_parts_stack())
        print("products exist; set FORCE_REGEN=1 to recompute")
        P = load_products()
    run_gates(P)
    fig09(P)
    print(f"assets written to {ASSETS}")


if __name__ == "__main__":
    main()
