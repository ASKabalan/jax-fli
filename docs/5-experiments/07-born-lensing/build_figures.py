"""Experiment 07 — Born lensing vs CosmoGrid: convergence (κ) figures.

The lensing analog of experiment 06. We Born-integrate the exp-06 PM density lightcones into
tomographic convergence maps and compare them, **statistically**, to a reference convergence — for two
weak-lensing source distributions:

* **Stage-3** — our lensed PM-sim κ vs **CosmoGrid's own** convergence (`cosmogrid_sample_kappa`,
  nside 512, Stage-3 forecast). The end-to-end check: does our pipeline reproduce CosmoGrid's lensing?
* **DES Y3** — our lensed PM-sim κ vs our **Born-on-CosmoGrid-density** κ (`kappa_born_des`). There is
  no native CosmoGrid DES κ, so the Born-on-the-published-density map (which matches Limber theory) is
  the reference.

Both source sets are also compared to **Limber weak-lensing theory** (`compute_theory_cl`, NOT the
density `_for_density` one), put on the measured pixel-window footing (theory × pixwin²(nside) before
binning, per series' nside — the `1-build.py` pattern).

The lensed PM-sim and the reference are **independent realisations**, so the comparison is statistical
(C_ℓ, PDF, starlet) and the map difference is uncorrelated (full amplitude, no cancellation). κ is
already dimensionless (no unit conversion), but CosmoGrid's κ carries the non-zero mean convergence
(monopole) while ours is mean-zero — so for the PDF/starlet/maps we subtract the per-bin mean (κ − κ̄);
the C_ℓ at ℓ ≥ 2 is monopole-independent. We show our lensed κ at native **nside 2048** and downsampled
to **512** (theory pixwin-matched to each); pixel-wise **differences are taken at the matched 512**.

Figures (-> ``assets/``):
  S3:  fig01 spectra+theory, fig02 PDF, fig03 starlet, fig04 starlet maps, fig05 κ maps.
  DES: fig06 spectra+theory, fig07 PDF, fig08 starlet, fig09 starlet maps, fig10 κ maps.
The starlet coefficient distributions stack the three tomographic bins as rows. The κ maps (fig05/fig10)
use a magma colormap; the starlet figures need the optional ``pycs``/CosmoStat backend:
``uv sync --extra starlet``.

Run (CPU is fine; float64 — the weak-lensing theory C_ℓ NaNs above ℓ≈215 in float32):
    JAX_PLATFORMS=cpu uv run --no-sync python docs/5-experiments/07-born-lensing/build_figures.py
"""

import jax

# float64 globally BEFORE jax_cosmo (the enable_x64() context manager breaks jax_cosmo's pure_callback
# distance cache; the global flag is the safe route) — also required so the lensing theory C_ℓ does not
# go NaN above ℓ≈215.
jax.config.update("jax_enable_x64", True)

import sys
from pathlib import Path

import equinox as eqx
import healpy as hp
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from huggingface_hub import snapshot_download

from jax_fli import compute_theory_cl
from jax_fli.io import Catalog, get_des_y3_nz_shear, get_stage3_nz_shear

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _exputils import savefig, set_style  # noqa: E402

ASSETS = HERE / "assets"
REPO = "ASKabalan/jax-fli-experiments"

NSIDE_HI, NSIDE_LO = 2048, 512
LMAX = 1500  # all published κ spectra stop here
NLB = 16  # multipoles per bandpower bin
NSCALES = 5  # starlet scales
LENSED = "07-cosmogrid-lensing"

# colours: reference vs our lensed sim (at the two nsides)
C_REF = "tab:orange"
C_LENS = "tab:blue"  # lensed at native 2048
C_LENS_LO = "tab:green"  # lensed downsampled to 512
C_2BIN = "tab:purple"  # the shallower 2-bin sim (box-depth cross-check) — solid, not grey like theory
C_TH = "0.3"  # Limber theory (the only grey line)
THEORY_LS = {NSIDE_LO: ":", NSIDE_HI: "--"}

root = Path(snapshot_download(REPO, repo_type="dataset", local_files_only=True))


def _load(path):
    return Catalog.from_dataset(load_dataset("parquet", data_files=str(root / path), split="train"))


def _spec(path):
    return _load(path).field[0]


def lensed_spec(nbin, src):
    return f"{LENSED}/spectra/cosmogrid_{nbin}bin_fullsky_slab/{src}.parquet"


def lensed_map(nbin, src):
    return f"{LENSED}/kappa/cosmogrid_{nbin}bin_fullsky_slab/{src}/BORN_exp6_{nbin}bin_fullsky_slab_s0.parquet"


# --- helpers --------------------------------------------------------------------------------------
def pixwin_match(theory, nside):
    """theory × pixwin²(nside) at full ℓ resolution (BEFORE binning); ℓ-dependent within a band."""
    pw2 = hp.pixwin(nside, lmax=LMAX) ** 2
    return eqx.tree_at(lambda p: p.array, theory, theory.array * pw2)


def binned(ps):
    """(ell_eff, C_ℓ[n_bins, n_bp]) bandpower-binned at the shared edges (lmin=2, nlb=16)."""
    b = ps.bin(nlb=NLB, lmin=2)
    return np.asarray(b.wavenumber), np.asarray(b.array)


def mean_sub(field):
    """κ − κ̄ per tomographic bin (remove the monopole CosmoGrid carries but our sim does not)."""
    a = field.array
    return field.replace(array=a - a.mean(axis=-1, keepdims=True))


# =============================================================================================
# fig01 / fig05 — per-bin κ C_ℓ vs Limber theory (spectra over data/theory ratio)
# =============================================================================================
def plot_spectra(s, stem):
    """One tomographic-bin column per shown bin; series = reference + lensed (2048 & 512) + 2-bin."""
    n_show = s["n_show"]
    theory = compute_theory_cl(s["cosmo"], jnp.arange(LMAX + 1), s["nz"])  # (4, LMAX+1)
    nsides = sorted({ser["nside"] for ser in s["series"]})
    theory_m = {ns: pixwin_match(theory, ns) for ns in nsides}
    theory_b = {ns: binned(theory_m[ns]) for ns in nsides}  # (ell, (4,nbp))
    series_b = [(ser, binned(ser["spec"])) for ser in s["series"]]
    ell_full = np.arange(LMAX + 1)

    fig, axes = plt.subplots(
        2, n_show, figsize=(4.3 * n_show, 5.2), gridspec_kw={"height_ratios": [3, 1]}, sharex="col"
    )
    fig.suptitle(s["spectra_title"], fontsize=14)
    for i in range(n_show):
        axs, axr = axes[0, i], axes[1, i]
        for ns in nsides:
            axs.plot(
                ell_full[2:],
                np.asarray(theory_m[ns].array)[i][2:],
                color=C_TH,
                ls=THEORY_LS[ns],
                lw=1.1,
                label=rf"Limber $\times\,w_\ell^2$({ns})" if i == 0 else None,
            )
        for ser, (ell_b, cl_b) in series_b:
            if i >= cl_b.shape[0]:
                continue
            axs.plot(ell_b, cl_b[i], color=ser["color"], ls=ser["ls"], lw=1.6, label=ser["label"] if i == 0 else None)
        axs.set(xscale="log", yscale="log")
        axs.set_title(rf"bin {i + 1}  ($z\approx{s['z'][i]:.2f}$)", fontsize=11)
        axs.grid(True, which="both", ls=":", alpha=0.4)
        if i == 0:
            axs.set_ylabel(r"$C_\ell^{\kappa\kappa}$")
            axs.legend(frameon=False, fontsize=7.5)

        axr.axhspan(0.95, 1.05, color="0.7", alpha=0.3)
        axr.axhline(1.0, color="0.4", ls="--", lw=0.9)
        for ser, (ell_b, cl_b) in series_b:
            if i >= cl_b.shape[0]:
                continue
            _, tb = theory_b[ser["nside"]]
            axr.plot(ell_b, cl_b[i] / tb[i], color=ser["color"], ls=ser["ls"], lw=1.6)
        axr.set(xscale="log", ylim=(0.6, 1.3))
        axr.set_xlabel(r"multipole $\ell$")
        axr.grid(True, which="both", ls=":", alpha=0.4)
        if i == 0:
            axr.set_ylabel("data / theory")
    fig.tight_layout()
    savefig(ASSETS / stem, fig)


# =============================================================================================
# fig02 / fig06 — per-bin κ PDF, lensed (512) vs reference (512), monopole removed
# =============================================================================================
def plot_pdf(s, stem):
    n_show = s["n_show"]
    lens = mean_sub(s["lens512"])
    ref = mean_sub(s["ref512"])
    fig, axes = plt.subplots(1, n_show, figsize=(4.7 * n_show, 4.3))
    for i in range(n_show):
        ax = axes[i]
        kref = np.asarray(ref[i].array)
        rng = (float(np.percentile(kref, 0.2)), float(np.percentile(kref, 99.8)))
        for fld, col, lab in [(lens, C_LENS_LO, s["lens_label"]), (ref, C_REF, s["ref_label"])]:
            p = fld[i].compute_pdf(bins=60, range=rng)
            ax.plot(np.asarray(p.bin_centers), np.asarray(p.array), color=col, lw=1.7, label=lab if i == 0 else None)
        ax.set(yscale="log", xlabel=r"$\kappa - \bar\kappa$", title=rf"bin {i + 1}  ($z\approx{s['z'][i]:.2f}$)")
        ax.grid(alpha=0.3, which="both")
        if i == 0:
            ax.set_ylabel("count")
            ax.legend(frameon=False)
    fig.suptitle(s["pdf_title"], y=1.02, fontsize=14)
    fig.tight_layout()
    savefig(ASSETS / stem, fig)


# =============================================================================================
# fig03 / fig08 — starlet per-scale coefficient distributions, lensed vs reference (512).
# One figure, rows = tomographic bins (1/2/3), columns = starlet scales.
# =============================================================================================
def plot_starlet(s, stem):
    lens_all = mean_sub(s["lens512"])
    ref_all = mean_sub(s["ref512"])
    n = s["n_show"]
    fig, axes = plt.subplots(n, NSCALES, figsize=(4.0 * NSCALES, 3.7 * n))
    for r in range(n):
        lens = lens_all[r].starlet_coefficients(nscales=NSCALES)
        ref = ref_all[r].starlet_coefficients(nscales=NSCALES)
        for sc in range(NSCALES):
            ax = axes[r, sc]
            lo, hi = np.percentile(np.asarray(ref.array[sc]), [0.5, 99.5])
            bins = np.linspace(lo, hi, 60)
            for st, col, lab in [(lens, C_LENS_LO, s["lens_label"]), (ref, C_REF, s["ref_label"])]:
                ax.hist(
                    np.asarray(st.array[sc]),
                    bins=bins,
                    density=True,
                    histtype="step",
                    lw=1.6,
                    color=col,
                    label=lab if (r == 0 and sc == 0) else None,
                )
            ax.set(yscale="log")
            ax.grid(alpha=0.3, which="both")
            if r == 0:
                ax.set_title(f"starlet scale {sc}")
            if r == n - 1:
                ax.set_xlabel("coefficient")
            if sc == 0:
                ax.set_ylabel(f"bin {r + 1} (z≈{s['z'][r]:.2f})\nprobability density")
                if r == 0:
                    ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"{s['starlet_title']} — rows are tomographic bins: coarse-scale agreement, "
        "fine-scale divergence (CIC window + resolution)",
        y=1.0,
    )
    fig.tight_layout()
    savefig(ASSETS / stem, fig)


# =============================================================================================
# fig04 / fig09 — starlet coefficient MAPS per scale (mollview), lensed vs reference + difference
# =============================================================================================
def plot_starlet_maps(s, stem):
    i = s["n_show"] - 1  # deepest shown bin (strongest lensing signal)
    z = float(s["z"][i])
    st_l = np.asarray(mean_sub(s["lens512"])[i].starlet_coefficients(nscales=NSCALES).array)
    st_r = np.asarray(mean_sub(s["ref512"])[i].starlet_coefficients(nscales=NSCALES).array)
    rows = [(s["lens_label"], st_l), (s["ref_label"], st_r), ("difference (lensed − reference)", st_l - st_r)]
    vmax = [float(np.percentile(np.abs(st_r[sc]), 99)) for sc in range(NSCALES)]
    vmax_d = [float(np.percentile(np.abs(st_l[sc] - st_r[sc]), 99)) for sc in range(NSCALES)]

    nrows = len(rows)
    fig = plt.figure(figsize=(3.4 * NSCALES, 3.7 * nrows))
    for r, (lab, arr) in enumerate(rows):
        vm = vmax_d if r == 2 else vmax
        for sc in range(NSCALES):
            ttl = f"scale {sc}\n{lab}" if r == 0 else lab
            hp.mollview(
                arr[sc],
                sub=(nrows, NSCALES, NSCALES * r + sc + 1),
                title=ttl,
                min=-vm[sc],
                max=vm[sc],
                cmap="RdBu_r",
                cbar=True,
                unit="starlet coefficient",
                notext=True,
                bgcolor=(0.0,) * 4,
            )
    fig.suptitle(
        f"{s['starlet_title']} maps — bin {i + 1} (z≈{z:.2f}): fine scales (left) carry small-scale "
        "detail, coarse scales (right) the smooth field. Bottom row = difference (independent "
        "realisations → uncorrelated)",
        y=1.0,
        fontsize=13,
    )
    savefig(ASSETS / stem, fig)


# =============================================================================================
# fig05 / fig10 — per-bin κ maps (mollview): rows {lensed, reference, difference}, matched 512
# =============================================================================================
def plot_maps(s, stem):
    n_show = s["n_show"]
    lens = mean_sub(s["lens512"])
    ref = mean_sub(s["ref512"])
    diff = lens.replace(array=lens.array[:n_show] - ref.array[:n_show])
    # κ − κ̄ in magma (clusters bright, voids dark); the difference is a symmetric residual → RdBu_r
    rows = [
        (s["lens_label"], lens, ref, "magma"),
        (s["ref_label"], ref, ref, "magma"),
        ("difference (lensed − reference)", diff, diff, "RdBu_r"),
    ]
    fig = plt.figure(figsize=(4.7 * n_show, 3.6 * len(rows)))
    for r, (lab, fld, scale_src, cmap) in enumerate(rows):
        for i in range(n_show):
            vmax = float(np.percentile(np.abs(np.asarray(scale_src[i].array)), 99))
            ttl = f"bin {i + 1} (z≈{s['z'][i]:.2f})\n{lab}" if r == 0 else lab
            hp.mollview(
                np.asarray(fld[i].array),
                sub=(len(rows), n_show, n_show * r + i + 1),
                title=ttl,
                min=-vmax,
                max=vmax,
                cmap=cmap,
                cbar=True,
                unit=r"$\kappa-\bar\kappa$",
                notext=True,
                bgcolor=(0.0,) * 4,
            )
    fig.suptitle(
        f"{s['maps_title']} (matched nside 512). Bottom row = difference: independent "
        "realisations → uncorrelated, full amplitude retained",
        y=1.0,
        fontsize=13,
    )
    savefig(ASSETS / stem, fig)


# --- assemble each source set -----------------------------------------------------------------
def build_set(src, ref_spec_path, ref_spec_nside, ref512_map_path, ref_label, nz, name):
    """Load the reference + lensed (2-bin & 3-bin) products for one source distribution.

    ``ref_spec_nside`` is the reference *spectrum's* native nside (512 for CosmoGrid, 2048 for the
    Born-on-CG-density κ); ``ref512_map_path`` is its nside-512 map for the PDF/starlet/map panels.
    """
    lensed3_spec = _spec(lensed_spec(3, src))
    lensed2_spec = _spec(lensed_spec(2, src))
    cosmo = _load(lensed_spec(3, src)).cosmology[0]
    n_show = np.asarray(lensed3_spec.array).shape[0]  # 3

    lensed3_map = _spec(lensed_map(3, src))  # native 2048
    lens512 = lensed3_map.ud_sample(NSIDE_LO)
    del lensed3_map  # free the 1.2 GB native map; keep only the 512 version
    lensed3_cl512 = lens512.angular_cl(method="healpy", lmax=LMAX)

    ref512 = _spec(ref512_map_path)  # nside-512 reference map
    ref_spec = _spec(ref_spec_path)

    series = [
        {
            "spec": ref_spec,
            "nside": ref_spec_nside,
            "label": f"{ref_label} ({ref_spec_nside})",
            "color": C_REF,
            "ls": "-",
        },
        {"spec": lensed3_spec, "nside": NSIDE_HI, "label": "lensed 3-bin (2048)", "color": C_LENS, "ls": "-"},
        {"spec": lensed3_cl512, "nside": NSIDE_LO, "label": "lensed 3-bin (512)", "color": C_LENS_LO, "ls": "--"},
        {"spec": lensed2_spec, "nside": NSIDE_HI, "label": "lensed 2-bin (2048)", "color": C_2BIN, "ls": ":"},
    ]
    return {
        "name": name,
        "nz": nz,
        "cosmo": cosmo,
        "n_show": n_show,
        "z": np.asarray(lensed3_spec.z_sources),
        "series": series,
        "lens512": lens512,
        "ref512": ref512,
        "lens_label": "lensed PM-sim (512)",
        "ref_label": ref_label,
        "spectra_title": f"{name}: convergence $C_\\ell$ — lensed PM-sim vs {ref_label} vs Limber theory",
        "pdf_title": f"{name}: convergence PDF — lensed PM-sim vs {ref_label}",
        "starlet_title": f"{name}: starlet coefficients — lensed PM-sim vs {ref_label}",
        "maps_title": f"{name}: convergence maps — lensed PM-sim vs {ref_label}",
    }


def main():
    set_style()

    s3 = build_set(
        "s3",
        "00-cosmogrid/kappa_spectra/spectra_cosmogrid_sample_kappa.parquet",
        NSIDE_LO,
        "00-cosmogrid/kappa/cosmogrid_sample_kappa.parquet",
        "CosmoGrid native κ",
        get_stage3_nz_shear(),
        "Stage-3 source n(z)",
    )
    plot_spectra(s3, "fig01-s3-spectra")
    plot_pdf(s3, "fig02-s3-pdf")
    plot_starlet(s3, "fig03-s3-starlet")
    plot_starlet_maps(s3, "fig04-s3-starlet-maps")
    plot_maps(s3, "fig05-s3-maps")
    del s3

    des = build_set(
        "des_y3",
        "00-cosmogrid/kappa_spectra/spectra_kappa_born_des.parquet",
        NSIDE_HI,
        "00-cosmogrid/kappa/kappa_born_des_512.parquet",
        "Born-on-CG-density κ",
        get_des_y3_nz_shear(),
        "DES Y3 source n(z)",
    )
    plot_spectra(des, "fig06-des-spectra")
    plot_pdf(des, "fig07-des-pdf")
    plot_starlet(des, "fig08-des-starlet")
    plot_starlet_maps(des, "fig09-des-starlet-maps")
    plot_maps(des, "fig10-des-maps")

    print(f"assets written to {ASSETS}")


if __name__ == "__main__":
    main()
