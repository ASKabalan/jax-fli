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
    JAX_PLATFORMS=cpu uv run --no-sync python docs/5-experiments/07-born-lensing/build.py
"""

import jax

# float64 globally BEFORE jax_cosmo (the enable_x64() context manager breaks jax_cosmo's pure_callback
# distance cache; the global flag is the safe route) — also required so the lensing theory C_ℓ does not
# go NaN above ℓ≈215.
jax.config.update("jax_enable_x64", True)

import sys
from pathlib import Path

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
NLB = 32  # multipoles per bandpower bin
NSCALES = 5  # starlet scales

# colours: reference vs our lensed sim (at the two nsides)
C_REF = "tab:orange"
C_LENS = "tab:blue"  # lensed at native 2048
C_LENS_LO = "tab:green"  # lensed downsampled to 512
C_2BIN = "tab:purple"  # the shallower 2-bin sim (box-depth cross-check) — solid, not grey like theory
C_TH = "0.3"  # Limber theory (the only grey line)
THEORY_LS = {NSIDE_LO: ":", NSIDE_HI: "--"}
LENS_LABEL = "lensed PM-sim (512)"  # our downsampled sim in the PDF/starlet/map panels

root = Path(snapshot_download(REPO, repo_type="dataset", local_files_only=True))

# =============================================================================================
# Load every published product explicitly (glanceable HF paths). Per source set we load the
# reference spectrum + its nside-512 map, and our lensed PM-sim: 3-bin & 2-bin spectra plus the
# native-2048 3-bin map that we downsample to 512 for the pixel-wise PDF/starlet/map panels.
# =============================================================================================
# ---- Stage-3 source n(z): reference = CosmoGrid's own nside-512 kappa ----
S3_REF_SPEC = "00-cosmogrid/kappa_spectra/spectra_cosmogrid_sample_kappa.parquet"
S3_REF_MAP512 = "00-cosmogrid/kappa/cosmogrid_sample_kappa.parquet"
S3_LENSED3_SPEC = "07-cosmogrid-lensing/spectra/cosmogrid_3bin_fullsky_slab/s3.parquet"
S3_LENSED2_SPEC = "07-cosmogrid-lensing/spectra/cosmogrid_2bin_fullsky_slab/s3.parquet"
S3_LENSED3_MAP = "07-cosmogrid-lensing/kappa/cosmogrid_3bin_fullsky_slab/s3/BORN_exp6_3bin_fullsky_slab_s0.parquet"

s3_lensed3_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{S3_LENSED3_SPEC}", split="train"))
s3_lensed2_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{S3_LENSED2_SPEC}", split="train"))
s3_ref_spec_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{S3_REF_SPEC}", split="train"))
s3_ref512_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{S3_REF_MAP512}", split="train"))
s3_map_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{S3_LENSED3_MAP}", split="train"))

lensed3_spec_s3, cosmo = s3_lensed3_cat.field[0], s3_lensed3_cat.cosmology[0]  # cosmo shared by every set (same run)
lensed2_spec_s3 = s3_lensed2_cat.field[0]
ref_spec_s3 = s3_ref_spec_cat.field[0]
ref512_s3 = s3_ref512_cat.field[0]
lens512_s3 = s3_map_cat.field[0].ud_sample(NSIDE_LO)  # downsample native 2048 -> 512 for pixel-wise panels
del s3_map_cat  # free the ~1.2 GB native-2048 catalog
lensed3_cl512_s3 = lens512_s3.angular_cl(method="healpy", lmax=LMAX)

# ---- DES Y3 source n(z): reference = our Born-on-CosmoGrid-density kappa (nside 2048) ----
DES_REF_SPEC = "00-cosmogrid/kappa_spectra/spectra_kappa_born_des.parquet"
DES_REF_MAP512 = "00-cosmogrid/kappa/kappa_born_des_512.parquet"
DES_LENSED3_SPEC = "07-cosmogrid-lensing/spectra/cosmogrid_3bin_fullsky_slab/des_y3.parquet"
DES_LENSED2_SPEC = "07-cosmogrid-lensing/spectra/cosmogrid_2bin_fullsky_slab/des_y3.parquet"
DES_LENSED3_MAP = "07-cosmogrid-lensing/kappa/cosmogrid_3bin_fullsky_slab/des_y3/BORN_exp6_3bin_fullsky_slab_s0.parquet"

des_lensed3_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DES_LENSED3_SPEC}", split="train"))
des_lensed2_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DES_LENSED2_SPEC}", split="train"))
des_ref_spec_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DES_REF_SPEC}", split="train"))
des_ref512_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DES_REF_MAP512}", split="train"))
des_map_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DES_LENSED3_MAP}", split="train"))

lensed3_spec_des = des_lensed3_cat.field[0]
lensed2_spec_des = des_lensed2_cat.field[0]
ref_spec_des = des_ref_spec_cat.field[0]
ref512_des = des_ref512_cat.field[0]
lens512_des = des_map_cat.field[0].ud_sample(NSIDE_LO)  # downsample native 2048 -> 512 for pixel-wise panels
del des_map_cat  # free the ~1.2 GB native-2048 catalog
lensed3_cl512_des = lens512_des.angular_cl(method="healpy", lmax=LMAX)


def pixwin_match(theory, nside):
    """theory × pixwin²(nside) at full ℓ resolution (BEFORE binning); ℓ-dependent within a band."""
    pw2 = hp.pixwin(nside, lmax=LMAX) ** 2
    return theory * pw2


def mean_sub(field):
    """κ − κ̄ per tomographic bin (remove the monopole CosmoGrid carries but our sim does not)."""
    a = field.array
    return field.replace(array=a - a.mean(axis=-1, keepdims=True))


# =============================================================================================
# fig01 / fig06 — per-bin κ C_ℓ vs Limber theory (spectra over data/theory ratio)
# =============================================================================================
def plot_spectra(series, cosmo, nz, z, stem):
    """One tomographic-bin column per shown bin. series = list of {spec, nside, label, color, ls}:
    the reference + our lensed sim (2048 & 512) + the 2-bin cross-check, each compared to the same
    Limber theory pixwin-matched to that series' own nside."""
    n_show = len(z)
    theory = compute_theory_cl(cosmo, jnp.arange(LMAX + 1), nz)  # (nbins, LMAX+1)
    nsides = sorted({ser["nside"] for ser in series})
    theory_m = {ns: pixwin_match(theory, ns) for ns in nsides}
    ell_full = np.arange(LMAX + 1)

    # bandpower-bin theory (per nside) and each series once (mode-weighted, nlb=NLB, shared edges)
    ell_b = np.asarray(theory_m[nsides[0]].bin(nlb=NLB).wavenumber)
    theory_b = {ns: np.asarray(theory_m[ns].bin(nlb=NLB).array) for ns in nsides}
    series_b = [(ser, np.asarray(ser["spec"].bin(nlb=NLB).array)) for ser in series]

    fig, axes = plt.subplots(
        2, n_show, figsize=(4.3 * n_show, 5.2), gridspec_kw={"height_ratios": [3, 1]}, sharex="col"
    )
    for i in range(n_show):
        axs, axr = axes[0, i], axes[1, i]
        for ns in nsides:
            th = np.asarray(theory_m[ns].array)[i][2:]
            axs.plot(
                ell_full[2:],
                ell_full[2:] * (ell_full[2:] + 1) / (2 * np.pi) * th,
                color=C_TH,
                ls=THEORY_LS[ns],
                lw=1.1,
                label=rf"Limber $\times\,w_\ell^2$({ns})" if i == 0 else None,
            )
        for ser, cl_b in series_b:
            if i >= cl_b.shape[0]:
                continue
            dl = ell_b * (ell_b + 1) / (2 * np.pi) * cl_b[i]
            axs.plot(ell_b, dl, color=ser["color"], ls=ser["ls"], lw=1.6, label=ser["label"] if i == 0 else None)
        axs.set(xscale="log", yscale="log")
        axs.set_title(rf"bin {i + 1}  ($z\approx{z[i]:.2f}$)", fontsize=11)
        axs.grid(True, which="both", ls=":", alpha=0.4)
        if i == 0:
            axs.set_ylabel(r"$\ell(\ell+1)\,C_\ell^{\kappa\kappa}/2\pi$")
            axs.legend(frameon=False, fontsize=7.5)

        axr.axhspan(-0.05, 0.05, color="0.7", alpha=0.3)
        axr.axhline(0.0, color="0.4", ls="--", lw=0.9)
        for ser, cl_b in series_b:
            if i >= cl_b.shape[0]:
                continue
            tb = theory_b[ser["nside"]]
            axr.plot(ell_b, cl_b[i] / tb[i] - 1, color=ser["color"], ls=ser["ls"], lw=1.6)
        axr.set(xscale="log", ylim=(-0.4, 0.3))
        axr.set_xlabel(r"multipole $\ell$")
        axr.grid(True, which="both", ls=":", alpha=0.4)
        if i == 0:
            axr.set_ylabel("data / theory - 1")
    fig.tight_layout()
    savefig(ASSETS / stem, fig)


# =============================================================================================
# fig02 / fig07 — per-bin κ PDF, lensed (512) vs reference (512), monopole removed
# =============================================================================================
def plot_pdf(lens512, ref512, lens_label, ref_label, z, stem):
    n_show = len(z)
    lens = mean_sub(lens512)
    ref = mean_sub(ref512)
    fig, axes = plt.subplots(1, n_show, figsize=(4.7 * n_show, 4.3))
    for i in range(n_show):
        ax = axes[i]
        kref = np.asarray(ref[i].array)
        rng = (float(np.percentile(kref, 0.2)), float(np.percentile(kref, 99.8)))
        for fld, col, lab in [(lens, C_LENS_LO, lens_label), (ref, C_REF, ref_label)]:
            p = fld[i].compute_pdf(bins=60, range=rng)
            ax.plot(np.asarray(p.bin_centers), np.asarray(p.array), color=col, lw=1.7, label=lab if i == 0 else None)
        ax.set(yscale="log", xlabel=r"$\kappa - \bar\kappa$", title=rf"bin {i + 1}  ($z\approx{z[i]:.2f}$)")
        ax.grid(alpha=0.3, which="both")
        if i == 0:
            ax.set_ylabel("count")
            ax.legend(frameon=False)
    fig.tight_layout()
    savefig(ASSETS / stem, fig)


# =============================================================================================
# fig03 / fig08 — starlet per-scale coefficient distributions, lensed vs reference (512).
# One figure, rows = tomographic bins (1/2/3), columns = starlet scales.
# =============================================================================================
def plot_starlet(lens512, ref512, lens_label, ref_label, z, stem):
    lens_all = mean_sub(lens512)
    ref_all = mean_sub(ref512)
    n = len(z)
    fig, axes = plt.subplots(n, NSCALES, figsize=(4.0 * NSCALES, 3.7 * n))
    for r in range(n):
        lens = lens_all[r].starlet_coefficients(nscales=NSCALES)
        ref = ref_all[r].starlet_coefficients(nscales=NSCALES)
        for sc in range(NSCALES):
            ax = axes[r, sc]
            lo, hi = np.percentile(np.asarray(ref.array[sc]), [0.5, 99.5])
            bins = np.linspace(lo, hi, 60)
            for st, col, lab in [(lens, C_LENS_LO, lens_label), (ref, C_REF, ref_label)]:
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
                ax.set_ylabel(f"bin {r + 1} ($z\\approx{z[r]:.2f}$)\nprobability density")
                if r == 0:
                    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    savefig(ASSETS / stem, fig)


# =============================================================================================
# fig04 / fig09 — starlet coefficient MAPS per scale (mollview), lensed vs reference + difference
# =============================================================================================
def plot_starlet_maps(lens512, ref512, lens_label, ref_label, z, stem):
    i = len(z) - 1  # deepest shown bin (strongest lensing signal)
    st_l = np.asarray(mean_sub(lens512)[i].starlet_coefficients(nscales=NSCALES).array)
    st_r = np.asarray(mean_sub(ref512)[i].starlet_coefficients(nscales=NSCALES).array)
    rows = [(lens_label, st_l), (ref_label, st_r), (r"difference (lensed $-$ reference)", st_l - st_r)]
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
                unit="",
                notext=True,
                bgcolor=(0.0,) * 4,
            )
    savefig(ASSETS / stem, fig)


# =============================================================================================
# fig05 / fig10 — per-bin κ maps (mollview): rows {lensed, reference, difference}, matched 512
# =============================================================================================
def plot_maps(lens512, ref512, lens_label, ref_label, z, stem):
    n_show = len(z)
    lens = mean_sub(lens512)
    ref = mean_sub(ref512)
    diff = lens.replace(array=lens.array[:n_show] - ref.array[:n_show])
    # κ − κ̄ in magma (clusters bright, voids dark); the difference is a symmetric residual → RdBu_r
    rows = [
        (lens_label, lens, ref, "magma"),
        (ref_label, ref, ref, "magma"),
        (r"difference (lensed $-$ reference)", diff, diff, "RdBu_r"),
    ]
    fig = plt.figure(figsize=(4.7 * n_show, 3.6 * len(rows)))
    for r, (lab, fld, scale_src, cmap) in enumerate(rows):
        for i in range(n_show):
            vmax = float(np.percentile(np.abs(np.asarray(scale_src[i].array)), 99))
            ttl = f"bin {i + 1} ($z\\approx{z[i]:.2f}$)\n{lab}" if r == 0 else lab
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
    savefig(ASSETS / stem, fig)


def main():
    set_style()

    # ---- Stage-3: CosmoGrid native κ (512) vs our lensed PM-sim (2048 & 512) + 2-bin ----
    z_s3 = np.asarray(lensed3_spec_s3.z_sources)
    ref_label_s3 = r"CosmoGrid native $\kappa$"
    series_s3 = [
        {"spec": ref_spec_s3, "nside": NSIDE_LO, "label": rf"{ref_label_s3} ({NSIDE_LO})", "color": C_REF, "ls": "-"},
        {"spec": lensed3_spec_s3, "nside": NSIDE_HI, "label": "lensed 3-bin (2048)", "color": C_LENS, "ls": "-"},
        {"spec": lensed3_cl512_s3, "nside": NSIDE_LO, "label": "lensed 3-bin (512)", "color": C_LENS_LO, "ls": "--"},
        {"spec": lensed2_spec_s3, "nside": NSIDE_HI, "label": "lensed 2-bin (2048)", "color": C_2BIN, "ls": ":"},
    ]
    plot_spectra(series_s3, cosmo, get_stage3_nz_shear(), z_s3, "fig01-s3-spectra")
    plot_pdf(lens512_s3, ref512_s3, LENS_LABEL, ref_label_s3, z_s3, "fig02-s3-pdf")
    plot_starlet(lens512_s3, ref512_s3, LENS_LABEL, ref_label_s3, z_s3, "fig03-s3-starlet")
    plot_starlet_maps(lens512_s3, ref512_s3, LENS_LABEL, ref_label_s3, z_s3, "fig04-s3-starlet-maps")
    plot_maps(lens512_s3, ref512_s3, LENS_LABEL, ref_label_s3, z_s3, "fig05-s3-maps")

    # ---- DES Y3: Born-on-CG-density κ (2048) vs our lensed PM-sim (2048 & 512) + 2-bin ----
    z_des = np.asarray(lensed3_spec_des.z_sources)
    ref_label_des = r"Born-on-CG-density $\kappa$"
    series_des = [
        {"spec": ref_spec_des, "nside": NSIDE_HI, "label": rf"{ref_label_des} ({NSIDE_HI})", "color": C_REF, "ls": "-"},
        {"spec": lensed3_spec_des, "nside": NSIDE_HI, "label": "lensed 3-bin (2048)", "color": C_LENS, "ls": "-"},
        {"spec": lensed3_cl512_des, "nside": NSIDE_LO, "label": "lensed 3-bin (512)", "color": C_LENS_LO, "ls": "--"},
        {"spec": lensed2_spec_des, "nside": NSIDE_HI, "label": "lensed 2-bin (2048)", "color": C_2BIN, "ls": ":"},
    ]
    plot_spectra(series_des, cosmo, get_des_y3_nz_shear(), z_des, "fig06-des-spectra")
    plot_pdf(lens512_des, ref512_des, LENS_LABEL, ref_label_des, z_des, "fig07-des-pdf")
    plot_starlet(lens512_des, ref512_des, LENS_LABEL, ref_label_des, z_des, "fig08-des-starlet")
    plot_starlet_maps(lens512_des, ref512_des, LENS_LABEL, ref_label_des, z_des, "fig09-des-starlet-maps")
    plot_maps(lens512_des, ref512_des, LENS_LABEL, ref_label_des, z_des, "fig10-des-maps")

    print(f"assets written to {ASSETS}")


if __name__ == "__main__":
    main()
