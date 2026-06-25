from __future__ import annotations

import jax

# float64 globally BEFORE jax_cosmo (the enable_x64() context manager breaks jax_cosmo's
# pure_callback comoving-distance cache; the global config flag is the safe route).
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
from matplotlib.lines import Line2D

from jax_fli import compute_theory_cl_for_density
from jax_fli.io import Catalog

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _exputils import savefig, set_style  # noqa: E402

ASSETS = HERE / "assets"
REPO = "ASKabalan/jax-fli-experiments"

CIC = "02-mass-assignement/density_spectra/spectra_exp2_cic.parquet"
CIC_DECONV = "02-mass-assignement/density_spectra/spectra_exp2_cic_deconv.parquet"
PCS = "02-mass-assignement/density_spectra/spectra_exp2_pcs.parquet"
PCS_DECONV = "02-mass-assignement/density_spectra/spectra_exp2_pcs_deconv.parquet"
TSC = "02-mass-assignement/density_spectra/spectra_exp2_tsc.parquet"
TSC_DECONV = "02-mass-assignement/density_spectra/spectra_exp2_tsc_deconv.parquet"

root = snapshot_download(REPO, repo_type="dataset", local_files_only=True)

cic_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CIC}", split="train"))
cic_deconv_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CIC_DECONV}", split="train"))
pcs_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{PCS}", split="train"))
pcs_deconv_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{PCS_DECONV}", split="train"))
tsc_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{TSC}", split="train"))
tsc_deconv_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{TSC_DECONV}", split="train"))

cic_spectra, cic_cosmo = cic_cat.field[0], cic_cat.cosmology[0]
cic_deconv_spectra = cic_deconv_cat.field[0]
pcs_spectra = pcs_cat.field[0]
pcs_deconv_spectra = pcs_deconv_cat.field[0]
tsc_spectra = tsc_cat.field[0]
tsc_deconv_spectra = tsc_deconv_cat.field[0]

NSIDE = 2048
NLB = 16  # multipoles per bandpower bin

raw = {"cic": cic_spectra, "tsc": tsc_spectra, "pcs": pcs_spectra}
deconv = {"cic": cic_deconv_spectra, "tsc": tsc_deconv_spectra, "pcs": pcs_deconv_spectra}

# Per-scheme palette + the raw/deconvolved colours for the per-scheme figures.
SCHEME_COLORS = {"cic": "tab:blue", "tsc": "tab:green", "pcs": "tab:red"}
SCHEME_STYLE = {"cic": "-", "tsc": "--", "pcs": ":"}  # line style per scheme (no markers)
SCHEME_LABEL = {
    "cic": "CIC (cloud-in-cell)",
    "tsc": "TSC (triangular-shaped cloud)",
    "pcs": "PCS (piecewise-cubic spline)",
}
C_RAW = "0.5"  # raw (carries the force-window deficit)
C_DECONV = "tab:blue"  # force-window-deconvolved

LMAX = int(cic_spectra.wavenumber.max())
ell_full = np.asarray(cic_spectra.wavenumber)
z_shells = np.asarray(cic_spectra.z_sources)
n_shells = np.asarray(cic_spectra.array).shape[0]

# -----------------------------------------------------------------------------
# Theory (comoving-volume Limber number counts) × pixwin²(nside). The deconvolution is the 3D
# force/mass-assignment window (a PM effect), NOT the HEALPix pixel window, so theory carries
# × pixwin²(2048) in every comparison. pixwin is ell-dependent within a bin → apply at full
# resolution BEFORE binning (same convention as exp 00/01).
# -----------------------------------------------------------------------------
theory_cls = compute_theory_cl_for_density(cic_cosmo, cic_spectra, jnp.arange(LMAX + 1))
pixwin2 = hp.pixwin(NSIDE, lmax=LMAX) ** 2  # numpy (LMAX+1,)
theory_pw = eqx.tree_at(lambda p: p.array, theory_cls, theory_cls.array * pixwin2)
theory_pw_arr = np.asarray(theory_pw.array)

# Bandpower-binned theory + measured spectra (identical edges via lmin=2).
_theory_b = theory_pw.bin(nlb=NLB, lmin=2)
theory_b = np.asarray(_theory_b.array)
leff = np.asarray(_theory_b.wavenumber)
raw_b = {k: np.asarray(v.bin(nlb=NLB, lmin=2).array) for k, v in raw.items()}
deconv_b = {k: np.asarray(v.bin(nlb=NLB, lmin=2).array) for k, v in deconv.items()}


# =============================================================================
# fig01 / fig02 — per-shell binned C_ell vs theory (top) + ratio (bottom), CIC vs TSC vs PCS (raw)
# =============================================================================
def plot_schemes_batch(data_b, shell_idxs, title, stem):
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
    fig.suptitle(title, y=1.0)
    for col, sh in enumerate(shell_idxs):
        ax_s = axes[0, col]
        ax_r = axes[1, col]
        ax_s.plot(ell_full[2:], theory_pw_arr[sh][2:], color="k", ls="--", lw=1.3, zorder=5)
        for k in SCHEME_COLORS:
            ax_s.plot(leff, data_b[k][sh], color=SCHEME_COLORS[k], ls=SCHEME_STYLE[k], lw=1.6, zorder=4)
        ax_s.set_xscale("log")
        ax_s.set_yscale("log")
        ax_s.set_title(f"shell {sh}:  z = {z_shells[sh]:.3f}", fontsize=11)
        ax_s.grid(True, which="both", ls=":", alpha=0.4)
        if col == 0:
            ax_s.set_ylabel(r"$C_\ell$")
        ax_r.axhspan(0.95, 1.05, color="0.7", alpha=0.3)
        ax_r.axhline(1.0, color="0.4", ls="--", lw=0.9)
        for k in SCHEME_COLORS:
            ax_r.plot(leff, data_b[k][sh] / theory_b[sh], color=SCHEME_COLORS[k], ls=SCHEME_STYLE[k], lw=1.4)
        ax_r.set_xscale("log")
        ax_r.set_ylim(0.4, 1.25)
        ax_r.set_xlabel(r"multipole $\ell$")
        ax_r.grid(True, which="both", ls=":", alpha=0.4)
        if col == 0:
            ax_r.set_ylabel("meas / theory")
    handles = [
        Line2D([], [], color=SCHEME_COLORS[k], ls=SCHEME_STYLE[k], lw=1.6, label=SCHEME_LABEL[k]) for k in SCHEME_COLORS
    ]
    handles += [
        Line2D([], [], color="k", ls="--", lw=1.4, label=r"Limber number-counts theory $\times\,w_\ell^2$"),
        Line2D([], [], color="0.7", lw=6, alpha=0.5, label=r"$\pm5\%$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    savefig(ASSETS / stem, fig)


# =============================================================================
# fig03 / fig04 / fig05 — one scheme, raw vs force-deconvolved vs theory, all ten shells (dense 2×5)
# =============================================================================
def plot_deconv_grid(scheme, stem):
    rb, db = raw_b[scheme], deconv_b[scheme]
    fig = plt.figure(figsize=(20, 9.5))
    gs = fig.add_gridspec(2, 5, hspace=0.32, wspace=0.22)
    for i in range(n_shells):
        cell = gs[i // 5, i % 5].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.0)
        ax_cl = fig.add_subplot(cell[0])
        ax_r = fig.add_subplot(cell[1], sharex=ax_cl)
        ax_cl.plot(ell_full[2:], theory_pw_arr[i][2:], "k--", lw=1.2, zorder=5)
        ax_cl.plot(leff, rb[i], color=C_RAW, ls=":", lw=1.5, zorder=3)
        ax_cl.plot(leff, db[i], color=C_DECONV, ls="-", lw=1.5, zorder=4)
        ax_cl.set_xscale("log")
        ax_cl.set_yscale("log")
        ax_cl.set_title(f"shell {i}:  z = {z_shells[i]:.3f}", fontsize=10)
        ax_cl.grid(alpha=0.2, which="both")
        ax_cl.tick_params(labelbottom=False)
        if i % 5 == 0:
            ax_cl.set_ylabel(r"$C_\ell$")
        ax_r.axhspan(0.95, 1.05, color="0.7", alpha=0.3)
        ax_r.axhline(1.0, color="0.4", ls="--", lw=0.9)
        ax_r.plot(leff, rb[i] / theory_b[i], color=C_RAW, ls=":", lw=1.4)
        ax_r.plot(leff, db[i] / theory_b[i], color=C_DECONV, ls="-", lw=1.4)
        ax_r.set_xscale("log")
        ax_r.set_ylim(0.5, 1.2)
        ax_r.grid(alpha=0.2, which="both")
        if i % 5 == 0:
            ax_r.set_ylabel("meas/thy", fontsize=8)
        if i >= 5:
            ax_r.set_xlabel(r"$\ell$")
    handles = [
        Line2D([], [], color=C_RAW, ls=":", lw=1.6, label="raw (carries force-window deficit)"),
        Line2D([], [], color=C_DECONV, ls="-", lw=1.6, label="force-window deconvolved"),
        Line2D([], [], color="k", ls="--", lw=1.4, label=r"Limber number-counts theory $\times\,w_\ell^2$"),
        Line2D([], [], color="0.7", lw=6, alpha=0.5, label=r"$\pm5\%$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(
        f"{SCHEME_LABEL[scheme]}: the Fourier-space force-window deconvolution recovers the small-scale power",
        y=1.02,
    )
    savefig(ASSETS / stem, fig)


def main():
    set_style()
    plot_schemes_batch(
        raw_b,
        range(0, 5),
        "Per-shell angular $C_\\ell$ vs theory and ratio — CIC vs TSC vs PCS (no deconvolution), shells 0–4",
        "fig01-schemes-shells-0-4",
    )
    plot_schemes_batch(
        raw_b,
        range(5, 10),
        "Per-shell angular $C_\\ell$ vs theory and ratio — CIC vs TSC vs PCS (no deconvolution), shells 5–9",
        "fig02-schemes-shells-5-9",
    )
    plot_deconv_grid("cic", "fig03-cic-deconv")
    plot_deconv_grid("pcs", "fig04-pcs-deconv")
    plot_deconv_grid("tsc", "fig05-tsc-deconv")
    plot_schemes_batch(
        deconv_b,
        range(5, 10),
        "Per-shell angular $C_\\ell$ vs theory and ratio — CIC vs TSC vs PCS (force-window deconvolved), shells 5–9",
        "fig06-schemes-deconv-shells-5-9",
    )
    print(f"assets written to {ASSETS}")


if __name__ == "__main__":
    main()
