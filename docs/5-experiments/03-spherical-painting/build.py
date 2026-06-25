"""Experiment 03 — spherical painting scheme + pixel window: figures.

Renders the SVG figures for ``docs/5-experiments/03-spherical-painting/README.md`` from the per-shell
spherical angular power spectra of the painting-scheme runs (NGP / bilinear / RBF kernel 0.8 px /
RBF kernel 1.5 px), each painted natively at nside 1024 and at nside 2048, plus the paint@2048 →
ud_grade→1024 variant computed here from the native-2048 density maps. Each measured overdensity
C_ell is compared to the analytic Limber number-counts theory
(``jax_fli.compute_theory_cl_for_density``) on the matching HEALPix pixel-window footing
(× pixwin²(nside) of that map).

Figures (-> ``assets/``):
  fig01 / fig02   native nside 1024: NGP / bilinear / RBF08 / RBF15 vs theory, first / last 5 shells.
  fig03 / fig04   native nside 2048: same four schemes vs theory, first / last 5 shells.
  fig05           paint@2048 vs paint@2048→ud_grade→1024 vs theory (NGP / RBF08 / RBF15), near+far shells.
  fig06           native nside 1024 vs paint@2048→ud_grade→1024 vs theory, near+far shells — the
                  pixel-window recovery check (both on the nside-1024 footing).

Run from the repo root (CPU is fine; loads the near/far nside-2048 density maps for fig05/fig06):
    JAX_PLATFORMS=cpu uv run --no-sync python docs/5-experiments/03-spherical-painting/spherical-painting.py
"""

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

import jax_fli as jfli
from jax_fli import compute_theory_cl_for_density
from jax_fli.io import Catalog

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _exputils import savefig, set_style  # noqa: E402

ASSETS = HERE / "assets"
REPO = "ASKabalan/jax-fli-experiments"
SPEC = "03-spherical-painting/spectra"
DENS = "03-spherical-painting/density"

NLB = 16  # multipoles per bandpower bin
NEAR, FAR = 2, 8  # the two highlighted shell indices (shell_0002, shell_0008)

SCHEMES = ["ngp", "bilinear", "rbf08", "rbf15"]
SCHEME_LABEL = {
    "ngp": "NGP (nearest grid point)",
    "bilinear": "bilinear",
    "rbf08": "RBF kernel 0.8 px",
    "rbf15": "RBF kernel 1.5 px",
}
SCHEME_COLORS = {"ngp": "tab:orange", "bilinear": "tab:blue", "rbf08": "tab:green", "rbf15": "tab:red"}
SCHEME_STYLE = {"ngp": "-", "bilinear": "--", "rbf08": ":", "rbf15": "-."}  # line style per scheme (no markers)
UD_SCHEMES = ["ngp", "rbf08", "rbf15"]  # schemes carried into the ud_grade comparison (fig05/fig06)
C_NATIVE = "0.5"  # native sampling
C_UD = "tab:blue"  # paint@2048 -> ud_grade -> 1024

root = snapshot_download(REPO, repo_type="dataset", local_files_only=True)


def _load(path):
    return Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{path}", split="train"))


# -----------------------------------------------------------------------------
# Measured spectra: every scheme painted natively at nside 1024 and at nside 2048.
# -----------------------------------------------------------------------------
spec1024 = {s: _load(f"{SPEC}/spectra_exp3_{s}_native1024.parquet").field[0] for s in SCHEMES}
spec2048 = {s: _load(f"{SPEC}/spectra_exp3_{s}_native2048.parquet").field[0] for s in SCHEMES}
cosmo = _load(f"{SPEC}/spectra_exp3_ngp_native1024.parquet").cosmology[0]

LMAX = int(spec1024["ngp"].wavenumber.max())
ell_full = np.asarray(spec1024["ngp"].wavenumber)
z_shells = np.asarray(spec1024["ngp"].z_sources)
n_shells = np.asarray(spec1024["ngp"].array).shape[0]

# paint@2048 -> ud_grade -> 1024 spectra for the near/far shells (overdensity, same lmax grid).
ud_b = {}  # (scheme, shell_idx) -> binned bandpowers (n_bins,)
for s in UD_SCHEMES:
    for sh, tag in [(NEAR, "0002"), (FAR, "0008")]:
        d = _load(f"{DENS}/exp3_{s}_native2048/shell_{tag}.parquet").field[0]
        cl = d.ud_sample(1024).to(jfli.units.OVERDENSITY).angular_cl(method="healpy", lmax=LMAX)
        ud_b[(s, sh)] = np.asarray(cl.bin(nlb=NLB, lmin=2).array)

# -----------------------------------------------------------------------------
# Theory (comoving-volume Limber number counts) on each pixel-window footing. The ud_grade map is a
# crude 4-pixel average (NOT an alm resample), so it is treated as a clean nside-1024 map here — a
# diagnostic framing, expect it to overshoot 1 at high ell (retained + aliased small-scale power).
# pixwin is ell-dependent within a bin, so multiply the FULL-res theory BEFORE binning.
# -----------------------------------------------------------------------------
theory = compute_theory_cl_for_density(cosmo, spec1024["ngp"], jnp.arange(LMAX + 1))


def _pixwin_match(nside):
    pw2 = hp.pixwin(nside, lmax=LMAX) ** 2  # numpy (LMAX+1,)
    return eqx.tree_at(lambda p: p.array, theory, theory.array * pw2)


theory_pw = {1024: _pixwin_match(1024), 2048: _pixwin_match(2048)}
theory_pw_arr = {ns: np.asarray(t.array) for ns, t in theory_pw.items()}
theory_b = {ns: np.asarray(t.bin(nlb=NLB, lmin=2).array) for ns, t in theory_pw.items()}
leff = np.asarray(theory_pw[1024].bin(nlb=NLB, lmin=2).wavenumber)

spec_b = {
    1024: {s: np.asarray(spec1024[s].bin(nlb=NLB, lmin=2).array) for s in SCHEMES},
    2048: {s: np.asarray(spec2048[s].bin(nlb=NLB, lmin=2).array) for s in SCHEMES},
}


# =============================================================================
# fig01–fig04 — per-shell binned C_ell + ratio, the four painting schemes at one native nside
# =============================================================================
def plot_schemes_batch(nside, shell_idxs, title, stem):
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
    fig.suptitle(title, y=1.0)
    for col, sh in enumerate(shell_idxs):
        ax_s = axes[0, col]
        ax_r = axes[1, col]
        ax_s.plot(ell_full[2:], theory_pw_arr[nside][sh][2:], color="k", ls="--", lw=1.3, zorder=5)
        for s in SCHEMES:
            ax_s.plot(leff, spec_b[nside][s][sh], color=SCHEME_COLORS[s], ls=SCHEME_STYLE[s], lw=1.5, zorder=4)
        ax_s.set_xscale("log")
        ax_s.set_yscale("log")
        ax_s.set_title(f"shell {sh}:  z = {z_shells[sh]:.3f}", fontsize=11)
        ax_s.grid(True, which="both", ls=":", alpha=0.4)
        if col == 0:
            ax_s.set_ylabel(r"$C_\ell$")
        ax_r.axhspan(0.95, 1.05, color="0.7", alpha=0.3)
        ax_r.axhline(1.0, color="0.4", ls="--", lw=0.9)
        for s in SCHEMES:
            ax_r.plot(
                leff, spec_b[nside][s][sh] / theory_b[nside][sh], color=SCHEME_COLORS[s], ls=SCHEME_STYLE[s], lw=1.3
            )
        ax_r.set_xscale("log")
        ax_r.set_ylim(0.3, 1.3)
        ax_r.set_xlabel(r"multipole $\ell$")
        ax_r.grid(True, which="both", ls=":", alpha=0.4)
        if col == 0:
            ax_r.set_ylabel("meas / theory")
    handles = [
        Line2D([], [], color=SCHEME_COLORS[s], ls=SCHEME_STYLE[s], lw=1.6, label=SCHEME_LABEL[s]) for s in SCHEMES
    ]
    handles += [
        Line2D([], [], color="k", ls="--", lw=1.4, label=rf"Limber theory $\times\,w_\ell^2$({nside})"),
        Line2D([], [], color="0.7", lw=6, alpha=0.5, label=r"$\pm5\%$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    savefig(ASSETS / stem, fig)


# =============================================================================
# fig05 / fig06 — native sampling vs paint@2048→ud_grade→1024, near + far shells, three schemes
# =============================================================================
def plot_udsample_grid(native_nside, title, stem):
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.2)
    for r, sh in enumerate([NEAR, FAR]):
        for c, s in enumerate(UD_SCHEMES):
            cell = gs[r, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.0)
            ax = fig.add_subplot(cell[0])
            axr = fig.add_subplot(cell[1], sharex=ax)
            # theory: native-nside footing (dashed); the ud map is on the 1024 footing (dotted)
            ax.plot(ell_full[2:], theory_pw_arr[native_nside][sh][2:], "k--", lw=1.1, zorder=5)
            if native_nside != 1024:
                ax.plot(ell_full[2:], theory_pw_arr[1024][sh][2:], "k:", lw=1.1, zorder=5)
            ax.plot(leff, spec_b[native_nside][s][sh], color=C_NATIVE, ls=":", lw=1.6, zorder=3)
            ax.plot(leff, ud_b[(s, sh)], color=C_UD, ls="-", lw=1.6, zorder=4)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{SCHEME_LABEL[s]} — shell {sh} (z={z_shells[sh]:.3f})", fontsize=10)
            ax.grid(alpha=0.2, which="both")
            ax.tick_params(labelbottom=False)
            if c == 0:
                ax.set_ylabel(r"$C_\ell$")
            axr.axhspan(0.95, 1.05, color="0.7", alpha=0.3)
            axr.axhline(1.0, color="0.4", ls="--", lw=0.9)
            axr.plot(leff, spec_b[native_nside][s][sh] / theory_b[native_nside][sh], color=C_NATIVE, ls=":", lw=1.4)
            axr.plot(leff, ud_b[(s, sh)] / theory_b[1024][sh], color=C_UD, ls="-", lw=1.4)
            axr.set_xscale("log")
            axr.set_ylim(0.3, 1.3)
            axr.grid(alpha=0.2, which="both")
            if c == 0:
                axr.set_ylabel("meas/thy", fontsize=8)
            axr.set_xlabel(r"$\ell$")
    handles = [
        Line2D([], [], color=C_NATIVE, ls=":", lw=1.6, label=f"native nside {native_nside}"),
        Line2D([], [], color=C_UD, ls="-", lw=1.6, label="paint@2048 → ud_grade → 1024"),
        Line2D([], [], color="k", ls="--", lw=1.4, label=rf"Limber theory $\times\,w_\ell^2$({native_nside})"),
    ]
    if native_nside != 1024:
        handles.append(Line2D([], [], color="k", ls=":", lw=1.4, label=r"Limber theory $\times\,w_\ell^2$(1024)"))
    handles.append(Line2D([], [], color="0.7", lw=6, alpha=0.5, label=r"$\pm5\%$"))
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, y=1.02)
    savefig(ASSETS / stem, fig)


# =============================================================================
# fig07 — native nside 2048 vs 1024 for two schemes (NGP, RBF08), per shell + ratio
# =============================================================================
def plot_nside_compare(schemes_sel, shell_idxs, title, stem):
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
    fig.suptitle(title, y=1.0)
    # nside distinguished by line style (1024 dotted, 2048 solid); scheme by colour.
    style = {1024: dict(ls=":"), 2048: dict(ls="-")}
    for col, sh in enumerate(shell_idxs):
        ax_s = axes[0, col]
        ax_r = axes[1, col]
        ax_s.plot(ell_full[2:], theory_pw_arr[1024][sh][2:], "k:", lw=1.0, zorder=5)
        ax_s.plot(ell_full[2:], theory_pw_arr[2048][sh][2:], "k--", lw=1.0, zorder=5)
        for s in schemes_sel:
            for ns in (1024, 2048):
                ax_s.plot(leff, spec_b[ns][s][sh], color=SCHEME_COLORS[s], lw=1.5, zorder=4, **style[ns])
        ax_s.set_xscale("log")
        ax_s.set_yscale("log")
        ax_s.set_title(f"shell {sh}:  z = {z_shells[sh]:.3f}", fontsize=11)
        ax_s.grid(True, which="both", ls=":", alpha=0.4)
        if col == 0:
            ax_s.set_ylabel(r"$C_\ell$")
        ax_r.axhspan(0.95, 1.05, color="0.7", alpha=0.3)
        ax_r.axhline(1.0, color="0.4", ls="--", lw=0.9)
        for s in schemes_sel:
            for ns in (1024, 2048):
                ax_r.plot(leff, spec_b[ns][s][sh] / theory_b[ns][sh], color=SCHEME_COLORS[s], lw=1.4, **style[ns])
        ax_r.set_xscale("log")
        ax_r.set_ylim(0.3, 1.3)
        ax_r.set_xlabel(r"multipole $\ell$")
        ax_r.grid(True, which="both", ls=":", alpha=0.4)
        if col == 0:
            ax_r.set_ylabel("meas / theory")
    handles = []
    for s in schemes_sel:
        for ns in (1024, 2048):
            handles.append(
                Line2D(
                    [], [], color=SCHEME_COLORS[s], ls=style[ns]["ls"], lw=1.6, label=f"{SCHEME_LABEL[s]} — nside {ns}"
                )
            )
    handles += [
        Line2D([], [], color="k", ls=":", lw=1.4, label=r"Limber theory $\times\,w_\ell^2$(1024)"),
        Line2D([], [], color="k", ls="--", lw=1.4, label=r"Limber theory $\times\,w_\ell^2$(2048)"),
        Line2D([], [], color="0.7", lw=6, alpha=0.5, label=r"$\pm5\%$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    savefig(ASSETS / stem, fig)


def main():
    set_style()
    plot_schemes_batch(
        1024,
        range(0, 5),
        "Native nside 1024 — painting schemes vs theory, shells 0–4",
        "fig01-schemes-native1024-shells-0-4",
    )
    plot_schemes_batch(
        1024,
        range(5, 10),
        "Native nside 1024 — painting schemes vs theory, shells 5–9",
        "fig02-schemes-native1024-shells-5-9",
    )
    plot_schemes_batch(
        2048,
        range(0, 5),
        "Native nside 2048 — painting schemes vs theory, shells 0–4",
        "fig03-schemes-native2048-shells-0-4",
    )
    plot_schemes_batch(
        2048,
        range(5, 10),
        "Native nside 2048 — painting schemes vs theory, shells 5–9",
        "fig04-schemes-native2048-shells-5-9",
    )
    plot_udsample_grid(
        2048, "Paint@2048 vs paint@2048 → ud_grade → 1024 vs theory (near + far shells)", "fig05-udsample-vs-native2048"
    )
    plot_udsample_grid(
        1024,
        "Native nside 1024 vs paint@2048 → ud_grade → 1024 vs theory — pixel-window recovery (near + far shells)",
        "fig06-udsample-vs-native1024",
    )
    plot_nside_compare(
        ["ngp", "rbf08"],
        range(5, 10),
        "Native nside 2048 vs 1024 — NGP and RBF 0.8 px vs theory, shells 5–9",
        "fig07-nside-compare-ngp-rbf08-shells-5-9",
    )
    print(f"assets written to {ASSETS}")


if __name__ == "__main__":
    main()
