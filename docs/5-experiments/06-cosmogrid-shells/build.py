"""Experiment 06 — match the CosmoGrid density shells: figures.

Renders the SVG figures for ``docs/5-experiments/06-cosmogrid-shells/README.md`` by loading the
already-published HuggingFace products (``ASKabalan/jax-fli-experiments``) and comparing our PM
density shells to the CosmoGrid reference. The 8 runs are
``{2bin,3bin} × {fullsky,quadrant} × {slab,pencil}`` (BullFrog 50-step PM, CIC paint with NO
force-window deconvolution, NGP spherical paint at nside 2048, CosmoGrid run000 cosmology, float64).

The two simulations are **independent realisations** (not CosmoGrid's IC phases), so every comparison
here is statistical — C_ℓ, PDF, peak counts, starlet — never pixel-level. The side-by-side maps show
matching *texture/geometry*, not matching structures.

Everything is compared as **overdensity** δ = ρ/ρ̄ − 1: our shells are stored as ``DENSITY`` and
CosmoGrid's as ``COUNTS`` (different units, different mean level), so counts/density are not
comparable — ``.to(OVERDENSITY)`` is the only common ground. For a HEALPix shell the pixel volume is
uniform, so δ just normalises each shell by its own mean (``normalization="per_plane"`` for the
batched lightcones; the off-centre quadrant observer normalises over its visible footprint).

Spectra are compared at the **same nside**, where the HEALPix pixel window cancels in the ratio — no
pixwin needed (the faint Limber theory overlay carries ×pixwin²(2048) only to sit on the measured
footing). The match degrades at high ℓ from the PM force-mesh Nyquist (ℓ ≈ π·χ/dx), the
un-deconvolved CIC force window, and the shot-noise floor; the per-shell ℓ_max guide marks where the
comparison becomes resolution/painting- rather than geometry-limited.

Figures (-> ``assets/``):
  fig01  per-shell C_ℓ vs CosmoGrid + ratio, representative near/mid/far shells (full sky).
  fig02  (2ℓ+1)-weighted SIM/CosmoGrid band ratio vs comoving distance, all 8 variants.
  fig03  nside-512 matched-resolution cross-check (ties the spectra to the higher-order stats).
  fig04  δ maps, near/mid/far × {2-bin, 3-bin, CosmoGrid}, full sky (texture/geometry only).
  fig05  δ maps, near/mid/far × {2-bin, 3-bin}, big-quadrant footprint.
  fig06  overdensity PDF, SIM vs CosmoGrid, full sky.
  fig07  peak counts, SIM vs CosmoGrid, full sky.
  fig08  masked higher-order: quadrant PDF + peak counts on its footprint vs CosmoGrid cut to the
         same footprint (footprint-pixel selection / apodize-and-taper — no MASTER deconvolution).
  fig09  starlet (spherical wavelet) per-scale coefficient distributions, SIM vs CosmoGrid.
  fig10  starlet coefficient maps per scale (mollview), SIM vs CosmoGrid.
         fig09/fig10 need the optional ``pycs``/CosmoStat backend: ``uv sync --extra starlet``.

Run (CPU is fine; float64; loads ~1.2 GB of nside-512 maps from the HF cache):
    JAX_PLATFORMS=cpu uv run --no-sync python docs/5-experiments/06-cosmogrid-shells/build.py
"""

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
from jaxpm.spherical import spherical_visibility_mask
from matplotlib.lines import Line2D

import jax_fli as jfli
from jax_fli import compute_theory_cl_for_density
from jax_fli.io import Catalog

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _exputils import savefig, set_style  # noqa: E402

ASSETS = HERE / "assets"
REPO = "ASKabalan/jax-fli-experiments"

NSIDE_HI = 2048  # native spherical-painting resolution (== CosmoGrid); spectra capped at ℓ = 1500
NSIDE_LO = 512  # matched resolution for the maps + higher-order statistics
NLB = 16  # multipoles per bandpower bin
BOX = {2: 4200.0, 3: 5000.0}  # full-sky box side [Mpc/h] per source-depth set (README §2)
NEAR, MID, FAR = 5, 25, 38  # representative shell indices (all inside the 2-bin set's 0..39)
OBS_QUAD = (0.1, 0.5, 0.9)  # big-quadrant observer (same as Exp 08); footprint via threshold=1.0

# 8 variants keyed by (nbin, geometry, decomposition).
NBINS = (2, 3)
GEOMS = ("fullsky", "quadrant")
DECOMPS = ("slab", "pencil")


# Style / palette ------------------------------------------------------------------------------
NBIN_COLOR = {2: "tab:blue", 3: "tab:green"}
GEOMKEY_COLOR = {
    (2, "fullsky"): "tab:blue",
    (3, "fullsky"): "tab:green",
    (2, "quadrant"): "tab:red",
    (3, "quadrant"): "tab:purple",
}
DECOMP_LS = {"slab": "-", "pencil": ":"}
C_CG = "tab:orange"  # CosmoGrid reference
C_TH = "0.35"  # Limber theory

root = Path(snapshot_download(REPO, repo_type="dataset", local_files_only=True))

SPEC_2BIN_FULLSKY_SLAB_2048 = "06-cosmogrid-shells/spectra_2048/spectra_cosmogrid_2bin_fullsky_slab_2048.parquet"
SPEC_2BIN_FULLSKY_PENCIL_2048 = "06-cosmogrid-shells/spectra_2048/spectra_cosmogrid_2bin_fullsky_pencil_2048.parquet"
SPEC_2BIN_QUADRANT_SLAB_2048 = "06-cosmogrid-shells/spectra_2048/spectra_cosmogrid_2bin_quadrant_slab_2048.parquet"
SPEC_2BIN_QUADRANT_PENCIL_2048 = "06-cosmogrid-shells/spectra_2048/spectra_cosmogrid_2bin_quadrant_pencil_2048.parquet"
SPEC_3BIN_FULLSKY_SLAB_2048 = "06-cosmogrid-shells/spectra_2048/spectra_cosmogrid_3bin_fullsky_slab_2048.parquet"
SPEC_3BIN_FULLSKY_PENCIL_2048 = "06-cosmogrid-shells/spectra_2048/spectra_cosmogrid_3bin_fullsky_pencil_2048.parquet"
SPEC_3BIN_QUADRANT_SLAB_2048 = "06-cosmogrid-shells/spectra_2048/spectra_cosmogrid_3bin_quadrant_slab_2048.parquet"
SPEC_3BIN_QUADRANT_PENCIL_2048 = "06-cosmogrid-shells/spectra_2048/spectra_cosmogrid_3bin_quadrant_pencil_2048.parquet"

SPEC_2BIN_FULLSKY_SLAB_512 = "06-cosmogrid-shells/spectra_512/spectra_cosmogrid_2bin_fullsky_slab_density_512.parquet"
SPEC_2BIN_FULLSKY_PENCIL_512 = (
    "06-cosmogrid-shells/spectra_512/spectra_cosmogrid_2bin_fullsky_pencil_density_512.parquet"
)
SPEC_2BIN_QUADRANT_SLAB_512 = "06-cosmogrid-shells/spectra_512/spectra_cosmogrid_2bin_quadrant_slab_density_512.parquet"
SPEC_2BIN_QUADRANT_PENCIL_512 = (
    "06-cosmogrid-shells/spectra_512/spectra_cosmogrid_2bin_quadrant_pencil_density_512.parquet"
)
SPEC_3BIN_FULLSKY_SLAB_512 = "06-cosmogrid-shells/spectra_512/spectra_cosmogrid_3bin_fullsky_slab_density_512.parquet"
SPEC_3BIN_FULLSKY_PENCIL_512 = (
    "06-cosmogrid-shells/spectra_512/spectra_cosmogrid_3bin_fullsky_pencil_density_512.parquet"
)
SPEC_3BIN_QUADRANT_SLAB_512 = "06-cosmogrid-shells/spectra_512/spectra_cosmogrid_3bin_quadrant_slab_density_512.parquet"
SPEC_3BIN_QUADRANT_PENCIL_512 = (
    "06-cosmogrid-shells/spectra_512/spectra_cosmogrid_3bin_quadrant_pencil_density_512.parquet"
)

MAP_2BIN_FULLSKY_SLAB = "06-cosmogrid-shells/density_512/cosmogrid_2bin_fullsky_slab_density_512.parquet"
MAP_2BIN_QUADRANT_SLAB = "06-cosmogrid-shells/density_512/cosmogrid_2bin_quadrant_slab_density_512.parquet"
MAP_3BIN_FULLSKY_SLAB = "06-cosmogrid-shells/density_512/cosmogrid_3bin_fullsky_slab_density_512.parquet"
MAP_3BIN_QUADRANT_SLAB = "06-cosmogrid-shells/density_512/cosmogrid_3bin_quadrant_slab_density_512.parquet"

CG_SPEC_2048 = "00-cosmogrid/density_spectra/spectra_cosmogrid_density_nside2048.parquet"
CG_SPEC_512 = "00-cosmogrid/density_spectra/spectra_cosmogrid_density_nside512.parquet"
CG_MAP = "00-cosmogrid/density/cosmogrid_density_nside512.parquet"

# -----------------------------------------------------------------------------
# Explicit per-file loads — every HF parquet on its own line.
# -----------------------------------------------------------------------------
spec_2bin_fs_slab_2048_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=str(root / SPEC_2BIN_FULLSKY_SLAB_2048), split="train")
)
spec_2bin_fs_pencil_2048_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=str(root / SPEC_2BIN_FULLSKY_PENCIL_2048), split="train")
)
spec_2bin_q_slab_2048_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=str(root / SPEC_2BIN_QUADRANT_SLAB_2048), split="train")
)
spec_2bin_q_pencil_2048_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=str(root / SPEC_2BIN_QUADRANT_PENCIL_2048), split="train")
)
spec_3bin_fs_slab_2048_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=str(root / SPEC_3BIN_FULLSKY_SLAB_2048), split="train")
)
spec_3bin_fs_pencil_2048_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=str(root / SPEC_3BIN_FULLSKY_PENCIL_2048), split="train")
)
spec_3bin_q_slab_2048_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=str(root / SPEC_3BIN_QUADRANT_SLAB_2048), split="train")
)
spec_3bin_q_pencil_2048_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=str(root / SPEC_3BIN_QUADRANT_PENCIL_2048), split="train")
)

spec_2bin_fs_slab_512_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=str(root / SPEC_2BIN_FULLSKY_SLAB_512), split="train")
)
spec_2bin_fs_pencil_512_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=str(root / SPEC_2BIN_FULLSKY_PENCIL_512), split="train")
)
spec_2bin_q_slab_512_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=str(root / SPEC_2BIN_QUADRANT_SLAB_512), split="train")
)
spec_2bin_q_pencil_512_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=str(root / SPEC_2BIN_QUADRANT_PENCIL_512), split="train")
)
spec_3bin_fs_slab_512_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=str(root / SPEC_3BIN_FULLSKY_SLAB_512), split="train")
)
spec_3bin_fs_pencil_512_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=str(root / SPEC_3BIN_FULLSKY_PENCIL_512), split="train")
)
spec_3bin_q_slab_512_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=str(root / SPEC_3BIN_QUADRANT_SLAB_512), split="train")
)
spec_3bin_q_pencil_512_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=str(root / SPEC_3BIN_QUADRANT_PENCIL_512), split="train")
)

map_2bin_fs_slab_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=str(root / MAP_2BIN_FULLSKY_SLAB), split="train")
)
map_2bin_q_slab_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=str(root / MAP_2BIN_QUADRANT_SLAB), split="train")
)
map_3bin_fs_slab_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=str(root / MAP_3BIN_FULLSKY_SLAB), split="train")
)
map_3bin_q_slab_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=str(root / MAP_3BIN_QUADRANT_SLAB), split="train")
)

cg_hi_cat = Catalog.from_dataset(load_dataset("parquet", data_files=str(root / CG_SPEC_2048), split="train"))
cg_lo_cat = Catalog.from_dataset(load_dataset("parquet", data_files=str(root / CG_SPEC_512), split="train"))
cg_od_cat = Catalog.from_dataset(load_dataset("parquet", data_files=str(root / CG_MAP), split="train"))

cosmo = spec_2bin_fs_slab_2048_cat.cosmology[0]

sim_spec_hi = {
    (2, "fullsky", "slab"): spec_2bin_fs_slab_2048_cat.field[0],
    (2, "fullsky", "pencil"): spec_2bin_fs_pencil_2048_cat.field[0],
    (2, "quadrant", "slab"): spec_2bin_q_slab_2048_cat.field[0],
    (2, "quadrant", "pencil"): spec_2bin_q_pencil_2048_cat.field[0],
    (3, "fullsky", "slab"): spec_3bin_fs_slab_2048_cat.field[0],
    (3, "fullsky", "pencil"): spec_3bin_fs_pencil_2048_cat.field[0],
    (3, "quadrant", "slab"): spec_3bin_q_slab_2048_cat.field[0],
    (3, "quadrant", "pencil"): spec_3bin_q_pencil_2048_cat.field[0],
}
sim_spec_lo = {
    (2, "fullsky", "slab"): spec_2bin_fs_slab_512_cat.field[0],
    (2, "fullsky", "pencil"): spec_2bin_fs_pencil_512_cat.field[0],
    (2, "quadrant", "slab"): spec_2bin_q_slab_512_cat.field[0],
    (2, "quadrant", "pencil"): spec_2bin_q_pencil_512_cat.field[0],
    (3, "fullsky", "slab"): spec_3bin_fs_slab_512_cat.field[0],
    (3, "fullsky", "pencil"): spec_3bin_fs_pencil_512_cat.field[0],
    (3, "quadrant", "slab"): spec_3bin_q_slab_512_cat.field[0],
    (3, "quadrant", "pencil"): spec_3bin_q_pencil_512_cat.field[0],
}
od_fs = {2: map_2bin_fs_slab_cat.field[0], 3: map_3bin_fs_slab_cat.field[0]}
od_q = {2: map_2bin_q_slab_cat.field[0], 3: map_3bin_q_slab_cat.field[0]}
cg_hi = cg_hi_cat.field[0]
cg_lo = cg_lo_cat.field[0]
cg_od = cg_od_cat.field[0]


# Spectra helpers ------------------------------------------------------------------------------
def bandpowers(ps):
    """(ell_eff, C_ℓ[n_shells, n_bp]) for a PowerSpectrum, whether it is stored per-ℓ (full sky /
    CosmoGrid -> bandpower-bin it) or already as decoupled bandpowers (the masked quadrant)."""
    w = np.asarray(ps.wavenumber)
    if w.size > 200 and np.allclose(np.diff(w[:5]), 1.0):  # dense integer ℓ grid -> bin
        b = ps.bin(nlb=NLB, lmin=2)
        return np.asarray(b.wavenumber), np.asarray(b.array)
    return w, np.asarray(ps.array)  # already decoupled bandpowers


def ratio_to_cg(ell_v, cl_v, ell_cg, cl_cg):
    """cl_v / cl_cg, interpolating the CosmoGrid bandpowers onto the variant's bandpower ℓ
    (shell i of the sim aligns to shell i of CosmoGrid; the sim covers a subset of CG's shells)."""
    out = np.empty_like(cl_v)
    for i in range(cl_v.shape[0]):
        out[i] = cl_v[i] / np.interp(ell_v, ell_cg, cl_cg[i])
    return out


def ell_max_shell(nbin, chi):
    """Per-shell PM Nyquist multipole ℓ_max ≈ π·χ/dx, dx = box/mesh (full-sky 2560³ template)."""
    dx = BOX[nbin] / 2560.0
    return np.pi * chi / dx


# =============================================================================================
# fig01 — per-shell C_ℓ vs CosmoGrid + ratio, representative near/mid/far shells (full sky)
# =============================================================================================
def fig01_spectra(sim_fs, cg_hi, cosmo):
    shells = [NEAR, MID, FAR]
    ell_cg, cg_b = bandpowers(cg_hi)
    z = np.asarray(cg_hi.z_sources)
    chi = np.asarray(cg_hi.comoving_centers)

    # faint continuous Limber number-counts theory on the measured (×pixwin²) footing
    LMAX = int(np.asarray(cg_hi.wavenumber).max())
    ell = jnp.arange(LMAX + 1)
    theory = compute_theory_cl_for_density(cosmo, cg_hi, ell)
    pw2 = hp.pixwin(NSIDE_HI, lmax=LMAX) ** 2
    theory = np.asarray(eqx.tree_at(lambda p: p.array, theory, theory.array * pw2).array)
    ell_np = np.asarray(ell)

    binned = {nb: bandpowers(sim_fs[nb]) for nb in NBINS}  # nb -> (ell, cl)

    fig, axes = plt.subplots(2, 3, figsize=(16, 6.2), gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
    for col, sh in enumerate(shells):
        axs, axr = axes[0, col], axes[1, col]
        axs.plot(ell_np[2:], theory[sh][2:], color=C_TH, ls="--", lw=1.2, zorder=2)
        axs.plot(ell_cg, cg_b[sh], color=C_CG, ls="-", lw=1.8, zorder=4)
        for nb in NBINS:
            ev, cv = binned[nb]
            axs.plot(ev, cv[sh], color=NBIN_COLOR[nb], ls="-", lw=1.5, zorder=3)
        lmax_sh = ell_max_shell(2, chi[sh])
        for ax in (axs, axr):
            ax.axvline(lmax_sh, color="0.6", ls=":", lw=1.0)
        axs.set(xscale="log", yscale="log")
        axs.set_title(f"shell {sh}:  z = {z[sh]:.3f}", fontsize=11)
        axs.grid(True, which="both", ls=":", alpha=0.4)
        if col == 0:
            axs.set_ylabel(r"$C_\ell$")

        axr.axhspan(0.95, 1.05, color="0.7", alpha=0.3)
        axr.axhline(1.0, color="0.4", ls="--", lw=0.9)
        for nb in NBINS:
            ev, cv = binned[nb]
            axr.plot(ev, ratio_to_cg(ev, cv[[sh]], ell_cg, cg_b[[sh]])[0], color=NBIN_COLOR[nb], lw=1.5)
        axr.set(xscale="log", ylim=(0.4, 1.25))
        axr.set_xlabel(r"multipole $\ell$")
        axr.grid(True, which="both", ls=":", alpha=0.4)
        if col == 0:
            axr.set_ylabel("sim / CosmoGrid")

    handles = [Line2D([], [], color=NBIN_COLOR[nb], lw=1.6, label=f"jax-fli {nb}-bin (full sky, slab)") for nb in NBINS]
    handles += [
        Line2D([], [], color=C_CG, lw=1.8, label="CosmoGrid (nside 2048)"),
        Line2D([], [], color=C_TH, ls="--", lw=1.4, label=r"Limber number-counts theory $\times\,w_\ell^2$"),
        Line2D([], [], color="0.6", ls=":", lw=1.0, label=r"$\ell_{\max}\approx\pi\chi/\mathrm{d}x$ (PM Nyquist)"),
        Line2D([], [], color="0.7", lw=6, alpha=0.5, label=r"$\pm5\%$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    savefig(ASSETS / "fig01-spectra-near-mid-far", fig)


# =============================================================================================
# fig02 — (2ℓ+1)-weighted SIM/CosmoGrid band ratio vs comoving distance, all 8 variants
# =============================================================================================
def fig02_band_vs_distance(sim_spec, cg_hi):
    ell_cg, cg_b = bandpowers(cg_hi)
    targets = [200, 400, 800]

    fig, axes = plt.subplots(1, len(targets), figsize=(5.2 * len(targets), 4.8), sharey=True)
    for ax, lt in zip(axes, targets):
        ax.axhspan(0.95, 1.05, color="0.82", lw=0, zorder=0)
        ax.axhline(1.0, color="0.4", ls="--", lw=1.0)
        for nb in NBINS:
            for geom in GEOMS:
                col = GEOMKEY_COLOR[(nb, geom)]
                for dec in DECOMPS:
                    ps = sim_spec[(nb, geom, dec)]
                    ev, cv = bandpowers(ps)
                    chi = np.asarray(ps.comoving_centers)
                    j = int(np.argmin(np.abs(ev - lt)))
                    r = np.array([cv[i, j] / np.interp(ev[j], ell_cg, cg_b[i]) for i in range(cv.shape[0])])
                    ax.plot(
                        chi,
                        r,
                        ls=DECOMP_LS[dec],
                        color=col,
                        lw=1.5 if dec == "slab" else 1.0,
                        alpha=0.95 if dec == "slab" else 0.7,
                    )
        ax.set_title(rf"$\ell \approx {lt}$ (nearest bandpower)", fontsize=11)
        ax.set_xlabel(r"comoving distance $\chi$  [Mpc/$h$]")
        ax.set_ylim(0.4, 1.2)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"sim / CosmoGrid  ($(2\ell+1)$-weighted bandpower)")

    handles = [
        Line2D([], [], color=GEOMKEY_COLOR[(nb, g)], lw=1.8, label=f"{nb}-bin {g}") for nb in NBINS for g in GEOMS
    ]
    handles += [Line2D([], [], color="0.3", ls=DECOMP_LS[d], lw=1.5, label=f"{d} decomposition") for d in DECOMPS]
    handles += [Line2D([], [], color="0.82", lw=6, label=r"$\pm5\%$")]
    fig.legend(handles=handles, loc="upper center", ncol=7, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    savefig(ASSETS / "fig02-band-vs-distance", fig)


# =============================================================================================
# fig03 — nside-512 matched-resolution cross-check (ties spectra to the higher-order stats)
# =============================================================================================
def fig03_nside512(sim_fs_512, cg_lo):
    ell_cg, cg_b = bandpowers(cg_lo)
    z = np.asarray(cg_lo.z_sources)
    binned = {nb: bandpowers(sim_fs_512[nb]) for nb in NBINS}
    shells = [NEAR, MID, FAR]

    fig, axes = plt.subplots(2, 3, figsize=(16, 6.0), gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
    for col, sh in enumerate(shells):
        axs, axr = axes[0, col], axes[1, col]
        axs.plot(ell_cg, cg_b[sh], color=C_CG, lw=1.8)
        for nb in NBINS:
            ev, cv = binned[nb]
            axs.plot(ev, cv[sh], color=NBIN_COLOR[nb], lw=1.5)
        axs.set(xscale="log", yscale="log")
        axs.set_title(f"shell {sh}:  z = {z[sh]:.3f}", fontsize=11)
        axs.grid(True, which="both", ls=":", alpha=0.4)
        if col == 0:
            axs.set_ylabel(r"$C_\ell$")
        axr.axhspan(0.95, 1.05, color="0.7", alpha=0.3)
        axr.axhline(1.0, color="0.4", ls="--", lw=0.9)
        for nb in NBINS:
            ev, cv = binned[nb]
            axr.plot(ev, ratio_to_cg(ev, cv[[sh]], ell_cg, cg_b[[sh]])[0], color=NBIN_COLOR[nb], lw=1.5)
        axr.set(xscale="log", ylim=(0.4, 1.25))
        axr.set_xlabel(r"multipole $\ell$")
        axr.grid(True, which="both", ls=":", alpha=0.4)
        if col == 0:
            axr.set_ylabel("sim / CosmoGrid")
    handles = [Line2D([], [], color=NBIN_COLOR[nb], lw=1.6, label=f"jax-fli {nb}-bin (nside 512)") for nb in NBINS]
    handles += [
        Line2D([], [], color=C_CG, lw=1.8, label="CosmoGrid (nside 512)"),
        Line2D([], [], color="0.7", lw=6, alpha=0.5, label=r"$\pm5\%$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    savefig(ASSETS / "fig03-nside512-crosscheck", fig)


# Map helpers ----------------------------------------------------------------------------------
def _logdelta(delta_1d):
    """log10(1+δ) for one shell's overdensity map; empty (δ=−1) pixels clip to the floor (dark)."""
    return np.log10(np.clip(1.0 + np.asarray(delta_1d, dtype=np.float64), 1e-3, None))


def _od_map(field, sh):
    """log10(1+δ) for shell ``sh`` of a raw DENSITY/COUNTS lightcone (converted to overdensity)."""
    return _logdelta(np.asarray(_overdensity_shell(field, sh).array))


# =============================================================================================
# fig04 / fig05 — δ maps, near/mid/far × variants (texture/geometry only, independent realisations)
# =============================================================================================
def fig04_maps_fullsky(od_fs, cg_od):
    z = np.asarray(cg_od.z_sources)
    cols = [("2-bin (slab)", od_fs[2]), ("3-bin (slab)", od_fs[3]), ("CosmoGrid", cg_od)]
    fig = plt.figure(figsize=(15, 10.2))
    for r, sh in enumerate([NEAR, MID, FAR]):
        vlims = np.percentile(_od_map(cg_od, sh), [45, 99.5])  # per-row scale (each depth on its own)
        for c, (lab, fld) in enumerate(cols):
            d = _od_map(fld, sh)
            ttl = f"{lab}\n(shell {sh}, z={z[sh]:.2f})" if r == 0 else f"shell {sh}, z={z[sh]:.2f}"
            hp.orthview(
                d,
                sub=(3, 3, 3 * r + c + 1),
                rot=(20, 35),
                half_sky=True,
                xsize=420,
                title=ttl,
                min=vlims[0],
                max=vlims[1],
                cmap="inferno",
                cbar=False,
                notext=True,
                bgcolor=(0.0,) * 4,
            )
    savefig(ASSETS / "fig04-maps-fullsky", fig)


def fig05_maps_quadrant(od_q):
    z = np.asarray(od_q[2].z_sources)
    cols = [("2-bin quadrant (slab)", od_q[2]), ("3-bin quadrant (slab)", od_q[3])]
    fig = plt.figure(figsize=(11, 15.0))
    for r, sh in enumerate([NEAR, MID, FAR]):
        # percentiles above the (large) out-of-footprint dark floor, so the in-footprint texture shows
        vlims = np.percentile(_od_map(cols[0][1], sh), [70, 99.5])
        for c, (lab, fld) in enumerate(cols):
            d = _od_map(fld, sh)
            ttl = f"{lab}\n(shell {sh}, z={z[sh]:.2f})" if r == 0 else f"shell {sh}, z={z[sh]:.2f}"
            hp.mollview(
                d,
                sub=(3, 2, 2 * r + c + 1),
                title=ttl,
                min=vlims[0],
                max=vlims[1],
                cmap="inferno",
                cbar=False,
                notext=True,
                bgcolor=(0.0,) * 4,
            )
    savefig(ASSETS / "fig05-maps-quadrant", fig)


# Higher-order helpers -------------------------------------------------------------------------
def _pdf_xy(field_shell, bins, rng):
    p = field_shell.compute_pdf(bins=bins, range=rng)
    return np.asarray(p.bin_centers), np.asarray(p.array)


def _peaks_xy(field_shell, bins, rng):
    p = field_shell.compute_peak_counts(bins=bins, range=rng)
    return np.asarray(p.bin_centers), np.asarray(p.array)


def _overdensity_shell(field, sh, normalization="per_plane"):
    return field[sh].to(jfli.units.OVERDENSITY, normalization=normalization)


# =============================================================================================
# fig06 / fig07 — overdensity PDF and peak counts, SIM vs CosmoGrid (full sky)
# =============================================================================================
def _higher_order_grid(od_fs, cg_od, xy, title, stem, xlabel):
    shells = [NEAR, MID, FAR]
    z = np.asarray(cg_od.z_sources)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, sh in zip(axes, shells):
        rng = (-1.0, float(np.percentile(np.asarray(_overdensity_shell(cg_od, sh).array), 99.9)))
        for nb in NBINS:
            x, y = xy(_overdensity_shell(od_fs[nb], sh), 50, rng)
            ax.plot(x, y, color=NBIN_COLOR[nb], lw=1.6, label=f"jax-fli {nb}-bin" if sh == NEAR else None)
        x, y = xy(_overdensity_shell(cg_od, sh), 50, rng)
        ax.plot(x, y, color=C_CG, lw=1.8, label="CosmoGrid" if sh == NEAR else None)
        ax.set(yscale="log", title=f"shell {sh}:  z = {z[sh]:.3f}", xlabel=xlabel)
        ax.grid(alpha=0.3, which="both")
        if sh == NEAR:
            ax.set_ylabel("count")
            ax.legend(frameon=False)
    fig.tight_layout()
    savefig(ASSETS / stem, fig)


# =============================================================================================
# fig08 — masked higher-order: quadrant on its footprint vs CosmoGrid cut to the same footprint
# =============================================================================================
def _masked_pdf_xy(od_shell, footprint, bins, rng):
    vals = np.asarray(od_shell.array)[footprint]
    h, edges = np.histogram(vals, bins=bins, range=rng)
    return 0.5 * (edges[1:] + edges[:-1]), h


def fig08_masked(od_q, cg_od):
    # Conservative all-radii footprint (the threshold=1.0 visibility cap, same observer as Exp 08).
    # Applied IDENTICALLY to the quadrant sim and to the full-sky CosmoGrid map, so PDF/peaks compare
    # on the very same pixels (these statistics have no MASTER-style mask deconvolution).
    footprint = np.asarray(spherical_visibility_mask(NSIDE_LO, OBS_QUAD, threshold=1.0)).astype(bool)
    apo = np.asarray(jfli.data.apodize(footprint.astype(float), 2.0))
    z = np.asarray(cg_od.z_sources)
    shells = [MID, FAR]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6))
    for r, sh in enumerate(shells):
        od_q2 = _overdensity_shell(od_q[2], sh)
        od_cg = _overdensity_shell(cg_od, sh)
        rng = (-1.0, float(np.percentile(np.asarray(od_cg.array)[footprint], 99.9)))

        # PDF: select only footprint pixels (never the δ=−1 out-of-footprint pixels)
        ax = axes[r, 0]
        xq, yq = _masked_pdf_xy(od_q2, footprint, 50, rng)
        xc, yc = _masked_pdf_xy(od_cg, footprint, 50, rng)
        ax.plot(xq, yq, color=GEOMKEY_COLOR[(2, "quadrant")], lw=1.7, label="jax-fli 2-bin quadrant")
        ax.plot(xc, yc, color=C_CG, lw=1.8, label="CosmoGrid (same footprint)")
        ax.set(yscale="log", xlabel=r"$\delta$", ylabel="count", title=f"PDF on footprint — shell {sh}, z={z[sh]:.2f}")
        ax.grid(alpha=0.3, which="both")
        if r == 0:
            ax.legend(frameon=False)

        # peak counts: taper to the apodized footprint, so boundary pixels can't seed spurious peaks
        ax = axes[r, 1]
        for fld, col in [(od_q2, GEOMKEY_COLOR[(2, "quadrant")]), (od_cg, C_CG)]:
            x, y = _peaks_xy(fld.replace(array=fld.array * apo), 50, rng)
            ax.plot(x, y, color=col, lw=1.7)
        ax.set(
            yscale="log",
            xlabel=r"peak height $\delta$",
            ylabel="count",
            title=f"peak counts on footprint — shell {sh}, z={z[sh]:.2f}",
        )
        ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    savefig(ASSETS / "fig08-masked-higher-order", fig)


# =============================================================================================
# fig09 — starlet (spherical wavelet) per-scale coefficient distributions (needs pycs/CosmoStat)
# =============================================================================================
def fig09_starlet(od_fs, cg_od, nscales=5):
    shells = [NEAR, MID, FAR]  # three rows: near / mid / far representative shells
    z = np.asarray(cg_od.z_sources)
    fig, axes = plt.subplots(len(shells), nscales, figsize=(4.0 * nscales, 3.7 * len(shells)))
    for r, sh in enumerate(shells):
        series = [(f"jax-fli {nb}-bin", _overdensity_shell(od_fs[nb], sh), NBIN_COLOR[nb]) for nb in NBINS]
        series.append(("CosmoGrid", _overdensity_shell(cg_od, sh), C_CG))
        coeffs = [(lab, fld.starlet_coefficients(nscales=nscales), col) for lab, fld, col in series]
        for s in range(nscales):
            ax = axes[r, s]
            lo, hi = np.percentile(np.asarray(coeffs[-1][1].array[s]), [0.5, 99.5])
            bins = np.linspace(lo, hi, 60)
            for lab, st, col in coeffs:
                ax.hist(
                    np.asarray(st.array[s]),
                    bins=bins,
                    density=True,
                    histtype="step",
                    lw=1.6,
                    color=col,
                    label=lab if (r == 0 and s == 0) else None,
                )
            ax.set(yscale="log")
            ax.grid(alpha=0.3, which="both")
            if r == 0:
                ax.set_title(f"starlet scale {s}")
            if r == len(shells) - 1:
                ax.set_xlabel("coefficient")
            if s == 0:
                ax.set_ylabel(f"shell {sh} (z={z[sh]:.2f})\nprobability density")
                if r == 0:
                    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    savefig(ASSETS / "fig09-starlet", fig)


# =============================================================================================
# fig10 — starlet coefficient MAPS per scale (mollview), jax-fli vs CosmoGrid (needs pycs)
# =============================================================================================
def fig10_starlet_maps(od_fs, cg_od, nscales=5):
    sh = MID
    st_sim = np.asarray(_overdensity_shell(od_fs[2], sh).starlet_coefficients(nscales=nscales).array)
    st_cg = np.asarray(_overdensity_shell(cg_od, sh).starlet_coefficients(nscales=nscales).array)
    # The two are INDEPENDENT realisations: the third row (sim − CosmoGrid) is not a residual but a
    # check that the fields are uncorrelated — it keeps the full per-scale amplitude (no cancellation).
    rows = [("jax-fli 2-bin", st_sim), ("CosmoGrid", st_cg), ("difference (jax-fli − CosmoGrid)", st_sim - st_cg)]
    # per-scale symmetric scale: rows 0/1 share one (comparable textures), the diff gets its own
    vmax = [float(np.percentile(np.abs(st_cg[s]), 99)) for s in range(nscales)]
    vmax_d = [float(np.percentile(np.abs(st_sim[s] - st_cg[s]), 99)) for s in range(nscales)]

    nrows = len(rows)
    fig = plt.figure(figsize=(3.4 * nscales, 3.7 * nrows))
    for r, (lab, arr) in enumerate(rows):
        vm = vmax_d if r == 2 else vmax
        for s in range(nscales):
            ttl = f"scale {s}\n{lab}" if r == 0 else lab
            hp.mollview(
                arr[s],
                sub=(nrows, nscales, nscales * r + s + 1),
                title=ttl,
                min=-vm[s],
                max=vm[s],
                cmap="RdBu_r",
                cbar=True,
                unit="",
                notext=True,
                bgcolor=(0.0,) * 4,
            )
    savefig(ASSETS / "fig10-starlet-maps", fig)


def main():
    set_style()

    sim_fs_hi = {nb: sim_spec_hi[(nb, "fullsky", "slab")] for nb in NBINS}
    sim_fs_lo = {nb: sim_spec_lo[(nb, "fullsky", "slab")] for nb in NBINS}

    fig01_spectra(sim_fs_hi, cg_hi, cosmo)
    fig02_band_vs_distance(sim_spec_hi, cg_hi)
    fig03_nside512(sim_fs_lo, cg_lo)

    fig04_maps_fullsky(od_fs, cg_od)
    fig05_maps_quadrant(od_q)
    _higher_order_grid(
        od_fs,
        cg_od,
        _pdf_xy,
        "Overdensity PDF — jax-fli PM vs CosmoGrid (full sky, nside 512)",
        "fig06-pdf",
        r"$\delta$",
    )
    _higher_order_grid(
        od_fs,
        cg_od,
        _peaks_xy,
        "Peak counts — jax-fli PM vs CosmoGrid (full sky, nside 512)",
        "fig07-peak-counts",
        r"peak height $\delta$",
    )
    fig08_masked(od_q, cg_od)
    fig09_starlet(od_fs, cg_od)  # needs pycs/CosmoStat (uv sync --extra starlet)
    fig10_starlet_maps(od_fs, cg_od)  # needs pycs/CosmoStat

    print(f"assets written to {ASSETS}")


if __name__ == "__main__":
    main()
