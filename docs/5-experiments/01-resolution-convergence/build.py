"""Experiment 01 — resolution convergence: figures.

Renders the SVG figures for ``docs/5-experiments/01-resolution-convergence/README.md`` from the
per-shell spherical angular power spectra of the five slab runs (``spectra_m*.parquet``), the
two pencil re-runs (precomputed ``spectra_m*_pencils.parquet``), and the density maps themselves
(``m*.parquet`` / ``m*_pencils.parquet``). Each shell's measured C_ell is compared to the
analytic Limber number-counts theory (``jax_fli.compute_theory_cl_for_density``), put on the
same HEALPix pixel-window footing (× pixwin²(512)). Spectra are bandpower-binned with
``.bin(nlb=16)``.

Figures (-> ``assets/``):
  fig01 / fig02   per-shell binned C_ell vs theory + ratio, first / last five shells, all 5 slabs.
  fig03           convergence diagnostic (measured/theory vs mesh, ell in [270,330]) + halo-vs-
                  displacement panel — the headline (anti-convergence + its cause), slabs only.
  fig04 / fig05   slab vs pencil for m2560 / m3072: band-[270,330] measured/theory vs distance.
  fig06           the same diagnostic as fig03 with m2560/m3072 swapped to pencil (convergence restored).
  fig07           the delta maps (shell 9, all five slabs): gnomonic / orthographic / mollview rows.
  fig08           the delta maps for m2560 / m3072, slab vs pencil: gnomonic + mollview rows.

Run from the repo root (CPU is fine; ~minutes, loads the 7 density maps for fig07/fig08):
    JAX_PLATFORMS=cpu uv run --no-sync python docs/5-experiments/01-resolution-convergence/build.py
"""

from __future__ import annotations

import jax

# float64 globally BEFORE jax_cosmo (the enable_x64() context manager breaks jax_cosmo's
# pure_callback comoving-distance cache; the global config flag is the safe route).
jax.config.update("jax_enable_x64", True)

import sys
from pathlib import Path

import healpy as hp
import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from huggingface_hub import snapshot_download
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from jax_fli import compute_theory_cl_for_density
from jax_fli.io import Catalog

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _exputils import savefig, set_style  # noqa: E402

ASSETS = HERE / "assets"
REPO = "ASKabalan/jax-fli-experiments"

NSIDE = 512
BOX = 2000.0  # Mpc/h
MESHES = [512, 1024, 2048, 2560, 3072]
PENCILS = [2560, 3072]
PX = {512: 4, 1024: 8, 2048: 64, 2560: 128, 3072: 256}  # x-decomposition (GPUs) per rung
HALO = {m: 0.5 * BOX / PX[m] for m in MESHES}  # physical ghost-zone width = halo_multiplier*box/px
COLORS = {m: c for m, c in zip(MESHES, cm.viridis(np.linspace(0.0, 0.9, len(MESHES))))}
MESH_STYLES = ["-", "--", "-.", ":", "-"]  # per-mesh line style (no markers); fig01/fig02 only
NLB = 16  # multipoles per bandpower bin
BAND = (270, 330)  # CV-clean convergence window (the [30,300] band is cosmic-variance dominated)
LPLOT = 2 * NSIDE  # trust boundary: beyond ell = 2*nside the maps are pixel-noise/aliasing dominated

root = snapshot_download(REPO, repo_type="dataset", local_files_only=True)

SPEC_512 = "01-resolution/spectra/spectra_m512.parquet"
SPEC_1024 = "01-resolution/spectra/spectra_m1024.parquet"
SPEC_2048 = "01-resolution/spectra/spectra_m2048.parquet"
SPEC_2560 = "01-resolution/spectra/spectra_m2560.parquet"
SPEC_3072 = "01-resolution/spectra/spectra_m3072.parquet"

SPEC_2560_PENCIL = "01-resolution/spectra/spectra_m2560_pencils.parquet"
SPEC_3072_PENCIL = "01-resolution/spectra/spectra_m3072_pencils.parquet"

DENS_512 = "01-resolution/density/m512.parquet"
DENS_1024 = "01-resolution/density/m1024.parquet"
DENS_2048 = "01-resolution/density/m2048.parquet"
DENS_2560 = "01-resolution/density/m2560.parquet"
DENS_3072 = "01-resolution/density/m3072.parquet"

DENS_2560_PENCIL = "01-resolution/density/m2560_pencils.parquet"
DENS_3072_PENCIL = "01-resolution/density/m3072_pencils.parquet"

# -----------------------------------------------------------------------------
# Explicit per-file loads — every HF parquet used is visible on its own line. Slab angular spectra and
# the precomputed pencil spectra (fig01–06), slab + pencil density maps (fig07/fig08 + the pencil halo).
# -----------------------------------------------------------------------------
spec_512_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{SPEC_512}", split="train"))
spec_1024_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{SPEC_1024}", split="train"))
spec_2048_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{SPEC_2048}", split="train"))
spec_2560_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{SPEC_2560}", split="train"))
spec_3072_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{SPEC_3072}", split="train"))

spec_2560_pencil_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{SPEC_2560_PENCIL}", split="train")
)
spec_3072_pencil_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{SPEC_3072_PENCIL}", split="train")
)

dens_512_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DENS_512}", split="train"))
dens_1024_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DENS_1024}", split="train"))
dens_2048_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DENS_2048}", split="train"))
dens_2560_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DENS_2560}", split="train"))
dens_3072_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DENS_3072}", split="train"))

dens_2560_pencil_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_2560_PENCIL}", split="train")
)
dens_3072_pencil_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_3072_PENCIL}", split="train")
)


cosmo = spec_512_cat.cosmology[0]  # one fiducial cosmology shared by every run

# Mesh-indexed views the figure loops iterate over (built by hand from the explicit loads above).
spectra = {
    512: spec_512_cat.field[0],
    1024: spec_1024_cat.field[0],
    2048: spec_2048_cat.field[0],
    2560: spec_2560_cat.field[0],
    3072: spec_3072_cat.field[0],
}
density = {
    512: dens_512_cat.field[0],
    1024: dens_1024_cat.field[0],
    2048: dens_2048_cat.field[0],
    2560: dens_2560_cat.field[0],
    3072: dens_3072_cat.field[0],
}
density_pencil = {2560: dens_2560_pencil_cat.field[0], 3072: dens_3072_pencil_cat.field[0]}
PENCIL_HALO = {m: float(np.asarray(density_pencil[m].halo_size).ravel()[0]) * BOX / m for m in PENCILS}

# Every parquet must carry the same fiducial cosmology — check silently, stop if a file disagrees.
for _c in (spec_1024_cat, spec_2048_cat, spec_2560_cat, spec_3072_cat, spec_2560_pencil_cat, spec_3072_pencil_cat):
    assert np.isclose(float(_c.cosmology[0].Omega_c), float(cosmo.Omega_c))
    assert np.isclose(float(_c.cosmology[0].sigma8), float(cosmo.sigma8))

LMAX = int(spectra[MESHES[0]].wavenumber.max())
ell = jnp.arange(LMAX + 1)
ell_full = np.asarray(spectra[MESHES[0]].wavenumber)
z_shells = np.asarray(spectra[MESHES[0]].z_sources)
chi_shells = np.asarray(spectra[MESHES[0]].comoving_centers)
n_shells = np.asarray(spectra[MESHES[0]].array).shape[0]


def _on_slab_ell(arr):
    """Right-pad a precomputed pencil C_ell (lmax 1500) up to the slab ell grid (lmax 1535) with NaN.

    The two were computed at slightly different lmax; every figure here works on ell <= 2*nside = 1024,
    well inside both, so the pad only realigns the axis and never enters a band/plot.
    """
    arr = np.asarray(arr)
    if arr.shape[1] == ell_full.size:
        return arr
    out = np.full((arr.shape[0], ell_full.size), np.nan)
    out[:, : arr.shape[1]] = arr
    return out


# Precomputed pencil C_ell (HF), on the same ell footing as the slab spectra (was angular_cl on the maps).
spectra_pencil = {
    2560: _on_slab_ell(spec_2560_pencil_cat.field[0].array),
    3072: _on_slab_ell(spec_3072_pencil_cat.field[0].array),
}

# -----------------------------------------------------------------------------
# Theory (comoving-volume Limber number counts), on the same HEALPix pixel-window footing.
# pixwin is ell-dependent within a bin, so multiply the FULL-res theory BEFORE any binning.
# -----------------------------------------------------------------------------
theory = compute_theory_cl_for_density(cosmo, spectra[MESHES[0]], ell)  # (n_shells, n_ell)
pixwin2 = hp.pixwin(NSIDE, lmax=LMAX) ** 2  # numpy (LMAX+1,)
theory_pw = theory * pixwin2  # abstract field: broadcasts (n_shells, n_ell) * (n_ell,) over the array
theory_pw_arr = np.asarray(theory_pw.array)
raw_s = {m: np.asarray(spectra[m].array) for m in MESHES}  # full-grid measured (fig03/04/05)

# Bandpower-binned spectra (fig01/fig02): bin theory and every mesh on identical edges.
theory_b = theory_pw.bin(nlb=NLB, lmin=2)
meas_b = {m: spectra[m].bin(nlb=NLB, lmin=2) for m in MESHES}
leff = np.asarray(theory_b.wavenumber)


def band_ratio(ell_grid, meas_2d, theo_2d, lo, hi):
    """(2l+1)-weighted measured/theory over [lo,hi] per shell, plus the f_sky=1 CV sigma."""
    m = (ell_grid >= lo) & (ell_grid <= hi)
    w = 2.0 * ell_grid[m] + 1.0
    mb = (meas_2d[:, m] * w).sum(axis=1) / w.sum()
    tb = (theo_2d[:, m] * w).sum(axis=1) / w.sum()
    return mb / tb, float(np.sqrt(2.0 / w.sum()))


def _sigma_displacement(z):
    """rms 1D Zel'dovich displacement from the linear P(k) at redshift z."""
    a = 1.0 / (1.0 + z)
    k = jnp.logspace(-4, 1.3, 4000)
    pk = jc.power.linear_matter_power(cosmo, k, a=a)
    return float(jnp.sqrt(jnp.trapezoid(pk, k) / (6 * np.pi**2)))


# =============================================================================
# fig01 / fig02 — per-shell binned C_ell vs theory (top) + measured/theory ratio (bottom)
# =============================================================================
def plot_spectra_batch(shell_idxs, title, stem):
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
    for col, sh in enumerate(shell_idxs):
        ax_s = axes[0, col]
        ax_r = axes[1, col]
        # top: continuous theory (×pixwin²) + the five binned slab spectra
        ax_s.plot(ell_full[2:], theory_pw_arr[sh][2:], color="k", ls="--", lw=1.3, zorder=5)
        for mi, m in enumerate(MESHES):
            ax_s.plot(leff, np.asarray(meas_b[m].array)[sh], color=COLORS[m], ls=MESH_STYLES[mi], lw=1.5, zorder=4)
        ax_s.axvline(LPLOT, color="0.6", ls=":", lw=0.9)
        ax_s.set_xscale("log")
        ax_s.set_yscale("log")
        ax_s.set_xlim(max(2.0, leff.min() * 0.8), LPLOT * 1.07)
        ax_s.set_title(f"shell {sh}:  z = {z_shells[sh]:.3f}", fontsize=11)
        ax_s.grid(True, which="both", ls=":", alpha=0.4)
        if col == 0:
            ax_s.set_ylabel(r"$C_\ell$")
        # bottom (3:1): binned measured / binned theory per resolution
        ax_r.axhspan(0.95, 1.05, color="0.7", alpha=0.3)
        ax_r.axhline(1.0, color="0.4", ls="--", lw=0.9)
        for mi, m in enumerate(MESHES):
            ax_r.plot(
                leff,
                np.asarray(meas_b[m].array)[sh] / np.asarray(theory_b.array)[sh],
                color=COLORS[m],
                ls=MESH_STYLES[mi],
                lw=1.3,
            )
        ax_r.axvline(LPLOT, color="0.6", ls=":", lw=0.9)
        ax_r.set_xscale("log")
        ax_r.set_xlim(max(2.0, leff.min() * 0.8), LPLOT * 1.07)
        ax_r.set_ylim(0.4, 1.4)
        ax_r.set_xlabel(r"multipole $\ell$")
        ax_r.grid(True, which="both", ls=":", alpha=0.4)
        if col == 0:
            ax_r.set_ylabel("meas / theory")
    handles = [
        Line2D([], [], color=COLORS[m], ls=MESH_STYLES[mi], lw=1.6, label=f"m{m} (px={PX[m]})")
        for mi, m in enumerate(MESHES)
    ]
    handles += [
        Line2D([], [], color="k", ls="--", lw=1.4, label=r"Limber number-counts theory $\times\,w_\ell^2$"),
        Line2D([], [], color="0.7", lw=6, alpha=0.5, label=r"$\pm5\%$"),
        Line2D([], [], color="0.6", ls=":", lw=0.9, label=r"$\ell=2\,n_{\rm side}$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=8, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    savefig(ASSETS / stem, fig)


# =============================================================================
# fig03 / fig06 — convergence diagnostic + the halo-vs-displacement mechanism.
# fig03: all five slab runs. fig06: the same, with m2560/m3072 swapped to their pencil decomposition.
# =============================================================================
def _fig_convergence(raw_map, halo_map, labels, suptitle, stem, annotations):
    lo, hi = BAND
    ratios = {m: band_ratio(ell_full, raw_map[m], theory_pw_arr, lo, hi)[0] for m in MESHES}
    chi_cut = 400.0  # drop the 4 innermost (near) shells — noisy / Limber-breakdown
    kept = [i for i in range(n_shells) if chi_shells[i] >= chi_cut]
    cmap = cm.plasma
    norm = Normalize(vmin=float(chi_shells[kept].min()), vmax=float(chi_shells[kept].max()))
    xs = np.arange(len(MESHES))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.8, 5.4))

    # left: measured/theory over the CV-clean band, per outer shell, vs resolution
    axL.axhspan(0.95, 1.05, color="0.82", lw=0, label=r"±5%")
    axL.axhline(1.0, color="0.4", ls="--", lw=1.0)
    for i in kept:
        axL.plot(xs, [ratios[m][i] for m in MESHES], "o-", color=cmap(norm(chi_shells[i])), lw=1.8, alpha=0.95, ms=4)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axL, pad=0.02)
    cbar.set_label(r"comoving distance $\chi$  [Mpc/$h$]")
    axL.set_xticks(xs)
    axL.set_xticklabels(labels)
    axL.set_ylabel(rf"measured / theory  ($\ell\in[{lo},{hi}]$, $(2\ell+1)$-weighted)")
    axL.set_title("Intermediate-ℓ agreement vs resolution (outer shells, χ ≥ 450 Mpc/h)")
    axL.grid(alpha=0.25)
    axL.legend(loc="lower left", fontsize=9)
    for ax_, ay, atext, acolor in annotations:
        axL.annotate(atext, (ax_, ay), fontsize=9, color=acolor, ha="center")

    # right: physical halo (ghost zone) vs the particle-displacement band
    s_lo, s_hi = _sigma_displacement(0.35), np.sqrt(3) * _sigma_displacement(0.0)  # 1D@far .. 3D@z0
    axR.axhspan(
        s_lo,
        s_hi,
        color="firebrick",
        alpha=0.13,
        lw=0,
        label=f"rms particle displacement\n({s_lo:.1f}–{s_hi:.1f} Mpc/h)",
    )
    for m in MESHES:
        ok = halo_map[m] > s_hi
        axR.plot(
            xs[MESHES.index(m)],
            halo_map[m],
            "o",
            ms=11,
            color=COLORS[m],
            markeredgecolor="green" if ok else "firebrick",
            markeredgewidth=2.2,
        )
        axR.annotate(
            f"{halo_map[m]:.1f}",
            (xs[MESHES.index(m)], halo_map[m]),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8.5,
        )
    axR.set_yscale("log")
    axR.set_xticks(xs)
    axR.set_xticklabels(labels)
    axR.set_ylabel(r"physical halo (ghost zone)  [Mpc/h]")
    axR.set_title("Halo vs the displacement scale")
    axR.grid(alpha=0.25, which="both")
    axR.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    savefig(ASSETS / stem, fig)


def fig03_convergence():
    _fig_convergence(
        raw_s,
        HALO,
        [f"m{m}\npx={PX[m]}" for m in MESHES],
        "Anti-convergence of the over-fine runs, and its cause: a starved distributed-PM ghost zone",
        "fig03-convergence",
        [(1.5, 1.02, "converged", "green"), (3.5, 0.80, "power lost\n(finer → worse)", "firebrick")],
    )


def fig06_convergence_pencil():
    # m2560/m3072 from their pencil decomposition (the larger ghost zone) instead of the starved slab
    raw_mixed = {m: (spectra_pencil[m] if m in PENCILS else raw_s[m]) for m in MESHES}
    halo_mixed = {m: (PENCIL_HALO[m] if m in PENCILS else HALO[m]) for m in MESHES}
    labels = [f"m{m}\n{'pencil' if m in PENCILS else f'px={PX[m]}'}" for m in MESHES]
    _fig_convergence(
        raw_mixed,
        halo_mixed,
        labels,
        "Pencil decomposition restores convergence: the larger ghost zone recovers the lost small-scale power",
        "fig06-convergence-pencil",
        [(1.5, 1.02, "converged", "green"), (3.5, 1.02, "recovered\n(pencil halo)", "green")],
    )


# =============================================================================
# fig04 / fig05 — slab vs pencil for one mesh: band-[270,330] measured/theory vs distance
# =============================================================================
def plot_slab_pencil(m, stem):
    lo, hi = BAND
    r_slab, _ = band_ratio(ell_full, raw_s[m], theory_pw_arr, lo, hi)
    r_pencil, _ = band_ratio(ell_full, spectra_pencil[m], theory_pw_arr, lo, hi)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.axhspan(0.95, 1.05, color="0.82", lw=0, label=r"±5%")
    ax.axhline(1.0, color="0.4", ls="--", lw=1.0)
    ax.plot(chi_shells, r_slab, "o-", color="firebrick", lw=1.6, ms=5, label=f"slab — starved halo {HALO[m]:.1f} Mpc/h")
    ax.plot(
        chi_shells, r_pencil, "o-", color="tab:blue", lw=1.6, ms=5, label=f"pencil — halo {PENCIL_HALO[m]:.1f} Mpc/h"
    )
    ax.set_xlabel(r"comoving distance $\chi$  [Mpc/$h$]")
    ax.set_ylabel(rf"measured / theory  ($\ell\in[{lo},{hi}]$, $(2\ell+1)$-weighted)")
    ax.set_ylim(0.4, 1.1)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=10)
    fig.tight_layout()
    savefig(ASSETS / stem, fig)


# =============================================================================
# fig07 — the delta maps (shell 9), all five slabs, shared scale (gnomonic / ortho / mollview rows)
# =============================================================================
def _logdelta(field, shell):
    a = np.asarray(field.array[shell], dtype=np.float64)
    return np.log10(np.clip(a / a.mean(), 0.05, None))  # log10(1 + delta)


def fig07_maps(shell=9):
    d1 = {m: _logdelta(density[m], shell) for m in MESHES}
    v = np.percentile(d1[2048], [1, 99])
    mv = dict(min=v[0], max=v[1], cmap="inferno", cbar=False, notext=True, bgcolor=(0.0,) * 4)
    fig = plt.figure(figsize=(20, 12.5))
    for i, m in enumerate(MESHES):
        hp.gnomview(
            d1[m],
            sub=(3, 5, i + 1),
            rot=(20, 35),
            reso=3.0,
            xsize=520,
            title=f"m{m}  (halo {HALO[m]:.1f} Mpc/h)",
            min=v[0],
            max=v[1],
            cmap="inferno",
            cbar=False,
            notext=True,
        )
        hp.orthview(d1[m], sub=(3, 5, i + 6), rot=(20, 35), half_sky=True, xsize=400, title="", **mv)
        hp.mollview(d1[m], sub=(3, 5, i + 11), title="", **mv)
    savefig(ASSETS / "fig07-maps-slabs", fig)


# =============================================================================
# fig08 — delta maps for m2560 / m3072, slab vs pencil (gnomonic + mollview rows)
# =============================================================================
def fig08_maps_slab_pencil(shell=9):
    panels = []
    for m in PENCILS:
        panels.append((f"m{m} slab  (halo {HALO[m]:.1f} Mpc/h)", _logdelta(density[m], shell)))
        panels.append((f"m{m} pencil  (halo {PENCIL_HALO[m]:.1f} Mpc/h)", _logdelta(density_pencil[m], shell)))
    v = np.percentile(panels[0][1], [1, 99])
    fig = plt.figure(figsize=(20, 9.5))
    for i, (ttl, d) in enumerate(panels):
        hp.gnomview(
            d,
            sub=(2, 4, i + 1),
            rot=(20, 35),
            reso=3.0,
            xsize=520,
            title=ttl,
            min=v[0],
            max=v[1],
            cmap="inferno",
            cbar=False,
            notext=True,
        )
        hp.mollview(
            d,
            sub=(2, 4, i + 5),
            title="",
            min=v[0],
            max=v[1],
            cmap="inferno",
            cbar=False,
            notext=True,
            bgcolor=(0.0,) * 4,
        )
    savefig(ASSETS / "fig08-maps-slab-pencil", fig)


def main():
    set_style()
    plot_spectra_batch(
        range(0, 5),
        "Per-shell angular $C_\\ell$ vs theory and the binned measured/theory ratio — shells 0–4 (nside=512, full sky)",
        "fig01-spectra-shells-0-4",
    )
    plot_spectra_batch(
        range(5, 10),
        "Per-shell angular $C_\\ell$ vs theory and the binned measured/theory ratio — shells 5–9 (nside=512, full sky)",
        "fig02-spectra-shells-5-9",
    )
    fig03_convergence()
    plot_slab_pencil(2560, "fig04-slab-vs-pencil-2560")
    plot_slab_pencil(3072, "fig05-slab-vs-pencil-3072")
    fig06_convergence_pencil()
    fig07_maps()
    fig08_maps_slab_pencil()
    print(f"assets written to {ASSETS}")


if __name__ == "__main__":
    main()
