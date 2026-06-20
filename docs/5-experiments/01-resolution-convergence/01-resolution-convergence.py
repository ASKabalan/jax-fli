"""Experiment 01 — resolution convergence: figures.

Renders the SVG figures for ``docs/5-experiments/01-resolution-convergence/README.md`` from the
per-shell spherical angular power spectra produced by ``fli-summary-stats`` (two sets: raw
``spectra_m*.parquet`` and pixel-window-deconvolved ``spectra_deconv_m*.parquet``) plus the
density maps (``m*.parquet``). Each shell's measured C_ell is compared to the analytic Limber
number-counts theory (``jax_fli.compute_theory_cl_for_density``), put on the same HEALPix
pixel-window footing.

Figures (→ ``assets/``):
  fig01-spectra        all 10 shells × 5 resolutions, raw C_ell vs theory (+cosmic-variance band).
  fig02-convergence    intermediate-ℓ (CV-clean) measured/theory vs mesh, beside the
                       halo-vs-displacement panel — the headline (anti-convergence + its cause).
  fig03-deconvolution  best resolutions (2048, 2560): raw vs pixwin-deconvolved → recovery.
  fig04-maps           the δ maps (shell 9, all resolutions) — no gross artifact, m512 smoother.

Run from the repo root (CPU is fine; ~minutes, loads the 5 maps for fig04):
    JAX_PLATFORMS=cpu uv run --no-sync python docs/5-experiments/01-resolution-convergence/01-resolution-convergence.py
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
from matplotlib import cm
from matplotlib.lines import Line2D

import jax_fli
from jax_fli.io import Catalog

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _exputils import savefig, set_style  # noqa: E402

ASSETS = HERE / "assets"
DATA_DIR = HERE.parents[1] / "000_RUNS" / "results" / "exp1"  # local fallback (docs/000_RUNS/results/exp1)
REPO = "ASKabalan/jax-fli-experiments"
BASE = "01-resolution-convergence"  # HF dataset config prefix / folder

NSIDE = 512
BOX = 2000.0  # Mpc/h
MESHES = [512, 1024, 2048, 2560, 3072]
PX = {512: 4, 1024: 8, 2048: 64, 2560: 128, 3072: 256}  # x-decomposition (GPUs) per rung
HALO = {m: 0.5 * BOX / PX[m] for m in MESHES}  # physical ghost-zone width = halo_multiplier·box/px
COLORS = {m: c for m, c in zip(MESHES, cm.viridis(np.linspace(0.0, 0.9, len(MESHES))))}
NBINS = 18
BAND = (30, 200)  # intermediate-ℓ window where the full-sky cosmic variance is ~0.7%


# --------------------------------------------------------------------------------------------
# Loading + theory
# --------------------------------------------------------------------------------------------
def _catalog(config, local_path):
    """Load a Catalog from the published HF dataset config; fall back to the local parquet."""
    try:
        from datasets import load_dataset

        return Catalog.from_dataset(load_dataset(REPO, config, split="train").with_format("numpy"))
    except Exception:
        return Catalog.from_parquet(str(local_path))


def load_ps(mesh, deconv=False):
    kind = "spectra-deconv" if deconv else "spectra"
    fname = f"spectra{'_deconv' if deconv else ''}_m{mesh}.parquet"
    cat = _catalog(f"{BASE}-{kind}-m{mesh}", DATA_DIR / fname)
    return cat.field[0], cat.cosmology[0]


def load_map(mesh):
    return _catalog(f"{BASE}-map-m{mesh}", DATA_DIR / f"m{mesh}.parquet").field[0]


def log_bin(ell, cl_2d, edges):
    """(2ℓ+1)-weighted bandpowers. cl_2d: (n_shells, n_ell). Returns leff, cb, nmodes."""
    w = 2.0 * ell + 1.0
    leff, cb, nmodes = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ell >= lo) & (ell < hi)
        if not m.any():
            continue
        ww = w[m]
        leff.append(float((ww * ell[m]).sum() / ww.sum()))
        cb.append((cl_2d[:, m] * ww).sum(axis=1) / ww.sum())
        nmodes.append(float(ww.sum()))
    return np.array(leff), np.stack(cb, axis=1), np.array(nmodes)


def band_ratio(ell, meas_2d, theo_2d, lo, hi):
    """(2ℓ+1)-weighted measured/theory over [lo,hi] per shell, + the f_sky=1 CV sigma."""
    m = (ell >= lo) & (ell <= hi)
    w = 2.0 * ell[m] + 1.0
    mb = (meas_2d[:, m] * w).sum(axis=1) / w.sum()
    tb = (theo_2d[:, m] * w).sum(axis=1) / w.sum()
    return mb / tb, float(np.sqrt(2.0 / w.sum()))


print("Loading spectra + computing theory …")
raw = {m: np.asarray(load_ps(m)[0].array) for m in MESHES}
dec = {m: np.asarray(load_ps(m, deconv=True)[0].array) for m in MESHES}
ps_ref, cosmo = load_ps(MESHES[0])
ell_full = np.asarray(ps_ref.wavenumber)
z_shells = np.asarray(ps_ref.z_sources)
chi_shells = np.asarray(ps_ref.comoving_centers)
n_shells = raw[MESHES[0]].shape[0]

sel = ell_full >= 2
ell = ell_full[sel]
theory_cont = np.asarray(
    jax_fli.compute_theory_cl_for_density(
        cosmo, ps_ref, ells=jnp.asarray(ell), nonlinear_fn="halofit", nz_zmax=0.5
    ).array
)
pixwin2 = hp.pixwin(NSIDE)[ell.astype(int)] ** 2
theory_pw = theory_cont * pixwin2[None, :]  # pixel-window-matched (for the RAW measured)
raw_s = {m: raw[m][:, sel] for m in MESHES}
dec_s = {m: dec[m][:, sel] for m in MESHES}
edges = np.unique(np.geomspace(2, ell.max(), NBINS + 1).astype(int))


# --------------------------------------------------------------------------------------------
# fig01 — spectra: all shells × all resolutions, raw vs theory (+CV band)
# --------------------------------------------------------------------------------------------
def fig01_spectra():
    leff, theory_b, nmodes_b = log_bin(ell, theory_pw, edges)
    meas_b = {m: log_bin(ell, raw_s[m], edges)[1] for m in MESHES}
    sigma_b = np.sqrt(2.0 / nmodes_b)
    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(2, 5, hspace=0.3, wspace=0.24)
    for i in range(n_shells):
        cell = gs[i // 5, i % 5].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.0)
        ax_cl = fig.add_subplot(cell[0])
        ax_r = fig.add_subplot(cell[1], sharex=ax_cl)
        # top: C_ell vs theory (+ CV band)
        ax_cl.fill_between(leff, theory_b[i] * (1 - sigma_b), theory_b[i] * (1 + sigma_b), color="0.78", lw=0, zorder=0)
        ax_cl.loglog(leff, theory_b[i], "k-", lw=1.7, zorder=5)
        for m in MESHES:
            ax_cl.loglog(leff, meas_b[m][i], "o-", color=COLORS[m], ms=2.5, lw=1.0, zorder=4)
        ax_cl.axvline(2 * NSIDE, color="0.6", ls=":", lw=0.9)
        ax_cl.set_title(f"shell {i}:  z = {z_shells[i]:.3f}", fontsize=10)
        ax_cl.set_xlim(leff.min(), leff.max())
        ax_cl.grid(alpha=0.2, which="both")
        ax_cl.tick_params(labelbottom=False)
        if i % 5 == 0:
            ax_cl.set_ylabel(r"$C_\ell$")
        # bottom (3:1 height): measured / theory per resolution
        ax_r.fill_between(leff, 1 - sigma_b, 1 + sigma_b, color="0.78", lw=0, zorder=0)
        ax_r.axhline(1.0, color="0.4", ls="--", lw=0.9)
        for m in MESHES:
            ax_r.semilogx(leff, meas_b[m][i] / theory_b[i], "-", color=COLORS[m], lw=1.0)
        ax_r.axvline(2 * NSIDE, color="0.6", ls=":", lw=0.9)
        ax_r.set_ylim(0.45, 1.45)
        ax_r.set_xlim(leff.min(), leff.max())
        ax_r.grid(alpha=0.2, which="both")
        if i % 5 == 0:
            ax_r.set_ylabel("meas/thy", fontsize=8)
        if i >= 5:
            ax_r.set_xlabel(r"$\ell$")
    handles = [Line2D([], [], color=COLORS[m], marker="o", ms=4, lw=1.4, label=f"m{m} (px={PX[m]})") for m in MESHES]
    handles += [
        Line2D([], [], color="k", lw=1.8, label="Limber number-counts theory × pixwin$^2$"),
        Line2D([], [], color="0.78", lw=6, label=r"$\pm1\sigma$ cosmic variance (full sky)"),
        Line2D([], [], color="0.6", ls=":", lw=0.9, label=r"$\ell=2\,n_{\rm side}$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(
        "Per-shell angular $C_\\ell$ vs theory (top) and the measured/theory ratio (bottom) — raw, nside=512, per_plane $\\delta$, full sky",
        y=1.02,
    )
    savefig(ASSETS / "fig01-spectra", fig)


# --------------------------------------------------------------------------------------------
# fig02 — convergence diagnostic + the halo-vs-displacement mechanism
# --------------------------------------------------------------------------------------------
def _sigma_displacement(z):
    a = 1.0 / (1.0 + z)
    k = jnp.logspace(-4, 1.3, 4000)
    pk = jc.power.linear_matter_power(cosmo, k, a=a)
    return float(jnp.sqrt(jnp.trapezoid(pk, k) / (6 * np.pi**2)))  # rms 1D Zel'dovich displacement


def fig02_convergence():
    ratios = {m: band_ratio(ell, raw_s[m], theory_pw, *BAND)[0] for m in MESHES}
    cv = band_ratio(ell, raw_s[MESHES[0]], theory_pw, *BAND)[1]
    xs = np.arange(len(MESHES))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.4))

    # left: measured/theory over the CV-clean band, per shell, vs resolution
    axL.axhspan(1 - cv, 1 + cv, color="0.82", lw=0, label=f"±1σ CV ≈ {cv:.1%}")
    axL.axhline(1.0, color="0.4", ls="--", lw=1.0)
    for i in range(n_shells):  # outer shells (the trustworthy, well-sampled ones) emphasized
        outer = chi_shells[i] >= 400
        axL.plot(
            xs,
            [ratios[m][i] for m in MESHES],
            "-",
            color=cm.plasma(i / n_shells),
            lw=1.8 if outer else 0.8,
            alpha=0.95 if outer else 0.4,
            marker="o" if outer else None,
            ms=4,
        )
    axL.set_xticks(xs)
    axL.set_xticklabels([f"m{m}\npx={PX[m]}" for m in MESHES])
    axL.set_ylabel(r"measured / theory  ($\ell\in[30,200]$, $(2\ell+1)$-weighted)")
    axL.set_title("Intermediate-ℓ agreement vs resolution (bold = outer shells, χ≥400)")
    axL.grid(alpha=0.25)
    axL.legend(loc="lower left", fontsize=9)
    axL.annotate("converged", (1.5, 1.02), fontsize=9, color="green", ha="center")
    axL.annotate("power lost\n(finer → worse)", (3.5, 0.80), fontsize=9, color="firebrick", ha="center")

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
        ok = HALO[m] > s_hi
        axR.plot(
            xs[MESHES.index(m)],
            HALO[m],
            "o",
            ms=11,
            color=COLORS[m],
            markeredgecolor="green" if ok else "firebrick",
            markeredgewidth=2.2,
        )
        axR.annotate(
            f"{HALO[m]:.1f}",
            (xs[MESHES.index(m)], HALO[m]),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8.5,
        )
    axR.set_yscale("log")
    axR.set_xticks(xs)
    axR.set_xticklabels([f"m{m}\npx={PX[m]}" for m in MESHES])
    axR.set_ylabel(r"physical halo (ghost zone) = $h_m\,\mathrm{box}/p_x$  [Mpc/h]")
    axR.set_title("Halo shrinks with decomposition, crossing the displacement scale")
    axR.grid(alpha=0.25, which="both")
    axR.legend(loc="upper right", fontsize=9)
    fig.suptitle("Anti-convergence of the over-fine runs, and its cause: a starved distributed-PM ghost zone", y=1.0)
    fig.tight_layout()
    savefig(ASSETS / "fig02-convergence", fig)


# --------------------------------------------------------------------------------------------
# fig03 — pixel-window deconvolution recovers the best-resolution spectra
# --------------------------------------------------------------------------------------------
def fig03_deconvolution(shell=9):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, m in zip(axes, (2048, 2560)):
        # ratio to the CONTINUOUS theory (deconvolution removes the pixwin the raw map carries)
        lb, raw_r, _ = log_bin(ell, raw_s[m] / theory_cont, edges)
        _, dec_r, _ = log_bin(ell, dec_s[m] / theory_cont, edges)
        ax.semilogx(lb, raw_r[shell], "o-", color="0.45", ms=4, lw=1.4, label="raw (carries pixwin²)")
        ax.semilogx(lb, dec_r[shell], "s-", color=COLORS[m], ms=4, lw=1.6, label="pixel-window-deconvolved")
        ax.axhspan(0.98, 1.02, color="forestgreen", alpha=0.13, lw=0, label="±2% target")
        ax.axhline(1.0, color="0.4", ls="--", lw=1.0)
        ax.axvline(2 * NSIDE, color="0.6", ls=":", lw=0.9)
        ax.set_title(f"m{m}  —  shell {shell} (z={z_shells[shell]:.2f})")
        ax.set_xlabel(r"$\ell$")
        ax.set_xlim(8, ell.max())  # drop the few-mode lowest bins (huge CV)
        ax.set_ylim(0.45, 1.4)
        ax.grid(alpha=0.25, which="both")
        ax.legend(loc="lower left", fontsize=9)
    axes[0].set_ylabel(r"measured / continuous theory")
    fig.suptitle("Pixel-window deconvolution recovers the high-ℓ power of the converged resolutions", y=1.0)
    fig.tight_layout()
    savefig(ASSETS / "fig03-deconvolution", fig)


# --------------------------------------------------------------------------------------------
# fig04 — the δ maps (shell 9), shared scale: no gross artifact, m512 smoother
# --------------------------------------------------------------------------------------------
def fig04_maps(shell=9):
    d1 = {}
    for m in MESHES:
        a = np.asarray(load_map(m).array[shell], dtype=np.float64)
        d1[m] = np.log10(np.clip(a / a.mean(), 0.05, None))  # log10(1 + delta)
    v = np.percentile(d1[2048], [1, 99])
    fig = plt.figure(figsize=(20, 8.8))
    for i, m in enumerate(MESHES):
        hp.gnomview(
            d1[m],
            sub=(2, 5, i + 1),
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
        hp.orthview(
            d1[m],
            sub=(2, 5, i + 6),
            rot=(20, 35),
            half_sky=True,
            xsize=400,  # keep the embedded raster small enough to rasterize/commit
            title="",
            min=v[0],
            max=v[1],
            cmap="inferno",
            cbar=False,
            notext=True,
            bgcolor=(0.0,) * 4,
        )
    fig.suptitle(
        f"δ maps, shell {shell} (z={z_shells[shell]:.2f}), $\\log_{{10}}(1+\\delta)$ — "
        r"top row flat gnomonic patch (~26°), bottom row orthographic globe (one hemisphere); no boundary artifact, m512 smoother",
        y=1.0,
    )
    savefig(ASSETS / "fig04-maps", fig)


def main():
    set_style()
    fig01_spectra()
    fig02_convergence()
    fig03_deconvolution()
    fig04_maps()
    print(f"assets written to {ASSETS}")


if __name__ == "__main__":
    main()
