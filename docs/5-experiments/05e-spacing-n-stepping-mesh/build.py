"""Experiment 05e — mesh ladder at the production step budget: Born-kappa figures.

Renders the SVG figures for ``docs/5-experiments/05e-spacing-n-stepping-mesh/README.md``. The ladder holds
the 05c drift-anchor physics fixed (BullFrog, D-stepping, 50 steps, equal-volume 20 shells, drift-on-lightcone,
nside 2048) and moves the PM mesh 512³ → 4096³. The IC white noise is drawn per mesh cell, so every mesh is a
different universe: each figure is judged against the CosmoGrid ``cosmo_172798`` Born reference, never
mesh-to-mesh. The 2560³ point is not re-run — it is 05c's ``exp5c_drift_20`` anchor, byte-identical in the
published ``spectra_exp5e_m2560.parquet``.

  fig01-fig06  one figure per mesh (512 / 1024 / 2048 / 2560 / 3072 / 4096), all three tomographic bins
               overlaid: the D_ell power (solid = the mesh, dashed = the 2560³ anchor, dotted = the
               pixwin-matched Limber theory), the C_ell/CosmoGrid − 1 ratio with the acceptance band
               (expected cosmology + nside-512-pixwin offset ± √2 × the empirical CV of the 200 fiducial
               CosmoGrid permutations), and the median bandpower ratios in five ℓ bands.

Run from the repo root (CPU is fine):
    JAX_PLATFORMS=cpu uv run --no-sync python docs/5-experiments/05e-spacing-n-stepping-mesh/build.py
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
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from huggingface_hub import snapshot_download
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from jax_fli import compute_theory_cl
from jax_fli.io import Catalog, get_stage3_nz_shear

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _exputils import savefig, set_style  # noqa: E402

ASSETS = HERE / "assets"
REPO = "ASKabalan/jax-fli-experiments"
DATA_ROOT = (
    HERE.parents[3] / "jax-fli-experiments"
)  # the dataset clone beside jax-fli; the 05e kappa_spectra are not pushed yet

LMAX = 1500  # the published spectra stop at ell 1500

# Cosmic-variance band (experiment 00, published under 00-cosmogrid/fiducial_kappa_spectra):
# the 200 fiducial perm spectra binned like every figure; the empirical fractional CV is the
# std across perms of the bandpowers, worst of the three plotted bins, times sqrt(2) because
# the ratio compares two independent single realizations (our run vs one CosmoGrid perm).
FIDUCIAL_KAPPA = "00-cosmogrid/fiducial_kappa_spectra"

# The CosmoGrid reference the ratio panels divide by: the Stage-3 forecast kappa at grid point
# 172798 (the grid cosmology closest to the run cosmology), binned like every figure, plus
# its Limber theory under the nside-512 pixel window the CosmoGrid maps carry.
CG_172798 = "00-cosmogrid/cosmo_172798/kappa_spectra/spectra_cosmogrid_sample_kappa.parquet"

root = snapshot_download(REPO, repo_type="dataset", local_files_only=True)

_fid_cats = [
    Catalog.from_dataset(
        load_dataset("parquet", data_files=f"{root}/{FIDUCIAL_KAPPA}/cosmo_fiducial_part{p}.parquet", split="train")
    )
    for p in range(4)
]
_fid_b = np.stack([np.asarray(f.bin(nlb=32, lmin=2).array) for cat in _fid_cats for f in cat.field])
CV_FRAC = np.sqrt(2.0) * (_fid_b.std(axis=0, ddof=1) / _fid_b.mean(axis=0))[:3].max(axis=0)

_cg17_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CG_172798}", split="train"))
cg17_b = np.asarray(_cg17_cat.field[0].bin(nlb=32, lmin=2).array)[:3]
cosmo_172798 = _cg17_cat.cosmology[0]
th_172798_b = np.asarray(
    (compute_theory_cl(cosmo_172798, jnp.arange(LMAX + 1), get_stage3_nz_shear()[:3]) * hp.pixwin(512, lmax=LMAX) ** 2)
    .bin(nlb=32, lmin=2)
    .array
)[:3]

SPECTRA_M512 = f"{DATA_ROOT}/05-spacing-n-stepping/05e-mesh/kappa_spectra/spectra_exp5e_m512.parquet"
SPECTRA_M1024 = f"{DATA_ROOT}/05-spacing-n-stepping/05e-mesh/kappa_spectra/spectra_exp5e_m1024.parquet"
SPECTRA_M2048 = f"{DATA_ROOT}/05-spacing-n-stepping/05e-mesh/kappa_spectra/spectra_exp5e_m2048.parquet"
SPECTRA_M2560 = f"{DATA_ROOT}/05-spacing-n-stepping/05e-mesh/kappa_spectra/spectra_exp5e_m2560.parquet"
SPECTRA_M3072 = f"{DATA_ROOT}/05-spacing-n-stepping/05e-mesh/kappa_spectra/spectra_exp5e_m3072.parquet"
SPECTRA_M4096 = f"{DATA_ROOT}/05-spacing-n-stepping/05e-mesh/kappa_spectra/spectra_exp5e_m4096.parquet"

m512_cat = Catalog.from_dataset(load_dataset("parquet", data_files=SPECTRA_M512, split="train"))
m1024_cat = Catalog.from_dataset(load_dataset("parquet", data_files=SPECTRA_M1024, split="train"))
m2048_cat = Catalog.from_dataset(load_dataset("parquet", data_files=SPECTRA_M2048, split="train"))
m2560_cat = Catalog.from_dataset(load_dataset("parquet", data_files=SPECTRA_M2560, split="train"))
m3072_cat = Catalog.from_dataset(load_dataset("parquet", data_files=SPECTRA_M3072, split="train"))
m4096_cat = Catalog.from_dataset(load_dataset("parquet", data_files=SPECTRA_M4096, split="train"))

cosmo = m512_cat.cosmology[0]
MESH_CATS = {512: m512_cat, 1024: m1024_cat, 2048: m2048_cat, 2560: m2560_cat, 3072: m3072_cat, 4096: m4096_cat}
for _cat in MESH_CATS.values():
    for _k in ("Omega_c", "Omega_b", "h", "n_s", "sigma8", "w0", "wa"):
        assert np.isclose(getattr(_cat.cosmology[0], _k), getattr(cosmo, _k))

# The Limber theory under the nside-2048 pixel window the run maps carry, binned like every figure,
# and its mean-over-bins offset against the 172798 theory (the expected cosmology + pixwin bias line).
pw2 = hp.pixwin(2048, lmax=LMAX) ** 2
theory_b = (compute_theory_cl(cosmo, jnp.arange(LMAX + 1), get_stage3_nz_shear()[:3]) * pw2).bin(nlb=32, lmin=2)
theory = np.asarray(theory_b.array)
bc = np.asarray(theory_b.wavenumber)
dl = bc * (bc + 1) / (2 * np.pi)
band_c = (theory / th_172798_b - 1.0).mean(axis=0)


ELL_MAX_PLOT = 1300  # above ~1300 both C_ell hit the PM-resolution floor
BIN_COLOURS = ("#4477aa", "#ee7733", "#117733")  # the thesis tomographic-bin colours


def lensing_mesh(mesh, stem):
    """Born kappa C_ell of one mesh run against the CosmoGrid cosmo_172798 reference, all three tomographic
    bins overlaid (Stage-3 [:3], coloured by bin) — the thesis `lensing_vs_cosmogrid` layout. Top: the D_ell
    power — solid = the mesh, dashed = the 2560³ anchor (05c drift anchor; absent on the 2560³ figure itself),
    dash-dotted = the CosmoGrid reference itself. Middle: C_ell/CosmoGrid − 1 with the acceptance band = the
    expected cosmology + nside-512-pixwin offset (th_run/th_172798 − 1, thin dotted) ± √2 × the empirical CV of
    the 200 fiducial CosmoGrid permutations (worst bin) — the run and the reference are independent single
    realizations, so a second CosmoGrid realization would live inside the band. The ℓ axis is numbered here;
    the bottom row carries its own band-start axis. Bottom: the median bandpower ratio per ℓ band
    ([30,100/150/200/250/300)) as bars — one solid and one hatched bar per bin colour, one bar group per
    variant — with the same acceptance band behind them. Spectra are bandpower-binned in linear bins of 32
    multipoles; the multipole axis starts at the first bandpower."""
    anchor = mesh != 2560
    run_b = np.asarray(MESH_CATS[mesh].field[0].bin(nlb=32, lmin=2).array)
    anchor_b = np.asarray(MESH_CATS[2560].field[0].bin(nlb=32, lmin=2).array)
    ratio_run = run_b / cg17_b - 1.0
    ratio_anchor = anchor_b / cg17_b - 1.0
    z = np.asarray(MESH_CATS[mesh].field[0].z_sources)[:3]
    keep = bc <= ELL_MAX_PLOT

    series = [("-", "", True, ratio_run)]  # (line style, bar hatch, bar fill, ratio spectra)
    if anchor:
        series.append(("--", "///", False, ratio_anchor))

    fig, (ax_spec, ax_cg, ax_band) = plt.subplots(
        3,
        1,
        figsize=(4.98, 5.2),
        height_ratios=[1.5, 1.0, 1.05],
        layout="constrained",
    )
    fig.get_layout_engine().set(h_pad=0.02, w_pad=0.02, hspace=0.03, wspace=0.03)

    ax_spec.set_title(rf"${mesh}^3$ mesh", pad=4)
    for b, colour in enumerate(BIN_COLOURS):
        ax_spec.plot(bc[keep], (dl * cg17_b[b])[keep], "-.", color=colour, lw=1.2)
        ax_spec.plot(bc[keep], (dl * run_b[b])[keep], "-", color=colour, lw=1.2)
        if anchor:
            ax_spec.plot(bc[keep], (dl * anchor_b[b])[keep], "--", color=colour, lw=1.2)
    ax_spec.set(xscale="log", yscale="log")
    ax_spec.set_ylabel(r"$\ell(\ell+1)\,C_\ell^{\kappa\kappa}/2\pi$")
    ax_spec.tick_params(labelbottom=False)
    ax_spec.set_xlim(20, ELL_MAX_PLOT)

    ax_cg.fill_between(bc[keep], (band_c - CV_FRAC)[keep], (band_c + CV_FRAC)[keep], color="0.88", lw=0, zorder=0)
    for b, colour in enumerate(BIN_COLOURS):
        ax_cg.plot(bc[keep], ratio_run[b][keep], "-", color=colour, lw=1.2)
        if anchor:
            ax_cg.plot(bc[keep], ratio_anchor[b][keep], "--", color=colour, lw=1.2)
    ax_cg.plot(bc[keep], band_c[keep], ":", color="0.3", lw=0.9)
    ax_cg.axhline(0.0, color="k", lw=0.8)
    ax_cg.set_xscale("log")
    ax_cg.set_xlim(20, ELL_MAX_PLOT)
    ax_cg.set_xlabel(r"$\ell$")
    ax_cg.set_ylabel("over CosmoGrid\n" + r"$-\,1$")
    # cap the y-range at the interesting scale: the bias line keeps rising (pixwin term) and would
    # otherwise compress the +-5-8% band that carries the verdict into invisibility
    m_le = (bc <= 400) & keep
    lo_r = ratio_run[:, keep].min() if anchor else ratio_anchor[:, keep].min()
    hi_r = max(ratio_run[:, m_le].max(), (band_c + CV_FRAC)[m_le].max())
    if anchor:
        hi_r = max(hi_r, ratio_anchor[:, m_le].max())
    pad_r = 0.06 * (hi_r - lo_r)
    ax_cg.set_ylim(lo_r - pad_r, hi_r + pad_r)

    band_starts = (30, 100, 150, 200, 250)
    band_edges = band_starts + (300,)
    sel_band = [(bc >= lo) & (bc < hi) for lo, hi in zip(band_edges[:-1], band_edges[1:])]
    centres = np.arange(len(band_starts))
    width = 0.8 / (len(series) * len(BIN_COLOURS))
    bar_vals = []
    for s, (_, hatch, filled, ratios) in enumerate(series):
        for b, colour in enumerate(BIN_COLOURS):
            med = np.array([np.median(ratios[b][sel]) for sel in sel_band])
            bar_vals.append(1.0 + med)
            ax_band.bar(
                centres + (len(series) * b + s - (len(series) * 3 - 1) / 2) * width,
                1.0 + med,
                width,
                color=colour if filled else "white",
                edgecolor=colour,
                hatch=hatch,
                lw=0.8,
                zorder=2,
            )
    # the acceptance band sits behind the bars: per-band worst-bin +-sqrt(2) x empirical CV
    acc = np.array([CV_FRAC[sel].max() for sel in sel_band])
    acc_lo, acc_hi = 1.0 - acc, 1.0 + acc
    ax_band.fill_between(
        np.concatenate([[centres[0] - 0.5], centres, [centres[-1] + 0.5]]),
        np.concatenate([acc_lo[:1], acc_lo, acc_lo[-1:]]),
        np.concatenate([acc_hi[:1], acc_hi, acc_hi[-1:]]),
        step="mid",
        color="0.88",
        lw=0,
        zorder=0,
    )
    ax_band.axhline(1.0, color="k", lw=0.8, zorder=1)
    ax_band.set_xticks(centres)
    ax_band.set_xticklabels([rf"${lo}$" for lo in band_starts])
    ax_band.set(xlabel=r"band start $\ell$", ylabel="median over\nCosmoGrid")
    ax_band.set_xlim(-0.55, len(band_starts) - 0.45)
    bars = np.concatenate(bar_vals)
    ax_band.set_ylim(min(bars.min(), acc_lo.min()) - 0.03, max(bars.max(), acc_hi.max()) + 0.03)

    fig.legend(
        handles=[
            Patch(color=c, label=rf"bin {b + 1}, $z_s = {zb:.2f}$") for b, (c, zb) in enumerate(zip(BIN_COLOURS, z))
        ]
        + [
            Line2D([], [], color="0.3", ls="-", label=rf"{mesh}$^3$ PM mesh"),
            *(
                []
                if not anchor
                else [Line2D([], [], color="0.3", ls="--", label=r"2560$^3$ anchor (the Exp 05c drift anchor)")]
            ),
            Line2D([], [], color="0.3", ls="-.", label=r"CosmoGrid $N$-body"),
            Patch(color="0.88", label=r"$\sqrt{2}\times$ empirical cosmic variance (200 CosmoGrid permutations)"),
            Line2D([], [], color="0.3", ls=":", lw=0.9, label="expected cosmology + pixel-window offset"),
        ],
        frameon=False,
        loc="outside upper center",
        ncol=4,
        handlelength=1.8,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    savefig(ASSETS / stem, fig)


def main():
    set_style()
    lensing_mesh(512, "fig01-lensing-mesh-512")
    lensing_mesh(1024, "fig02-lensing-mesh-1024")
    lensing_mesh(2048, "fig03-lensing-mesh-2048")
    lensing_mesh(2560, "fig04-lensing-mesh-2560")
    lensing_mesh(3072, "fig05-lensing-mesh-3072")
    lensing_mesh(4096, "fig06-lensing-mesh-4096")
    print(f"assets written to {ASSETS}")


if __name__ == "__main__":
    main()
