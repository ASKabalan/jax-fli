"""Experiment 05d — step & stepping convergence at the production geometry: Born-kappa figures.

Renders the SVG figures for ``docs/5-experiments/05d-spacing-n-stepping-steps/README.md``. The sweep moves
``--nb-steps`` through 20 / 30 / 40 / 50 with BullFrog in both time variables — D-stepping (``bfbf``, the
production choice) and a-stepping (``bfkdk``) — at the production geometry (2560³, 5 Gpc/h box, 20
equal-volume shells, drift-on-lightcone, nside 2048), judged on the 3-bin Gauss–Legendre Born κ C_ℓ. Every
run shares the 05c anchor's seed, mesh and cosmology, so step counts are phase-matched; the figures judge
against the CosmoGrid ``cosmo_172798`` Born reference. The 50-step D-stepping point is not re-run — it is
05c's ``exp5c_drift_20`` anchor, byte-identical in the published ``spectra_exp5d_bfbf_50.parquet``.

  fig01-fig04  one figure per step count (20 / 30 / 40 / 50), all three tomographic bins overlaid: the
               D_ell power (solid = BullFrog D-stepping, dashed = BullFrog KDK a-stepping, dotted = the
               pixwin-matched Limber theory), the C_ell/CosmoGrid − 1 ratio with the acceptance band
               (expected cosmology + nside-512-pixwin offset ± √2 × the empirical CV of the 200 fiducial
               CosmoGrid permutations), and the median bandpower ratios in five ℓ bands.

Run from the repo root (CPU is fine):
    JAX_PLATFORMS=cpu uv run --no-sync python docs/5-experiments/05d-spacing-n-stepping-steps/build.py
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
)  # the dataset clone beside jax-fli; the 05d kappa_spectra are not pushed yet

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

SPECTRA_BF_20 = f"{DATA_ROOT}/05-spacing-n-stepping/05d-steps/kappa_spectra/spectra_exp5d_bfbf_20.parquet"
SPECTRA_BF_30 = f"{DATA_ROOT}/05-spacing-n-stepping/05d-steps/kappa_spectra/spectra_exp5d_bfbf_30.parquet"
SPECTRA_BF_40 = f"{DATA_ROOT}/05-spacing-n-stepping/05d-steps/kappa_spectra/spectra_exp5d_bfbf_40.parquet"
SPECTRA_BF_50 = f"{DATA_ROOT}/05-spacing-n-stepping/05d-steps/kappa_spectra/spectra_exp5d_bfbf_50.parquet"
SPECTRA_KDK_20 = f"{DATA_ROOT}/05-spacing-n-stepping/05d-steps/kappa_spectra/spectra_exp5d_bfkdk_20.parquet"
SPECTRA_KDK_30 = f"{DATA_ROOT}/05-spacing-n-stepping/05d-steps/kappa_spectra/spectra_exp5d_bfkdk_30.parquet"
SPECTRA_KDK_40 = f"{DATA_ROOT}/05-spacing-n-stepping/05d-steps/kappa_spectra/spectra_exp5d_bfkdk_40.parquet"
SPECTRA_KDK_50 = f"{DATA_ROOT}/05-spacing-n-stepping/05d-steps/kappa_spectra/spectra_exp5d_bfkdk_50.parquet"

bfbf_20_cat = Catalog.from_dataset(load_dataset("parquet", data_files=SPECTRA_BF_20, split="train"))
bfbf_30_cat = Catalog.from_dataset(load_dataset("parquet", data_files=SPECTRA_BF_30, split="train"))
bfbf_40_cat = Catalog.from_dataset(load_dataset("parquet", data_files=SPECTRA_BF_40, split="train"))
bfbf_50_cat = Catalog.from_dataset(load_dataset("parquet", data_files=SPECTRA_BF_50, split="train"))
bfkdk_20_cat = Catalog.from_dataset(load_dataset("parquet", data_files=SPECTRA_KDK_20, split="train"))
bfkdk_30_cat = Catalog.from_dataset(load_dataset("parquet", data_files=SPECTRA_KDK_30, split="train"))
bfkdk_40_cat = Catalog.from_dataset(load_dataset("parquet", data_files=SPECTRA_KDK_40, split="train"))
bfkdk_50_cat = Catalog.from_dataset(load_dataset("parquet", data_files=SPECTRA_KDK_50, split="train"))

cosmo = bfbf_20_cat.cosmology[0]
BF_CATS = {20: bfbf_20_cat, 30: bfbf_30_cat, 40: bfbf_40_cat, 50: bfbf_50_cat}
KDK_CATS = {20: bfkdk_20_cat, 30: bfkdk_30_cat, 40: bfkdk_40_cat, 50: bfkdk_50_cat}
for _cat in (*BF_CATS.values(), *KDK_CATS.values()):
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


def lensing_steps(steps, stem):
    """Born kappa C_ell of the step sweep against the CosmoGrid cosmo_172798 reference, all three tomographic
    bins overlaid (Stage-3 [:3], coloured by bin) — the thesis `lensing_vs_cosmogrid` layout. Top: the D_ell
    power — solid = BullFrog D-stepping (the production choice; at 50 steps this is the 05c drift anchor),
    dashed = KDK a-stepping, dash-dotted = the CosmoGrid reference itself. Middle: C_ell/CosmoGrid − 1 with
    the acceptance band = the expected cosmology + nside-512-pixwin offset (th_run/th_172798 − 1, thin dotted)
    ± √2 × the empirical CV of the 200 fiducial CosmoGrid permutations (worst bin) — the run and the reference
    are independent single realizations, so a second CosmoGrid realization would live inside the band. The ℓ
    axis is numbered here; the bottom row carries its own band-start axis. Bottom: the median bandpower ratio
    per ℓ band ([30,100/150/200/250/300)) as bars — one solid and one hatched bar per bin colour, one bar
    group per stepping — with the same acceptance band behind them. Spectra are bandpower-binned in linear
    bins of 32 multipoles; the multipole axis starts at the first bandpower."""
    bf_b = np.asarray(BF_CATS[steps].field[0].bin(nlb=32, lmin=2).array)
    kdk_b = np.asarray(KDK_CATS[steps].field[0].bin(nlb=32, lmin=2).array)
    ratio_bf = bf_b / cg17_b - 1.0
    ratio_kdk = kdk_b / cg17_b - 1.0
    z = np.asarray(BF_CATS[steps].field[0].z_sources)[:3]
    keep = bc <= ELL_MAX_PLOT

    series = (  # (line style, bar hatch, bar fill, ratio spectra); the production stepping first
        ("-", "", True, ratio_bf),
        ("--", "///", False, ratio_kdk),
    )

    fig, (ax_spec, ax_cg, ax_band) = plt.subplots(
        3,
        1,
        figsize=(4.98, 5.2),
        height_ratios=[1.5, 1.0, 1.05],
        layout="constrained",
    )
    fig.get_layout_engine().set(h_pad=0.02, w_pad=0.02, hspace=0.03, wspace=0.03)

    ax_spec.set_title(rf"${steps}$ steps", pad=4)
    for b, colour in enumerate(BIN_COLOURS):
        ax_spec.plot(bc[keep], (dl * cg17_b[b])[keep], "-.", color=colour, lw=1.2)
        for ls, _, _, _ in series:
            spec = bf_b if ls == "-" else kdk_b
            ax_spec.plot(bc[keep], (dl * spec[b])[keep], ls, color=colour, lw=1.2)
    ax_spec.set(xscale="log", yscale="log")
    ax_spec.set_ylabel(r"$\ell(\ell+1)\,C_\ell^{\kappa\kappa}/2\pi$")
    ax_spec.tick_params(labelbottom=False)
    ax_spec.set_xlim(20, ELL_MAX_PLOT)

    ax_cg.fill_between(bc[keep], (band_c - CV_FRAC)[keep], (band_c + CV_FRAC)[keep], color="0.88", lw=0, zorder=0)
    for b, colour in enumerate(BIN_COLOURS):
        ax_cg.plot(bc[keep], ratio_bf[b][keep], "-", color=colour, lw=1.2)
        ax_cg.plot(bc[keep], ratio_kdk[b][keep], "--", color=colour, lw=1.2)
    ax_cg.plot(bc[keep], band_c[keep], ":", color="0.3", lw=0.9)
    ax_cg.axhline(0.0, color="k", lw=0.8)
    ax_cg.set_xscale("log")
    ax_cg.set_xlim(20, ELL_MAX_PLOT)
    ax_cg.set_xlabel(r"$\ell$")
    ax_cg.set_ylabel("over CosmoGrid\n" + r"$-\,1$")
    # cap the y-range at the interesting scale: the bias line keeps rising (pixwin term) and would
    # otherwise compress the +-5-8% band that carries the verdict into invisibility
    m_le = (bc <= 400) & keep
    lo_r = min(ratio_bf[:, keep].min(), ratio_kdk[:, keep].min())
    hi_r = max(ratio_bf[:, m_le].max(), ratio_kdk[:, m_le].max(), (band_c + CV_FRAC)[m_le].max())
    pad_r = 0.06 * (hi_r - lo_r)
    ax_cg.set_ylim(lo_r - pad_r, hi_r + pad_r)

    band_starts = (30, 100, 150, 200, 250)
    band_edges = band_starts + (300,)
    sel_band = [(bc >= lo) & (bc < hi) for lo, hi in zip(band_edges[:-1], band_edges[1:])]
    centres = np.arange(len(band_starts))
    width = 0.8 / (len(series) * len(BIN_COLOURS))
    bar_vals = []
    for s, (ls, hatch, filled, ratios) in enumerate(series):
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

    bf_label = "BullFrog D-stepping"
    if steps == 50:
        bf_label += " (the Exp 05c anchor)"
    fig.legend(
        handles=[Patch(color=c, label=rf"bin {b + 1}, $z_s = {zb:.2f}$") for b, (c, zb) in enumerate(zip(BIN_COLOURS, z))]
        + [
            Line2D([], [], color="0.3", ls="-", label=bf_label),
            Line2D([], [], color="0.3", ls="--", label="KDK a-stepping"),
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
    lensing_steps(20, "fig01-lensing-steps-20")
    lensing_steps(30, "fig02-lensing-steps-30")
    lensing_steps(40, "fig03-lensing-steps-40")
    lensing_steps(50, "fig04-lensing-steps-50")
    print(f"assets written to {ASSETS}")


if __name__ == "__main__":
    main()
