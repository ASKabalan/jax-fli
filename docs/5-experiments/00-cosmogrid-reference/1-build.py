import jax

jax.config.update("jax_enable_x64", True)

import sys
from pathlib import Path

import equinox as eqx
import healpy as hp
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset
from huggingface_hub import snapshot_download

from jax_fli import compute_theory_cl
from jax_fli.io import Catalog, get_des_y3_nz_shear, get_stage3_nz_shear

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # -> docs/5-experiments
import matplotlib.pyplot as plt
from _exputils import savefig, set_style

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

REPO = "ASKabalan/jax-fli-experiments"
CG_KAPPA = "00-cosmogrid/kappa_spectra/spectra_cosmogrid_sample_kappa.parquet"
BORN_KAPPA_S3 = "00-cosmogrid/kappa_spectra/spectra_kappa_born_s3.parquet"
BORN_KAPPA_S3_512 = "00-cosmogrid/kappa_spectra/spectra_kappa_born_s3_512.parquet"
BORN_KAPPA_DES_Y3 = "00-cosmogrid/kappa_spectra/spectra_kappa_born_des.parquet"

NLB = 16  # multipoles per bandpower bin

root = snapshot_download(REPO, repo_type="dataset", local_files_only=True)

# -----------------------------------------------------------------------------
# Load the measured convergence (kappa) spectra. Each is a (4-bin, n_ell) auto-only
# angular C_ell measured from a HEALPix map at the dataset's own nside:
#   * cg_kappa   — CosmoGrid's OWN Stage-3 forecast kappa (native nside 512)
#   * born_s3    — jax-fli Born kappa, Stage-3 source n(z)   (nside 2048)
#   * born_s3_512 — same Born kappa downsampled to nside 512 (matches cg_kappa's nside)
#   * born_des   — jax-fli Born kappa, DES Y3 source n(z)    (nside 2048)
# -----------------------------------------------------------------------------
cg_kappa_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CG_KAPPA}", split="train"))
cg_kappa, cg_cosmo = cg_kappa_cat.field[0], cg_kappa_cat.cosmology[0]

born_kappa_s3_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{BORN_KAPPA_S3}", split="train"))
born_kappa_s3_512_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{BORN_KAPPA_S3_512}", split="train")
)
born_kappa_des_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{BORN_KAPPA_DES_Y3}", split="train"))

born_kappa_s3 = born_kappa_s3_cat.field[0]
born_kappa_s3_512 = born_kappa_s3_512_cat.field[0]
born_kappa_des = born_kappa_des_cat.field[0]
des_cosmo = born_kappa_des_cat.cosmology[0]

LMAX = int(cg_kappa.wavenumber.max())
ell = jnp.arange(LMAX + 1)

# Limber weak-lensing theory C_ell, one auto-spectrum per tomographic bin (cross=False).
nz_s3 = get_stage3_nz_shear()
nz_des = get_des_y3_nz_shear()
theory_s3 = compute_theory_cl(cg_cosmo, ell, nz_s3)  # (4, n_ell), Stage-3 n(z)
theory_des = compute_theory_cl(des_cosmo, ell, nz_des)  # (4, n_ell), DES Y3 n(z)


# =============================================================================
# Style + colors
# =============================================================================
set_style()

# Distinguish the two pixel windows on the theory line, and the measured series by line style.
THEORY_LS = {512: ":", 2048: "--"}
DATA_STYLES = ["-", "-."]  # one per measured series in a set (no markers)


def pixwin_match(theory, nside):
    """Multiply a theory PowerSpectrum by pixwin^2(nside) so it sits on the same footing as a
    measured map at that nside. The pixel window is ell-dependent within a bin, so this MUST be
    applied at full resolution, BEFORE bandpower binning."""
    pw2 = hp.pixwin(nside, lmax=LMAX) ** 2  # numpy (LMAX+1,)
    return eqx.tree_at(lambda p: p.array, theory, theory.array * pw2)


# =============================================================================
# Helper — one figure per comparison set: 4 tomographic-bin columns, spectra over ratio.
# =============================================================================
def plot_kappa_set(series, theory, title, stem):
    """series: list of dicts {field, nside, label, color}. Every measured series is compared to
    the SAME base theory, pixwin-matched to that series' own nside (so a 512 series and a 2048
    series compare to different effective theory at high ell)."""
    nsides = sorted({s["nside"] for s in series})
    theory_m = {ns: pixwin_match(theory, ns) for ns in nsides}  # full-res, per nside
    theory_b = {ns: theory_m[ns].bin(nlb=NLB, lmin=2) for ns in nsides}
    data_b = [s["field"].bin(nlb=NLB, lmin=2) for s in series]  # binned measured, same edges
    n_bins = theory.spectra.shape[0]
    z = np.asarray(series[0]["field"].z_sources)  # effective source redshift per bin

    fig, axes = plt.subplots(
        nrows=2,
        ncols=n_bins,
        figsize=(4.2 * n_bins, 5.0),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex="col",
    )
    fig.suptitle(title, fontsize=15)

    for i in range(n_bins):
        ax_spec = axes[0, i]
        ax_ratio = axes[1, i]

        # --- spectra (log-log): one matched-theory line per distinct nside + binned data ---
        for ns in nsides:
            ax_spec.plot(
                ell[2:],
                theory_m[ns].spectra[i][2:],
                color="0.25",
                ls=THEORY_LS[ns],
                lw=1.2,
                label=rf"Limber $\times\,w_\ell^2$({ns})",
            )
        for j, (s, db) in enumerate(zip(series, data_b)):
            ax_spec.plot(db.wavenumber, db.spectra[i], color=s["color"], ls=DATA_STYLES[j], lw=1.7, label=s["label"])
        ax_spec.set_xscale("log")
        ax_spec.set_yscale("log")
        ax_spec.set_ylabel(r"$C_\ell^{\kappa\kappa}$")
        ax_spec.grid(True, which="both", ls=":", alpha=0.4)
        ax_spec.set_title(rf"Bin {i + 1}  ($z\approx{z[i]:.2f}$)")
        if i == 0:
            ax_spec.legend(frameon=False, fontsize=8)

        # --- ratio (log-x): each measured series over its own pixwin-matched theory ---
        ax_ratio.axhspan(0.95, 1.05, color="0.7", alpha=0.3, label=r"$\pm 5\%$")
        ax_ratio.axhline(1.0, color="0.4", ls="--", lw=1.0)
        for j, (s, db) in enumerate(zip(series, data_b)):
            tb = theory_b[s["nside"]]
            ax_ratio.plot(db.wavenumber, db.spectra[i] / tb.spectra[i], color=s["color"], ls=DATA_STYLES[j], lw=1.7)
        ax_ratio.set_xscale("log")
        ax_ratio.set_ylim(0.8, 1.2)
        ax_ratio.set_xlabel(r"multipole $\ell$")
        ax_ratio.grid(True, which="both", ls=":", alpha=0.4)
        if i == 0:
            ax_ratio.set_ylabel("data / theory")
            ax_ratio.legend(loc="upper right", frameon=False, fontsize=8)

    fig.tight_layout()
    savefig(ASSETS / stem, fig)


# =============================================================================
# Set 1 — Stage-3 n(z): CosmoGrid native kappa (512) vs jax-fli Born kappa (2048)
# =============================================================================
plot_kappa_set(
    [
        {"field": cg_kappa, "nside": 512, "label": "CosmoGrid native κ (nside 512)", "color": "tab:orange"},
        {"field": born_kappa_s3, "nside": 2048, "label": "jax-fli Born κ (nside 2048)", "color": "tab:blue"},
    ],
    theory_s3,
    "Stage-3 source n(z) — CosmoGrid native κ vs jax-fli Born κ vs Limber theory",
    "fig05-kappa-s3",
)

# =============================================================================
# Set 2 — Stage-3 n(z) at matched nside 512: CosmoGrid native vs Born (downsampled)
# =============================================================================
plot_kappa_set(
    [
        {"field": cg_kappa, "nside": 512, "label": "CosmoGrid native κ (nside 512)", "color": "tab:orange"},
        {"field": born_kappa_s3_512, "nside": 512, "label": "jax-fli Born κ (nside 512)", "color": "tab:green"},
    ],
    theory_s3,
    "Stage-3 source n(z) at matched nside 512 — CosmoGrid native κ vs jax-fli Born κ vs Limber theory",
    "fig06-kappa-s3-512",
)

# =============================================================================
# Set 3 — DES Y3 n(z): jax-fli Born kappa (2048) vs Limber theory (no native CosmoGrid κ for DES)
# =============================================================================
plot_kappa_set(
    [
        {"field": born_kappa_des, "nside": 2048, "label": "jax-fli Born κ, DES Y3 (nside 2048)", "color": "tab:blue"},
    ],
    theory_des,
    "DES Y3 source n(z) — jax-fli Born κ vs Limber theory",
    "fig07-kappa-des",
)

print(f"assets written to {ASSETS}")
