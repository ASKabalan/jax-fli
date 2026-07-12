import jax

jax.config.update("jax_enable_x64", True)

import sys
from pathlib import Path

import healpy as hp
import jax.numpy as jnp
import jax_cosmo as jc
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

NLB = 32  # multipoles per bandpower bin

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
born_kappa_des_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{BORN_KAPPA_DES_Y3}", split="train")
)

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

# Convert to CL * ell * (ell + 1) / (2 * pi) for plottiing
cg_kappa = cg_kappa * cg_kappa.wavenumber * (cg_kappa.wavenumber + 1) / (2 * jnp.pi)
born_kappa_s3 = born_kappa_s3 * born_kappa_s3.wavenumber * (born_kappa_s3.wavenumber + 1) / (2 * jnp.pi)
born_kappa_s3_512 = born_kappa_s3_512 * born_kappa_s3_512.wavenumber * (born_kappa_s3_512.wavenumber + 1) / (2 * jnp.pi)
born_kappa_des = born_kappa_des * born_kappa_des.wavenumber * (born_kappa_des.wavenumber + 1) / (2 * jnp.pi)
theory_s3 = theory_s3 * born_kappa_des.wavenumber * (born_kappa_des.wavenumber + 1) / (2 * jnp.pi)
theory_des = theory_des * born_kappa_des.wavenumber * (born_kappa_des.wavenumber + 1) / (2 * jnp.pi)

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
    return theory * pw2


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
        ax_spec.set_ylabel(r"$\ell(\ell+1)\,C_\ell^{\kappa\kappa}/2\pi$")
        ax_spec.grid(True, which="both", ls=":", alpha=0.4)
        ax_spec.set_title(rf"Bin {i + 1}  ($z\approx{z[i]:.2f}$)")
        if i == 0:
            ax_spec.legend(frameon=False, fontsize=8)

        # --- ratio (log-x): each measured series over its own pixwin-matched theory ---
        ax_ratio.axhspan(-0.05, 0.05, color="0.7", alpha=0.3, label=r"$\pm 5\%$")
        ax_ratio.axhline(0.0, color="0.4", ls="--", lw=1.0)
        for j, (s, db) in enumerate(zip(series, data_b)):
            tb = theory_b[s["nside"]]
            ax_ratio.plot(db.wavenumber, db.spectra[i] / tb.spectra[i] - 1, color=s["color"], ls=DATA_STYLES[j], lw=1.7)
        ax_ratio.set_xscale("log")
        ax_ratio.set_ylim(-0.2, 0.2)
        ax_ratio.set_xlabel(r"multipole $\ell$")
        ax_ratio.grid(True, which="both", ls=":", alpha=0.4)
        if i == 0:
            ax_ratio.set_ylabel("data / theory - 1")
            ax_ratio.legend(loc="upper right", frameon=False, fontsize=8)

    fig.tight_layout()
    savefig(ASSETS / stem, fig)


# =============================================================================
# Set 1 — Stage-3 n(z): CosmoGrid native kappa (512) vs jax-fli Born kappa (2048)
# =============================================================================
plot_kappa_set(
    [
        {"field": cg_kappa, "nside": 512, "label": r"CosmoGrid native $\kappa$ (nside 512)", "color": "tab:orange"},
        {"field": born_kappa_s3, "nside": 2048, "label": r"jax-fli Born $\kappa$ (nside 2048)", "color": "tab:blue"},
    ],
    theory_s3,
    r"Stage-3 source n(z) — CosmoGrid native $\kappa$ vs jax-fli Born $\kappa$ vs Limber theory",
    "fig05-kappa-s3",
)

# =============================================================================
# Set 2 — Stage-3 n(z) at matched nside 512: CosmoGrid native vs Born (downsampled)
# =============================================================================
plot_kappa_set(
    [
        {"field": cg_kappa, "nside": 512, "label": r"CosmoGrid native $\kappa$ (nside 512)", "color": "tab:orange"},
        {"field": born_kappa_s3_512, "nside": 512, "label": r"jax-fli Born $\kappa$ (nside 512)", "color": "tab:green"},
    ],
    theory_s3,
    r"Stage-3 source n(z) at matched nside 512 — CosmoGrid native $\kappa$ vs jax-fli Born $\kappa$ vs Limber theory",
    "fig06-kappa-s3-512",
)

# =============================================================================
# Set 3 — DES Y3 n(z): jax-fli Born kappa (2048) vs Limber theory (no native CosmoGrid κ for DES)
# =============================================================================
plot_kappa_set(
    [
        {
            "field": born_kappa_des,
            "nside": 2048,
            "label": r"jax-fli Born $\kappa$, DES Y3 (nside 2048)",
            "color": "tab:blue",
        },
    ],
    theory_des,
    r"DES Y3 source n(z) — jax-fli Born $\kappa$ vs Limber theory",
    "fig07-kappa-des",
)

# =============================================================================
# fig08 — the two source n(z) distributions and their lensing efficiency q(z)
# (built from jax_fli.data.get_{stage3,des_y3}_nz_shear, the same WeakLensing kernel as data.plot_nz)
# =============================================================================
z_grid = jnp.linspace(0.0, 2.0, 300)
_cm = plt.get_cmap("YlOrRd")

fig, axes = plt.subplots(2, 2, sharex="col", figsize=(13, 6), gridspec_kw={"hspace": 0.05})
for col, (nz_list, src_cosmo, name) in enumerate([(nz_s3, cg_cosmo, "Stage-3"), (nz_des, des_cosmo, "DES Y3")]):
    colors = [_cm(x) for x in np.linspace(0.35, 0.95, len(nz_list))]
    qz = np.asarray(jc.probes.WeakLensing(nz_list).kernel(src_cosmo, z_grid, 1000.0))  # (nbins, nz)
    for i, nz in enumerate(nz_list):
        axes[0, col].plot(z_grid, nz(z_grid), color=colors[i], lw=2, label=f"bin {i + 1}")
        axes[1, col].plot(z_grid, qz[i], color=colors[i], lw=2)
    axes[0, col].set_title(f"{name} source n(z)")
    axes[1, col].set_xlabel(r"redshift $z$")
    axes[0, col].set_ylim(bottom=0.0)
    axes[1, col].set_ylim(bottom=0.0)
    axes[0, col].legend(frameon=False, fontsize=8)
    axes[0, col].grid(alpha=0.3)
    axes[1, col].grid(alpha=0.3)
axes[0, 0].set_ylabel(r"$n(z)$")
axes[1, 0].set_ylabel(r"$q(z)$  (lensing efficiency)")
fig.tight_layout()
savefig(ASSETS / "fig08-nz-bins", fig)

print(f"assets written to {ASSETS}")
