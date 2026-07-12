import jax

jax.config.update("jax_enable_x64", True)

import sys
from pathlib import Path

import healpy as hp
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset
from huggingface_hub import snapshot_download

from jax_fli import compute_theory_cl_for_density
from jax_fli.io import Catalog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # -> docs/5-experiments
import matplotlib.pyplot as plt
from _exputils import savefig, set_style

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

REPO = "ASKabalan/jax-fli-experiments"
SHELLS_SPECTRA = "00-cosmogrid/density_spectra/spectra_cosmogrid_density_nside2048.parquet"

N_SHELLS = 56
NSIDE = 2048

NLB = 32  # multipoles per bandpower bin
TARGETS = [200, 300, 400, 500, 600, 700]  # convergence-point multipoles

root = snapshot_download(REPO, repo_type="dataset", local_files_only=True)

spectra_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{SHELLS_SPECTRA}", split="train"))
spectra, cosmo = spectra_cat.field[0], spectra_cat.cosmology[0]

LMAX = spectra.wavenumber.max()

theory_cls = compute_theory_cl_for_density(cosmo, spectra, jnp.arange(LMAX + 1))

# multiply by ell (ell + 1) / (2 * jnp.pi) to get the dimensionless power spectrum
spectra = spectra * spectra.wavenumber * (spectra.wavenumber + 1) / (2 * jnp.pi)
theory_cls = theory_cls * spectra.wavenumber * (spectra.wavenumber + 1) / (2 * jnp.pi)

# The data are HEALPix nside-2048 maps, so the measured C_ell is suppressed by the pixel
# window. Multiply the continuous-sky theory by pixwin^2 to compare like-for-like.
pixwin2 = hp.pixwin(NSIDE, lmax=int(LMAX)) ** 2  # numpy (LMAX+1,)
theory_cls = theory_cls * pixwin2

# =============================================================================
# Style + colors
# =============================================================================
set_style()

C_DATA = "tab:blue"  # CosmoGrid measured (binned bandpowers)
C_TH = "k"  # Limber theory


# =============================================================================
# Helper Function for Plotting
# =============================================================================
def plot_shell_batch(shells_data, shells_th, ell_array, title, start_idx, stem):
    # Bandpower-bin BOTH data and theory with the SAME operator (mode-weighted, nlb=32),
    # so the bins line up exactly and the ratio is well-defined at each bin's effective ell.
    data_b = shells_data.bin(nlb=NLB)
    theory_b = shells_th.bin(nlb=NLB)
    leff = data_b.wavenumber  # shared bin centers (effective multipole per bin)
    ratio = data_b.spectra / theory_b.spectra - 1.0

    # 2 rows, 5 columns with a 3:1 height ratio (spectra over ratio)
    fig, axes = plt.subplots(
        nrows=2,
        ncols=5,
        figsize=(20, 4),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex="col",
    )

    for i in range(5):
        ax_spec = axes[0, i]
        ax_ratio = axes[1, i]

        # --- First row: spectra (log-log) ---
        # Continuous theory line over the full multipole grid.
        ax_spec.plot(
            ell_array, shells_th.spectra[i], color=C_TH, ls="--", lw=1.3, label=r"Limber theory $\times\,w_\ell^2$"
        )
        # Binned data: a clean solid line (no markers).
        ax_spec.plot(leff, data_b.spectra[i], color=C_DATA, ls="-", lw=1.6, label=f"CosmoGrid (binned, nlb={NLB})")
        ax_spec.set_xscale("log")
        ax_spec.set_yscale("log")
        ax_spec.set_ylabel(r"$\ell(\ell+1)\,C_\ell/2\pi$")
        ax_spec.grid(True, which="both", ls=":", alpha=0.4)
        ax_spec.set_title(f"Shell {start_idx + i}")
        if i == 0:
            ax_spec.legend(frameon=False)

        # --- Second row: binned ratio (log-x) ---
        ax_ratio.axhspan(-0.05, 0.05, color="0.7", alpha=0.3, label=r"$\pm 5\%$")
        ax_ratio.axhline(0.0, color="0.4", ls="--", lw=1.0)
        ax_ratio.plot(leff, ratio[i], color=C_DATA, ls="-", lw=1.6)
        ax_ratio.set_xscale("log")
        ax_ratio.set_ylim(-0.2, 0.2)
        ax_ratio.set_xlabel(r"multipole $\ell$")
        ax_ratio.grid(True, which="both", ls=":", alpha=0.4)
        if i == 0:
            ax_ratio.set_ylabel("data / theory - 1")
            ax_ratio.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    savefig(ASSETS / stem, fig)


# Define the multipole array matching the LMAX range
ell = jnp.arange(LMAX + 1)

# =============================================================================
# Plotting for the first 5 shells
# =============================================================================
plot_shell_batch(spectra[:5], theory_cls[:5], ell, "First 5 Shells", start_idx=0, stem="fig01-first5-shells")

# =============================================================================
# Plotting for the mid 5 shells
# =============================================================================
mid_idx = (N_SHELLS // 2) - 2  # Starting around shell 26
plot_shell_batch(
    spectra[mid_idx : mid_idx + 5],
    theory_cls[mid_idx : mid_idx + 5],
    ell,
    "Middle 5 Shells",
    start_idx=mid_idx,
    stem="fig02-mid5-shells",
)

# =============================================================================
# Plotting for the last 5 shells
# =============================================================================
last_idx = N_SHELLS - 5
plot_shell_batch(
    spectra[last_idx:], theory_cls[last_idx:], ell, "Last 5 Shells", start_idx=last_idx, stem="fig03-last5-shells"
)


# =============================================================================
# fig04 — convergence to the pixwin^2-matched theory across all 56 shells
# =============================================================================
# For each target multipole we take the (2ell+1)-weighted measured/theory ratio in a narrow
# band around it, per shell, and plot it against the shell's comoving distance chi. Theory is
# already x pixwin^2(2048), so this is a like-for-like (pixel-window-matched) comparison.
ell_np = np.asarray(spectra.wavenumber)
meas = np.asarray(spectra.spectra)  # (n_shells, n_ell)
theory_pw = np.asarray(theory_cls.spectra)  # (n_shells, n_ell), already x pixwin^2
chi = np.asarray(spectra.comoving_centers)  # (n_shells,) comoving distance [Mpc/h]


def band_ratio(lo, hi):
    """Per-shell (2ell+1)-weighted measured/theory over ell in [lo, hi], plus the full-sky CV sigma."""
    sel = (ell_np >= lo) & (ell_np <= hi)
    w = 2.0 * ell_np[sel] + 1.0
    mb = (meas[:, sel] * w).sum(axis=1) / w.sum()
    tb = (theory_pw[:, sel] * w).sum(axis=1) / w.sum()
    return mb / tb, float(np.sqrt(2.0 / w.sum()))


fig, axes = plt.subplots(2, len(TARGETS) // 2, figsize=(2.5 * len(TARGETS), 4.6), sharey=True)
axes = axes.flatten()
for ax, ltarget in zip(axes, TARGETS):
    lo, hi = round(0.9 * ltarget), round(1.1 * ltarget)
    ratio, cv = band_ratio(lo, hi)
    ax.axhspan(-0.05, 0.05, color="orange", alpha=0.18, lw=0, label=r"$\pm5\%$")
    ax.axhspan(-cv, cv, color="0.8", lw=0, label=rf"$\pm1\sigma$ CV ({cv:.1%})")
    ax.axhline(0.0, color="0.4", ls="--", lw=1.0)
    ax.plot(chi, ratio - 1, "-", color=C_DATA, lw=1.4)
    ax.set_title(rf"$\ell \approx$ {ltarget}  (band {lo}–{hi})", fontsize=11)
    ax.set_xlabel(r"comoving distance $\chi$  [Mpc/$h$]")
    ax.set_ylim(-0.2, 0.2)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
axes[0].set_ylabel(r"measured / theory - 1  ($(2\ell+1)$-weighted)")
fig.tight_layout()
savefig(ASSETS / "fig04-convergence-pixwin", fig)

print(f"assets written to {ASSETS}")
