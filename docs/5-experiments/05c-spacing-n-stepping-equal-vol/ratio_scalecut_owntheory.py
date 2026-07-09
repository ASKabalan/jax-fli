"""Ratio of the scale-cut (ell_max=200) Born kappa to the Limber theory at its OWN cosmology.

Single-row ratio (3 tomographic bins) overlaying the two spacings + the CosmoGrid reference, each
measurement divided by compute_theory_cl at ITS OWN cosmology x pixwin(2048)^2:
  - 05c equal-volume (red) / Planck15 theory
  - 05b scale-factor  (blue) / Planck15 theory
  - CosmoGrid N-body  (black) / CosmoGrid-cosmology theory (sigma8=0.90, h=0.73, w0=-1.1665)

The scale-cut taper (SphericalDensity.scale_cut, lightcone.py:1277; l_cut=200, width=8) is flat (=1)
below ell=192, so within the retained band this ratio IS what the scale-cut pixel likelihood compares.
Drift shown (no-drift within <0.4% for 05b; slightly worse for 05c: bin1 1.49 vs 1.35).

Run (CPU, cached spectra only):
    JAX_PLATFORMS=cpu uv run --no-sync python docs/5-experiments/05c-spacing-n-stepping-equal-vol/ratio_scalecut_owntheory.py
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)  # before jax_cosmo

import sys
from pathlib import Path

import healpy as hp
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from huggingface_hub import snapshot_download
from matplotlib.lines import Line2D

from jax_fli import compute_theory_cl
from jax_fli.io import Catalog, get_stage3_nz_shear

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _exputils import savefig, set_style  # noqa: E402

ASSETS = HERE / "assets"
REPO = "ASKabalan/jax-fli-experiments"
LMAX, L_CUT, L_WIDTH, N = 1500, 200, 8, 20

root = snapshot_download(REPO, repo_type="dataset", local_files_only=True)


def load(p):
    return Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{p}", split="train"))


cg = load("00-cosmogrid/kappa_spectra/spectra_kappa_born_s3.parquet")
b05 = load(f"05-spacing-n-stepping/05b-3bins/kappa_spectra/spectra_born_drift_{N}.parquet")
c05 = load(f"05-spacing-n-stepping/05c-equal-volume/spectra_midpoint/spectra_born_drift_{N}.parquet")

k_cg, k_b, k_c = (np.asarray(x.field[0].array)[:3] for x in (cg, b05, c05))
cosmo_cg, cosmo_pl = cg.cosmology[0], b05.cosmology[0]

ell = np.arange(LMAX + 1)
pw2 = hp.pixwin(2048, lmax=LMAX) ** 2
nzs = get_stage3_nz_shear()[:3]
th_cg = np.asarray((compute_theory_cl(cosmo_cg, jnp.arange(LMAX + 1), nzs) * pw2).array)  # own cosmology
th_pl = np.asarray((compute_theory_cl(cosmo_pl, jnp.arange(LMAX + 1), nzs) * pw2).array)  # Planck15

# ratio to own-cosmology theory (scale cut is flat below 192 -> this is the retained-band ratio)
r_cg, r_b, r_c = k_cg / th_cg, k_b / th_pl, k_c / th_pl


def _logbin(e, y, nb=26):
    m = e >= 2
    ee, yy = e[m], np.asarray(y)[m]
    edges = np.unique(np.round(np.logspace(np.log10(2), np.log10(ee.max()), nb)).astype(int))
    bc, bv = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (ee >= lo) & (ee < hi)
        if s.any():
            bc.append(np.sqrt(lo * hi))
            bv.append(np.nanmean(yy[s]))
    return np.asarray(bc), np.asarray(bv)


def _med(y):
    m = (ell >= 20) & (ell <= 192)  # the band the ell_max=200 cut retains
    return float(np.nanmedian(y[m]))


print("median ratio to OWN-cosmology theory over the retained band [20,192]:")
for name, r in [("05c equal-vol ", r_c), ("05b scale-fac.", r_b), ("cosmogrid ref ", r_cg)]:
    print(f"  {name}  bin1/2/3 = {_med(r[0]):.3f} / {_med(r[1]):.3f} / {_med(r[2]):.3f}")


def fig15_scalecut_ratio():
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4), sharey=True)
    series = [(r_c, "tab:red", "05c equal-volume, drift"),
              (r_b, "tab:blue", "05b scale-factor, drift"),
              (r_cg, "k", "CosmoGrid Born (N-body)")]
    for b in range(3):
        ax = axes[b]
        ax.axhspan(0.95, 1.05, color="0.7", alpha=0.3)
        ax.axhline(1.0, color="0.4", ls="--", lw=0.9)
        ax.axvspan(L_CUT - L_WIDTH, L_CUT, color="0.85", alpha=0.7)
        ax.axvline(L_CUT, color="0.3", lw=0.8)
        for r, col, _ in series:
            bc, rb = _logbin(ell, r[b])
            ax.semilogx(bc, rb, color=col, lw=1.7)
        ax.set_title(f"bin {b + 1}", fontsize=11)
        ax.set_xlim(8, 210)
        ax.set_ylim(0.5, 1.7)
        ax.set_xlabel(r"multipole $\ell$")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        if b == 0:
            ax.set_ylabel(r"scale-cut $C_\ell^{\kappa\kappa}$ / own-cosmology theory")
    handles = [Line2D([], [], color=c, lw=1.7, label=lab) for _, c, lab in series]
    handles.append(Line2D([], [], color="0.85", lw=6, alpha=0.8, label=r"scale-cut taper ($\ell_{\max}=200$, width 8)"))
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    savefig(ASSETS / "fig15-scalecut-ratio-owntheory", fig)


set_style()
fig15_scalecut_ratio()
print(f"figure -> {ASSETS}/fig15-scalecut-ratio-owntheory.svg")
