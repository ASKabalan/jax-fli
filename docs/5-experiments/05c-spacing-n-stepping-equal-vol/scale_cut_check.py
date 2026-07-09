"""Scale-cut go/no-go: 05b/05c 20-shell Born kappa vs the CosmoGrid Born reference at ell <= 200.

Pick the spacing experiment with a CLI arg (default 05c): 05b = scale-factor spacing (thin shells),
05c = equal-volume spacing (fat inner ball). Same box / cosmology (Planck15) / n(z) — only the shelling
differs, so this isolates whether the shell spacing is what makes the Born kappa usable under the cut.

The inference (branch ``scale_cut``) band-limits each observable map to ``ell_max`` before the pixel
Gaussian. So the real question is not "is the equal-volume Born perfect at all ell" but: once scales
above ell=200 are cut, is the 05c equal-volume 20-shell Born kappa consistent with CosmoGrid — good
enough to use, or not?

For a tomographic AUTO-spectrum the map-level ``SphericalDensity.scale_cut`` (map2alm -> cosine ell-taper
w(ell) -> alm2map, ``src/jax_fli/fields/lightcone.py:1276``) reduces EXACTLY to multiplying C_ell by
w(ell)**2 (the taper is diagonal in ell; the band-limited round-trip is lossless). So the REAL cut is
applied here directly on the cached published spectra — no kappa maps, no downloads. With the config
default ``ell_taper_width=8`` the taper is 1 for ell<=192, so a cut at 200 leaves the whole ell<=192 band
untouched: it cannot remove an excess that lives at ell~100-200. The ratio to theory is therefore
cut-invariant below 192 — which is precisely the point of the check.

CosmoGrid (sigma8=0.90) and 05c (Planck15, sigma8=0.816) are different realisations at DIFFERENT
cosmologies, so each measurement is ratioed to the Limber weak-lensing theory at ITS OWN cosmology
(x pixwin(2048)**2), identical to fig11.

Run (CPU, ~minutes, cached data only):
    JAX_PLATFORMS=cpu uv run --no-sync python docs/5-experiments/05c-spacing-n-stepping-equal-vol/scale_cut_check.py 05b
"""

from __future__ import annotations

import jax

# float64 globally BEFORE jax_cosmo (the enable_x64() context manager breaks jax_cosmo's pure_callback
# comoving-distance cache; the global config flag is the safe route). Matches born_quadrature_fix.py.
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

from jax_fli import compute_theory_cl
from jax_fli.io import Catalog, get_stage3_nz_shear

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent
sys.path.insert(0, str(DOCS))
from _exputils import savefig, set_style  # noqa: E402

# which spacing experiment to check: `python scale_cut_check.py 05b|05c` (default equal-volume 05c).
# 05b = scale-factor spacing (thin shells), 05c = equal-volume (fat inner ball) — same box/cosmo/n(z).
EXPS = {
    "05b": ("05b-spacing-n-stepping-3bin", "05-spacing-n-stepping/05b-3bins", "scale-factor", "kappa_spectra"),
    "05c": ("05c-spacing-n-stepping-equal-vol", "05-spacing-n-stepping/05c-equal-volume", "equal-volume", "spectra_midpoint"),
}
WHICH = sys.argv[1] if len(sys.argv) > 1 else "05c"
EXP_DIR, HF_SUB, SPACING, SPECTRA_DIR = EXPS[WHICH]
ASSETS = DOCS / EXP_DIR / "assets"
N_SHELLS = 20  # target shell count (matches the annotated fig10 run)

REPO = "ASKabalan/jax-fli-experiments"
LMAX = 1500
L_CUT, L_WIDTH = 200, 8  # config defaults: ell_max, ell_taper_width

CG = "00-cosmogrid/kappa_spectra/spectra_kappa_born_s3.parquet"
DRIFT = f"{HF_SUB}/{SPECTRA_DIR}/spectra_born_drift_{N_SHELLS}.parquet"
NODRIFT = f"{HF_SUB}/{SPECTRA_DIR}/spectra_born_nodrift_{N_SHELLS}.parquet"

root = snapshot_download(REPO, repo_type="dataset", local_files_only=True)


def load_ps(path):
    return Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{path}", split="train"))


cg_cat, dr_cat, no_cat = load_ps(CG), load_ps(DRIFT), load_ps(NODRIFT)

kappa_cg = np.asarray(cg_cat.field[0].array)[:3]  # (3, 1501) CosmoGrid Born, s3 bins 1-3
kappa_dr = np.asarray(dr_cat.field[0].array)[:3]  # (3, 1501) equal-vol 20-shell, with drift
kappa_no = np.asarray(no_cat.field[0].array)[:3]  # (3, 1501) equal-vol 20-shell, no drift

cosmo_cg = cg_cat.cosmology[0]  # sigma8=0.90, h=0.73, w0=-1.1665
cosmo_pl = dr_cat.cosmology[0]  # Planck15, sigma8=0.816
print(f"CosmoGrid cosmo:  Om_c={float(cosmo_cg.Omega_c):.4f} h={float(cosmo_cg.h):.3f} s8={float(cosmo_cg.sigma8):.3f}")
print(f"equal-vol cosmo:  Om_c={float(cosmo_pl.Omega_c):.4f} h={float(cosmo_pl.h):.3f} s8={float(cosmo_pl.sigma8):.3f}")

# --- Limber weak-lensing theory at each own cosmology x pixwin^2 (same recipe as fig11) ---
ell = np.arange(LMAX + 1)
pw2 = hp.pixwin(2048, lmax=LMAX) ** 2
nzs = get_stage3_nz_shear()[:3]
th_cg = np.asarray((compute_theory_cl(cosmo_cg, jnp.arange(LMAX + 1), nzs) * pw2).array)  # (3, 1501)
th_pl = np.asarray((compute_theory_cl(cosmo_pl, jnp.arange(LMAX + 1), nzs) * pw2).array)  # (3, 1501)

# --- the REAL scale-cut taper w(ell), replicated from lightcone.py:1276-1277 (l_cut=200, l_width=8) ---
_x = (ell - (L_CUT - L_WIDTH)) / L_WIDTH
w = np.where(ell <= L_CUT - L_WIDTH, 1.0, np.where(ell >= L_CUT, 0.0, 0.5 * (1.0 + np.cos(np.pi * _x))))
w2 = w**2  # auto-spectrum effect of scale_cut == multiply C_ell by w(ell)^2 (exact for autos)

# scale-cut spectra (what the pixel likelihood sees) — for the top-row visual
kappa_cg_sc, kappa_dr_sc, kappa_no_sc = kappa_cg * w2, kappa_dr * w2, kappa_no * w2
th_cg_sc, th_pl_sc = th_cg * w2, th_pl * w2

# ratio to own theory (cut cancels for ell<=192 -> this is what decides "good or not" in the retained band)
r_cg = kappa_cg / th_cg
r_dr = kappa_dr / th_pl
r_no = kappa_no / th_pl


def _logbin(e, y, nb=24):
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


def _band(y, lo, hi):
    m = (ell >= lo) & (ell <= hi)
    return float(np.nanmedian(y[m])), float(np.nanmax(np.abs(y[m] - 1.0)))


def _crossover(rb_ell, rb):
    """Largest binned ell (from low ell up) where the ratio stays within +/-5% of 1 continuously."""
    ok = np.abs(rb - 1.0) <= 0.05
    last = 0.0
    for e_, o_ in zip(rb_ell, ok):
        if e_ < 20:  # judge from ell>=20
            continue
        if o_:
            last = e_
        else:
            break
    return last


# ---------------------------------------------------------------------------------------------
# verdict table
# ---------------------------------------------------------------------------------------------
print("=" * 92)
print("GATE — CosmoGrid Born / CosmoGrid theory should hug 1 across [20,200] (it's the N-body reference)")
for b in range(3):
    med, mx = _band(r_cg[b], 20, 200)
    print(f"  cosmogrid bin{b + 1}: median[20-200]={med:.3f}   max|r-1|[20-200]={mx:.3f}")
print("=" * 92)
print(f"RESULT — {WHICH} ({SPACING} spacing, N={N_SHELLS}): median ratio to own theory (retained band); "
      "crossover = highest ell within +/-5%")
labels = [(f"{SPACING} drift", r_dr), (f"{SPACING} no-drift", r_no), ("cosmogrid ref", r_cg)]
for name, r in labels:
    print(f"  {name}:")
    for b in range(3):
        m200, x200 = _band(r[b], 20, 200)
        m100, _ = _band(r[b], 20, 100)
        rb_ell, rb = _logbin(ell, r[b])
        xo = _crossover(rb_ell, rb)
        print(
            f"    bin{b + 1}:  med[20-200]={m200:.3f}  med[20-100]={m100:.3f}  "
            f"max|r-1|[20-200]={x200:.3f}  crossover_ell~{xo:.0f}"
        )
print("=" * 92)


# ---------------------------------------------------------------------------------------------
# fig14 — scale-cut applied, equal-volume vs CosmoGrid, each vs own theory
# ---------------------------------------------------------------------------------------------
def fig14_scale_cut_cosmogrid():
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 6.6), gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
    for b in range(3):
        ax_s, ax_r = axes[0, b], axes[1, b]
        # top: scale-cut spectra (roll-off at 200 visible) + own theory (also cut)
        bc, cg_b = _logbin(ell, kappa_cg_sc[b])
        _, dr_b = _logbin(ell, kappa_dr_sc[b])
        _, no_b = _logbin(ell, kappa_no_sc[b])
        _, thc_b = _logbin(ell, th_cg_sc[b])
        _, thp_b = _logbin(ell, th_pl_sc[b])
        ax_s.loglog(bc, cg_b, color="k", lw=1.6)
        ax_s.loglog(bc, thc_b, color="k", ls=":", lw=1.1)
        ax_s.loglog(bc, dr_b, color="tab:blue", lw=1.8)
        ax_s.loglog(bc, no_b, color="tab:red", lw=1.1, ls="--")  # dashed so overlapping blue (drift) stays visible
        ax_s.loglog(bc, thp_b, color="0.45", ls="--", lw=1.1)
        ax_s.axvspan(L_CUT - L_WIDTH, L_CUT, color="0.85", alpha=0.6)
        ax_s.axvline(L_CUT, color="0.3", ls="-", lw=0.8)
        ax_s.set_title(f"bin {b + 1}", fontsize=11)
        ax_s.set_xlim(8, 300)
        ax_s.grid(True, which="both", ls=":", alpha=0.4)
        ax_s.tick_params(labelbottom=False)
        if b == 0:
            ax_s.set_ylabel(r"$C_\ell^{\kappa\kappa}$ (scale-cut)")
        # bottom: ratio to own theory (the verdict; cut cancels below 192)
        _, rcg = _logbin(ell, r_cg[b])
        _, rdr = _logbin(ell, r_dr[b])
        _, rno = _logbin(ell, r_no[b])
        ax_r.axhspan(0.95, 1.05, color="0.7", alpha=0.3)
        ax_r.axhline(1.0, color="0.4", ls="--", lw=0.9)
        ax_r.axvspan(L_CUT - L_WIDTH, L_CUT, color="0.85", alpha=0.6)
        ax_r.axvline(L_CUT, color="0.3", ls="-", lw=0.8)
        ax_r.semilogx(bc, rcg, color="k", lw=1.5)
        ax_r.semilogx(bc, rdr, color="tab:blue", lw=1.8)
        ax_r.semilogx(bc, rno, color="tab:red", lw=1.1, ls="--")  # dashed so overlapping blue (drift) stays visible
        med, _ = _band(r_dr[b], 20, 200)
        ax_r.text(
            0.04, 0.86, rf"{SPACING} med$_{{[20,200]}}$={med:.2f}", transform=ax_r.transAxes,
            fontsize=8.5, color="tab:blue", va="top",
        )
        ax_r.set_ylim(0.3, 1.95)
        ax_r.set_xlim(8, 300)
        ax_r.set_xlabel(r"multipole $\ell$")
        ax_r.grid(True, which="both", ls=":", alpha=0.4)
        if b == 0:
            ax_r.set_ylabel("meas / own theory")
    handles = [
        Line2D([], [], color="k", lw=1.6, label="CosmoGrid Born (thin shells, N-body)"),
        Line2D([], [], color="tab:blue", lw=1.8, label=f"{SPACING} {N_SHELLS} shells, with drift"),
        Line2D([], [], color="tab:red", lw=1.1, ls="--", label=f"{SPACING} {N_SHELLS} shells, no drift (overlaps drift)"),
        Line2D([], [], color="0.45", ls="--", lw=1.4, label=r"Limber theory $\times\,w_\ell^2$ (own cosmology)"),
        Line2D([], [], color="0.85", lw=6, alpha=0.8, label=r"scale-cut taper ($\ell_{\max}=200$, width 8)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    savefig(ASSETS / "fig14-scale-cut-cosmogrid", fig)


def main():
    set_style()
    fig14_scale_cut_cosmogrid()
    print(f"figure written to {ASSETS}/fig14-scale-cut-cosmogrid.svg")


if __name__ == "__main__":
    main()
