"""Experiment 05c — equal-volume shells (density-shell C_ell + Born lensing): figures.

Renders the SVG figures for ``docs/5-experiments/05c-spacing-n-stepping-equal-vol/README.md``. Equal-volume
shell spacing gives every shell the same comoving volume: the innermost shell is a *fat ball* and the outer
shells are thin (floored to ``--min-width 60``). The frozen-epoch redshift error of a thick shell is therefore
largest in the fat inner shell — exactly where the drift (moving each particle to the scale factor at which it
crosses the lightcone) buys the most.

  fig01   illustration (small local sim): one particle cloud coloured by the redshift it is assigned under a
          10-shell equal-volume freeze, the drift's smooth z(r), and a 30-shell equal-volume freeze.
  fig02   density C_ell over the near (fat inner ball) / mid / far radial region: the 10-shell drift / no-drift
          runs vs a continuous-lightcone reference summed from the 40-shell no-drift run (counts -> overdensity).
  fig03   per-shell density C_ell census vs Limber theory for the 5- / 8- / 10-shell runs (drift + no-drift,
          one subplot per shell: a log-log C_ell panel over a ratio-to-theory strip).
  fig04-fig08  the same census for the 16- / 20- / 25- / 30- / 40-shell runs.
  fig09   Born shell windows on the 10-shell geometry: the exact lensing kernel with each shell drawn as a box
          whose area is its Born weight under midpoint vs Gauss-Legendre, equal-volume (5c) vs scale-factor (5b)
          at z_s=1.2 — the midpoint box overshoots the fat inner shell.
  fig10   Born convergence C_ell per tomographic bin (Stage-3 [:3]) vs the number of shells, ratioed to the own
          40-shell run — midpoint quadrature (three source-bin row-pairs x no-drift / with-drift columns).
  fig11   the same, composite-Simpson quadrature.
  fig12   the same, Gauss-Legendre (exact-kernel) quadrature.
  fig13   the Gauss-Legendre convergence ratioed to the Limber weak-lensing theory (the residual PM roll-off).
  fig14-fig18  equal-volume vs scale-factor (05b) spacing at Gauss-Legendre (drift), all bins overlaid: the
          D_ell power + the C_ell/theory-1 residual (nlb=32), at N = 30 / 40 / 20 / 12 / 25 shells.

Every 05c run is published for both drift and no-drift up to 40 shells. Equal-volume spacing is excellent for
per-shell density statistics but its fat inner shell breaks the *midpoint* Born quadrature: one midpoint kernel
weight x one volume-averaged map per shell is a bad rule when the lensing kernel varies strongly across a shell,
and equal-volume makes the inner shells (where the low-z kernel lives) as fat as possible, so the coarse-N
midpoint convergence overshoots badly (fig09/fig10). Integrating the kernel across each shell with composite
Simpson or Gauss-Legendre removes the overshoot (fig11/fig12); the Gauss-Legendre runs then follow the
PM-resolution roll-off, close to the scale-factor 05b runs (fig13-fig18). The drift removes the shells'
frozen-epoch error (~20%, the census story), a separate effect from the quadrature.

Run from the repo root (CPU is fine; fig01 runs a small sim, fig02 sums nside-2048 maps):
    JAX_PLATFORMS=cpu uv run --no-sync python docs/5-experiments/05c-spacing-n-stepping-equal-vol/build.py
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
from matplotlib.lines import Line2D

import jax_fli as jfli
from jax_fli import compute_theory_cl, compute_theory_cl_for_density
from jax_fli.io import Catalog, get_stage3_nz_shear
from jax_fli.lensing import plot_born_windows

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _exputils import savefig, set_style  # noqa: E402

ASSETS = HERE / "assets"
REPO = "ASKabalan/jax-fli-experiments"

LMAX = 1500  # the published spectra stop at ell 1500
BOX, MESH = 5000.0, 2560.0  # per-shell PM-Nyquist line ell_max ~ pi*chi/dx, dx = box/mesh (equal-volume box)
CENSUS_RATIO_YLIM = (-0.3, 0.35)  # ratio-to-theory strip range, centered on 0 (wide enough for the fat inner shell)
NSHELLS_KAPPA = [5, 8, 10, 12, 16, 20, 25, 30, 40]  # Born-convergence shell-count sweep (fig10-fig13)

root = snapshot_download(REPO, repo_type="dataset", local_files_only=True)

# Per-shell density census (fig03-fig08): drift + no-drift spectra for every published shell count. fig01/fig02
# additionally read only the shell geometry (equal-volume edges) from nodrift_10 / nodrift_30 / nodrift_40.
DRIFT_5_SPECTRA = "05-spacing-n-stepping/05c-equal-volume/density_spectra/spectra_exp5c_drift_5.parquet"
NODRIFT_5_SPECTRA = "05-spacing-n-stepping/05c-equal-volume/density_spectra/spectra_exp5c_nodrift_5.parquet"
DRIFT_8_SPECTRA = "05-spacing-n-stepping/05c-equal-volume/density_spectra/spectra_exp5c_drift_8.parquet"
NODRIFT_8_SPECTRA = "05-spacing-n-stepping/05c-equal-volume/density_spectra/spectra_exp5c_nodrift_8.parquet"
DRIFT_10_SPECTRA = "05-spacing-n-stepping/05c-equal-volume/density_spectra/spectra_exp5c_drift_10.parquet"
NODRIFT_10_SPECTRA = "05-spacing-n-stepping/05c-equal-volume/density_spectra/spectra_exp5c_nodrift_10.parquet"
DRIFT_16_SPECTRA = "05-spacing-n-stepping/05c-equal-volume/density_spectra/spectra_exp5c_drift_16.parquet"
NODRIFT_16_SPECTRA = "05-spacing-n-stepping/05c-equal-volume/density_spectra/spectra_exp5c_nodrift_16.parquet"
DRIFT_20_SPECTRA = "05-spacing-n-stepping/05c-equal-volume/density_spectra/spectra_exp5c_drift_20.parquet"
NODRIFT_20_SPECTRA = "05-spacing-n-stepping/05c-equal-volume/density_spectra/spectra_exp5c_nodrift_20.parquet"
DRIFT_25_SPECTRA = "05-spacing-n-stepping/05c-equal-volume/density_spectra/spectra_exp5c_drift_25.parquet"
NODRIFT_25_SPECTRA = "05-spacing-n-stepping/05c-equal-volume/density_spectra/spectra_exp5c_nodrift_25.parquet"
DRIFT_30_SPECTRA = "05-spacing-n-stepping/05c-equal-volume/density_spectra/spectra_exp5c_drift_30.parquet"
NODRIFT_30_SPECTRA = "05-spacing-n-stepping/05c-equal-volume/density_spectra/spectra_exp5c_nodrift_30.parquet"
DRIFT_40_SPECTRA = "05-spacing-n-stepping/05c-equal-volume/density_spectra/spectra_exp5c_drift_40.parquet"
NODRIFT_40_SPECTRA = "05-spacing-n-stepping/05c-equal-volume/density_spectra/spectra_exp5c_nodrift_40.parquet"

drift_5_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DRIFT_5_SPECTRA}", split="train"))
nodrift_5_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{NODRIFT_5_SPECTRA}", split="train"))
drift_8_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DRIFT_8_SPECTRA}", split="train"))
nodrift_8_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{NODRIFT_8_SPECTRA}", split="train"))
drift_10_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DRIFT_10_SPECTRA}", split="train"))
nodrift_10_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{NODRIFT_10_SPECTRA}", split="train"))
drift_16_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DRIFT_16_SPECTRA}", split="train"))
nodrift_16_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{NODRIFT_16_SPECTRA}", split="train"))
drift_20_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DRIFT_20_SPECTRA}", split="train"))
nodrift_20_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{NODRIFT_20_SPECTRA}", split="train"))
drift_25_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DRIFT_25_SPECTRA}", split="train"))
nodrift_25_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{NODRIFT_25_SPECTRA}", split="train"))
drift_30_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DRIFT_30_SPECTRA}", split="train"))
nodrift_30_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{NODRIFT_30_SPECTRA}", split="train"))
drift_40_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DRIFT_40_SPECTRA}", split="train"))
nodrift_40_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{NODRIFT_40_SPECTRA}", split="train"))

cosmo = nodrift_10_cat.cosmology[0]  # one fiducial cosmology shared by every run
for _c in (
    drift_5_cat,
    nodrift_5_cat,
    drift_8_cat,
    nodrift_8_cat,
    drift_10_cat,
    drift_16_cat,
    nodrift_16_cat,
    drift_20_cat,
    nodrift_20_cat,
    drift_25_cat,
    nodrift_25_cat,
    drift_30_cat,
    nodrift_30_cat,
    drift_40_cat,
    nodrift_40_cat,
):
    assert np.isclose(float(_c.cosmology[0].Omega_c), float(cosmo.Omega_c))
    assert np.isclose(float(_c.cosmology[0].sigma8), float(cosmo.sigma8))

drift_5, nodrift_5 = drift_5_cat.field[0], nodrift_5_cat.field[0]
drift_8, nodrift_8 = drift_8_cat.field[0], nodrift_8_cat.field[0]
drift_10, nodrift_10 = drift_10_cat.field[0], nodrift_10_cat.field[0]
drift_16, nodrift_16 = drift_16_cat.field[0], nodrift_16_cat.field[0]
drift_20, nodrift_20 = drift_20_cat.field[0], nodrift_20_cat.field[0]
drift_25, nodrift_25 = drift_25_cat.field[0], nodrift_25_cat.field[0]
drift_30, nodrift_30 = drift_30_cat.field[0], nodrift_30_cat.field[0]
drift_40, nodrift_40 = drift_40_cat.field[0], nodrift_40_cat.field[0]

# Equal-volume shell edges (near, far) from the spectra metadata. NOTE: reconstruct them from the stored
# density_width (comoving_centers ± width/2) — the centre-reflection helper jax_fli.utils.edges assumes
# uniform-ish widths and produces oscillating edges for equal-volume spacing.
chi10 = np.asarray(nodrift_10.comoving_centers)
w10 = np.asarray(nodrift_10.density_width)
edges10 = np.stack([chi10 - 0.5 * w10, chi10 + 0.5 * w10], axis=0)
chi30 = np.asarray(nodrift_30.comoving_centers)
w30 = np.asarray(nodrift_30.density_width)
chi40 = np.asarray(nodrift_40.comoving_centers)

# 05b (scale-factor) 10-shell shell edges — for fig09's side-by-side window comparison against equal-volume.
DENSITY_5B_10 = "05-spacing-n-stepping/05b-3bins/density_spectra/spectra_exp5b_nodrift_10.parquet"
density_5b_10_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DENSITY_5B_10}", split="train"))
chi_5b = np.asarray(density_5b_10_cat.field[0].comoving_centers)
w_5b = np.asarray(density_5b_10_cat.field[0].density_width)

# Born-convergence (kappa) spectra: 3-bin tomographic (Stage-3 [:3]) auto C_ell per shell count, drift and
# no-drift, under three quadrature schemes — midpoint (legacy), Simpson, and Gauss-Legendre (exact, the
# reference). Each parquet holds a PowerSpectrum of (3, n_ell) — one auto spectrum per source bin. Feeds
# fig10/fig11/fig12 (per quadrature, ratioed to each quadrature's own 40-shell run) and fig13 (GL vs theory).

# ---- midpoint quadrature ----
KAPPA_MID_DRIFT_5 = "05-spacing-n-stepping/05c-equal-volume/spectra_midpoint/spectra_midpoint_drift_5.parquet"
KAPPA_MID_DRIFT_8 = "05-spacing-n-stepping/05c-equal-volume/spectra_midpoint/spectra_midpoint_drift_8.parquet"
KAPPA_MID_DRIFT_10 = "05-spacing-n-stepping/05c-equal-volume/spectra_midpoint/spectra_midpoint_drift_10.parquet"
KAPPA_MID_DRIFT_12 = "05-spacing-n-stepping/05c-equal-volume/spectra_midpoint/spectra_midpoint_drift_12.parquet"
KAPPA_MID_DRIFT_16 = "05-spacing-n-stepping/05c-equal-volume/spectra_midpoint/spectra_midpoint_drift_16.parquet"
KAPPA_MID_DRIFT_20 = "05-spacing-n-stepping/05c-equal-volume/spectra_midpoint/spectra_midpoint_drift_20.parquet"
KAPPA_MID_DRIFT_25 = "05-spacing-n-stepping/05c-equal-volume/spectra_midpoint/spectra_midpoint_drift_25.parquet"
KAPPA_MID_DRIFT_30 = "05-spacing-n-stepping/05c-equal-volume/spectra_midpoint/spectra_midpoint_drift_30.parquet"
KAPPA_MID_DRIFT_40 = "05-spacing-n-stepping/05c-equal-volume/spectra_midpoint/spectra_midpoint_drift_40.parquet"
KAPPA_MID_NODRIFT_5 = "05-spacing-n-stepping/05c-equal-volume/spectra_midpoint/spectra_midpoint_nodrift_5.parquet"
KAPPA_MID_NODRIFT_8 = "05-spacing-n-stepping/05c-equal-volume/spectra_midpoint/spectra_midpoint_nodrift_8.parquet"
KAPPA_MID_NODRIFT_10 = "05-spacing-n-stepping/05c-equal-volume/spectra_midpoint/spectra_midpoint_nodrift_10.parquet"
KAPPA_MID_NODRIFT_12 = "05-spacing-n-stepping/05c-equal-volume/spectra_midpoint/spectra_midpoint_nodrift_12.parquet"
KAPPA_MID_NODRIFT_16 = "05-spacing-n-stepping/05c-equal-volume/spectra_midpoint/spectra_midpoint_nodrift_16.parquet"
KAPPA_MID_NODRIFT_20 = "05-spacing-n-stepping/05c-equal-volume/spectra_midpoint/spectra_midpoint_nodrift_20.parquet"
KAPPA_MID_NODRIFT_25 = "05-spacing-n-stepping/05c-equal-volume/spectra_midpoint/spectra_midpoint_nodrift_25.parquet"
KAPPA_MID_NODRIFT_30 = "05-spacing-n-stepping/05c-equal-volume/spectra_midpoint/spectra_midpoint_nodrift_30.parquet"
KAPPA_MID_NODRIFT_40 = "05-spacing-n-stepping/05c-equal-volume/spectra_midpoint/spectra_midpoint_nodrift_40.parquet"

# ---- composite Simpson quadrature ----
KAPPA_SIMP_DRIFT_5 = "05-spacing-n-stepping/05c-equal-volume/spectra_simpson/spectra_simpson_drift_5.parquet"
KAPPA_SIMP_DRIFT_8 = "05-spacing-n-stepping/05c-equal-volume/spectra_simpson/spectra_simpson_drift_8.parquet"
KAPPA_SIMP_DRIFT_10 = "05-spacing-n-stepping/05c-equal-volume/spectra_simpson/spectra_simpson_drift_10.parquet"
KAPPA_SIMP_DRIFT_12 = "05-spacing-n-stepping/05c-equal-volume/spectra_simpson/spectra_simpson_drift_12.parquet"
KAPPA_SIMP_DRIFT_16 = "05-spacing-n-stepping/05c-equal-volume/spectra_simpson/spectra_simpson_drift_16.parquet"
KAPPA_SIMP_DRIFT_20 = "05-spacing-n-stepping/05c-equal-volume/spectra_simpson/spectra_simpson_drift_20.parquet"
KAPPA_SIMP_DRIFT_25 = "05-spacing-n-stepping/05c-equal-volume/spectra_simpson/spectra_simpson_drift_25.parquet"
KAPPA_SIMP_DRIFT_30 = "05-spacing-n-stepping/05c-equal-volume/spectra_simpson/spectra_simpson_drift_30.parquet"
KAPPA_SIMP_DRIFT_40 = "05-spacing-n-stepping/05c-equal-volume/spectra_simpson/spectra_simpson_drift_40.parquet"
KAPPA_SIMP_NODRIFT_5 = "05-spacing-n-stepping/05c-equal-volume/spectra_simpson/spectra_simpson_nodrift_5.parquet"
KAPPA_SIMP_NODRIFT_8 = "05-spacing-n-stepping/05c-equal-volume/spectra_simpson/spectra_simpson_nodrift_8.parquet"
KAPPA_SIMP_NODRIFT_10 = "05-spacing-n-stepping/05c-equal-volume/spectra_simpson/spectra_simpson_nodrift_10.parquet"
KAPPA_SIMP_NODRIFT_12 = "05-spacing-n-stepping/05c-equal-volume/spectra_simpson/spectra_simpson_nodrift_12.parquet"
KAPPA_SIMP_NODRIFT_16 = "05-spacing-n-stepping/05c-equal-volume/spectra_simpson/spectra_simpson_nodrift_16.parquet"
KAPPA_SIMP_NODRIFT_20 = "05-spacing-n-stepping/05c-equal-volume/spectra_simpson/spectra_simpson_nodrift_20.parquet"
KAPPA_SIMP_NODRIFT_25 = "05-spacing-n-stepping/05c-equal-volume/spectra_simpson/spectra_simpson_nodrift_25.parquet"
KAPPA_SIMP_NODRIFT_30 = "05-spacing-n-stepping/05c-equal-volume/spectra_simpson/spectra_simpson_nodrift_30.parquet"
KAPPA_SIMP_NODRIFT_40 = "05-spacing-n-stepping/05c-equal-volume/spectra_simpson/spectra_simpson_nodrift_40.parquet"

# ---- Gauss-Legendre quadrature (exact kernel; the reference) ----
KAPPA_GL_DRIFT_5 = "05-spacing-n-stepping/05c-equal-volume/spectra_gauss_legendre/spectra_gl_drift_5.parquet"
KAPPA_GL_DRIFT_8 = "05-spacing-n-stepping/05c-equal-volume/spectra_gauss_legendre/spectra_gl_drift_8.parquet"
KAPPA_GL_DRIFT_10 = "05-spacing-n-stepping/05c-equal-volume/spectra_gauss_legendre/spectra_gl_drift_10.parquet"
KAPPA_GL_DRIFT_12 = "05-spacing-n-stepping/05c-equal-volume/spectra_gauss_legendre/spectra_gl_drift_12.parquet"
KAPPA_GL_DRIFT_16 = "05-spacing-n-stepping/05c-equal-volume/spectra_gauss_legendre/spectra_gl_drift_16.parquet"
KAPPA_GL_DRIFT_20 = "05-spacing-n-stepping/05c-equal-volume/spectra_gauss_legendre/spectra_gl_drift_20.parquet"
KAPPA_GL_DRIFT_25 = "05-spacing-n-stepping/05c-equal-volume/spectra_gauss_legendre/spectra_gl_drift_25.parquet"
KAPPA_GL_DRIFT_30 = "05-spacing-n-stepping/05c-equal-volume/spectra_gauss_legendre/spectra_gl_drift_30.parquet"
KAPPA_GL_DRIFT_40 = "05-spacing-n-stepping/05c-equal-volume/spectra_gauss_legendre/spectra_gl_drift_40.parquet"
KAPPA_GL_NODRIFT_5 = "05-spacing-n-stepping/05c-equal-volume/spectra_gauss_legendre/spectra_gl_nodrift_5.parquet"
KAPPA_GL_NODRIFT_8 = "05-spacing-n-stepping/05c-equal-volume/spectra_gauss_legendre/spectra_gl_nodrift_8.parquet"
KAPPA_GL_NODRIFT_10 = "05-spacing-n-stepping/05c-equal-volume/spectra_gauss_legendre/spectra_gl_nodrift_10.parquet"
KAPPA_GL_NODRIFT_12 = "05-spacing-n-stepping/05c-equal-volume/spectra_gauss_legendre/spectra_gl_nodrift_12.parquet"
KAPPA_GL_NODRIFT_16 = "05-spacing-n-stepping/05c-equal-volume/spectra_gauss_legendre/spectra_gl_nodrift_16.parquet"
KAPPA_GL_NODRIFT_20 = "05-spacing-n-stepping/05c-equal-volume/spectra_gauss_legendre/spectra_gl_nodrift_20.parquet"
KAPPA_GL_NODRIFT_25 = "05-spacing-n-stepping/05c-equal-volume/spectra_gauss_legendre/spectra_gl_nodrift_25.parquet"
KAPPA_GL_NODRIFT_30 = "05-spacing-n-stepping/05c-equal-volume/spectra_gauss_legendre/spectra_gl_nodrift_30.parquet"
KAPPA_GL_NODRIFT_40 = "05-spacing-n-stepping/05c-equal-volume/spectra_gauss_legendre/spectra_gl_nodrift_40.parquet"

# ---- scale-factor spacing (05b) at N=12,20,25,30,40 for the spacing comparison (fig14-fig18). 05b was only
# run with the midpoint quadrature, but its thin scale-factor shells make the midpoint/GL quadrature difference
# < ~0.2% on the total per-bin lensing weight (see fig09), so 05b midpoint stands in for 05b GL here. ----
KAPPA_5B_DRIFT_12 = "05-spacing-n-stepping/05b-3bins/kappa_spectra/spectra_born_drift_12.parquet"
KAPPA_5B_DRIFT_20 = "05-spacing-n-stepping/05b-3bins/kappa_spectra/spectra_born_drift_20.parquet"
KAPPA_5B_DRIFT_25 = "05-spacing-n-stepping/05b-3bins/kappa_spectra/spectra_born_drift_25.parquet"
KAPPA_5B_DRIFT_30 = "05-spacing-n-stepping/05b-3bins/kappa_spectra/spectra_born_drift_30.parquet"
KAPPA_5B_DRIFT_40 = "05-spacing-n-stepping/05b-3bins/kappa_spectra/spectra_born_drift_40.parquet"

# fmt: off  (one HF spectra file per line — kept single-line so the loaded files are glanceable)
# midpoint
kappa_mid_drift_5_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_MID_DRIFT_5}", split="train")
)
kappa_mid_drift_8_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_MID_DRIFT_8}", split="train")
)
kappa_mid_drift_10_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_MID_DRIFT_10}", split="train")
)
kappa_mid_drift_12_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_MID_DRIFT_12}", split="train")
)
kappa_mid_drift_16_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_MID_DRIFT_16}", split="train")
)
kappa_mid_drift_20_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_MID_DRIFT_20}", split="train")
)
kappa_mid_drift_25_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_MID_DRIFT_25}", split="train")
)
kappa_mid_drift_30_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_MID_DRIFT_30}", split="train")
)
kappa_mid_drift_40_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_MID_DRIFT_40}", split="train")
)
kappa_mid_nodrift_5_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_MID_NODRIFT_5}", split="train")
)
kappa_mid_nodrift_8_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_MID_NODRIFT_8}", split="train")
)
kappa_mid_nodrift_10_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_MID_NODRIFT_10}", split="train")
)
kappa_mid_nodrift_12_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_MID_NODRIFT_12}", split="train")
)
kappa_mid_nodrift_16_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_MID_NODRIFT_16}", split="train")
)
kappa_mid_nodrift_20_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_MID_NODRIFT_20}", split="train")
)
kappa_mid_nodrift_25_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_MID_NODRIFT_25}", split="train")
)
kappa_mid_nodrift_30_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_MID_NODRIFT_30}", split="train")
)
kappa_mid_nodrift_40_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_MID_NODRIFT_40}", split="train")
)
# Simpson
kappa_simp_drift_5_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_SIMP_DRIFT_5}", split="train")
)
kappa_simp_drift_8_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_SIMP_DRIFT_8}", split="train")
)
kappa_simp_drift_10_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_SIMP_DRIFT_10}", split="train")
)
kappa_simp_drift_12_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_SIMP_DRIFT_12}", split="train")
)
kappa_simp_drift_16_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_SIMP_DRIFT_16}", split="train")
)
kappa_simp_drift_20_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_SIMP_DRIFT_20}", split="train")
)
kappa_simp_drift_25_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_SIMP_DRIFT_25}", split="train")
)
kappa_simp_drift_30_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_SIMP_DRIFT_30}", split="train")
)
kappa_simp_drift_40_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_SIMP_DRIFT_40}", split="train")
)
kappa_simp_nodrift_5_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_SIMP_NODRIFT_5}", split="train")
)
kappa_simp_nodrift_8_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_SIMP_NODRIFT_8}", split="train")
)
kappa_simp_nodrift_10_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_SIMP_NODRIFT_10}", split="train")
)
kappa_simp_nodrift_12_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_SIMP_NODRIFT_12}", split="train")
)
kappa_simp_nodrift_16_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_SIMP_NODRIFT_16}", split="train")
)
kappa_simp_nodrift_20_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_SIMP_NODRIFT_20}", split="train")
)
kappa_simp_nodrift_25_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_SIMP_NODRIFT_25}", split="train")
)
kappa_simp_nodrift_30_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_SIMP_NODRIFT_30}", split="train")
)
kappa_simp_nodrift_40_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_SIMP_NODRIFT_40}", split="train")
)
# Gauss-Legendre
kappa_gl_drift_5_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_GL_DRIFT_5}", split="train")
)
kappa_gl_drift_8_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_GL_DRIFT_8}", split="train")
)
kappa_gl_drift_10_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_GL_DRIFT_10}", split="train")
)
kappa_gl_drift_12_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_GL_DRIFT_12}", split="train")
)
kappa_gl_drift_16_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_GL_DRIFT_16}", split="train")
)
kappa_gl_drift_20_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_GL_DRIFT_20}", split="train")
)
kappa_gl_drift_25_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_GL_DRIFT_25}", split="train")
)
kappa_gl_drift_30_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_GL_DRIFT_30}", split="train")
)
kappa_gl_drift_40_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_GL_DRIFT_40}", split="train")
)
kappa_gl_nodrift_5_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_GL_NODRIFT_5}", split="train")
)
kappa_gl_nodrift_8_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_GL_NODRIFT_8}", split="train")
)
kappa_gl_nodrift_10_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_GL_NODRIFT_10}", split="train")
)
kappa_gl_nodrift_12_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_GL_NODRIFT_12}", split="train")
)
kappa_gl_nodrift_16_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_GL_NODRIFT_16}", split="train")
)
kappa_gl_nodrift_20_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_GL_NODRIFT_20}", split="train")
)
kappa_gl_nodrift_25_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_GL_NODRIFT_25}", split="train")
)
kappa_gl_nodrift_30_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_GL_NODRIFT_30}", split="train")
)
kappa_gl_nodrift_40_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_GL_NODRIFT_40}", split="train")
)
# scale-factor spacing (05b), midpoint
kappa_5b_drift_12_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_5B_DRIFT_12}", split="train")
)
kappa_5b_drift_20_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_5B_DRIFT_20}", split="train")
)
kappa_5b_drift_25_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_5B_DRIFT_25}", split="train")
)
kappa_5b_drift_30_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_5B_DRIFT_30}", split="train")
)
kappa_5b_drift_40_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_5B_DRIFT_40}", split="train")
)
# fmt: on

for _c in (
    kappa_mid_drift_5_cat,
    kappa_mid_nodrift_40_cat,
    kappa_simp_drift_10_cat,
    kappa_simp_nodrift_20_cat,
    kappa_gl_drift_5_cat,
    kappa_gl_nodrift_40_cat,
    kappa_5b_drift_20_cat,
):
    assert np.isclose(float(_c.cosmology[0].Omega_c), float(cosmo.Omega_c))
    assert np.isclose(float(_c.cosmology[0].sigma8), float(cosmo.sigma8))

# kappa C_ell keyed by shell count; each entry is a PowerSpectrum of (3, n_ell) — one auto spectrum per bin.
# One dict per (quadrature, drift) so fig10/fig11/fig12 can ratio each quadrature to its OWN 40-shell run.
kappa_mid_dr = {
    n: c.field[0]
    for n, c in zip(
        NSHELLS_KAPPA,
        [
            kappa_mid_drift_5_cat,
            kappa_mid_drift_8_cat,
            kappa_mid_drift_10_cat,
            kappa_mid_drift_12_cat,
            kappa_mid_drift_16_cat,
            kappa_mid_drift_20_cat,
            kappa_mid_drift_25_cat,
            kappa_mid_drift_30_cat,
            kappa_mid_drift_40_cat,
        ],
    )
}
kappa_mid_no = {
    n: c.field[0]
    for n, c in zip(
        NSHELLS_KAPPA,
        [
            kappa_mid_nodrift_5_cat,
            kappa_mid_nodrift_8_cat,
            kappa_mid_nodrift_10_cat,
            kappa_mid_nodrift_12_cat,
            kappa_mid_nodrift_16_cat,
            kappa_mid_nodrift_20_cat,
            kappa_mid_nodrift_25_cat,
            kappa_mid_nodrift_30_cat,
            kappa_mid_nodrift_40_cat,
        ],
    )
}
kappa_simp_dr = {
    n: c.field[0]
    for n, c in zip(
        NSHELLS_KAPPA,
        [
            kappa_simp_drift_5_cat,
            kappa_simp_drift_8_cat,
            kappa_simp_drift_10_cat,
            kappa_simp_drift_12_cat,
            kappa_simp_drift_16_cat,
            kappa_simp_drift_20_cat,
            kappa_simp_drift_25_cat,
            kappa_simp_drift_30_cat,
            kappa_simp_drift_40_cat,
        ],
    )
}
kappa_simp_no = {
    n: c.field[0]
    for n, c in zip(
        NSHELLS_KAPPA,
        [
            kappa_simp_nodrift_5_cat,
            kappa_simp_nodrift_8_cat,
            kappa_simp_nodrift_10_cat,
            kappa_simp_nodrift_12_cat,
            kappa_simp_nodrift_16_cat,
            kappa_simp_nodrift_20_cat,
            kappa_simp_nodrift_25_cat,
            kappa_simp_nodrift_30_cat,
            kappa_simp_nodrift_40_cat,
        ],
    )
}
kappa_gl_dr = {
    n: c.field[0]
    for n, c in zip(
        NSHELLS_KAPPA,
        [
            kappa_gl_drift_5_cat,
            kappa_gl_drift_8_cat,
            kappa_gl_drift_10_cat,
            kappa_gl_drift_12_cat,
            kappa_gl_drift_16_cat,
            kappa_gl_drift_20_cat,
            kappa_gl_drift_25_cat,
            kappa_gl_drift_30_cat,
            kappa_gl_drift_40_cat,
        ],
    )
}
kappa_gl_no = {
    n: c.field[0]
    for n, c in zip(
        NSHELLS_KAPPA,
        [
            kappa_gl_nodrift_5_cat,
            kappa_gl_nodrift_8_cat,
            kappa_gl_nodrift_10_cat,
            kappa_gl_nodrift_12_cat,
            kappa_gl_nodrift_16_cat,
            kappa_gl_nodrift_20_cat,
            kappa_gl_nodrift_25_cat,
            kappa_gl_nodrift_30_cat,
            kappa_gl_nodrift_40_cat,
        ],
    )
}

# scale-factor spacing (05b), midpoint, drift only (fig14-fig18 spacing comparison).
kappa_5b_dr = {
    12: kappa_5b_drift_12_cat.field[0],
    20: kappa_5b_drift_20_cat.field[0],
    25: kappa_5b_drift_25_cat.field[0],
    30: kappa_5b_drift_30_cat.field[0],
    40: kappa_5b_drift_40_cat.field[0],
}


# =============================================================================
# fig01 — local illustration: one particle cloud, three redshift-assignment colourings (equal volume)
# =============================================================================
def fig01_illustration():
    """Small local sim in the real 5 Gpc/h box: paint particles across the lightcone radius and colour each by
    the redshift it is assigned under (a) a 10-shell equal-volume freeze, (b) the drift's smooth z(r), (c) a
    30-shell equal-volume freeze — banded against the *real* equal-volume shell edges of the 05c runs."""
    key = jax.random.PRNGKey(7)
    mesh_size, box_size, nside, n_steps = (256, 256, 256), (5000.0, 5000.0, 5000.0), 256, 10
    R0, R1 = 80.0, 2450.0  # radial window, inside the box half-width (2500 Mpc/h, no tiling)
    cosmo_ill = jc.Planck18()  # a fresh cosmology for the illustration IC (the parquet-loaded one trips IC interp)

    def a_of_chi(chi):
        return np.asarray(jc.background.a_of_chi(cosmo, jnp.atleast_1d(jnp.asarray(chi))))

    def z_of(chi):
        return 1.0 / a_of_chi(chi) - 1.0

    ic = jfli.gaussian_initial_conditions(key, mesh_size, box_size, cosmo=cosmo_ill, nside=nside)
    dx, p = jfli.lpt(cosmo_ill, ic, ts=0.1, order=1)
    part = jfli.PaintingOptions(target="particles")
    pf = jfli.nbody(
        cosmo_ill,
        dx,
        p,
        solver=jfli.DoubleKickDrift(interp_kernel=jfli.NoInterp(painting=part), t0=0.1, t1=1.0, n_steps=n_steps),
        ts=jnp.array([a_of_chi(0.5 * (R0 + R1))[0]]),
        density_widths=jnp.array([R1 - R0]),
        shell_spacing="comoving",
        min_width=1.0,
    )
    obs = 0.5 * np.array(box_size)
    pos = np.asarray(pf.to(jfli.PositionUnit.MPC_H).array).reshape(-1, 3) - obs
    r = np.linalg.norm(pos, axis=1)
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    sel = (np.abs(pos[:, 2]) < 120.0) & (phi > 0.1) & (phi < 1.3) & (r > R0) & (r < R1)
    idx = np.where(sel)[0]
    rng = np.random.default_rng(0)
    if idx.size > 40000:
        idx = rng.choice(idx, 40000, replace=False)
    x, y, rr = pos[idx, 0], pos[idx, 1], r[idx]

    def banded_z(centers, widths):
        """Assign each particle the redshift of the equal-volume shell (centre) whose [near, far] it falls in."""
        edges = np.append(centers - 0.5 * widths, centers[-1] + 0.5 * widths[-1])
        b = np.clip(np.digitize(rr, edges) - 1, 0, len(centers) - 1)
        return z_of(centers)[b]

    panels = [
        ("10 shells, no drift", banded_z(chi10, w10)),
        ("10 shells, with drift", z_of(rr)),
        ("30 shells, no drift", banded_z(chi30, w30)),
    ]
    vmin, vmax = float(z_of(R0)[0]), float(z_of(R1)[0])  # z grows with distance: near R0 -> min, far R1 -> max
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), sharex=True, sharey=True)
    sc = None
    for ax, (title, zc) in zip(axes, panels):
        sc = ax.scatter(x, y, c=zc, s=2, cmap="turbo", vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("x [Mpc/h]")
        ax.set_aspect("equal")
    axes[0].set_ylabel("y [Mpc/h]")
    cbar = fig.colorbar(sc, ax=axes, fraction=0.018, pad=0.01)
    cbar.set_label("assigned redshift z")
    savefig(ASSETS / "fig01-redshift-assignment", fig)


# =============================================================================
# fig02 — density C_ell: 10-shell drift / no-drift vs a 40-shell continuous-lightcone reference
# =============================================================================
# Each column is a near / mid / far radial region: a group of consecutive 10-shell shells whose extent lines up
# (to within half a floored 40-shell, ~30 Mpc/h) with whole 40-shell edges, so the reference sums WHOLE 40-shell
# thin shells — no fractional weighting. Equal volume makes "near" a single fat inner ball and "far" a couple of
# thin shells. 10-shell {0}=[0,1160]≈40 {0..17}; {4,5,6,7}=[1842,2321]≈40 {29..36}; {8,9}=[2321,2500]≈40 {37,38,39}.
COLUMNS = [
    ("near", [0]),
    ("mid", [4, 5, 6, 7]),
    ("far", [8, 9]),
]


def _region_cl(run_dir, indices, lo, hi):
    """Sum whole nside-2048 shells `indices` of run `run_dir` into one [lo, hi] slab in COUNTS (per-pixel
    volume cancels on conversion), then overdensity -> angular_cl. Returns the PowerSpectrum (bin it at plot
    time). Used for both the 10-shell measurement and the 40-shell continuous-lightcone reference."""
    files = [f"{root}/05-spacing-n-stepping/05c-equal-volume/density/{run_dir}/shell_{i:04d}.parquet" for i in indices]
    maps = [Catalog.from_dataset(load_dataset("parquet", data_files=f, split="train")).field[0] for f in files]
    counts = sum(np.asarray(m.to(jfli.DensityUnit.COUNTS).array) for m in maps)
    slab = (
        maps[0]
        .to(jfli.DensityUnit.COUNTS)
        .replace(
            array=jnp.asarray(counts),
            comoving_centers=jnp.asarray(0.5 * (lo + hi)),
            density_width=jnp.asarray(hi - lo),
        )
    )
    return slab.to(jfli.DensityUnit.OVERDENSITY).angular_cl(method="healpy", lmax=LMAX)


def fig02_density_shells():
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16.5, 6.4), gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
    for col, (label, sh) in enumerate(COLUMNS):
        lo, hi = float(edges10[0, sh[0]]), float(edges10[1, sh[-1]])
        members40 = np.where((chi40 >= lo) & (chi40 <= hi))[0]  # whole 40-shells whose centre is inside the region
        ref_b_ps = _region_cl("exp5c_nodrift_40", members40, lo, hi).bin(nlb=32, lmin=2)
        no_b_ps = _region_cl("exp5c_nodrift_10", sh, lo, hi).bin(nlb=32, lmin=2)
        dr_b_ps = _region_cl("exp5c_drift_10", sh, lo, hi).bin(nlb=32, lmin=2)
        bc = np.asarray(ref_b_ps.wavenumber)
        dl = bc * (bc + 1) / (2 * np.pi)
        ref_b = np.asarray(ref_b_ps.array).reshape(-1)
        no_b = np.asarray(no_b_ps.array).reshape(-1)
        dr_b = np.asarray(dr_b_ps.array).reshape(-1)
        ax_s, ax_r = axes[0, col], axes[1, col]
        ax_s.loglog(bc, dl * ref_b, color="k", lw=1.6, label="40-shell reference")
        ax_s.loglog(bc, dl * no_b, color="tab:red", lw=1.5, label="10-shell, no drift")
        ax_s.loglog(bc, dl * dr_b, color="tab:blue", lw=1.5, label="10-shell, with drift")
        ax_s.set_title(rf"{label} region:  $\chi \in [{lo:.0f},\,{hi:.0f}]$ Mpc/h", fontsize=11)
        ax_s.grid(True, which="both", ls=":", alpha=0.4)
        # quantify the frozen-epoch bias each run carries vs the reference (largest for the fat inner region)
        inb = (bc >= 50) & (bc <= 800)
        dev_no = np.nanmedian(no_b[inb] / ref_b[inb]) - 1
        dev_dr = np.nanmedian(dr_b[inb] / ref_b[inb]) - 1
        ax_s.text(
            0.04,
            0.05,
            f"median bias ($\\ell\\in[50,800]$):\nno drift: {dev_no * 100:+.2f}\\%\nwith drift: {dev_dr * 100:+.2f}\\%",
            transform=ax_s.transAxes,
            fontsize=8.5,
            va="bottom",
            ha="left",
            family="monospace",
        )
        if col == 0:
            ax_s.set_ylabel(r"$\ell(\ell+1)\,C_\ell/2\pi$")
        ax_r.axhspan(-0.05, 0.05, color="0.7", alpha=0.3)
        ax_r.axhline(0.0, color="0.4", ls="--", lw=0.9)
        ax_r.semilogx(bc, no_b / ref_b - 1.0, color="tab:red", lw=1.4)
        ax_r.semilogx(bc, dr_b / ref_b - 1.0, color="tab:blue", lw=1.4)
        ax_r.set_ylim(-0.05, 0.05)
        ax_r.set_xlabel(r"multipole $\ell$")
        ax_r.grid(True, which="both", ls=":", alpha=0.4)
        if col == 0:
            ax_r.set_ylabel("meas / 40-shell - 1")
    handles = [
        Line2D([], [], color="k", lw=1.8, label="40-shell reference (continuous lightcone)"),
        Line2D([], [], color="tab:red", lw=1.6, label="10-shell, no drift"),
        Line2D([], [], color="tab:blue", lw=1.6, label="10-shell, with drift"),
        Line2D([], [], color="0.7", lw=6, alpha=0.5, label=r"$\pm5\%$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    savefig(ASSETS / "fig02-density-shells", fig)


# =============================================================================
# fig03-fig08 — per-shell density C_ell census vs Limber theory (drift + no-drift), one subplot per shell
# =============================================================================
def _draw_census(container, spec_no, spec_dr, nrows, ncols, *, top=0.9, bottom=0.08):
    """Draw one run's per-shell density census into `container` (a Figure or SubFigure): for every shell a
    log-log C_ell panel (Limber theory dashed, no-drift red, with-drift blue) over a measured/theory ratio
    strip. Theory is the comoving-volume Limber number-counts prediction x pixwin^2(nside=2048); drift and
    no-drift share the shell geometry, so it is computed once from the no-drift run. The shot noise in the
    measurement (absent from theory) lifts the ratio at high ell, so read the red<->blue gap (shared shot
    noise cancels between the two runs) as the drift's frozen-epoch correction, not the distance from 0."""
    pw2 = hp.pixwin(2048, lmax=LMAX) ** 2
    theory_b = (compute_theory_cl_for_density(cosmo, spec_no, jnp.arange(LMAX + 1)) * pw2).bin(nlb=32, lmin=2)
    no_b_all = spec_no.bin(nlb=32, lmin=2)
    dr_b_all = spec_dr.bin(nlb=32, lmin=2)
    bc = np.asarray(no_b_all.wavenumber)
    dl = bc * (bc + 1) / (2 * np.pi)
    theory = np.asarray(theory_b.array)
    no = np.asarray(no_b_all.array)
    dr = np.asarray(dr_b_all.array)
    chi = np.asarray(spec_no.comoving_centers)
    dx = BOX / MESH
    gs = container.add_gridspec(
        2 * nrows,
        ncols,
        height_ratios=[3, 1] * nrows,
        hspace=0.45,
        wspace=0.32,
        left=0.06,
        right=0.99,
        top=top,
        bottom=bottom,
    )
    for i in range(no.shape[0]):
        r, c = divmod(i, ncols)
        ax_s = container.add_subplot(gs[2 * r, c])
        ax_r = container.add_subplot(gs[2 * r + 1, c], sharex=ax_s)
        th_b, no_b, dr_b = theory[i], no[i], dr[i]
        ax_s.loglog(bc, dl * th_b, "k--", lw=1.0)
        ax_s.loglog(bc, dl * no_b, color="tab:red", lw=1.0)
        ax_s.loglog(bc, dl * dr_b, color="tab:blue", lw=1.0)
        ax_r.axhspan(-0.05, 0.05, color="0.7", alpha=0.3)
        ax_r.axhline(0.0, color="0.4", ls="--", lw=0.8)
        ax_r.semilogx(bc, no_b / th_b - 1.0, color="tab:red", lw=1.0)
        ax_r.semilogx(bc, dr_b / th_b - 1.0, color="tab:blue", lw=1.0)
        ax_r.set_ylim(*CENSUS_RATIO_YLIM)
        lmax_sh = np.pi * chi[i] / dx  # PM Nyquist; beyond it the comparison is resolution-limited
        for ax in (ax_s, ax_r):
            ax.axvline(lmax_sh, color="0.6", ls=":", lw=0.8)
            ax.grid(True, which="both", ls=":", alpha=0.35)
            ax.tick_params(labelsize=7)
        ax_s.set_title(rf"$\chi={chi[i]:.0f}$", fontsize=8)
        ax_s.tick_params(labelbottom=False)
        if c == 0:
            ax_s.set_ylabel(r"$\ell(\ell+1)\,C_\ell/2\pi$", fontsize=8)
            ax_r.set_ylabel("meas/th - 1", fontsize=7)
        if r == nrows - 1:
            ax_r.set_xlabel(r"$\ell$", fontsize=8)


def _census_legend(target, **kwargs):
    handles = [
        Line2D([], [], color="k", ls="--", lw=1.4, label=r"Limber theory $\times\,w_\ell^2$"),
        Line2D([], [], color="tab:red", lw=1.6, label="no drift"),
        Line2D([], [], color="tab:blue", lw=1.6, label="with drift"),
        Line2D([], [], color="0.6", ls=":", lw=1.2, label=r"$\ell_{\max}\approx\pi\chi/\mathrm{d}x$ (PM Nyquist)"),
        Line2D([], [], color="0.7", lw=6, alpha=0.5, label=r"$\pm5\%$"),
    ]
    target.legend(handles=handles, ncol=5, fontsize=9, frameon=False, **kwargs)


def fig03_density_census_small():
    """The three small runs (5 / 8 / 10 shells) stacked as one figure of sub-blocks."""
    fig = plt.figure(figsize=(13.0, 14.0))
    subs = fig.subfigures(4, 1, height_ratios=[0.18, 1, 2, 2], hspace=0.05)
    _census_legend(subs[0], loc="center")  # dedicated legend strip on top
    _draw_census(subs[1], nodrift_5, drift_5, 1, 5, top=0.82, bottom=0.14)
    _draw_census(subs[2], nodrift_8, drift_8, 2, 4, top=0.88, bottom=0.09)
    _draw_census(subs[3], nodrift_10, drift_10, 2, 5, top=0.88, bottom=0.09)
    savefig(ASSETS / "fig03-density-census-small", fig)


def density_census(spec_no, spec_dr, nrows, ncols, stem):
    """One run's census on an nrows x ncols grid (one shell per cell)."""
    fig = plt.figure(figsize=(2.5 * ncols, 2.9 * nrows))
    _draw_census(fig, spec_no, spec_dr, nrows, ncols, top=0.9, bottom=0.06)
    _census_legend(fig, loc="upper center", bbox_to_anchor=(0.5, 0.99))
    savefig(ASSETS / stem, fig)


# =============================================================================
# fig09 — Born shell windows: midpoint vs Gauss-Legendre on the equal-volume (5c) and scale-factor (5b) geometry
# =============================================================================
def fig09_born_windows():
    """The Born lensing quadrature made visible, comparing the equal-volume (05c) and scale-factor (05b) 10-shell
    geometries side by side. The black curve is the exact lensing kernel w(chi) = chi (1+z)(1 - chi/chi_s) for a
    single source at z_s = 1.2; the grey area under it is the exact integral. Each shell is a box whose AREA is
    its Born weight (box height = weight / width): colour encodes the quadrature — midpoint vs the exact
    Gauss-Legendre integral (Simpson is identical to Gauss-Legendre for these smooth shells, so only the two
    distinct windows are drawn) — and line style the spacing (solid = equal-volume, dashed = scale-factor). The
    exact boxes tile the shaded kernel; the midpoint boxes (markers = kernel at each shell centre) poke far above
    it on equal-volume's fat inner shell while scale-factor's thin shells stay close — the whole equal-volume
    Born problem. The legend reports each scheme's summed weight and its % bias vs Gauss-Legendre; this is the
    first of the two figures plot_born_windows() returns."""
    # plot_born_windows labels carry Σ / % / en-dash; matplotlib snapshots text.usetex at Text creation, so build
    # the figure with usetex OFF (the rest of the experiment runs usetex ON via set_style).
    with plt.rc_context({"text.usetex": False}):
        fig_w, fig_r = plot_born_windows(
            get_stage3_nz_shear()[:3],
            cosmo=cosmo,
            comoving_centers={"5c": chi10, "5b": chi_5b},
            density_width={"5c": w10, "5b": w_5b},
            z_kernel=1.2,
            quadrature=["midpoint", "gauss_legendre"],
            legend_loc="outside",
        )
    plt.close(fig_r)  # 05c shows only the window plot; the per-bin scheme/GL ratio is the library's 2nd output
    savefig(ASSETS / "fig09-born-windows", fig_w)


# =============================================================================
# fig10 / fig11 / fig12 — Born convergence C_ell per bin vs shell count, ratioed to the own 40-shell run, one
# figure per quadrature scheme (midpoint / composite Simpson / Gauss-Legendre)
# =============================================================================
def lensing_vs_ref40(kappa_no, kappa_dr, stem):
    """Per source bin, the Born convergence C_ell for every shell count ratioed to its own 40-shell run, under one
    quadrature. Three tomographic bins (Stage-3 [:3]) are stacked as row-pairs; no-drift and with-drift fill the
    two columns, and the ratio window adapts per bin. Under the midpoint rule the low-z bin 1 (whose kernel weights
    the fat inner shell) overshoots its own 40-shell run and converges slowly; composite Simpson and Gauss-Legendre
    integrate the kernel across each shell and collapse that overshoot, so the coarse runs track the 40-shell
    reference. The drift lowers the residual (compare the two columns)."""
    counts = [n for n in NSHELLS_KAPPA if n != 40]
    colors = {n: c for n, c in zip(counts, cm.viridis(np.linspace(0.0, 0.88, len(counts))))}
    nodrift_b = {n: kappa_no[n].bin(nlb=32, lmin=2) for n in NSHELLS_KAPPA}
    drift_b = {n: kappa_dr[n].bin(nlb=32, lmin=2) for n in NSHELLS_KAPPA}
    bc = np.asarray(nodrift_b[40].wavenumber)
    dl = bc * (bc + 1) / (2 * np.pi)
    nbins = np.asarray(nodrift_b[40].array).shape[0]
    fig, axes = plt.subplots(
        2 * nbins, 2, figsize=(12.0, 13.5), gridspec_kw={"height_ratios": [3, 1] * nbins}, sharex="col"
    )
    for b in range(nbins):
        # data-driven ratio window shared by both columns of this source bin (bin 1 swings far more than bin 3)
        rr = []
        for kb in (nodrift_b, drift_b):
            ref_b = np.asarray(kb[40].array)[b]
            for n in counts:
                c_b = np.asarray(kb[n].array)[b]
                rr.append((c_b / ref_b - 1.0)[bc >= 20])
        rr = np.concatenate(rr)
        lo = min(-0.1, max(-0.6, 1.05 * float(np.nanmin(rr))))
        hi = max(0.1, min(2.2, 1.05 * float(np.nanmax(rr))))
        for col, (label, kb) in enumerate([("no drift", nodrift_b), ("with drift", drift_b)]):
            ax_s, ax_r = axes[2 * b, col], axes[2 * b + 1, col]
            ref_b = np.asarray(kb[40].array)[b]
            ax_s.loglog(bc, dl * ref_b, color="k", lw=1.8)
            for n in counts:
                c_b = np.asarray(kb[n].array)[b]
                ax_s.loglog(bc, dl * c_b, color=colors[n], lw=1.2)
            ax_s.set_title(f"bin {b + 1} — {label}", fontsize=11)
            ax_s.grid(True, which="both", ls=":", alpha=0.4)
            ax_s.tick_params(labelbottom=False)
            if col == 0:
                ax_s.set_ylabel(r"$\ell(\ell+1)\,C_\ell^{\kappa\kappa}/2\pi$")
            ax_r.axhspan(-0.03, 0.03, color="0.7", alpha=0.3)
            ax_r.axhline(0.0, color="0.4", ls="--", lw=0.9)
            for n in counts:
                c_b = np.asarray(kb[n].array)[b]
                ax_r.semilogx(bc, c_b / ref_b - 1.0, color=colors[n], lw=1.2)
            ax_r.set_ylim(lo, hi)  # per-bin window (bin 1 swings far more than bin 3)
            ax_r.grid(True, which="both", ls=":", alpha=0.4)
            if col == 0:
                ax_r.set_ylabel("meas / 40 sh. - 1")
            if b == nbins - 1:
                ax_r.set_xlabel(r"multipole $\ell$")
            else:
                ax_r.tick_params(labelbottom=False)
    handles = [Line2D([], [], color=colors[n], lw=1.6, label=f"{n} shells") for n in counts]
    handles += [
        Line2D([], [], color="k", lw=1.8, label="40 shells (reference)"),
        Line2D([], [], color="0.7", lw=6, alpha=0.5, label=r"$\pm3\%$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    savefig(ASSETS / stem, fig)


# =============================================================================
# fig13 — Gauss-Legendre Born convergence per tomographic bin ratioed to the Limber weak-lensing theory
# =============================================================================
def fig13_lensing_theory():
    """The Gauss-Legendre (exact-kernel) Born convergence for every shell count, ratioed to the Limber
    weak-lensing theory (the same Stage-3 [:3] source bins, x pixwin^2(2048)). With the exact quadrature the
    midpoint overshoot of fig10 is gone: every count tracks theory at large scales and falls below at small
    scales on the single common PM-resolution roll-off — the same ceiling the scale-factor 05b runs hit. The
    no-drift and with-drift columns now agree, since the radial projection no longer carries a quadrature bias."""
    counts = NSHELLS_KAPPA
    colors = {n: c for n, c in zip(counts, cm.viridis(np.linspace(0.0, 0.9, len(counts))))}
    pw2 = hp.pixwin(2048, lmax=LMAX) ** 2
    theory_b = (compute_theory_cl(cosmo, jnp.arange(LMAX + 1), get_stage3_nz_shear()[:3]) * pw2).bin(nlb=32, lmin=2)
    theory = np.asarray(theory_b.array)
    bc = np.asarray(theory_b.wavenumber)
    dl = bc * (bc + 1) / (2 * np.pi)
    nodrift_b = {n: kappa_gl_no[n].bin(nlb=32, lmin=2) for n in NSHELLS_KAPPA}
    drift_b = {n: kappa_gl_dr[n].bin(nlb=32, lmin=2) for n in NSHELLS_KAPPA}
    nbins = theory.shape[0]
    fig, axes = plt.subplots(
        2 * nbins, 2, figsize=(12.0, 13.5), gridspec_kw={"height_ratios": [3, 1] * nbins}, sharex="col"
    )
    for b in range(nbins):
        th_b = theory[b]
        # data-driven ratio window shared by both columns of this source bin
        rr = np.concatenate(
            [(np.asarray(kb[n].array)[b] / th_b - 1.0)[bc >= 20] for kb in (nodrift_b, drift_b) for n in counts]
        )
        lo = min(-0.1, max(-0.8, 1.05 * float(np.nanmin(rr))))
        hi = max(0.1, min(1.0, 1.05 * float(np.nanmax(rr))))
        for col, (label, kb) in enumerate([("no drift", nodrift_b), ("with drift", drift_b)]):
            ax_s, ax_r = axes[2 * b, col], axes[2 * b + 1, col]
            ax_s.loglog(bc, dl * th_b, color="k", ls="--", lw=1.8)
            for n in counts:
                c_b = np.asarray(kb[n].array)[b]
                ax_s.loglog(bc, dl * c_b, color=colors[n], lw=1.2)
            ax_s.set_title(f"bin {b + 1} — {label}", fontsize=11)
            ax_s.grid(True, which="both", ls=":", alpha=0.4)
            ax_s.tick_params(labelbottom=False)
            if col == 0:
                ax_s.set_ylabel(r"$\ell(\ell+1)\,C_\ell^{\kappa\kappa}/2\pi$")
            ax_r.axhspan(-0.05, 0.05, color="0.7", alpha=0.3)
            ax_r.axhline(0.0, color="0.4", ls="--", lw=0.9)
            for n in counts:
                c_b = np.asarray(kb[n].array)[b]
                ax_r.semilogx(bc, c_b / th_b - 1.0, color=colors[n], lw=1.2)
            ax_r.set_ylim(lo, hi)  # per-bin window
            ax_r.grid(True, which="both", ls=":", alpha=0.4)
            if col == 0:
                ax_r.set_ylabel("meas / theory - 1")
            if b == nbins - 1:
                ax_r.set_xlabel(r"multipole $\ell$")
            else:
                ax_r.tick_params(labelbottom=False)
    handles = [Line2D([], [], color=colors[n], lw=1.6, label=f"{n} shells") for n in counts]
    handles += [
        Line2D(
            [],
            [],
            color="k",
            ls="--",
            lw=1.8,
            label=r"Limber weak-lensing theory (Stage-3 bins 1--3) $\times\,w_\ell^2$",
        ),
        Line2D([], [], color="0.7", lw=6, alpha=0.5, label=r"$\pm5\%$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    savefig(ASSETS / "fig13-lensing-theory", fig)


# =============================================================================
# fig14-fig18 — equal-volume vs scale-factor (05b) spacing at Gauss-Legendre (drift), one per shell count
# =============================================================================
def lensing_spacing(n_shells, stem):
    """Equal-volume vs scale-factor spacing at a fixed Gauss-Legendre quadrature (drift, `n_shells` shells), with
    all three tomographic bins overlaid (Stage-3 [:3], coloured by bin). The top panel is the D_ell power
    ell(ell+1)/2pi C_ell — solid = equal-volume, dashed = scale-factor (05b), dotted = Limber weak-lensing theory
    (x pixwin^2(2048)) — and the bottom panel the fractional residual C_ell/theory - 1 for each spacing. Spectra
    are bandpower-binned in linear bins of 32 multipoles. Both spacings sit on theory around ell~50-100 and roll
    off together below it (the PM-resolution transfer); the solid-dashed gap is the spacing difference —
    equal-volume typically carries a ~10-25% small-scale excess (ell~500-1000) from its fat-shell geometry, and
    bin 1 sits lowest (its low-z sources inside the fat inner ball). 05b was only run with midpoint, but for its
    thin scale-factor shells midpoint = Gauss-Legendre to <0.2% on the total lensing weight, so its midpoint
    spectra stand in."""
    pw2 = hp.pixwin(2048, lmax=LMAX) ** 2
    theory_b = (compute_theory_cl(cosmo, jnp.arange(LMAX + 1), get_stage3_nz_shear()[:3]) * pw2).bin(nlb=32, lmin=2)
    theory = np.asarray(theory_b.array)
    bc = np.asarray(theory_b.wavenumber)
    dl = bc * (bc + 1) / (2 * np.pi)
    ev_b = np.asarray(kappa_gl_dr[n_shells].bin(nlb=32, lmin=2).array)
    sf_b = np.asarray(kappa_5b_dr[n_shells].bin(nlb=32, lmin=2).array)
    bincol = {0: "#4C72B0", 1: "#DD8452", 2: "#C44E52"}
    fig, (ax_s, ax_r) = plt.subplots(2, 1, figsize=(7.4, 6.8), sharex=True, gridspec_kw={"height_ratios": [2.7, 1]})
    for b in range(3):
        ev, sf, th = ev_b[b], sf_b[b], theory[b]
        ax_s.loglog(bc, dl * ev, color=bincol[b], ls="-", lw=1.6)
        ax_s.loglog(bc, dl * sf, color=bincol[b], ls="--", lw=1.6)
        ax_s.loglog(bc, dl * th, color=bincol[b], ls=":", lw=1.3)
        ax_r.semilogx(bc, ev / th - 1.0, color=bincol[b], ls="-", lw=1.6)
        ax_r.semilogx(bc, sf / th - 1.0, color=bincol[b], ls="--", lw=1.6)
    ax_s.set_ylabel(r"$\ell(\ell+1)\,C_\ell^{\kappa\kappa}/2\pi$")
    ax_s.grid(True, which="both", ls=":", alpha=0.4)
    ax_r.axhspan(-0.05, 0.05, color="0.7", alpha=0.25)
    ax_r.axhline(0.0, color="0.4", ls="--", lw=0.9)
    ax_r.set_ylim(-0.6, 0.15)
    ax_r.set_xlim(15, 1300)  # below ell~15 is cosmic-variance-noisy; above ~1300 both C_ell hit the PM-resolution floor
    ax_r.set_ylabel(r"$C_\ell\,/\,C_\ell^{\mathrm{th}} - 1$")
    ax_r.set_xlabel(r"multipole $\ell$")
    ax_r.grid(True, which="both", ls=":", alpha=0.4)
    handles = [Line2D([], [], color=bincol[b], lw=2.0, label=f"bin {b + 1}") for b in range(3)]
    handles += [
        Line2D([], [], color="0.3", ls="-", lw=1.8, label=f"Equal-volume spacing (N={n_shells})"),
        Line2D([], [], color="0.3", ls="--", lw=1.8, label=f"Uniform scale-factor spacing (N={n_shells})"),
        Line2D([], [], color="0.3", ls=":", lw=1.5, label="Limber theory"),
    ]
    ax_s.legend(handles=handles, loc="lower center", fontsize=8.5, ncol=2, frameon=False)
    fig.tight_layout()
    savefig(ASSETS / stem, fig)


def main():
    set_style()
    fig01_illustration()
    fig02_density_shells()
    fig03_density_census_small()
    density_census(nodrift_16, drift_16, 4, 4, "fig04-density-census-16")
    density_census(nodrift_20, drift_20, 4, 5, "fig05-density-census-20")
    density_census(nodrift_25, drift_25, 5, 5, "fig06-density-census-25")
    density_census(nodrift_30, drift_30, 5, 6, "fig07-density-census-30")
    density_census(nodrift_40, drift_40, 5, 8, "fig08-density-census-40")
    fig09_born_windows()
    lensing_vs_ref40(kappa_mid_no, kappa_mid_dr, "fig10-lensing-midpoint")
    lensing_vs_ref40(kappa_simp_no, kappa_simp_dr, "fig11-lensing-simpson")
    lensing_vs_ref40(kappa_gl_no, kappa_gl_dr, "fig12-lensing-gauss-legendre")
    fig13_lensing_theory()
    lensing_spacing(10, "fig14-lensing-spacing-10")
    lensing_spacing(12, "fig15-lensing-spacing-12")
    lensing_spacing(16, "fig16-lensing-spacing-16")
    lensing_spacing(20, "fig17-lensing-spacing-20")
    lensing_spacing(25, "fig18-lensing-spacing-25")
    lensing_spacing(30, "fig18-lensing-spacing-30")
    lensing_spacing(40, "fig18-lensing-spacing-40")
    print(f"assets written to {ASSETS}")


if __name__ == "__main__":
    main()
