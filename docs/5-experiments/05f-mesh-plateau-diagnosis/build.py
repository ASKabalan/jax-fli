"""Experiment 05f — mesh plateau diagnosis: mesh × solver × shell spacing × resolution cut, 100 steps.

Renders the SVG figures for ``docs/5-experiments/05f-mesh-plateau-diagnosis/README.md``. Every run shares the 05e
production physics (5 Gpc/h box, 20 shells, nside 2048, drift on the lightcone, 3-bin Stage-3 Born under
Gauss–Legendre) and moves four knobs: the PM mesh (512³ … 4096³), the solver (bf = BullFrog with D-stepping,
kdk = DoubleKickDrift with a-stepping), the shell spacing (uniform in a, plain equal volume, and three
equal-volume variants capped at 150 Mpc/h width starting at r_min = 0 / 150 / 300 Mpc/h), and the Born
resolution cut (off at nside 2048, on at nside 512).

  fig01  density C_ell / theory at ell ~ 200 against mesh, one line per shell coloured by its comoving centre
  fig02  the same at ell ~ 500
  fig03  Born kappa C_ell / theory at ell ~ 200 against mesh, no resolution cut
  fig04  the same with the resolution cut
  fig05  kappa with the cut over kappa without it, against ell, one line per mesh
  fig06  the candidate configurations against the CosmoGrid cosmo_172798 Born reference

Run from the repo root (CPU is fine):
    JAX_PLATFORMS=cpu uv run --no-sync python docs/5-experiments/05f-mesh-plateau-diagnosis/build.py
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
import pandas as pd
from datasets import load_dataset
from huggingface_hub import snapshot_download
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter

from jax_fli import compute_theory_cl, compute_theory_cl_for_density
from jax_fli.io import Catalog, get_stage3_nz_shear

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _exputils import savefig, set_style  # noqa: E402

ASSETS = HERE / "assets"
REPO = "ASKabalan/jax-fli-experiments"

LMAX = 1500  # the published spectra stop at ell 1500

MESHES = (512, 1024, 2048, 2560, 3072, 4096)
SOLVERS = ("bf", "kdk")
SPACINGS = ("a", "equal_vol", "eqvolc_w150_r0", "eqvolc_w150_r150", "eqvolc_w150_r300")

SOLVER_LABEL = {"bf": r"bf (BullFrog, $D$-stepping)", "kdk": r"kdk (DoubleKickDrift, $a$-stepping)"}
SPACING_LABEL = {
    "a": r"uniform in $a$",
    "equal_vol": r"equal volume",
    "eqvolc_w150_r0": r"equal vol., $w \leq 150$, $r_{\min} = 0$",
    "eqvolc_w150_r150": r"equal vol., $w \leq 150$, $r_{\min} = 150$",
    "eqvolc_w150_r300": r"equal vol., $w \leq 150$, $r_{\min} = 300$",
}

# Cosmic-variance band (experiment 00): the 200 fiducial CosmoGrid permutations, the empirical fractional CV of
# the bandpowers, worst of the three plotted bins, times sqrt(2) (our run and one CosmoGrid map are two
# independent single realisations).
FIDUCIAL_KAPPA = "00-cosmogrid/fiducial_kappa_spectra"

# The CosmoGrid Stage-3 Born reference at grid point 172798, the grid cosmology closest to the run cosmology.
CG_172798 = "00-cosmogrid/cosmo_172798/kappa_spectra/spectra_cosmogrid_sample_kappa.parquet"

PERF_PM = "05-spacing-n-stepping/05f-mesh/perf/perf_pm.csv"

root = snapshot_download(
    REPO,
    repo_type="dataset",
    allow_patterns=[
        "05-spacing-n-stepping/05f-mesh/*_spectra/*",
        PERF_PM,
        f"{FIDUCIAL_KAPPA}/cosmo_fiducial_part*.parquet",
        CG_172798,
    ],
)

DENS_M512_BF_A = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m512_bf_a.parquet"
DENS_M1024_BF_A = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m1024_bf_a.parquet"
DENS_M2048_BF_A = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m2048_bf_a.parquet"
DENS_M2560_BF_A = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m2560_bf_a.parquet"
DENS_M3072_BF_A = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m3072_bf_a.parquet"
DENS_M4096_BF_A = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m4096_bf_a.parquet"
DENS_M512_BF_EQVOL = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m512_bf_equal_vol.parquet"
DENS_M1024_BF_EQVOL = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m1024_bf_equal_vol.parquet"
DENS_M2048_BF_EQVOL = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m2048_bf_equal_vol.parquet"
DENS_M2560_BF_EQVOL = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m2560_bf_equal_vol.parquet"
DENS_M3072_BF_EQVOL = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m3072_bf_equal_vol.parquet"
DENS_M4096_BF_EQVOL = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m4096_bf_equal_vol.parquet"
DENS_M512_BF_R0 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m512_bf_eqvolc_w150_r0.parquet"
DENS_M1024_BF_R0 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m1024_bf_eqvolc_w150_r0.parquet"
DENS_M2048_BF_R0 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m2048_bf_eqvolc_w150_r0.parquet"
DENS_M2560_BF_R0 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m2560_bf_eqvolc_w150_r0.parquet"
DENS_M3072_BF_R0 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m3072_bf_eqvolc_w150_r0.parquet"
DENS_M4096_BF_R0 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m4096_bf_eqvolc_w150_r0.parquet"
DENS_M512_BF_R150 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m512_bf_eqvolc_w150_r150.parquet"
DENS_M1024_BF_R150 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m1024_bf_eqvolc_w150_r150.parquet"
DENS_M2048_BF_R150 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m2048_bf_eqvolc_w150_r150.parquet"
DENS_M2560_BF_R150 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m2560_bf_eqvolc_w150_r150.parquet"
DENS_M3072_BF_R150 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m3072_bf_eqvolc_w150_r150.parquet"
DENS_M4096_BF_R150 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m4096_bf_eqvolc_w150_r150.parquet"
DENS_M512_BF_R300 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m512_bf_eqvolc_w150_r300.parquet"
DENS_M1024_BF_R300 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m1024_bf_eqvolc_w150_r300.parquet"
DENS_M2048_BF_R300 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m2048_bf_eqvolc_w150_r300.parquet"
DENS_M2560_BF_R300 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m2560_bf_eqvolc_w150_r300.parquet"
DENS_M3072_BF_R300 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m3072_bf_eqvolc_w150_r300.parquet"
DENS_M4096_BF_R300 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m4096_bf_eqvolc_w150_r300.parquet"
DENS_M512_KDK_A = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m512_kdk_a.parquet"
DENS_M1024_KDK_A = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m1024_kdk_a.parquet"
DENS_M2048_KDK_A = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m2048_kdk_a.parquet"
DENS_M2560_KDK_A = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m2560_kdk_a.parquet"
DENS_M3072_KDK_A = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m3072_kdk_a.parquet"
DENS_M4096_KDK_A = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m4096_kdk_a.parquet"
DENS_M512_KDK_EQVOL = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m512_kdk_equal_vol.parquet"
DENS_M1024_KDK_EQVOL = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m1024_kdk_equal_vol.parquet"
DENS_M2048_KDK_EQVOL = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m2048_kdk_equal_vol.parquet"
DENS_M2560_KDK_EQVOL = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m2560_kdk_equal_vol.parquet"
DENS_M3072_KDK_EQVOL = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m3072_kdk_equal_vol.parquet"
DENS_M4096_KDK_EQVOL = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m4096_kdk_equal_vol.parquet"
DENS_M512_KDK_R0 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m512_kdk_eqvolc_w150_r0.parquet"
DENS_M1024_KDK_R0 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m1024_kdk_eqvolc_w150_r0.parquet"
DENS_M2048_KDK_R0 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m2048_kdk_eqvolc_w150_r0.parquet"
DENS_M2560_KDK_R0 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m2560_kdk_eqvolc_w150_r0.parquet"
DENS_M3072_KDK_R0 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m3072_kdk_eqvolc_w150_r0.parquet"
DENS_M4096_KDK_R0 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m4096_kdk_eqvolc_w150_r0.parquet"
DENS_M512_KDK_R150 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m512_kdk_eqvolc_w150_r150.parquet"
DENS_M1024_KDK_R150 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m1024_kdk_eqvolc_w150_r150.parquet"
DENS_M2048_KDK_R150 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m2048_kdk_eqvolc_w150_r150.parquet"
DENS_M2560_KDK_R150 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m2560_kdk_eqvolc_w150_r150.parquet"
DENS_M3072_KDK_R150 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m3072_kdk_eqvolc_w150_r150.parquet"
DENS_M4096_KDK_R150 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m4096_kdk_eqvolc_w150_r150.parquet"
DENS_M512_KDK_R300 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m512_kdk_eqvolc_w150_r300.parquet"
DENS_M1024_KDK_R300 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m1024_kdk_eqvolc_w150_r300.parquet"
DENS_M2048_KDK_R300 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m2048_kdk_eqvolc_w150_r300.parquet"
DENS_M2560_KDK_R300 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m2560_kdk_eqvolc_w150_r300.parquet"
DENS_M3072_KDK_R300 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m3072_kdk_eqvolc_w150_r300.parquet"
DENS_M4096_KDK_R300 = "05-spacing-n-stepping/05f-mesh/density_spectra/spectra_exp5f_m4096_kdk_eqvolc_w150_r300.parquet"

KAPPA_M512_BF_A = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m512_bf_a.parquet"
KAPPA_M1024_BF_A = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m1024_bf_a.parquet"
KAPPA_M2048_BF_A = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m2048_bf_a.parquet"
KAPPA_M2560_BF_A = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m2560_bf_a.parquet"
KAPPA_M3072_BF_A = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m3072_bf_a.parquet"
KAPPA_M4096_BF_A = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m4096_bf_a.parquet"
KAPPA_M512_BF_EQVOL = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m512_bf_equal_vol.parquet"
KAPPA_M1024_BF_EQVOL = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m1024_bf_equal_vol.parquet"
KAPPA_M2048_BF_EQVOL = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m2048_bf_equal_vol.parquet"
KAPPA_M2560_BF_EQVOL = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m2560_bf_equal_vol.parquet"
KAPPA_M3072_BF_EQVOL = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m3072_bf_equal_vol.parquet"
KAPPA_M4096_BF_EQVOL = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m4096_bf_equal_vol.parquet"
KAPPA_M512_BF_R0 = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m512_bf_eqvolc_w150_r0.parquet"
KAPPA_M1024_BF_R0 = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m1024_bf_eqvolc_w150_r0.parquet"
KAPPA_M2048_BF_R0 = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m2048_bf_eqvolc_w150_r0.parquet"
KAPPA_M2560_BF_R0 = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m2560_bf_eqvolc_w150_r0.parquet"
KAPPA_M3072_BF_R0 = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m3072_bf_eqvolc_w150_r0.parquet"
KAPPA_M4096_BF_R0 = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m4096_bf_eqvolc_w150_r0.parquet"
KAPPA_M512_BF_R150 = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m512_bf_eqvolc_w150_r150.parquet"
KAPPA_M1024_BF_R150 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m1024_bf_eqvolc_w150_r150.parquet"
)
KAPPA_M2048_BF_R150 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m2048_bf_eqvolc_w150_r150.parquet"
)
KAPPA_M2560_BF_R150 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m2560_bf_eqvolc_w150_r150.parquet"
)
KAPPA_M3072_BF_R150 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m3072_bf_eqvolc_w150_r150.parquet"
)
KAPPA_M4096_BF_R150 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m4096_bf_eqvolc_w150_r150.parquet"
)
KAPPA_M512_BF_R300 = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m512_bf_eqvolc_w150_r300.parquet"
KAPPA_M1024_BF_R300 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m1024_bf_eqvolc_w150_r300.parquet"
)
KAPPA_M2048_BF_R300 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m2048_bf_eqvolc_w150_r300.parquet"
)
KAPPA_M2560_BF_R300 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m2560_bf_eqvolc_w150_r300.parquet"
)
KAPPA_M3072_BF_R300 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m3072_bf_eqvolc_w150_r300.parquet"
)
KAPPA_M4096_BF_R300 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m4096_bf_eqvolc_w150_r300.parquet"
)
KAPPA_M512_KDK_A = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m512_kdk_a.parquet"
KAPPA_M1024_KDK_A = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m1024_kdk_a.parquet"
KAPPA_M2048_KDK_A = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m2048_kdk_a.parquet"
KAPPA_M2560_KDK_A = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m2560_kdk_a.parquet"
KAPPA_M3072_KDK_A = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m3072_kdk_a.parquet"
KAPPA_M4096_KDK_A = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m4096_kdk_a.parquet"
KAPPA_M512_KDK_EQVOL = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m512_kdk_equal_vol.parquet"
KAPPA_M1024_KDK_EQVOL = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m1024_kdk_equal_vol.parquet"
KAPPA_M2048_KDK_EQVOL = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m2048_kdk_equal_vol.parquet"
KAPPA_M2560_KDK_EQVOL = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m2560_kdk_equal_vol.parquet"
KAPPA_M3072_KDK_EQVOL = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m3072_kdk_equal_vol.parquet"
KAPPA_M4096_KDK_EQVOL = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m4096_kdk_equal_vol.parquet"
KAPPA_M512_KDK_R0 = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m512_kdk_eqvolc_w150_r0.parquet"
KAPPA_M1024_KDK_R0 = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m1024_kdk_eqvolc_w150_r0.parquet"
KAPPA_M2048_KDK_R0 = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m2048_kdk_eqvolc_w150_r0.parquet"
KAPPA_M2560_KDK_R0 = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m2560_kdk_eqvolc_w150_r0.parquet"
KAPPA_M3072_KDK_R0 = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m3072_kdk_eqvolc_w150_r0.parquet"
KAPPA_M4096_KDK_R0 = "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m4096_kdk_eqvolc_w150_r0.parquet"
KAPPA_M512_KDK_R150 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m512_kdk_eqvolc_w150_r150.parquet"
)
KAPPA_M1024_KDK_R150 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m1024_kdk_eqvolc_w150_r150.parquet"
)
KAPPA_M2048_KDK_R150 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m2048_kdk_eqvolc_w150_r150.parquet"
)
KAPPA_M2560_KDK_R150 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m2560_kdk_eqvolc_w150_r150.parquet"
)
KAPPA_M3072_KDK_R150 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m3072_kdk_eqvolc_w150_r150.parquet"
)
KAPPA_M4096_KDK_R150 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m4096_kdk_eqvolc_w150_r150.parquet"
)
KAPPA_M512_KDK_R300 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m512_kdk_eqvolc_w150_r300.parquet"
)
KAPPA_M1024_KDK_R300 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m1024_kdk_eqvolc_w150_r300.parquet"
)
KAPPA_M2048_KDK_R300 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m2048_kdk_eqvolc_w150_r300.parquet"
)
KAPPA_M2560_KDK_R300 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m2560_kdk_eqvolc_w150_r300.parquet"
)
KAPPA_M3072_KDK_R300 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m3072_kdk_eqvolc_w150_r300.parquet"
)
KAPPA_M4096_KDK_R300 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_spectra/spectra_born_gl_m4096_kdk_eqvolc_w150_r300.parquet"
)

CUT_M512_BF_A = "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m512_bf_a.parquet"
CUT_M1024_BF_A = "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m1024_bf_a.parquet"
CUT_M2048_BF_A = "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m2048_bf_a.parquet"
CUT_M2560_BF_A = "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m2560_bf_a.parquet"
CUT_M3072_BF_A = "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m3072_bf_a.parquet"
CUT_M4096_BF_A = "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m4096_bf_a.parquet"
CUT_M512_BF_EQVOL = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m512_bf_equal_vol.parquet"
)
CUT_M1024_BF_EQVOL = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m1024_bf_equal_vol.parquet"
)
CUT_M2048_BF_EQVOL = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m2048_bf_equal_vol.parquet"
)
CUT_M2560_BF_EQVOL = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m2560_bf_equal_vol.parquet"
)
CUT_M3072_BF_EQVOL = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m3072_bf_equal_vol.parquet"
)
CUT_M4096_BF_EQVOL = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m4096_bf_equal_vol.parquet"
)
CUT_M512_BF_R0 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m512_bf_eqvolc_w150_r0.parquet"
)
CUT_M1024_BF_R0 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m1024_bf_eqvolc_w150_r0.parquet"
)
CUT_M2048_BF_R0 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m2048_bf_eqvolc_w150_r0.parquet"
)
CUT_M2560_BF_R0 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m2560_bf_eqvolc_w150_r0.parquet"
)
CUT_M3072_BF_R0 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m3072_bf_eqvolc_w150_r0.parquet"
)
CUT_M4096_BF_R0 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m4096_bf_eqvolc_w150_r0.parquet"
)
CUT_M512_BF_R150 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m512_bf_eqvolc_w150_r150.parquet"
)
CUT_M1024_BF_R150 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m1024_bf_eqvolc_w150_r150.parquet"
)
CUT_M2048_BF_R150 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m2048_bf_eqvolc_w150_r150.parquet"
)
CUT_M2560_BF_R150 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m2560_bf_eqvolc_w150_r150.parquet"
)
CUT_M3072_BF_R150 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m3072_bf_eqvolc_w150_r150.parquet"
)
CUT_M4096_BF_R150 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m4096_bf_eqvolc_w150_r150.parquet"
)
CUT_M512_BF_R300 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m512_bf_eqvolc_w150_r300.parquet"
)
CUT_M1024_BF_R300 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m1024_bf_eqvolc_w150_r300.parquet"
)
CUT_M2048_BF_R300 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m2048_bf_eqvolc_w150_r300.parquet"
)
CUT_M2560_BF_R300 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m2560_bf_eqvolc_w150_r300.parquet"
)
CUT_M3072_BF_R300 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m3072_bf_eqvolc_w150_r300.parquet"
)
CUT_M4096_BF_R300 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m4096_bf_eqvolc_w150_r300.parquet"
)
CUT_M512_KDK_A = "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m512_kdk_a.parquet"
CUT_M1024_KDK_A = "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m1024_kdk_a.parquet"
CUT_M2048_KDK_A = "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m2048_kdk_a.parquet"
CUT_M2560_KDK_A = "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m2560_kdk_a.parquet"
CUT_M3072_KDK_A = "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m3072_kdk_a.parquet"
CUT_M4096_KDK_A = "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m4096_kdk_a.parquet"
CUT_M512_KDK_EQVOL = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m512_kdk_equal_vol.parquet"
)
CUT_M1024_KDK_EQVOL = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m1024_kdk_equal_vol.parquet"
)
CUT_M2048_KDK_EQVOL = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m2048_kdk_equal_vol.parquet"
)
CUT_M2560_KDK_EQVOL = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m2560_kdk_equal_vol.parquet"
)
CUT_M3072_KDK_EQVOL = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m3072_kdk_equal_vol.parquet"
)
CUT_M4096_KDK_EQVOL = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m4096_kdk_equal_vol.parquet"
)
CUT_M512_KDK_R0 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m512_kdk_eqvolc_w150_r0.parquet"
)
CUT_M1024_KDK_R0 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m1024_kdk_eqvolc_w150_r0.parquet"
)
CUT_M2048_KDK_R0 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m2048_kdk_eqvolc_w150_r0.parquet"
)
CUT_M2560_KDK_R0 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m2560_kdk_eqvolc_w150_r0.parquet"
)
CUT_M3072_KDK_R0 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m3072_kdk_eqvolc_w150_r0.parquet"
)
CUT_M4096_KDK_R0 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m4096_kdk_eqvolc_w150_r0.parquet"
)
CUT_M512_KDK_R150 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m512_kdk_eqvolc_w150_r150.parquet"
)
CUT_M1024_KDK_R150 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m1024_kdk_eqvolc_w150_r150.parquet"
)
CUT_M2048_KDK_R150 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m2048_kdk_eqvolc_w150_r150.parquet"
)
CUT_M2560_KDK_R150 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m2560_kdk_eqvolc_w150_r150.parquet"
)
CUT_M3072_KDK_R150 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m3072_kdk_eqvolc_w150_r150.parquet"
)
CUT_M4096_KDK_R150 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m4096_kdk_eqvolc_w150_r150.parquet"
)
CUT_M512_KDK_R300 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m512_kdk_eqvolc_w150_r300.parquet"
)
CUT_M1024_KDK_R300 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m1024_kdk_eqvolc_w150_r300.parquet"
)
CUT_M2048_KDK_R300 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m2048_kdk_eqvolc_w150_r300.parquet"
)
CUT_M2560_KDK_R300 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m2560_kdk_eqvolc_w150_r300.parquet"
)
CUT_M3072_KDK_R300 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m3072_kdk_eqvolc_w150_r300.parquet"
)
CUT_M4096_KDK_R300 = (
    "05-spacing-n-stepping/05f-mesh/kappa_gl_rescut_spectra/spectra_born_gl_rescut_m4096_kdk_eqvolc_w150_r300.parquet"
)

dens_m512_bf_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DENS_M512_BF_A}", split="train"))
dens_m1024_bf_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DENS_M1024_BF_A}", split="train"))
dens_m2048_bf_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DENS_M2048_BF_A}", split="train"))
dens_m2560_bf_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DENS_M2560_BF_A}", split="train"))
dens_m3072_bf_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DENS_M3072_BF_A}", split="train"))
dens_m4096_bf_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DENS_M4096_BF_A}", split="train"))
dens_m512_bf_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M512_BF_EQVOL}", split="train")
)
dens_m1024_bf_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M1024_BF_EQVOL}", split="train")
)
dens_m2048_bf_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M2048_BF_EQVOL}", split="train")
)
dens_m2560_bf_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M2560_BF_EQVOL}", split="train")
)
dens_m3072_bf_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M3072_BF_EQVOL}", split="train")
)
dens_m4096_bf_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M4096_BF_EQVOL}", split="train")
)
dens_m512_bf_r0 = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DENS_M512_BF_R0}", split="train"))
dens_m1024_bf_r0 = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DENS_M1024_BF_R0}", split="train"))
dens_m2048_bf_r0 = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DENS_M2048_BF_R0}", split="train"))
dens_m2560_bf_r0 = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DENS_M2560_BF_R0}", split="train"))
dens_m3072_bf_r0 = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DENS_M3072_BF_R0}", split="train"))
dens_m4096_bf_r0 = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DENS_M4096_BF_R0}", split="train"))
dens_m512_bf_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M512_BF_R150}", split="train")
)
dens_m1024_bf_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M1024_BF_R150}", split="train")
)
dens_m2048_bf_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M2048_BF_R150}", split="train")
)
dens_m2560_bf_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M2560_BF_R150}", split="train")
)
dens_m3072_bf_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M3072_BF_R150}", split="train")
)
dens_m4096_bf_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M4096_BF_R150}", split="train")
)
dens_m512_bf_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M512_BF_R300}", split="train")
)
dens_m1024_bf_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M1024_BF_R300}", split="train")
)
dens_m2048_bf_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M2048_BF_R300}", split="train")
)
dens_m2560_bf_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M2560_BF_R300}", split="train")
)
dens_m3072_bf_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M3072_BF_R300}", split="train")
)
dens_m4096_bf_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M4096_BF_R300}", split="train")
)
dens_m512_kdk_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DENS_M512_KDK_A}", split="train"))
dens_m1024_kdk_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DENS_M1024_KDK_A}", split="train"))
dens_m2048_kdk_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DENS_M2048_KDK_A}", split="train"))
dens_m2560_kdk_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DENS_M2560_KDK_A}", split="train"))
dens_m3072_kdk_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DENS_M3072_KDK_A}", split="train"))
dens_m4096_kdk_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DENS_M4096_KDK_A}", split="train"))
dens_m512_kdk_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M512_KDK_EQVOL}", split="train")
)
dens_m1024_kdk_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M1024_KDK_EQVOL}", split="train")
)
dens_m2048_kdk_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M2048_KDK_EQVOL}", split="train")
)
dens_m2560_kdk_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M2560_KDK_EQVOL}", split="train")
)
dens_m3072_kdk_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M3072_KDK_EQVOL}", split="train")
)
dens_m4096_kdk_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M4096_KDK_EQVOL}", split="train")
)
dens_m512_kdk_r0 = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DENS_M512_KDK_R0}", split="train"))
dens_m1024_kdk_r0 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M1024_KDK_R0}", split="train")
)
dens_m2048_kdk_r0 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M2048_KDK_R0}", split="train")
)
dens_m2560_kdk_r0 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M2560_KDK_R0}", split="train")
)
dens_m3072_kdk_r0 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M3072_KDK_R0}", split="train")
)
dens_m4096_kdk_r0 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M4096_KDK_R0}", split="train")
)
dens_m512_kdk_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M512_KDK_R150}", split="train")
)
dens_m1024_kdk_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M1024_KDK_R150}", split="train")
)
dens_m2048_kdk_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M2048_KDK_R150}", split="train")
)
dens_m2560_kdk_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M2560_KDK_R150}", split="train")
)
dens_m3072_kdk_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M3072_KDK_R150}", split="train")
)
dens_m4096_kdk_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M4096_KDK_R150}", split="train")
)
dens_m512_kdk_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M512_KDK_R300}", split="train")
)
dens_m1024_kdk_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M1024_KDK_R300}", split="train")
)
dens_m2048_kdk_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M2048_KDK_R300}", split="train")
)
dens_m2560_kdk_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M2560_KDK_R300}", split="train")
)
dens_m3072_kdk_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M3072_KDK_R300}", split="train")
)
dens_m4096_kdk_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{DENS_M4096_KDK_R300}", split="train")
)

kappa_m512_bf_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{KAPPA_M512_BF_A}", split="train"))
kappa_m1024_bf_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{KAPPA_M1024_BF_A}", split="train"))
kappa_m2048_bf_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{KAPPA_M2048_BF_A}", split="train"))
kappa_m2560_bf_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{KAPPA_M2560_BF_A}", split="train"))
kappa_m3072_bf_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{KAPPA_M3072_BF_A}", split="train"))
kappa_m4096_bf_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{KAPPA_M4096_BF_A}", split="train"))
kappa_m512_bf_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M512_BF_EQVOL}", split="train")
)
kappa_m1024_bf_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M1024_BF_EQVOL}", split="train")
)
kappa_m2048_bf_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M2048_BF_EQVOL}", split="train")
)
kappa_m2560_bf_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M2560_BF_EQVOL}", split="train")
)
kappa_m3072_bf_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M3072_BF_EQVOL}", split="train")
)
kappa_m4096_bf_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M4096_BF_EQVOL}", split="train")
)
kappa_m512_bf_r0 = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{KAPPA_M512_BF_R0}", split="train"))
kappa_m1024_bf_r0 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M1024_BF_R0}", split="train")
)
kappa_m2048_bf_r0 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M2048_BF_R0}", split="train")
)
kappa_m2560_bf_r0 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M2560_BF_R0}", split="train")
)
kappa_m3072_bf_r0 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M3072_BF_R0}", split="train")
)
kappa_m4096_bf_r0 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M4096_BF_R0}", split="train")
)
kappa_m512_bf_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M512_BF_R150}", split="train")
)
kappa_m1024_bf_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M1024_BF_R150}", split="train")
)
kappa_m2048_bf_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M2048_BF_R150}", split="train")
)
kappa_m2560_bf_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M2560_BF_R150}", split="train")
)
kappa_m3072_bf_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M3072_BF_R150}", split="train")
)
kappa_m4096_bf_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M4096_BF_R150}", split="train")
)
kappa_m512_bf_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M512_BF_R300}", split="train")
)
kappa_m1024_bf_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M1024_BF_R300}", split="train")
)
kappa_m2048_bf_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M2048_BF_R300}", split="train")
)
kappa_m2560_bf_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M2560_BF_R300}", split="train")
)
kappa_m3072_bf_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M3072_BF_R300}", split="train")
)
kappa_m4096_bf_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M4096_BF_R300}", split="train")
)
kappa_m512_kdk_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{KAPPA_M512_KDK_A}", split="train"))
kappa_m1024_kdk_a = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M1024_KDK_A}", split="train")
)
kappa_m2048_kdk_a = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M2048_KDK_A}", split="train")
)
kappa_m2560_kdk_a = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M2560_KDK_A}", split="train")
)
kappa_m3072_kdk_a = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M3072_KDK_A}", split="train")
)
kappa_m4096_kdk_a = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M4096_KDK_A}", split="train")
)
kappa_m512_kdk_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M512_KDK_EQVOL}", split="train")
)
kappa_m1024_kdk_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M1024_KDK_EQVOL}", split="train")
)
kappa_m2048_kdk_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M2048_KDK_EQVOL}", split="train")
)
kappa_m2560_kdk_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M2560_KDK_EQVOL}", split="train")
)
kappa_m3072_kdk_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M3072_KDK_EQVOL}", split="train")
)
kappa_m4096_kdk_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M4096_KDK_EQVOL}", split="train")
)
kappa_m512_kdk_r0 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M512_KDK_R0}", split="train")
)
kappa_m1024_kdk_r0 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M1024_KDK_R0}", split="train")
)
kappa_m2048_kdk_r0 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M2048_KDK_R0}", split="train")
)
kappa_m2560_kdk_r0 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M2560_KDK_R0}", split="train")
)
kappa_m3072_kdk_r0 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M3072_KDK_R0}", split="train")
)
kappa_m4096_kdk_r0 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M4096_KDK_R0}", split="train")
)
kappa_m512_kdk_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M512_KDK_R150}", split="train")
)
kappa_m1024_kdk_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M1024_KDK_R150}", split="train")
)
kappa_m2048_kdk_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M2048_KDK_R150}", split="train")
)
kappa_m2560_kdk_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M2560_KDK_R150}", split="train")
)
kappa_m3072_kdk_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M3072_KDK_R150}", split="train")
)
kappa_m4096_kdk_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M4096_KDK_R150}", split="train")
)
kappa_m512_kdk_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M512_KDK_R300}", split="train")
)
kappa_m1024_kdk_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M1024_KDK_R300}", split="train")
)
kappa_m2048_kdk_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M2048_KDK_R300}", split="train")
)
kappa_m2560_kdk_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M2560_KDK_R300}", split="train")
)
kappa_m3072_kdk_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M3072_KDK_R300}", split="train")
)
kappa_m4096_kdk_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{KAPPA_M4096_KDK_R300}", split="train")
)

cut_m512_bf_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CUT_M512_BF_A}", split="train"))
cut_m1024_bf_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CUT_M1024_BF_A}", split="train"))
cut_m2048_bf_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CUT_M2048_BF_A}", split="train"))
cut_m2560_bf_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CUT_M2560_BF_A}", split="train"))
cut_m3072_bf_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CUT_M3072_BF_A}", split="train"))
cut_m4096_bf_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CUT_M4096_BF_A}", split="train"))
cut_m512_bf_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M512_BF_EQVOL}", split="train")
)
cut_m1024_bf_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M1024_BF_EQVOL}", split="train")
)
cut_m2048_bf_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M2048_BF_EQVOL}", split="train")
)
cut_m2560_bf_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M2560_BF_EQVOL}", split="train")
)
cut_m3072_bf_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M3072_BF_EQVOL}", split="train")
)
cut_m4096_bf_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M4096_BF_EQVOL}", split="train")
)
cut_m512_bf_r0 = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CUT_M512_BF_R0}", split="train"))
cut_m1024_bf_r0 = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CUT_M1024_BF_R0}", split="train"))
cut_m2048_bf_r0 = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CUT_M2048_BF_R0}", split="train"))
cut_m2560_bf_r0 = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CUT_M2560_BF_R0}", split="train"))
cut_m3072_bf_r0 = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CUT_M3072_BF_R0}", split="train"))
cut_m4096_bf_r0 = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CUT_M4096_BF_R0}", split="train"))
cut_m512_bf_r150 = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CUT_M512_BF_R150}", split="train"))
cut_m1024_bf_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M1024_BF_R150}", split="train")
)
cut_m2048_bf_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M2048_BF_R150}", split="train")
)
cut_m2560_bf_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M2560_BF_R150}", split="train")
)
cut_m3072_bf_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M3072_BF_R150}", split="train")
)
cut_m4096_bf_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M4096_BF_R150}", split="train")
)
cut_m512_bf_r300 = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CUT_M512_BF_R300}", split="train"))
cut_m1024_bf_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M1024_BF_R300}", split="train")
)
cut_m2048_bf_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M2048_BF_R300}", split="train")
)
cut_m2560_bf_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M2560_BF_R300}", split="train")
)
cut_m3072_bf_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M3072_BF_R300}", split="train")
)
cut_m4096_bf_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M4096_BF_R300}", split="train")
)
cut_m512_kdk_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CUT_M512_KDK_A}", split="train"))
cut_m1024_kdk_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CUT_M1024_KDK_A}", split="train"))
cut_m2048_kdk_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CUT_M2048_KDK_A}", split="train"))
cut_m2560_kdk_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CUT_M2560_KDK_A}", split="train"))
cut_m3072_kdk_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CUT_M3072_KDK_A}", split="train"))
cut_m4096_kdk_a = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CUT_M4096_KDK_A}", split="train"))
cut_m512_kdk_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M512_KDK_EQVOL}", split="train")
)
cut_m1024_kdk_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M1024_KDK_EQVOL}", split="train")
)
cut_m2048_kdk_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M2048_KDK_EQVOL}", split="train")
)
cut_m2560_kdk_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M2560_KDK_EQVOL}", split="train")
)
cut_m3072_kdk_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M3072_KDK_EQVOL}", split="train")
)
cut_m4096_kdk_eqvol = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M4096_KDK_EQVOL}", split="train")
)
cut_m512_kdk_r0 = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CUT_M512_KDK_R0}", split="train"))
cut_m1024_kdk_r0 = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CUT_M1024_KDK_R0}", split="train"))
cut_m2048_kdk_r0 = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CUT_M2048_KDK_R0}", split="train"))
cut_m2560_kdk_r0 = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CUT_M2560_KDK_R0}", split="train"))
cut_m3072_kdk_r0 = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CUT_M3072_KDK_R0}", split="train"))
cut_m4096_kdk_r0 = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CUT_M4096_KDK_R0}", split="train"))
cut_m512_kdk_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M512_KDK_R150}", split="train")
)
cut_m1024_kdk_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M1024_KDK_R150}", split="train")
)
cut_m2048_kdk_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M2048_KDK_R150}", split="train")
)
cut_m2560_kdk_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M2560_KDK_R150}", split="train")
)
cut_m3072_kdk_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M3072_KDK_R150}", split="train")
)
cut_m4096_kdk_r150 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M4096_KDK_R150}", split="train")
)
cut_m512_kdk_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M512_KDK_R300}", split="train")
)
cut_m1024_kdk_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M1024_KDK_R300}", split="train")
)
cut_m2048_kdk_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M2048_KDK_R300}", split="train")
)
cut_m2560_kdk_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M2560_KDK_R300}", split="train")
)
cut_m3072_kdk_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M3072_KDK_R300}", split="train")
)
cut_m4096_kdk_r300 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{CUT_M4096_KDK_R300}", split="train")
)

DENS = {  # (solver, spacing) -> catalogues ordered like MESHES
    ("bf", "a"): (dens_m512_bf_a, dens_m1024_bf_a, dens_m2048_bf_a, dens_m2560_bf_a, dens_m3072_bf_a, dens_m4096_bf_a),
    ("bf", "equal_vol"): (
        dens_m512_bf_eqvol,
        dens_m1024_bf_eqvol,
        dens_m2048_bf_eqvol,
        dens_m2560_bf_eqvol,
        dens_m3072_bf_eqvol,
        dens_m4096_bf_eqvol,
    ),
    ("bf", "eqvolc_w150_r0"): (
        dens_m512_bf_r0,
        dens_m1024_bf_r0,
        dens_m2048_bf_r0,
        dens_m2560_bf_r0,
        dens_m3072_bf_r0,
        dens_m4096_bf_r0,
    ),
    ("bf", "eqvolc_w150_r150"): (
        dens_m512_bf_r150,
        dens_m1024_bf_r150,
        dens_m2048_bf_r150,
        dens_m2560_bf_r150,
        dens_m3072_bf_r150,
        dens_m4096_bf_r150,
    ),
    ("bf", "eqvolc_w150_r300"): (
        dens_m512_bf_r300,
        dens_m1024_bf_r300,
        dens_m2048_bf_r300,
        dens_m2560_bf_r300,
        dens_m3072_bf_r300,
        dens_m4096_bf_r300,
    ),
    ("kdk", "a"): (
        dens_m512_kdk_a,
        dens_m1024_kdk_a,
        dens_m2048_kdk_a,
        dens_m2560_kdk_a,
        dens_m3072_kdk_a,
        dens_m4096_kdk_a,
    ),
    ("kdk", "equal_vol"): (
        dens_m512_kdk_eqvol,
        dens_m1024_kdk_eqvol,
        dens_m2048_kdk_eqvol,
        dens_m2560_kdk_eqvol,
        dens_m3072_kdk_eqvol,
        dens_m4096_kdk_eqvol,
    ),
    ("kdk", "eqvolc_w150_r0"): (
        dens_m512_kdk_r0,
        dens_m1024_kdk_r0,
        dens_m2048_kdk_r0,
        dens_m2560_kdk_r0,
        dens_m3072_kdk_r0,
        dens_m4096_kdk_r0,
    ),
    ("kdk", "eqvolc_w150_r150"): (
        dens_m512_kdk_r150,
        dens_m1024_kdk_r150,
        dens_m2048_kdk_r150,
        dens_m2560_kdk_r150,
        dens_m3072_kdk_r150,
        dens_m4096_kdk_r150,
    ),
    ("kdk", "eqvolc_w150_r300"): (
        dens_m512_kdk_r300,
        dens_m1024_kdk_r300,
        dens_m2048_kdk_r300,
        dens_m2560_kdk_r300,
        dens_m3072_kdk_r300,
        dens_m4096_kdk_r300,
    ),
}

KAPPA = {  # (solver, spacing) -> catalogues ordered like MESHES
    ("bf", "a"): (
        kappa_m512_bf_a,
        kappa_m1024_bf_a,
        kappa_m2048_bf_a,
        kappa_m2560_bf_a,
        kappa_m3072_bf_a,
        kappa_m4096_bf_a,
    ),
    ("bf", "equal_vol"): (
        kappa_m512_bf_eqvol,
        kappa_m1024_bf_eqvol,
        kappa_m2048_bf_eqvol,
        kappa_m2560_bf_eqvol,
        kappa_m3072_bf_eqvol,
        kappa_m4096_bf_eqvol,
    ),
    ("bf", "eqvolc_w150_r0"): (
        kappa_m512_bf_r0,
        kappa_m1024_bf_r0,
        kappa_m2048_bf_r0,
        kappa_m2560_bf_r0,
        kappa_m3072_bf_r0,
        kappa_m4096_bf_r0,
    ),
    ("bf", "eqvolc_w150_r150"): (
        kappa_m512_bf_r150,
        kappa_m1024_bf_r150,
        kappa_m2048_bf_r150,
        kappa_m2560_bf_r150,
        kappa_m3072_bf_r150,
        kappa_m4096_bf_r150,
    ),
    ("bf", "eqvolc_w150_r300"): (
        kappa_m512_bf_r300,
        kappa_m1024_bf_r300,
        kappa_m2048_bf_r300,
        kappa_m2560_bf_r300,
        kappa_m3072_bf_r300,
        kappa_m4096_bf_r300,
    ),
    ("kdk", "a"): (
        kappa_m512_kdk_a,
        kappa_m1024_kdk_a,
        kappa_m2048_kdk_a,
        kappa_m2560_kdk_a,
        kappa_m3072_kdk_a,
        kappa_m4096_kdk_a,
    ),
    ("kdk", "equal_vol"): (
        kappa_m512_kdk_eqvol,
        kappa_m1024_kdk_eqvol,
        kappa_m2048_kdk_eqvol,
        kappa_m2560_kdk_eqvol,
        kappa_m3072_kdk_eqvol,
        kappa_m4096_kdk_eqvol,
    ),
    ("kdk", "eqvolc_w150_r0"): (
        kappa_m512_kdk_r0,
        kappa_m1024_kdk_r0,
        kappa_m2048_kdk_r0,
        kappa_m2560_kdk_r0,
        kappa_m3072_kdk_r0,
        kappa_m4096_kdk_r0,
    ),
    ("kdk", "eqvolc_w150_r150"): (
        kappa_m512_kdk_r150,
        kappa_m1024_kdk_r150,
        kappa_m2048_kdk_r150,
        kappa_m2560_kdk_r150,
        kappa_m3072_kdk_r150,
        kappa_m4096_kdk_r150,
    ),
    ("kdk", "eqvolc_w150_r300"): (
        kappa_m512_kdk_r300,
        kappa_m1024_kdk_r300,
        kappa_m2048_kdk_r300,
        kappa_m2560_kdk_r300,
        kappa_m3072_kdk_r300,
        kappa_m4096_kdk_r300,
    ),
}

CUT = {  # (solver, spacing) -> catalogues ordered like MESHES
    ("bf", "a"): (cut_m512_bf_a, cut_m1024_bf_a, cut_m2048_bf_a, cut_m2560_bf_a, cut_m3072_bf_a, cut_m4096_bf_a),
    ("bf", "equal_vol"): (
        cut_m512_bf_eqvol,
        cut_m1024_bf_eqvol,
        cut_m2048_bf_eqvol,
        cut_m2560_bf_eqvol,
        cut_m3072_bf_eqvol,
        cut_m4096_bf_eqvol,
    ),
    ("bf", "eqvolc_w150_r0"): (
        cut_m512_bf_r0,
        cut_m1024_bf_r0,
        cut_m2048_bf_r0,
        cut_m2560_bf_r0,
        cut_m3072_bf_r0,
        cut_m4096_bf_r0,
    ),
    ("bf", "eqvolc_w150_r150"): (
        cut_m512_bf_r150,
        cut_m1024_bf_r150,
        cut_m2048_bf_r150,
        cut_m2560_bf_r150,
        cut_m3072_bf_r150,
        cut_m4096_bf_r150,
    ),
    ("bf", "eqvolc_w150_r300"): (
        cut_m512_bf_r300,
        cut_m1024_bf_r300,
        cut_m2048_bf_r300,
        cut_m2560_bf_r300,
        cut_m3072_bf_r300,
        cut_m4096_bf_r300,
    ),
    ("kdk", "a"): (cut_m512_kdk_a, cut_m1024_kdk_a, cut_m2048_kdk_a, cut_m2560_kdk_a, cut_m3072_kdk_a, cut_m4096_kdk_a),
    ("kdk", "equal_vol"): (
        cut_m512_kdk_eqvol,
        cut_m1024_kdk_eqvol,
        cut_m2048_kdk_eqvol,
        cut_m2560_kdk_eqvol,
        cut_m3072_kdk_eqvol,
        cut_m4096_kdk_eqvol,
    ),
    ("kdk", "eqvolc_w150_r0"): (
        cut_m512_kdk_r0,
        cut_m1024_kdk_r0,
        cut_m2048_kdk_r0,
        cut_m2560_kdk_r0,
        cut_m3072_kdk_r0,
        cut_m4096_kdk_r0,
    ),
    ("kdk", "eqvolc_w150_r150"): (
        cut_m512_kdk_r150,
        cut_m1024_kdk_r150,
        cut_m2048_kdk_r150,
        cut_m2560_kdk_r150,
        cut_m3072_kdk_r150,
        cut_m4096_kdk_r150,
    ),
    ("kdk", "eqvolc_w150_r300"): (
        cut_m512_kdk_r300,
        cut_m1024_kdk_r300,
        cut_m2048_kdk_r300,
        cut_m2560_kdk_r300,
        cut_m3072_kdk_r300,
        cut_m4096_kdk_r300,
    ),
}

cosmo = dens_m512_bf_a.cosmology[0]
for _cats in (*DENS.values(), *KAPPA.values(), *CUT.values()):
    for _cat in _cats:
        for _k in ("Omega_c", "Omega_b", "h", "n_s", "sigma8", "w0", "wa"):
            assert np.isclose(getattr(_cat.cosmology[0], _k), getattr(cosmo, _k))

ell = jnp.arange(LMAX + 1)
pw2048 = hp.pixwin(2048, lmax=LMAX) ** 2
pw512 = hp.pixwin(512, lmax=LMAX) ** 2

# Density theory: the shell geometry depends on the spacing only, so one Limber prediction per spacing
# (pixel-window matched to nside 2048), checked against the shell centres of every mesh and solver.
DENS_THEORY = {}
for _sp in SPACINGS:
    _ref = DENS[("bf", _sp)][0].field[0]
    for _s in SOLVERS:
        for _cat in DENS[(_s, _sp)]:
            assert np.allclose(np.asarray(_cat.field[0].comoving_centers), np.asarray(_ref.comoving_centers))
    DENS_THEORY[_sp] = (compute_theory_cl_for_density(cosmo, _ref, ell) * pw2048).bin(nlb=32, lmin=2)
bc = np.asarray(DENS_THEORY["a"].wavenumber)

# Born kappa theory for the three Stage-3 source bins, pixel-window matched to each map resolution.
kappa_theory = compute_theory_cl(cosmo, ell, get_stage3_nz_shear()[:3])
KAPPA_THEORY_2048 = np.asarray((kappa_theory * pw2048).bin(nlb=32, lmin=2).array)
KAPPA_THEORY_512 = np.asarray((kappa_theory * pw512).bin(nlb=32, lmin=2).array)
KAPPA_THEORY_NOPIX = np.asarray(kappa_theory.bin(nlb=32, lmin=2).array)

# CosmoGrid reference, pixel window divided out, and the cosmology offset between the two grid points.
_fid_part0 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{FIDUCIAL_KAPPA}/cosmo_fiducial_part0.parquet", split="train")
)
_fid_part1 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{FIDUCIAL_KAPPA}/cosmo_fiducial_part1.parquet", split="train")
)
_fid_part2 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{FIDUCIAL_KAPPA}/cosmo_fiducial_part2.parquet", split="train")
)
_fid_part3 = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{FIDUCIAL_KAPPA}/cosmo_fiducial_part3.parquet", split="train")
)
_fid_b = np.stack(
    [
        np.asarray(f.bin(nlb=32, lmin=2).array)
        for cat in (_fid_part0, _fid_part1, _fid_part2, _fid_part3)
        for f in cat.field
    ]
)
CV_FRAC = np.sqrt(2.0) * (_fid_b.std(axis=0, ddof=1) / _fid_b.mean(axis=0))[:3].max(axis=0)

cg_172798 = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{CG_172798}", split="train"))
cg_b = np.asarray((cg_172798.field[0] / pw512).bin(nlb=32, lmin=2).array)[:3]
th_172798_b = np.asarray(
    compute_theory_cl(cg_172798.cosmology[0], ell, get_stage3_nz_shear()[:3]).bin(nlb=32, lmin=2).array
)[:3]
CG_OFFSET = (KAPPA_THEORY_NOPIX / th_172798_b - 1.0).mean(axis=0)

perf_pm = pd.read_csv(f"{root}/{PERF_PM}", header=None)
# columns: name, precision, x, y, z, px, py, backend, nodes, jit, min, max, mean, std, last (times in ms), ...
PM_MEAN_S = {
    row[0].split("_M")[0].removeprefix("pm100_exp5f_"): row[12] / 1e3 for row in perf_pm.itertuples(index=False)
}
PM_GPUS = {
    row[0].split("_M")[0].removeprefix("pm100_exp5f_"): row[5] * row[6] for row in perf_pm.itertuples(index=False)
}

BIN_COLOURS = ("#4477aa", "#ee7733", "#117733")  # the thesis tomographic-bin colours
Z_SOURCES = np.asarray(kappa_m2048_bf_a.field[0].z_sources)[:3]


def band_median(ratio, lo, hi):
    """Median over the bandpowers with centre in [lo, hi) of a (n, n_bins) ratio: one number per row."""
    return np.median(ratio[:, (bc >= lo) & (bc < hi)], axis=1)


X_MESH = np.arange(len(MESHES))  # meshes on an evenly spaced categorical axis
Y_MAX = 1.3  # points above are drawn as triangles on the top edge


def _mesh_axis(ax):
    ax.set_xticks(X_MESH)
    ax.set_xticklabels([rf"${m}^3$" for m in MESHES], rotation=45)
    ax.set_xlim(-0.3, len(MESHES) - 0.7)
    ax.set_ylim(0.0, Y_MAX)
    ax.grid(True, alpha=0.3)


def _plot_clipped(ax, y, colour):
    """Mesh line of one shell or source bin; values above Y_MAX sit on the top edge as triangles."""
    ax.plot(X_MESH, np.minimum(y, Y_MAX - 0.02), "-o", color=colour, ms=2.5, lw=1.0)
    off = y > Y_MAX
    ax.plot(X_MESH[off], np.full(off.sum(), Y_MAX - 0.02), "^", color=colour, ms=5)


def _grid_labels(axes, ylabel):
    for j, sp in enumerate(SPACINGS):
        axes[0, j].annotate(SPACING_LABEL[sp], xy=(0.5, 1.03), xycoords="axes fraction", ha="center", va="bottom")
        axes[-1, j].set_xlabel("PM mesh")
    for i, s in enumerate(SOLVERS):
        axes[i, 0].set_ylabel(SOLVER_LABEL[s] + "\n" + ylabel)


def density_mesh(lo, hi, stem):
    """Density C_ell / theory, median over ell in [lo, hi), against mesh. Rows = solver, columns = shell spacing;
    one line per shell, coloured by its comoving centre."""
    width = 13.0
    set_style(width_in=width)
    fig, axes = plt.subplots(2, 5, figsize=(width, 5.6), sharex=True, sharey=True, layout="constrained")
    norm = Normalize(0.0, 2500.0)
    cmap = plt.get_cmap("viridis")
    for i, s in enumerate(SOLVERS):
        for j, sp in enumerate(SPACINGS):
            ax = axes[i, j]
            theory = np.asarray(DENS_THEORY[sp].array)
            chi = np.asarray(DENS[(s, sp)][0].field[0].comoving_centers)
            med = np.stack(
                [band_median(np.asarray(c.field[0].bin(nlb=32, lmin=2).array) / theory, lo, hi) for c in DENS[(s, sp)]]
            )
            ax.axhspan(0.95, 1.05, color="0.9", lw=0, zorder=0)
            ax.axhline(1.0, color="k", lw=0.8)
            for sh in range(med.shape[1]):
                _plot_clipped(ax, med[:, sh], cmap(norm(chi[sh])))
            _mesh_axis(ax)
    _grid_labels(axes, rf"$C_\ell / C_\ell^{{\rm th}}$, $\ell \in [{lo}, {hi})$")
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, label=r"shell centre $\chi$ [Mpc/$h$]", pad=0.01)
    fig.legend(
        handles=[
            Line2D([], [], color="0.3", marker="^", ls="none", label=rf"off scale ($> {Y_MAX}$, shot noise)"),
            Patch(color="0.9", label=r"$\pm 5\%$ of theory"),
        ],
        frameon=False,
        loc="outside upper center",
        ncol=2,
    )
    savefig(ASSETS / stem, fig)


def kappa_mesh(cats, theory, stem):
    """Born kappa C_ell / theory, median over ell in [150, 250), against mesh. Rows = solver, columns = shell
    spacing; one line per source bin. The y-range is shared by the cut and no-cut figures."""
    width = 13.0
    set_style(width_in=width)
    fig, axes = plt.subplots(2, 5, figsize=(width, 5.6), sharex=True, sharey=True, layout="constrained")
    for i, s in enumerate(SOLVERS):
        for j, sp in enumerate(SPACINGS):
            ax = axes[i, j]
            med = np.stack(
                [
                    band_median(np.asarray(c.field[0].bin(nlb=32, lmin=2).array) / theory, 150, 250)
                    for c in cats[(s, sp)]
                ]
            )
            ax.axhspan(0.95, 1.05, color="0.9", lw=0, zorder=0)
            ax.axhline(1.0, color="k", lw=0.8)
            for b, colour in enumerate(BIN_COLOURS):
                _plot_clipped(ax, med[:, b], colour)
            _mesh_axis(ax)
    _grid_labels(axes, r"$C_\ell^{\kappa\kappa} / C_\ell^{\rm th}$, $\ell \in [150, 250)$")
    fig.legend(
        handles=[
            Line2D([], [], color=c, marker="o", ms=3, label=rf"source bin {b + 1}, $z_s = {z:.2f}$")
            for b, (c, z) in enumerate(zip(BIN_COLOURS, Z_SOURCES))
        ]
        + [
            Line2D([], [], color="0.3", marker="^", ls="none", label=rf"off scale ($> {Y_MAX}$)"),
            Patch(color="0.9", label=r"$\pm 5\%$"),
        ],
        frameon=False,
        loc="outside upper center",
        ncol=5,
    )
    savefig(ASSETS / stem, fig)


def kappa_cut_over_nocut(solver, spacing, stem):
    """Born kappa with the resolution cut over kappa without it (pixel windows divided out) against ell, one panel
    per source bin, one line per mesh."""
    width = 11.0
    set_style(width_in=width)
    fig, axes = plt.subplots(1, 3, figsize=(width, 3.4), sharey=True, layout="constrained")
    cmap = plt.get_cmap("plasma")
    for b, ax in enumerate(axes):
        for m, (cut_cat, nocut_cat) in enumerate(zip(CUT[(solver, spacing)], KAPPA[(solver, spacing)])):
            cut_b = np.asarray((cut_cat.field[0] / pw512).bin(nlb=32, lmin=2).array)[b]
            nocut_b = np.asarray((nocut_cat.field[0] / pw2048).bin(nlb=32, lmin=2).array)[b]
            ax.plot(bc, cut_b / nocut_b, color=cmap(m / (len(MESHES) - 0.5)), lw=1.2, label=rf"${MESHES[m]}^3$")
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set(xscale="log", xlim=(20, LMAX), ylim=(0.0, 1.1), xlabel=r"$\ell$")
        ax.grid(True, alpha=0.3)
        ax.annotate(rf"source bin {b + 1}, $z_s = {Z_SOURCES[b]:.2f}$", xy=(0.04, 0.06), xycoords="axes fraction")
    axes[0].set_ylabel(r"$C_\ell^{\rm cut} / C_\ell^{\rm no\ cut}$")
    fig.legend(*axes[0].get_legend_handles_labels(), title="PM mesh", frameon=False, loc="outside right center")
    savefig(ASSETS / stem, fig)


# (solver, spacing, mesh, cut) — the candidates fig06 compares against CosmoGrid
CANDIDATES = (
    ("kdk", "eqvolc_w150_r0", 4096, False),
    ("kdk", "eqvolc_w150_r0", 3072, False),
    ("bf", "eqvolc_w150_r0", 4096, False),
    ("kdk", "a", 4096, False),
)
CANDIDATE_COLOURS = ("#222222", "#cc3311", "#0077bb", "#009988")


def _candidate_ratio(solver, spacing, mesh, cut):
    """Candidate kappa over the CosmoGrid reference minus one, both with their pixel window divided out."""
    cats, pw = (CUT, pw512) if cut else (KAPPA, pw2048)
    run_b = np.asarray((cats[(solver, spacing)][MESHES.index(mesh)].field[0] / pw).bin(nlb=32, lmin=2).array)
    return run_b / cg_b - 1.0


def _candidate_label(solver, spacing, mesh, cut):
    return rf"{solver}, {SPACING_LABEL[spacing]}, ${mesh}^3$" + (", cut" if cut else "")


ELL_MAX_CG = 1000  # the CosmoGrid maps are nside 512


def cosmogrid_candidates(stem):
    """Candidate kappa against the CosmoGrid cosmo_172798 Born reference, one panel per source bin. Grey band:
    the expected cosmology offset between the two grid points +- sqrt(2) x the empirical CV."""
    width = 11.0
    set_style(width_in=width)
    fig, axes = plt.subplots(1, 3, figsize=(width, 3.6), sharey=True, layout="constrained")
    keep = bc <= ELL_MAX_CG
    for b, ax in enumerate(axes):
        ax.fill_between(
            bc[keep], (CG_OFFSET - CV_FRAC)[keep], (CG_OFFSET + CV_FRAC)[keep], color="0.88", lw=0, zorder=0
        )
        ax.plot(bc[keep], CG_OFFSET[keep], ":", color="0.3", lw=0.9)
        ax.axhline(0.0, color="k", lw=0.8)
        for cand, colour in zip(CANDIDATES, CANDIDATE_COLOURS):
            ax.plot(bc[keep], _candidate_ratio(*cand)[b][keep], color=colour, lw=1.2, label=_candidate_label(*cand))
        ax.set(xscale="log", xlim=(20, ELL_MAX_CG), xlabel=r"$\ell$")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, alpha=0.3)
        ax.annotate(rf"source bin {b + 1}, $z_s = {Z_SOURCES[b]:.2f}$", xy=(0.04, 0.06), xycoords="axes fraction")
    axes[0].set_ylabel(r"$C_\ell^{\kappa\kappa} / C_\ell^{\rm CosmoGrid} - 1$")
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [
        Patch(color="0.88", label=r"$\sqrt{2}\times$ empirical cosmic variance (200 CosmoGrid permutations)"),
        Line2D([], [], color="0.3", ls=":", lw=0.9, label="expected cosmology offset"),
    ]
    fig.legend(handles=handles, frameon=False, loc="outside upper center", ncol=3)
    savefig(ASSETS / stem, fig)


def print_decision_table():
    """Markdown table for the README: kappa / theory at ell ~ 200, the median kappa / CosmoGrid - 1 over
    ell in [30, 300), and the PM wall time per simulation."""
    print(
        "| solver | spacing | mesh | cut | κ/theory ℓ≈200 (bins 1/2/3) | κ/CosmoGrid − 1, ℓ∈[30,300) (bins 1/2/3) | PM time [s] × GPUs |"
    )
    print("|---|---|---|---|---|---|---|")
    for solver, spacing, mesh, cut in CANDIDATES:
        cats, theory = (CUT, KAPPA_THEORY_512) if cut else (KAPPA, KAPPA_THEORY_2048)
        run_b = np.asarray(cats[(solver, spacing)][MESHES.index(mesh)].field[0].bin(nlb=32, lmin=2).array)
        th = band_median(run_b / theory, 150, 250)
        cg = band_median(_candidate_ratio(solver, spacing, mesh, cut), 30, 300)
        tag = f"m{mesh}_{solver}_{spacing}"
        print(
            f"| {solver} | {spacing} | {mesh}³ | {'yes' if cut else 'no'} | "
            + " / ".join(f"{v:.2f}" for v in th)
            + " | "
            + " / ".join(f"{v:+.2f}" for v in cg)
            + f" | {PM_MEAN_S[tag]:.0f} × {PM_GPUS[tag]} |"
        )


def main():
    set_style()
    density_mesh(150, 250, "fig01-density-mesh-ell200")
    density_mesh(450, 550, "fig02-density-mesh-ell500")
    kappa_mesh(KAPPA, KAPPA_THEORY_2048, "fig03-kappa-mesh-nocut")
    kappa_mesh(CUT, KAPPA_THEORY_512, "fig04-kappa-mesh-rescut")
    kappa_cut_over_nocut("kdk", "eqvolc_w150_r0", "fig05-kappa-cut-over-nocut")
    cosmogrid_candidates("fig06-cosmogrid-candidates")
    print_decision_table()
    print(f"assets written to {ASSETS}")


if __name__ == "__main__":
    main()
