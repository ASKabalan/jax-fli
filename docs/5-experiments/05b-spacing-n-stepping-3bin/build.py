"""Experiment 05b — deep 3-bin box (scale-factor shells; density-shell C_ell + Born lensing): figures.

Renders the SVG figures for ``docs/5-experiments/05b-spacing-n-stepping-3bin/README.md``. 05b is 05a taken to a
5 Gpc/h box at 2560^3 so the inscribed full-sky lightcone reaches all three low-z Stage-3 source bins: the
shell spacing is still ``a`` (scale factor, like 05a), and the drift moves each particle to the scale factor at
which it actually crosses the lightcone, removing the frozen-epoch error of a thick shell.

  fig01   illustration (small local 256^3 sim in the 5 Gpc/h box): the same particle cloud coloured by the
          redshift it is assigned under a 10-shell freeze, the drift's smooth z(r), and a 40-shell freeze.
  fig02   per-shell density C_ell at the near / mid / far shell: the 10-shell drift / no-drift runs vs a
          continuous-lightcone reference built by summing the matching 40-shell run (counts -> overdensity).
  fig03   per-shell density C_ell census vs Limber theory for the 5- / 8- / 10-shell runs (drift + no-drift,
          one subplot per shell: a log-log C_ell panel over a ratio-to-theory strip).
  fig04-fig08  the same census for the 16- / 20- / 25- / 30- / 40-shell runs.
  fig09   Born convergence C_ell per tomographic bin (Stage-3 [:3]) vs the number of shells, drift vs no-drift,
          each ratioed to its own 40-shell run (three source-bin row-pairs x no-drift / with-drift columns).
  fig10   the same Born convergence ratioed to the Limber weak-lensing theory instead of the 40-shell run.
  fig11   the 20-shell convergence against the CosmoGrid Born reference (thin shells, same born() code, its own
          cosmology) — each measurement ratioed to the Limber theory at its own cosmology.

Because the scale-factor shells are thin, the drift's effect on the Born convergence is small — a few percent at
the coarsest 5-8 shells, negligible by ~16 — so fig09/fig10's no-drift and with-drift columns nearly coincide,
in contrast to the equal-volume 05c where the fat inner shell's error survives the projection. The shell-Born
quadrature error (see 05c fig12) is <= 2% for this geometry at 20 shells.

Run from the repo root (CPU is fine; fig01 runs a small sim, fig02 sums nside-2048 maps):
    JAX_PLATFORMS=cpu uv run --no-sync python docs/5-experiments/05b-spacing-n-stepping-3bin/build.py
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _exputils import savefig, set_style  # noqa: E402

ASSETS = HERE / "assets"
REPO = "ASKabalan/jax-fli-experiments"

NEAR_SHELL, MID_SHELL, FAR_SHELL = 1, 5, 9  # which 10-run shell each fig02 column zooms in on
NSHELLS_KAPPA = [5, 8, 10, 12, 16, 20, 25, 30, 40]  # Born-convergence shell-count sweep (fig09/fig10)
LMAX = 1500  # the published spectra stop at ell 1500
BOX, MESH = 5000.0, 2560.0  # per-shell PM-Nyquist line ell_max ~ pi*chi/dx, dx = box/mesh (5 Gpc/h box)
CENSUS_RATIO_YLIM = (-0.25, 0.25)  # density census ratio-to-theory strip range (centered on 0)

root = snapshot_download(REPO, repo_type="dataset", local_files_only=True)

# Density spectra: the full per-shell census (fig03-fig08), drift + no-drift at every shell count. fig02
# additionally reads only the 10-run measured C_ell (drift_10 / nodrift_10) and the 40-run shell geometry.
DRIFT_5_SPECTRA = "05-spacing-n-stepping/05b-3bins/density_spectra/spectra_exp5b_drift_5.parquet"
NODRIFT_5_SPECTRA = "05-spacing-n-stepping/05b-3bins/density_spectra/spectra_exp5b_nodrift_5.parquet"
DRIFT_8_SPECTRA = "05-spacing-n-stepping/05b-3bins/density_spectra/spectra_exp5b_drift_8.parquet"
NODRIFT_8_SPECTRA = "05-spacing-n-stepping/05b-3bins/density_spectra/spectra_exp5b_nodrift_8.parquet"
DRIFT_10_SPECTRA = "05-spacing-n-stepping/05b-3bins/density_spectra/spectra_exp5b_drift_10.parquet"
NODRIFT_10_SPECTRA = "05-spacing-n-stepping/05b-3bins/density_spectra/spectra_exp5b_nodrift_10.parquet"
DRIFT_16_SPECTRA = "05-spacing-n-stepping/05b-3bins/density_spectra/spectra_exp5b_drift_16.parquet"
NODRIFT_16_SPECTRA = "05-spacing-n-stepping/05b-3bins/density_spectra/spectra_exp5b_nodrift_16.parquet"
DRIFT_20_SPECTRA = "05-spacing-n-stepping/05b-3bins/density_spectra/spectra_exp5b_drift_20.parquet"
NODRIFT_20_SPECTRA = "05-spacing-n-stepping/05b-3bins/density_spectra/spectra_exp5b_nodrift_20.parquet"
DRIFT_25_SPECTRA = "05-spacing-n-stepping/05b-3bins/density_spectra/spectra_exp5b_drift_25.parquet"
NODRIFT_25_SPECTRA = "05-spacing-n-stepping/05b-3bins/density_spectra/spectra_exp5b_nodrift_25.parquet"
DRIFT_30_SPECTRA = "05-spacing-n-stepping/05b-3bins/density_spectra/spectra_exp5b_drift_30.parquet"
NODRIFT_30_SPECTRA = "05-spacing-n-stepping/05b-3bins/density_spectra/spectra_exp5b_nodrift_30.parquet"
DRIFT_40_SPECTRA = "05-spacing-n-stepping/05b-3bins/density_spectra/spectra_exp5b_drift_40.parquet"
NODRIFT_40_SPECTRA = "05-spacing-n-stepping/05b-3bins/density_spectra/spectra_exp5b_nodrift_40.parquet"

# Born-convergence (kappa) spectra: 3-bin tomographic (Stage-3 [:3]) auto C_ell per shell count, drift and
# no-drift. Each parquet holds a (3, n_ell) array — one scalar auto spectrum per source bin. Feeds fig09/fig10.
DRIFT_5_KAPPA = "05-spacing-n-stepping/05b-3bins/kappa_spectra/spectra_born_drift_5.parquet"
DRIFT_8_KAPPA = "05-spacing-n-stepping/05b-3bins/kappa_spectra/spectra_born_drift_8.parquet"
DRIFT_10_KAPPA = "05-spacing-n-stepping/05b-3bins/kappa_spectra/spectra_born_drift_10.parquet"
DRIFT_12_KAPPA = "05-spacing-n-stepping/05b-3bins/kappa_spectra/spectra_born_drift_12.parquet"
DRIFT_16_KAPPA = "05-spacing-n-stepping/05b-3bins/kappa_spectra/spectra_born_drift_16.parquet"
DRIFT_20_KAPPA = "05-spacing-n-stepping/05b-3bins/kappa_spectra/spectra_born_drift_20.parquet"
DRIFT_25_KAPPA = "05-spacing-n-stepping/05b-3bins/kappa_spectra/spectra_born_drift_25.parquet"
DRIFT_30_KAPPA = "05-spacing-n-stepping/05b-3bins/kappa_spectra/spectra_born_drift_30.parquet"
DRIFT_40_KAPPA = "05-spacing-n-stepping/05b-3bins/kappa_spectra/spectra_born_drift_40.parquet"

NODRIFT_5_KAPPA = "05-spacing-n-stepping/05b-3bins/kappa_spectra/spectra_born_nodrift_5.parquet"
NODRIFT_8_KAPPA = "05-spacing-n-stepping/05b-3bins/kappa_spectra/spectra_born_nodrift_8.parquet"
NODRIFT_10_KAPPA = "05-spacing-n-stepping/05b-3bins/kappa_spectra/spectra_born_nodrift_10.parquet"
NODRIFT_12_KAPPA = "05-spacing-n-stepping/05b-3bins/kappa_spectra/spectra_born_nodrift_12.parquet"
NODRIFT_16_KAPPA = "05-spacing-n-stepping/05b-3bins/kappa_spectra/spectra_born_nodrift_16.parquet"
NODRIFT_20_KAPPA = "05-spacing-n-stepping/05b-3bins/kappa_spectra/spectra_born_nodrift_20.parquet"
NODRIFT_25_KAPPA = "05-spacing-n-stepping/05b-3bins/kappa_spectra/spectra_born_nodrift_25.parquet"
NODRIFT_30_KAPPA = "05-spacing-n-stepping/05b-3bins/kappa_spectra/spectra_born_nodrift_30.parquet"
NODRIFT_40_KAPPA = "05-spacing-n-stepping/05b-3bins/kappa_spectra/spectra_born_nodrift_40.parquet"

# -----------------------------------------------------------------------------
# Density spectra feed the per-shell census (fig03-fig08); fig02 additionally uses the 10-run measured
# C_ell and the 40-run shell geometry. Kappa spectra feed the Born-convergence figures (fig09, fig10),
# drift and no-drift across the full shell-count sweep. Every HF parquet used is on its own line.
# -----------------------------------------------------------------------------
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

kappa_drift_5_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DRIFT_5_KAPPA}", split="train"))
kappa_drift_8_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DRIFT_8_KAPPA}", split="train"))
kappa_drift_10_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DRIFT_10_KAPPA}", split="train"))
kappa_drift_12_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DRIFT_12_KAPPA}", split="train"))
kappa_drift_16_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DRIFT_16_KAPPA}", split="train"))
kappa_drift_20_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DRIFT_20_KAPPA}", split="train"))
kappa_drift_25_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DRIFT_25_KAPPA}", split="train"))
kappa_drift_30_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DRIFT_30_KAPPA}", split="train"))
kappa_drift_40_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DRIFT_40_KAPPA}", split="train"))

kappa_nodrift_5_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{NODRIFT_5_KAPPA}", split="train")
)
kappa_nodrift_8_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{NODRIFT_8_KAPPA}", split="train")
)
kappa_nodrift_10_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{NODRIFT_10_KAPPA}", split="train")
)
kappa_nodrift_12_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{NODRIFT_12_KAPPA}", split="train")
)
kappa_nodrift_16_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{NODRIFT_16_KAPPA}", split="train")
)
kappa_nodrift_20_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{NODRIFT_20_KAPPA}", split="train")
)
kappa_nodrift_25_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{NODRIFT_25_KAPPA}", split="train")
)
kappa_nodrift_30_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{NODRIFT_30_KAPPA}", split="train")
)
kappa_nodrift_40_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{NODRIFT_40_KAPPA}", split="train")
)

# External reference for fig11: Born convergence on the CosmoGrid density shells (~70-100 Mpc/h thin shells,
# full N-body, nside 2048), computed with the SAME jax_fli born(). NOTE: this sim is at a DIFFERENT cosmology
# (sigma8=0.90, h=0.73) than the 05b runs — fig11 therefore ratios each measurement to the Limber theory at
# its OWN cosmology instead of ratioing the two measurements directly.
COSMOGRID_KAPPA = "00-cosmogrid/kappa_spectra/spectra_kappa_born_s3.parquet"

cosmogrid_kappa_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/{COSMOGRID_KAPPA}", split="train")
)
cosmo_cg = cosmogrid_kappa_cat.cosmology[0]  # deliberately NOT asserted equal to `cosmo` (different sim)
kappa_cosmogrid = cosmogrid_kappa_cat.field[0]  # 4 Stage-3 bins; fig11 uses the first 3

cosmo = drift_10_cat.cosmology[0]  # one fiducial cosmology shared by every run
for _c in (
    drift_5_cat,
    nodrift_5_cat,
    drift_8_cat,
    nodrift_8_cat,
    nodrift_10_cat,
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
    kappa_drift_40_cat,
    kappa_nodrift_40_cat,
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

# kappa C_ell keyed by shell count; each entry is the (3, n_ell) PowerSpectrum — one auto spectrum per bin.
kappa_drift = {
    n: c.field[0]
    for n, c in zip(
        NSHELLS_KAPPA,
        [
            kappa_drift_5_cat,
            kappa_drift_8_cat,
            kappa_drift_10_cat,
            kappa_drift_12_cat,
            kappa_drift_16_cat,
            kappa_drift_20_cat,
            kappa_drift_25_cat,
            kappa_drift_30_cat,
            kappa_drift_40_cat,
        ],
    )
}
kappa_nodrift = {
    n: c.field[0]
    for n, c in zip(
        NSHELLS_KAPPA,
        [
            kappa_nodrift_5_cat,
            kappa_nodrift_8_cat,
            kappa_nodrift_10_cat,
            kappa_nodrift_12_cat,
            kappa_nodrift_16_cat,
            kappa_nodrift_20_cat,
            kappa_nodrift_25_cat,
            kappa_nodrift_30_cat,
            kappa_nodrift_40_cat,
        ],
    )
}

# 10-run and 40-run shell edges (near, far) from the spectra metadata — used to match thin->thick shells.
chi10 = np.asarray(nodrift_10.comoving_centers)
w10 = np.asarray(nodrift_10.density_width)
edges10 = np.stack([chi10 - 0.5 * w10, chi10 + 0.5 * w10], axis=0)
chi40 = np.asarray(nodrift_40.comoving_centers)


# =============================================================================
# fig01 — local illustration: one particle cloud, three redshift-assignment colourings
# =============================================================================
def fig01_illustration():
    """Small local 256^3 sim in the real 5 Gpc/h box: paint particles in a thick radial bin, colour by the
    redshift each particle is assigned under (a) a 10-shell freeze, (b) the drift's smooth z(r), (c) a 40-shell
    freeze."""
    key = jax.random.PRNGKey(7)
    mesh_size, box_size, nside, n_steps = (256, 256, 256), (5000.0, 5000.0, 5000.0), 256, 10
    R0, R1 = 300.0, 2400.0  # thick radial bin, inside the box half-width (no tiling)
    cosmo_ill = jc.Planck18()  # a fresh cosmology for the illustration sim (the parquet-loaded one trips IC interp)

    def a_of_chi(chi):
        return np.asarray(jc.background.a_of_chi(cosmo_ill, jnp.atleast_1d(jnp.asarray(chi))))

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
    sel = (np.abs(pos[:, 2]) < 150.0) & (phi > 0.1) & (phi < 1.3) & (r > R0) & (r < R1)
    idx = np.where(sel)[0]
    rng = np.random.default_rng(0)
    if idx.size > 40000:
        idx = rng.choice(idx, 40000, replace=False)
    x, y, rr = pos[idx, 0], pos[idx, 1], r[idx]

    def banded_z(n):
        edges = np.linspace(R0, R1, n + 1)
        b = np.clip(np.digitize(rr, edges) - 1, 0, n - 1)
        return z_of(0.5 * (edges[:-1] + edges[1:]))[b]

    panels = [
        ("10 shells, no drift", banded_z(10)),
        ("10 shells, with drift", z_of(rr)),
        ("40 shells, no drift", banded_z(40)),
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
def _thick_ref_cl(target_shell):
    """Continuous-lightcone reference PowerSpectrum for one 10-run shell: sum the matching 40-run no-drift thin
    shells in COUNTS (per-pixel volume cancels on conversion), then overdensity -> angular_cl."""
    lo, hi = edges10[0, target_shell], edges10[1, target_shell]
    members = np.where((chi40 > lo) & (chi40 < hi))[0]
    files = [f"{root}/05-spacing-n-stepping/05b-3bins/density/exp5b_nodrift_40/shell_{i:04d}.parquet" for i in members]
    maps = [Catalog.from_dataset(load_dataset("parquet", data_files=f, split="train")).field[0] for f in files]
    counts = sum(np.asarray(m.to(jfli.DensityUnit.COUNTS).array) for m in maps)
    thick = (
        maps[0]
        .to(jfli.DensityUnit.COUNTS)
        .replace(
            array=jnp.asarray(counts),
            comoving_centers=jnp.asarray(0.5 * (lo + hi)),
            density_width=jnp.asarray(hi - lo),
        )
    )
    return thick.to(jfli.DensityUnit.OVERDENSITY).angular_cl(method="healpy", lmax=LMAX)


def fig02_density_shells():
    cols = [("near", NEAR_SHELL), ("mid", MID_SHELL), ("far", FAR_SHELL)]
    drift_b_all = drift_10.bin(nlb=32, lmin=2)
    nodrift_b_all = nodrift_10.bin(nlb=32, lmin=2)
    bc = np.asarray(drift_b_all.wavenumber)
    dl = bc * (bc + 1) / (2 * np.pi)
    drift_arr, nodrift_arr = np.asarray(drift_b_all.array), np.asarray(nodrift_b_all.array)
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16.5, 6.4), gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
    for col, (label, sh) in enumerate(cols):
        ref_b = np.asarray(_thick_ref_cl(sh).bin(nlb=32, lmin=2).array).reshape(-1)
        no_b = nodrift_arr[sh]
        dr_b = drift_arr[sh]
        ax_s, ax_r = axes[0, col], axes[1, col]
        ax_s.loglog(bc, dl * ref_b, color="k", lw=1.6, label="40-shell reference")
        ax_s.loglog(bc, dl * no_b, color="tab:red", lw=1.5, label="10-shell, no drift")
        ax_s.loglog(bc, dl * dr_b, color="tab:blue", lw=1.5, label="10-shell, with drift")
        ax_s.set_title(
            rf"{label} shell {sh}:  $\chi = {0.5 * (edges10[0, sh] + edges10[1, sh]):.0f}$ Mpc/h", fontsize=11
        )
        ax_s.grid(True, which="both", ls=":", alpha=0.4)
        # quantify the frozen-epoch bias each run carries vs the reference (largest for the thickest shells)
        inb = (bc >= 50) & (bc <= 800)
        dev_no = np.nanmedian(no_b[inb] / ref_b[inb]) - 1
        dev_dr = np.nanmedian(dr_b[inb] / ref_b[inb]) - 1
        ax_s.text(
            0.04,
            0.05,
            f"median bias ($\\ell\\in[50,800]$):\nno drift   {dev_no * 100:+.2f}\\%\nwith drift {dev_dr * 100:+.2f}\\%",
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
# fig09 — Born convergence C_ell per tomographic bin vs the number of shells (ratio to each 40-shell run)
# =============================================================================
def fig09_lensing():
    """Per source bin, the Born convergence C_ell for every shell count ratioed to its own 40-shell run. Three
    tomographic bins (Stage-3 [:3]) are stacked as row-pairs; no-drift and with-drift fill the two columns, and
    the ratio window adapts per bin. With scale-factor spacing the shells are thin, so the convergence settles
    cleanly onto the 40-shell run (the coarse 5-shell run is ~20-30% high in the low-z bin 1, within a few % by
    ~16 shells); the drift gives only a small head start (a few % at 5 shells, <1% by 16 shells), so the two
    columns nearly coincide — the Born projection washes the (already small) frozen-epoch error out of the
    convergence, in contrast to the equal-volume 05c where the fat inner shell defeats that averaging."""
    counts = [n for n in NSHELLS_KAPPA if n != 40]
    colors = {n: c for n, c in zip(counts, cm.viridis(np.linspace(0.0, 0.88, len(counts))))}
    nodrift_b = {n: kappa_nodrift[n].bin(nlb=32, lmin=2) for n in NSHELLS_KAPPA}
    drift_b = {n: kappa_drift[n].bin(nlb=32, lmin=2) for n in NSHELLS_KAPPA}
    bc = np.asarray(nodrift_b[40].wavenumber)
    dl = bc * (bc + 1) / (2 * np.pi)
    nbins = np.asarray(nodrift_b[40].array).shape[0]
    fig, axes = plt.subplots(
        2 * nbins, 2, figsize=(12.0, 13.5), gridspec_kw={"height_ratios": [3, 1] * nbins}, sharex="col"
    )
    for b in range(nbins):
        # data-driven ratio window shared by both columns of this source bin (bin 1 swings more than bin 3)
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
            ax_r.set_ylim(lo, hi)  # per-bin window (bin 1 swings more than bin 3)
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
    savefig(ASSETS / "fig09-lensing", fig)


# =============================================================================
# fig10 — Born convergence C_ell per tomographic bin ratioed to the Limber weak-lensing theory
# =============================================================================
def fig10_lensing_theory():
    """fig09 re-referenced to the Limber weak-lensing theory (the same Stage-3 [:3] source bins the Born sim
    used, x pixwin^2(2048)) instead of the 40-shell run. All counts track theory at large scales and fall below
    at small scales as the finite PM resolution and Born projection suppress power — a common deficit essentially
    independent of the drift; the per-bin ratio window adapts to each bin's spread."""
    counts = NSHELLS_KAPPA
    colors = {n: c for n, c in zip(counts, cm.viridis(np.linspace(0.0, 0.9, len(counts))))}
    pw2 = hp.pixwin(2048, lmax=LMAX) ** 2
    theory_b = (compute_theory_cl(cosmo, jnp.arange(LMAX + 1), get_stage3_nz_shear()[:3]) * pw2).bin(nlb=32, lmin=2)
    theory = np.asarray(theory_b.array)
    bc = np.asarray(theory_b.wavenumber)
    dl = bc * (bc + 1) / (2 * np.pi)
    nodrift_b = {n: kappa_nodrift[n].bin(nlb=32, lmin=2) for n in NSHELLS_KAPPA}
    drift_b = {n: kappa_drift[n].bin(nlb=32, lmin=2) for n in NSHELLS_KAPPA}
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
        hi = max(0.1, min(2.2, 1.05 * float(np.nanmax(rr))))
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
    savefig(ASSETS / "fig10-lensing-theory", fig)


# =============================================================================
# fig11 — 20-shell Born convergence vs the CosmoGrid Born reference (each ratioed to its OWN Limber theory)
# =============================================================================
def fig11_lensing_cosmogrid():
    """The 20-shell drift / no-drift Born convergence against the CosmoGrid Born convergence — the external
    reference computed with the SAME born() on CosmoGrid's ~70-100 Mpc/h thin shells (full N-body, nside 2048).
    CosmoGrid is a different realisation at a DIFFERENT cosmology (sigma8=0.90 vs 0.816), so each measurement is
    ratioed to the Limber weak-lensing theory at its own cosmology. With scale-factor shells the 20-shell runs
    track the CosmoGrid reference at low ell and depart only through the PM-resolution roll-off at high ell —
    the shelling itself is sound for lensing (contrast the equal-volume 05c fig11)."""
    pw2 = hp.pixwin(2048, lmax=LMAX) ** 2
    th_own_b = (compute_theory_cl(cosmo, jnp.arange(LMAX + 1), get_stage3_nz_shear()[:3]) * pw2).bin(nlb=32, lmin=2)
    th_cg_b = (compute_theory_cl(cosmo_cg, jnp.arange(LMAX + 1), get_stage3_nz_shear()[:3]) * pw2).bin(nlb=32, lmin=2)
    th_own = np.asarray(th_own_b.array)
    th_cg = np.asarray(th_cg_b.array)
    cg_b_all = kappa_cosmogrid.bin(nlb=32, lmin=2)
    no_b_all = kappa_nodrift[20].bin(nlb=32, lmin=2)
    dr_b_all = kappa_drift[20].bin(nlb=32, lmin=2)
    bc = np.asarray(th_own_b.wavenumber)
    dl = bc * (bc + 1) / (2 * np.pi)
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16.5, 6.4), gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
    for b in range(3):
        cg_b = np.asarray(cg_b_all.array)[b]
        no_b = np.asarray(no_b_all.array)[b]
        dr_b = np.asarray(dr_b_all.array)[b]
        tho_b = th_own[b]
        thc_b = th_cg[b]
        ax_s, ax_r = axes[0, b], axes[1, b]
        ax_s.loglog(bc, dl * cg_b, color="k", lw=1.6)
        ax_s.loglog(bc, dl * thc_b, color="k", ls=":", lw=1.1)
        ax_s.loglog(bc, dl * no_b, color="tab:red", lw=1.5)
        ax_s.loglog(bc, dl * dr_b, color="tab:blue", lw=1.5)
        ax_s.loglog(bc, dl * tho_b, color="0.45", ls="--", lw=1.1)
        ax_s.set_title(f"bin {b + 1}", fontsize=11)
        ax_s.grid(True, which="both", ls=":", alpha=0.4)
        if b == 0:
            ax_s.set_ylabel(r"$\ell(\ell+1)\,C_\ell^{\kappa\kappa}/2\pi$")
        ax_r.axhspan(-0.05, 0.05, color="0.7", alpha=0.3)
        ax_r.axhline(0.0, color="0.4", ls="--", lw=0.9)
        ax_r.semilogx(bc, cg_b / thc_b - 1.0, color="k", lw=1.5)
        ax_r.semilogx(bc, no_b / tho_b - 1.0, color="tab:red", lw=1.4)
        ax_r.semilogx(bc, dr_b / tho_b - 1.0, color="tab:blue", lw=1.4)
        ax_r.set_ylim(-0.7, 0.4)
        ax_r.set_xlabel(r"multipole $\ell$")
        ax_r.grid(True, which="both", ls=":", alpha=0.4)
        if b == 0:
            ax_r.set_ylabel("meas / own theory - 1")
    handles = [
        Line2D([], [], color="k", lw=1.6, label="CosmoGrid Born (thin shells, N-body)"),
        Line2D([], [], color="tab:red", lw=1.6, label="20 scale-factor shells, no drift"),
        Line2D([], [], color="tab:blue", lw=1.6, label="20 scale-factor shells, with drift"),
        Line2D([], [], color="0.45", ls="--", lw=1.4, label=r"Limber theory $\times\,w_\ell^2$ (run cosmology)"),
        Line2D([], [], color="k", ls=":", lw=1.4, label=r"Limber theory $\times\,w_\ell^2$ (CosmoGrid cosmology)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    savefig(ASSETS / "fig11-lensing-cosmogrid", fig)


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
    fig09_lensing()
    fig10_lensing_theory()
    fig11_lensing_cosmogrid()
    print(f"assets written to {ASSETS}")


if __name__ == "__main__":
    main()
