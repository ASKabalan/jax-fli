"""Experiment 05c — equal-volume shells (density-shell C_ell): figures.

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

Every 05c run is published for both drift and no-drift up to 40 shells, so the census covers both. Born lensing
is not yet published for 05c, so there is no convergence figure here.

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
from matplotlib.lines import Line2D

import jax_fli as jfli
from jax_fli import compute_theory_cl_for_density
from jax_fli.io import Catalog

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _exputils import savefig, set_style  # noqa: E402

ASSETS = HERE / "assets"
REPO = "ASKabalan/jax-fli-experiments"

LMAX = 1500  # the published spectra stop at ell 1500
BOX, MESH = 5000.0, 2560.0  # per-shell PM-Nyquist line ell_max ~ pi*chi/dx, dx = box/mesh (equal-volume box)
CENSUS_RATIO_YLIM = (0.7, 1.35)  # ratio-to-theory strip range (wide enough for the fat inner shell)

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
ell_full = np.asarray(nodrift_10.wavenumber)

# Equal-volume shell edges (near, far) from the spectra metadata. NOTE: reconstruct them from the stored
# density_width (comoving_centers ± width/2) — the centre-reflection helper jax_fli.utils.edges assumes
# uniform-ish widths and produces oscillating edges for equal-volume spacing.
chi10 = np.asarray(nodrift_10.comoving_centers)
w10 = np.asarray(nodrift_10.density_width)
edges10 = np.stack([chi10 - 0.5 * w10, chi10 + 0.5 * w10], axis=0)
chi30 = np.asarray(nodrift_30.comoving_centers)
w30 = np.asarray(nodrift_30.density_width)
chi40 = np.asarray(nodrift_40.comoving_centers)


def _logbin(ell, y, nb=20):
    """Geometric-mean bandpower binning of a 1-D C_ell for clean log-log curves and ratios."""
    m = ell >= 2
    e, yy = ell[m], np.asarray(y)[m]
    edges = np.unique(np.round(np.logspace(np.log10(2), np.log10(e.max()), nb)).astype(int))
    bc, bv = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (e >= lo) & (e < hi)
        if s.any():
            bc.append(np.sqrt(lo * hi))
            bv.append(np.nanmean(yy[s]))
    return np.asarray(bc), np.asarray(bv)


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
    volume cancels on conversion), then overdensity -> angular_cl. Used for both the 10-shell measurement
    and the 40-shell continuous-lightcone reference."""
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
    ps = slab.to(jfli.DensityUnit.OVERDENSITY).angular_cl(method="healpy", lmax=LMAX)
    return np.asarray(ps.array).reshape(-1)


def fig02_density_shells():
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16.5, 6.4), gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
    for col, (label, sh) in enumerate(COLUMNS):
        lo, hi = float(edges10[0, sh[0]]), float(edges10[1, sh[-1]])
        members40 = np.where((chi40 >= lo) & (chi40 <= hi))[0]  # whole 40-shells whose centre is inside the region
        ref = _region_cl("exp5c_nodrift_40", members40, lo, hi)
        no = _region_cl("exp5c_nodrift_10", sh, lo, hi)
        dr = _region_cl("exp5c_drift_10", sh, lo, hi)
        bc, ref_b = _logbin(ell_full, ref)
        _, no_b = _logbin(ell_full, no)
        _, dr_b = _logbin(ell_full, dr)
        ax_s, ax_r = axes[0, col], axes[1, col]
        ax_s.loglog(bc, ref_b, color="k", lw=1.6, label="40-shell reference")
        ax_s.loglog(bc, no_b, color="tab:red", lw=1.5, label="10-shell, no drift")
        ax_s.loglog(bc, dr_b, color="tab:blue", lw=1.5, label="10-shell, with drift")
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
            ax_s.set_ylabel(r"$C_\ell$")
        ax_r.axhspan(0.95, 1.05, color="0.7", alpha=0.3)
        ax_r.axhline(1.0, color="0.4", ls="--", lw=0.9)
        ax_r.semilogx(bc, no_b / ref_b, color="tab:red", lw=1.4)
        ax_r.semilogx(bc, dr_b / ref_b, color="tab:blue", lw=1.4)
        ax_r.set_ylim(0.95, 1.05)
        ax_r.set_xlabel(r"multipole $\ell$")
        ax_r.grid(True, which="both", ls=":", alpha=0.4)
        if col == 0:
            ax_r.set_ylabel("meas / 40-shell")
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
    noise cancels between the two runs) as the drift's frozen-epoch correction, not the distance from 1."""
    pw2 = hp.pixwin(2048, lmax=LMAX) ** 2
    theory = np.asarray((compute_theory_cl_for_density(cosmo, spec_no, jnp.arange(LMAX + 1)) * pw2).array)
    no = np.asarray(spec_no.array)
    dr = np.asarray(spec_dr.array)
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
        bc, th_b = _logbin(ell_full, theory[i])
        _, no_b = _logbin(ell_full, no[i])
        _, dr_b = _logbin(ell_full, dr[i])
        ax_s.loglog(bc, th_b, "k--", lw=1.0)
        ax_s.loglog(bc, no_b, color="tab:red", lw=1.0)
        ax_s.loglog(bc, dr_b, color="tab:blue", lw=1.0)
        ax_r.axhspan(0.95, 1.05, color="0.7", alpha=0.3)
        ax_r.axhline(1.0, color="0.4", ls="--", lw=0.8)
        ax_r.semilogx(bc, no_b / th_b, color="tab:red", lw=1.0)
        ax_r.semilogx(bc, dr_b / th_b, color="tab:blue", lw=1.0)
        ax_r.set_ylim(*CENSUS_RATIO_YLIM)
        lmax_sh = np.pi * chi[i] / dx  # PM Nyquist; beyond it the comparison is resolution-limited
        for ax in (ax_s, ax_r):
            ax.axvline(lmax_sh, color="0.6", ls=":", lw=0.8)
            ax.grid(True, which="both", ls=":", alpha=0.35)
            ax.tick_params(labelsize=7)
        ax_s.set_title(rf"$\chi={chi[i]:.0f}$", fontsize=8)
        ax_s.tick_params(labelbottom=False)
        if c == 0:
            ax_s.set_ylabel(r"$C_\ell$", fontsize=8)
            ax_r.set_ylabel("meas/th", fontsize=7)
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
    subs[1].suptitle("5 shells", fontsize=12)
    _draw_census(subs[2], nodrift_8, drift_8, 2, 4, top=0.88, bottom=0.09)
    subs[2].suptitle("8 shells", fontsize=12)
    _draw_census(subs[3], nodrift_10, drift_10, 2, 5, top=0.88, bottom=0.09)
    subs[3].suptitle("10 shells", fontsize=12)
    savefig(ASSETS / "fig03-density-census-small", fig)


def density_census(spec_no, spec_dr, nrows, ncols, stem):
    """One run's census on an nrows x ncols grid (one shell per cell)."""
    fig = plt.figure(figsize=(2.5 * ncols, 2.9 * nrows))
    _draw_census(fig, spec_no, spec_dr, nrows, ncols, top=0.9, bottom=0.06)
    _census_legend(fig, loc="upper center", bbox_to_anchor=(0.5, 0.99))
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
    print(f"assets written to {ASSETS}")


if __name__ == "__main__":
    main()
