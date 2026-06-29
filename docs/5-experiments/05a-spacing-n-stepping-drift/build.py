"""Experiment 05a — drift on the lightcone (thick shells; density-shell C_ell + Born lensing): figures.

Renders the SVG figures for ``docs/5-experiments/05a-spacing-n-stepping-drift/README.md``. The drift moves each
particle to the scale factor at which it actually crosses the lightcone, removing the frozen-epoch error
of a thick shell — so a *drifted* coarse lightcone matches a *much finer* undrifted one.

  fig01   illustration (small local 256³ sim): the same particle cloud coloured by the
          redshift it is assigned under a 10-shell freeze, the drift's smooth z(r), and a 40-shell freeze.
  fig02   per-shell density C_ell at the near / mid / far shell: the 10-shell drift / no-drift runs vs a
          continuous-lightcone reference built by summing the matching 40-shell run (counts -> overdensity).
  fig03   Born convergence C_ell vs the number of shells, drift vs no-drift, each ratioed to its 40-shell run.

Run from the repo root (CPU is fine; fig01 runs a small sim, fig02 loads a few nside-2048 maps):
    JAX_PLATFORMS=cpu uv run --no-sync python docs/5-experiments/05a-spacing-n-stepping-drift/build.py
"""

from __future__ import annotations

import jax

# float64 globally BEFORE jax_cosmo (the enable_x64() context manager breaks jax_cosmo's
# pure_callback comoving-distance cache; the global config flag is the safe route).
jax.config.update("jax_enable_x64", True)

import sys
from pathlib import Path

import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from huggingface_hub import snapshot_download
from matplotlib import cm
from matplotlib.lines import Line2D

import jax_fli as jfli
from jax_fli.io import Catalog

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _exputils import savefig, set_style  # noqa: E402

ASSETS = HERE / "assets"
REPO = "ASKabalan/jax-fli-experiments"

NEAR_SHELL, MID_SHELL, FAR_SHELL = 1, 5, 9  # which 10-run shell each fig02 column zooms in on
NSHELLS_KAPPA = [5, 8, 10, 12, 16, 20, 25, 30, 40]  # fig03 sweep (drift vs no-drift Born C_ell)
NLB = 16
LMAX = 1500  # the published spectra stop at ell 1500

root = snapshot_download(REPO, repo_type="dataset", local_files_only=True)

DRIFT_10_SPECTRA = "05-spacing-n-stepping/05a-drift/density_spectra/spectra_exp5a_drift_10.parquet"
NODRIFT_10_SPECTRA = "05-spacing-n-stepping/05a-drift/density_spectra/spectra_exp5a_nodrift_10.parquet"
NODRIFT_40_SPECTRA = "05-spacing-n-stepping/05a-drift/density_spectra/spectra_exp5a_nodrift_40.parquet"

DRIFT_5_KAPPA = "05-spacing-n-stepping/05a-drift/kappa_spectra/spectra_born_drift_5.parquet"
DRIFT_8_KAPPA = "05-spacing-n-stepping/05a-drift/kappa_spectra/spectra_born_drift_8.parquet"
DRIFT_10_KAPPA = "05-spacing-n-stepping/05a-drift/kappa_spectra/spectra_born_drift_10.parquet"
DRIFT_12_KAPPA = "05-spacing-n-stepping/05a-drift/kappa_spectra/spectra_born_drift_12.parquet"
DRIFT_16_KAPPA = "05-spacing-n-stepping/05a-drift/kappa_spectra/spectra_born_drift_16.parquet"
DRIFT_20_KAPPA = "05-spacing-n-stepping/05a-drift/kappa_spectra/spectra_born_drift_20.parquet"
DRIFT_25_KAPPA = "05-spacing-n-stepping/05a-drift/kappa_spectra/spectra_born_drift_25.parquet"
DRIFT_30_KAPPA = "05-spacing-n-stepping/05a-drift/kappa_spectra/spectra_born_drift_30.parquet"
DRIFT_40_KAPPA = "05-spacing-n-stepping/05a-drift/kappa_spectra/spectra_born_drift_40.parquet"

NODRIFT_5_KAPPA = "05-spacing-n-stepping/05a-drift/kappa_spectra/spectra_born_nodrift_5.parquet"
NODRIFT_8_KAPPA = "05-spacing-n-stepping/05a-drift/kappa_spectra/spectra_born_nodrift_8.parquet"
NODRIFT_10_KAPPA = "05-spacing-n-stepping/05a-drift/kappa_spectra/spectra_born_nodrift_10.parquet"
NODRIFT_12_KAPPA = "05-spacing-n-stepping/05a-drift/kappa_spectra/spectra_born_nodrift_12.parquet"
NODRIFT_16_KAPPA = "05-spacing-n-stepping/05a-drift/kappa_spectra/spectra_born_nodrift_16.parquet"
NODRIFT_20_KAPPA = "05-spacing-n-stepping/05a-drift/kappa_spectra/spectra_born_nodrift_20.parquet"
NODRIFT_25_KAPPA = "05-spacing-n-stepping/05a-drift/kappa_spectra/spectra_born_nodrift_25.parquet"
NODRIFT_30_KAPPA = "05-spacing-n-stepping/05a-drift/kappa_spectra/spectra_born_nodrift_30.parquet"
NODRIFT_40_KAPPA = "05-spacing-n-stepping/05a-drift/kappa_spectra/spectra_born_nodrift_40.parquet"

# -----------------------------------------------------------------------------
# Density spectra: the 10-shell drift / no-drift runs (fig02 lines) + the 40-shell no-drift run (its
# shell geometry tells fig02 which thin shells to sum for the reference). Kappa spectra: the full
# shell-count sweep, drift and no-drift (fig03). Every HF parquet used is on its own line.
# -----------------------------------------------------------------------------
drift_10_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{DRIFT_10_SPECTRA}", split="train"))
nodrift_10_cat = Catalog.from_dataset(load_dataset("parquet", data_files=f"{root}/{NODRIFT_10_SPECTRA}", split="train"))
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

cosmo = drift_10_cat.cosmology[0]  # one fiducial cosmology shared by every run
for _c in (nodrift_10_cat, nodrift_40_cat, kappa_drift_40_cat, kappa_nodrift_40_cat):
    assert np.isclose(float(_c.cosmology[0].Omega_c), float(cosmo.Omega_c))
    assert np.isclose(float(_c.cosmology[0].sigma8), float(cosmo.sigma8))

drift_10 = drift_10_cat.field[0]
nodrift_10 = nodrift_10_cat.field[0]
nodrift_40 = nodrift_40_cat.field[0]
ell_full = np.asarray(drift_10.wavenumber)

# kappa C_ell (one source bin -> 1-D array each), keyed by shell count.
kappa_drift = {
    n: np.asarray(c.field[0].array).reshape(-1)
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
    n: np.asarray(c.field[0].array).reshape(-1)
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
# fig01 — local illustration: one particle cloud, three redshift-assignment colourings
# =============================================================================
def fig01_illustration():
    """Small local 256³ sim: paint particles in a thick radial bin, colour by the redshift each
    particle is assigned under (a) a 10-shell freeze, (b) the drift's smooth z(r), (c) a 40-shell freeze."""
    key = jax.random.PRNGKey(7)
    mesh_size, box_size, nside, n_steps = (256, 256, 256), (2000.0, 2000.0, 2000.0), 256, 10
    R0, R1 = 120.0, 950.0  # thick radial bin, inside the box half-width (no tiling)
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
    sel = (np.abs(pos[:, 2]) < 80.0) & (phi > 0.1) & (phi < 1.3) & (r > R0) & (r < R1)
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
    """Continuous-lightcone reference C_ell for one 10-run shell: sum the matching 40-run no-drift thin
    shells in COUNTS (per-pixel volume cancels on conversion), then overdensity -> angular_cl."""
    lo, hi = edges10[0, target_shell], edges10[1, target_shell]
    members = np.where((chi40 > lo) & (chi40 < hi))[0]
    files = [f"{root}/05-spacing-n-stepping/05a-drift/density/exp5a_nodrift_40/shell_{i:04d}.parquet" for i in members]
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
    ps = thick.to(jfli.DensityUnit.OVERDENSITY).angular_cl(method="healpy", lmax=LMAX)
    return np.asarray(ps.array).reshape(-1)


def fig02_density_shells():
    cols = [("near", NEAR_SHELL), ("mid", MID_SHELL), ("far", FAR_SHELL)]
    drift_arr, nodrift_arr = np.asarray(drift_10.array), np.asarray(nodrift_10.array)
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16.5, 6.4), gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
    for col, (label, sh) in enumerate(cols):
        ref = _thick_ref_cl(sh)
        bc, ref_b = _logbin(ell_full, ref)
        _, no_b = _logbin(ell_full, nodrift_arr[sh])
        _, dr_b = _logbin(ell_full, drift_arr[sh])
        ax_s, ax_r = axes[0, col], axes[1, col]
        ax_s.loglog(bc, ref_b, color="k", lw=1.6, label="40-shell reference")
        ax_s.loglog(bc, no_b, color="tab:red", lw=1.5, label="10-shell, no drift")
        ax_s.loglog(bc, dr_b, color="tab:blue", lw=1.5, label="10-shell, with drift")
        ax_s.set_title(f"{label} shell {sh}:  χ = {0.5 * (edges10[0, sh] + edges10[1, sh]):.0f} Mpc/h", fontsize=11)
        ax_s.grid(True, which="both", ls=":", alpha=0.4)
        # quantify the (sub-percent at 10 shells) frozen-epoch bias each run carries vs the reference
        inb = (bc >= 50) & (bc <= 800)
        dev_no = np.nanmedian(no_b[inb] / ref_b[inb]) - 1
        dev_dr = np.nanmedian(dr_b[inb] / ref_b[inb]) - 1
        ax_s.text(
            0.04,
            0.05,
            f"median bias (ℓ∈[50,800]):\nno drift   {dev_no:+.2%}\nwith drift {dev_dr:+.2%}",
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
# fig03 — Born convergence C_ell vs the number of shells, drift vs no-drift (ratio to each 40-shell run)
# =============================================================================
def fig03_lensing():
    counts = [n for n in NSHELLS_KAPPA if n != 40]
    colors = {n: c for n, c in zip(counts, cm.viridis(np.linspace(0.0, 0.88, len(counts))))}
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13.5, 6.6), gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
    for col, (label, kappa) in enumerate([("no drift", kappa_nodrift), ("with drift", kappa_drift)]):
        bc, ref_b = _logbin(ell_full, kappa[40])
        ax_s, ax_r = axes[0, col], axes[1, col]
        ax_s.loglog(bc, ref_b, color="k", lw=1.8, label="40 shells")
        for n in counts:
            _, c_b = _logbin(ell_full, kappa[n])
            ax_s.loglog(bc, c_b, color=colors[n], lw=1.2)
        ax_s.set_title(f"Born convergence — {label}", fontsize=12)
        ax_s.grid(True, which="both", ls=":", alpha=0.4)
        ax_s.set_ylabel(r"$C_\ell^{\kappa\kappa}$") if col == 0 else None
        ax_r.axhspan(0.97, 1.03, color="0.7", alpha=0.3)
        ax_r.axhline(1.0, color="0.4", ls="--", lw=0.9)
        for n in counts:
            _, c_b = _logbin(ell_full, kappa[n])
            ax_r.semilogx(bc, c_b / ref_b, color=colors[n], lw=1.2)
        ax_r.set_ylim(0.93, 1.07)
        ax_r.set_xlabel(r"multipole $\ell$")
        ax_r.grid(True, which="both", ls=":", alpha=0.4)
        ax_r.set_ylabel("meas / 40 shells") if col == 0 else None
    handles = [Line2D([], [], color=colors[n], lw=1.6, label=f"{n} shells") for n in counts]
    handles += [
        Line2D([], [], color="k", lw=1.8, label="40 shells (reference)"),
        Line2D([], [], color="0.7", lw=6, alpha=0.5, label=r"$\pm3\%$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    savefig(ASSETS / "fig03-lensing", fig)


def main():
    set_style()
    fig01_illustration()
    fig02_density_shells()
    fig03_lensing()
    print(f"assets written to {ASSETS}")


if __name__ == "__main__":
    main()
