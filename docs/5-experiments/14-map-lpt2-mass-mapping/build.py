"""Experiment 14 figures: MAP mass mapping with 2LPT on 4 GPUs (pilot).

Reads the two pilot runs of ``docs/3-sampling-and-inference/15-LPT-Multihost-MassMapping.py`` (``run.sh``,
``WHAT=pilot``) from ``data/`` and draws the figures of ``14-LPT-MassMapping.ipynb`` for each survey, plus the
DES Y3 vs Euclid comparison. The runs are local (not yet on HuggingFace); CPU is enough:

    JAX_PLATFORMS=cpu uv run python build.py
"""

from __future__ import annotations

import jax

# float64 globally BEFORE jax_cosmo (the enable_x64() context manager breaks jax_cosmo's
# pure_callback comoving-distance cache; the global config flag is the safe route).
jax.config.update("jax_enable_x64", True)

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import datasets  # noqa: E402
import healpy as hp  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax_cosmo as jc  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import jax_fli as jfli  # noqa: E402
from jax_fli.data.nz import get_des_y3_nz_shear  # noqa: E402
from jax_fli.fields.density import plot_3d_density  # noqa: E402
from jax_fli.io import Catalog  # noqa: E402
from jax_fli.summary_statistics import starlet_coefficients_spherical  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _exputils import savefig, set_style  # noqa: E402

datasets.disable_progress_bar()

ASSETS = HERE / "assets"
DES = HERE / "data/PILOT_MESH512_DES"  # the run names say 512, the runs are 300^3 (run.sh PILOT)
EUCLID = HERE / "data/PILOT_MESH512_EUCLID"

# the PILOT line of run.sh and the defaults of the 15 script
MESH = 300
NSIDE = 128  # = PAINT_NSIDE in the pilot
ELL_MIN, ELL_MAX, ELL_TAPER = 2, 192, 32
N_SHELLS, MIN_WIDTH, MAX_WIDTH, R_MIN = 22, 50.0, 150.0, 300.0  # Mpc/h
DES_BINS = (1, 2)  # DES Y3 bins 2 and 3
MAX_Z = 1.0
KAPPA_PLOT_LIMIT = 5  # MAP columns in the filmstrip

cosmo = jc.Planck18()
box = tuple(float(x) for x in jfli.utils.compute_box_size_from_redshift(cosmo, MAX_Z, (0.5, 0.5, 0.5)))
L, chi_max = box[0], box[0] / 2
k_nyq = np.pi * MESH / L
n_bins = len(DES_BINS)
NZ_DES = [get_des_y3_nz_shear(zmax=MAX_Z)[i] for i in DES_BINS]
NZ_EUCLID = [get_des_y3_nz_shear(gals_per_arcmin2=[30.0 / 4] * 4, zmax=MAX_Z)[i] for i in DES_BINS]
zz = jnp.linspace(1e-3, MAX_Z, 512)
BIN = [
    f"source bin {i + 1} ($\\langle z\\rangle = {float(jnp.trapezoid(zz * nz(zz), zz) / jnp.trapezoid(nz(zz), zz)):.2f}$)"
    for i, nz in zip(DES_BINS, NZ_DES)
]

_, r_c, widths = jfli.resolve_geometry(
    cosmo, chi_max, nb_shells=N_SHELLS, shell_spacing="equal_vol", min_width=MIN_WIDTH, max_width=MAX_WIDTH, r_min=R_MIN
)
order = np.argsort(np.asarray(r_c))
r_lo = np.asarray(r_c)[order] - np.asarray(widths)[order] / 2
r_hi = np.asarray(r_c)[order] + np.asarray(widths)[order] / 2

ONE = dict(color="0.5", ls="--", lw=1)  # the reference line at 1
BANDS = ((2, 30), (30, 60), (60, 100), (100, 150), (150, ELL_MAX))
ZOOM_ROT, ZOOM_RESO, ZOOM_NPIX = (30.0, 25.0), 15.0, 64  # 16 deg x 16 deg at 15 arcmin

ST_NSIDE, N_SCALES = int(2 ** np.ceil(np.log2(ELL_MAX / 3))), 5
l_hi = 3 * ST_NSIDE / 2 ** np.arange(N_SCALES)
SCALE_BANDS = [f"$\\ell\\approx{l_hi[j] / 2:.0f}$--${l_hi[j]:.0f}$" for j in range(N_SCALES - 1)] + [
    f"coarse, $\\ell\\leq{l_hi[-1]:.0f}$"
]
NU_EDGES = np.linspace(-5.0, 5.0, 41)
NU = 0.5 * (NU_EDGES[1:] + NU_EDGES[:-1])


# ---------------------------------------------------------------- loading


def load_run(out, name, tag, color, nz, sigma_e):
    static = lambda n: Catalog.from_parquet(str(out / f"{n}.parquet")).field[0]
    prefix = {"ic": "ic", "density": "lc", "kappa": "kappa", "coh_ic": "coh_ic", "trans_ic": "trans_ic"}
    files = {
        n: sorted((out / f"{n}_evolution").glob(f"{p}_*.parquet"), key=lambda f: int(f.stem.split("_")[-1]))
        for n, p in prefix.items()
    }
    R = SimpleNamespace(out=out, name=name, tag=tag, color=color, nz=nz, sigma_e=sigma_e)
    R.truth_ic, R.truth_lc, R.lattice_lc, R.truth_kappa = (
        static(n) for n in ("true_ic", "truth_density_lightcone", "lattice_density_lightcone", "truth_kappa")
    )
    R.load = lambda n, i: Catalog.from_parquet(str(files[n][i])).field[0]
    R.n_frames = len(files["ic"])
    R.steps = [int(r["step"]) for r in sorted(json.loads((out / "metrics.json").read_text()), key=lambda r: r["frame"])]
    R.summary = json.loads((out / "summary.json").read_text())
    assert R.truth_ic.array.shape == (MESH,) * 3 and R.truth_kappa.nside == NSIDE
    assert np.isclose(R.summary["sigma_e"], sigma_e) and R.truth_lc.array.shape[0] == N_SHELLS
    R.ic_final = R.load("ic", -1)
    R.truth_arr, R.final_arr = np.asarray(R.truth_ic.array), np.asarray(R.ic_final.array, dtype=np.float64)
    R.kt = np.asarray(R.truth_kappa.array)
    R.kf = [np.asarray(R.load("kappa", i).array, dtype=np.float64) for i in range(R.n_frames)]
    R.r_kappa = np.array([[np.corrcoef(R.kt[b], k[b])[0, 1] for b in range(n_bins)] for k in R.kf])
    R.mid = 1 + int(np.argmin(np.abs(R.r_kappa[1:-1].mean(axis=1) - 0.5 * R.r_kappa[-1].mean())))
    R.sel = sorted(set(range(R.n_frames - 1, -1, -max(1, R.n_frames // KAPPA_PLOT_LIMIT))) | {0}, reverse=True)
    R.l1_truth, R.l1_final, R.starlet_r = {}, {}, {}
    print(f"{name}: {R.n_frames} frames, intermediate step {R.steps[R.mid]}")
    return R


# ---------------------------------------------------------------- figures, one survey


def plot_loss(R):
    hist = np.load(R.out / "loss_history.npz")
    step, U, g = hist["steps"], hist["loss"], hist["grad_norm"]
    dU = U - U.min()
    set_style(9.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.0))
    ax1.loglog(step[dU > 0], dU[dU > 0], color=R.color)
    ax1.set(xlabel="Adam step", ylabel="$U - U_{\\min}$ (negative log posterior)")
    ax2.loglog(step, g, color=R.color)
    ax2.set(xlabel="Adam step", ylabel="gradient norm $\\|\\nabla_w U\\|$")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    savefig(ASSETS / f"fig01-loss-{R.tag}", fig)


def plot_kappa_filmstrip(R):
    fields = [R.truth_kappa] + [R.load("kappa", i) for i in R.sel]
    labels = ["truth"] + [
        f"MAP, step {R.steps[i]}" + (" (final)" if i == R.n_frames - 1 else " (first guess)" if i == 0 else "")
        for i in R.sel
    ]
    vmin, vmax = np.percentile(np.concatenate([np.asarray(f.array).ravel() for f in fields]), [0.1, 99.9])
    set_style(2.6 * len(fields))
    fig, axes = plt.subplots(n_bins, len(fields), figsize=(2.6 * len(fields), 2.0 * n_bins), squeeze=False)
    for c, (lab, fld) in enumerate(zip(labels, fields)):
        fld.plot(ax=list(axes[:, c]), titles=[f"{lab}\n{z}" for z in BIN], vmin=vmin, vmax=vmax)
    savefig(ASSETS / f"fig02-kappa-filmstrip-{R.tag}", fig)


def plot_lightcone(R):
    shown = np.linspace(0, N_SHELLS - 1, 6).astype(int)
    lat = np.asarray(R.lattice_lc.array)[shown]
    rho_bar = lat.mean(axis=1, keepdims=True)
    rows = {
        "truth": (np.asarray(R.truth_lc.array)[shown] - lat) / rho_bar,
        "MAP": (np.asarray(R.load("density", -1).array, dtype=np.float64)[shown] - lat) / rho_bar,
    }
    set_style(15.0)
    fig, axes = plt.subplots(2, len(shown), figsize=(15.0, 4.0), squeeze=False)
    for r, (lab, arr) in enumerate(rows.items()):
        lim = float(np.percentile(np.abs(arr), 99.5))
        R.truth_lc.replace(array=jnp.asarray(arr)).plot(
            ax=list(axes[r]),
            titles=[f"{lab}, {r_lo[s]:.0f}--{r_hi[s]:.0f} Mpc/h" for s in shown],
            vmin=-lim,
            vmax=lim,
            cmap="RdBu_r",
        )
    savefig(ASSETS / f"fig03-lightcone-{R.tag}", fig)


def spectra(a, b):
    # per-ell coherence and transfer of map b against map a, ELL_MIN <= ell < ELL_MAX, plus per-band coherence
    caa, cbb, cab = hp.anafast(a, lmax=ELL_MAX), hp.anafast(b, lmax=ELL_MAX), hp.anafast(a, b, lmax=ELL_MAX)
    ell = np.arange(ELL_MAX + 1)
    w = 2 * ell + 1
    per_band = [
        np.sum((w * cab)[lo:hi]) / np.sqrt(np.sum((w * caa)[lo:hi]) * np.sum((w * cbb)[lo:hi])) for lo, hi in BANDS
    ]
    sl = slice(ELL_MIN, ELL_MAX)
    return ell[sl], (cab / np.sqrt(caa * cbb))[sl], np.sqrt(cbb / caa)[sl], per_band


def plot_ic_projection(R):
    proj_t = R.truth_ic.sky_projection(cosmo, R.nz, nside=NSIDE, r_min=R_MIN)
    pt = np.asarray(proj_t.array)
    pp = np.asarray(R.ic_final.sky_projection(cosmo, R.nz, nside=NSIDE, r_min=R_MIN).array, dtype=np.float64)
    R.pt, R.pp = pt, pp
    lim_t, lim_d = float(np.percentile(np.abs(pt), 99.8)), float(np.percentile(np.abs(pt - pp), 99.8))
    set_style(12.0)
    fig, axes = plt.subplots(n_bins, 3, figsize=(12.0, 3.0 * n_bins), squeeze=False)
    proj_t.plot(ax=list(axes[:, 0]), titles=[f"truth, projected IC\n{z}" for z in BIN], vmin=-lim_t, vmax=lim_t)
    proj_t.replace(array=jnp.asarray(pp)).plot(
        ax=list(axes[:, 1]), titles=[f"MAP, projected IC\n{z}" for z in BIN], vmin=-lim_t, vmax=lim_t
    )
    proj_t.replace(array=jnp.asarray(pt - pp)).plot(
        ax=list(axes[:, 2]), titles=[f"truth $-$ MAP\n{z}" for z in BIN], vmin=-lim_d, vmax=lim_d, cmap="RdBu_r"
    )
    savefig(ASSETS / f"fig04-ic-projection-{R.tag}", fig)

    patch = lambda m: hp.gnomview(
        m, rot=ZOOM_ROT, reso=ZOOM_RESO, xsize=ZOOM_NPIX, return_projected_map=True, no_plot=True
    )
    half = ZOOM_RESO * ZOOM_NPIX / 120
    set_style(11.0)
    fig, axes = plt.subplots(n_bins, 3, figsize=(11.0, 3.3 * n_bins), squeeze=False)
    for b in range(n_bins):
        t, p = patch(pt[b]), patch(pp[b])
        lim = float(np.abs(t).max())
        for c, (a, lab, cmap) in enumerate(
            ((t, "truth", "magma"), (p, "MAP", "magma"), (t - p, "truth $-$ MAP", "RdBu_r"))
        ):
            im = axes[b, c].imshow(a, origin="lower", extent=[-half, half, -half, half], cmap=cmap, vmin=-lim, vmax=lim)
            axes[b, c].set(title=f"{lab}, {BIN[b]}", xlabel="degrees", ylabel="degrees")
            plt.colorbar(im, ax=axes[b, c], fraction=0.046, label="projected IC")
    fig.tight_layout()
    savefig(ASSETS / f"fig05-ic-projection-patch-{R.tag}", fig)

    R.proj_rows, R.proj_curves = [], []
    for b in range(n_bins):
        ell, r_l, T_l, per_band = spectra(pt[b], pp[b])
        keep = ell <= ELL_MAX - ELL_TAPER
        R.proj_rows.append((BIN[b], float(r_l[keep].mean()), float(T_l[keep].mean()), per_band))
        R.proj_curves.append((ell, r_l))


def cube_faces(sky):
    # the sky map along the ray from the observer through each voxel of the three faces plot_3d_density draws
    g = np.arange(MESH) - 0.5 * MESH
    u, v = np.meshgrid(g, g, indexing="ij")
    near, far = np.full_like(u, g[0]), np.full_like(u, g[-1])
    vol = np.zeros((MESH,) * 3, np.float32)
    for face, (x, y, z) in (
        ((slice(None), slice(None), 0), (u, v, near)),
        ((slice(None), 0, slice(None)), (u, near, v)),
        ((MESH - 1, slice(None), slice(None)), (far, u, v)),
    ):
        vol[face] = sky[hp.vec2pix(NSIDE, x.ravel(), y.ravel(), z.ravel())].reshape(MESH, MESH)
    return vol


def plot_ic_cube(R):
    t, p = R.pt.mean(axis=0), R.pp.mean(axis=0)
    lim_t, lim_d = float(np.percentile(np.abs(t), 99.8)), float(np.percentile(np.abs(t - p), 99.8))
    set_style(15.0)
    plt.rcParams["savefig.dpi"] = 200  # the faces are rasterized: 3 x 300^2 contour patches as vectors weigh 250 MB
    fig = plt.figure(figsize=(15.0, 4.8))
    for i, (sky, lab, cmap, lim) in enumerate(
        ((t, "truth", "magma", lim_t), (p, "MAP", "magma", lim_t), (t - p, "truth $-$ MAP", "RdBu_r", lim_d))
    ):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        plot_3d_density(
            ax, cube_faces(sky), project_slices=1, vmin=-lim, vmax=lim, cmap=cmap, labels=("", "", ""), colorbar=True
        )
        for c in ax.collections:
            c.set_rasterized(True)
        ax.set_title(lab)
    savefig(ASSETS / f"fig06-ic-cube-{R.tag}", fig)


def plot_ic_slices(R):
    k0 = MESH // 2
    ext = [-chi_max, chi_max, -chi_max, chi_max]
    t_s, p_s = R.truth_arr[..., k0], R.final_arr[..., k0]
    set_style(14.0)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    for ax, a, lab, cmap, lim in (
        (axes[0], t_s, "truth IC", "magma", np.abs(t_s).max()),
        (axes[1], p_s, "MAP IC (own colour scale)", "magma", np.abs(p_s).max()),
        (axes[2], t_s - p_s, "truth $-$ MAP", "RdBu_r", np.abs(t_s).max()),
    ):
        im = ax.imshow(a.T, origin="lower", extent=ext, cmap=cmap, vmin=-lim, vmax=lim)
        for rad, ls in ((R_MIN, ":"), (chi_max, "--")):
            ax.add_patch(plt.Circle((0, 0), rad, fill=False, ls=ls, color="w" if cmap == "magma" else "k", lw=1))
        ax.set(title=lab, xlabel="$x$ [Mpc/h]", ylabel="$y$ [Mpc/h]")
        plt.colorbar(im, ax=ax, fraction=0.046, label="linear density contrast at $z = 0$")
    fig.tight_layout()
    savefig(ASSETS / f"fig07-ic-slice-{R.tag}", fig)

    preds = [np.asarray(R.load("ic", i).array, dtype=np.float64)[..., k0] for i in R.sel]
    lim_p, lim_t = max(np.abs(p).max() for p in preds), np.abs(t_s).max()
    set_style(2.5 * (len(R.sel) + 1))
    fig, axes = plt.subplots(2, len(R.sel) + 1, figsize=(2.5 * (len(R.sel) + 1), 5.2), squeeze=False)
    axes[0, 0].imshow(t_s.T, origin="lower", extent=ext, cmap="magma", vmin=-lim_t, vmax=lim_t)
    axes[0, 0].set_title("truth IC")
    axes[1, 0].axis("off")
    for c, (i, p) in enumerate(zip(R.sel, preds), start=1):
        axes[0, c].imshow(p.T, origin="lower", extent=ext, cmap="magma", vmin=-lim_p, vmax=lim_p)
        axes[0, c].set_title(f"MAP, step {R.steps[i]}")
        axes[1, c].imshow((t_s - p).T, origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim_t, vmax=lim_t)
        axes[1, c].set_title(f"truth $-$ MAP, step {R.steps[i]}")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    savefig(ASSETS / f"fig08-ic-slice-steps-{R.tag}", fig)


def plot_ic_radial(R):
    ax1d = (np.arange(MESH) - MESH // 2) * (L / MESH)
    r_vox = np.sqrt(ax1d[:, None, None] ** 2 + ax1d[None, :, None] ** 2 + ax1d[None, None, :] ** 2).ravel()
    r_edges = np.linspace(0.0, np.sqrt(3) * chi_max * 1.0001, 41)
    idx = np.digitize(r_vox, r_edges) - 1
    count = np.bincount(idx, minlength=40)[:40]
    rms = lambda a: np.sqrt(np.bincount(idx, weights=a.ravel() ** 2, minlength=40)[:40] / np.maximum(count, 1))
    x = 0.5 * (r_edges[1:] + r_edges[:-1]) / chi_max
    set_style(14.0)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.6))
    axes[0].plot(x, rms(R.truth_arr), "k", lw=2, label="truth")
    axes[0].plot(x, rms(R.final_arr), color=R.color, label="MAP")
    axes[0].plot(x, rms(R.truth_arr - R.final_arr), color="C2", label="truth $-$ MAP")
    axes[0].axvline(1.0, color="k", ls="--", lw=1)
    axes[0].axvline(R_MIN / chi_max, color="k", ls=":", lw=1)
    axes[0].set(xlabel="distance from the observer / lightcone edge", ylabel="RMS of the IC in the shell")
    axes[0].legend()
    R.load("coh_ic", -1).plot(ax=axes[1], logy=False, label="MAP vs truth", legend=True)
    R.load("trans_ic", -1).plot(ax=axes[2], logy=True, label="MAP vs truth", legend=True)
    axes[1].set(title="", xlabel="$k$ [h/Mpc]", ylabel="3D IC coherence $r(k)$")
    axes[2].set(title="", xlabel="$k$ [h/Mpc]", ylabel="3D IC transfer $T(k)=\\sqrt{P_{\\rm MAP}/P_{\\rm truth}}$")
    for ax in axes:
        ax.grid(alpha=0.3)
    for ax in axes[1:]:
        ax.axhline(1.0, **ONE)
    fig.tight_layout()
    savefig(ASSETS / f"fig09-ic-radial-spectra-{R.tag}", fig)
    R.rms_rows = []
    for name, m in (("inside the lightcone", r_vox < chi_max), ("corners", r_vox > chi_max)):
        t_, p_ = R.truth_arr.ravel()[m], R.final_arr.ravel()[m]
        R.rms_rows.append((name, p_.std() / t_.std(), (t_ - p_).std() / t_.std()))


def plot_kappa(R):
    kappa_pred = R.load("kappa", -1)
    kappa_diff = R.truth_kappa.replace(array=R.truth_kappa.array - kappa_pred.array)
    vals = np.concatenate([np.asarray(f.array).ravel() for f in (R.truth_kappa, kappa_pred)])
    dm = float(np.abs(np.asarray(kappa_diff.array)).max())
    set_style(12.0)
    fig, axes = plt.subplots(n_bins, 3, figsize=(12.0, 3.0 * n_bins), squeeze=False)
    R.truth_kappa.plot(
        ax=list(axes[:, 0]), titles=[f"truth $\\kappa$\n{z}" for z in BIN], vmin=vals.min(), vmax=vals.max()
    )
    kappa_pred.plot(ax=list(axes[:, 1]), titles=[f"MAP $\\kappa$\n{z}" for z in BIN], vmin=vals.min(), vmax=vals.max())
    kappa_diff.plot(ax=list(axes[:, 2]), titles=[f"truth $-$ MAP\n{z}" for z in BIN], vmin=-dm, vmax=dm, cmap="RdBu_r")
    savefig(ASSETS / f"fig10-kappa-maps-{R.tag}", fig)

    # best achievable coherence: the joint Wiener filter of both bins, r_a^2 = [C (C + N)^-1 C]_aa / C_aa
    ell = np.arange(ELL_MAX + 1)
    sel_l = (ell >= ELL_MIN) & (ell < ELL_MAX)  # the taper weight is 0 at ELL_MAX
    kp = np.asarray(kappa_pred.array)
    C = np.array([[hp.anafast(R.kt[a], R.kt[b], lmax=ELL_MAX) for b in range(n_bins)] for a in range(n_bins)])
    C = C.transpose(2, 0, 1)
    N_ell = R.sigma_e**2 / (np.array([float(nz.gals_per_arcmin2) for nz in R.nz]) * (180 * 60 / np.pi) ** 2)
    M = np.einsum("lab,lbc,lcd->lad", C[sel_l], np.linalg.inv(C[sel_l] + np.diag(N_ell)[None]), C[sel_l])
    set_style(12.0)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.6))
    R.kappa_rows, R.kappa_curves = [], []
    for b in range(n_bins):
        ctt, cpp, ctp = (
            hp.anafast(R.kt[b], lmax=ELL_MAX),
            hp.anafast(kp[b], lmax=ELL_MAX),
            hp.anafast(R.kt[b], kp[b], lmax=ELL_MAX),
        )
        r_l, T_l = (ctp / np.sqrt(ctt * cpp))[sel_l], np.sqrt(cpp / ctt)[sel_l]
        w_l = np.sqrt(M[:, b, b] / C[sel_l, b, b])
        for ax, y in zip(axes, (r_l, T_l)):
            ax.plot(ell[sel_l], y, color=f"C{b}", lw=1, label=f"MAP, {BIN[b]}")
            ax.plot(ell[sel_l], w_l, color=f"C{b}", ls=":", lw=1.5, label=f"joint Wiener filter, {BIN[b]}")
        keep = ell[sel_l] <= ELL_MAX - ELL_TAPER
        R.kappa_rows.append((BIN[b], r_l[keep].mean(), w_l[keep].mean(), T_l[keep].mean()))
        R.kappa_curves.append((ell[sel_l], r_l, w_l))
    axes[0].set(xlabel="multipole $\\ell$", ylabel="$\\kappa$ coherence $r(\\ell)$ with the truth")
    axes[1].set(
        xlabel="multipole $\\ell$",
        ylabel="$\\kappa$ transfer $T(\\ell)=\\sqrt{C_\\ell^{\\rm MAP}/C_\\ell^{\\rm truth}}$",
    )
    for ax in axes:
        ax.axhline(1.0, **ONE)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(ASSETS / f"fig11-kappa-coherence-{R.tag}", fig)


def starlet(m):
    # exact resampling through the harmonic coefficients onto ST_NSIDE, so every scale lies inside the data band
    m = hp.alm2map(hp.map2alm(np.asarray(m, dtype=np.float64), lmax=ELL_MAX), ST_NSIDE, lmax=ELL_MAX)
    return starlet_coefficients_spherical(m, nside=ST_NSIDE, nscales=N_SCALES)[0]


def l1_norm(coef, sigma):
    # per scale: nu = w_j / sigma_j, then the sum of |nu| over the pixels of each nu bin (Ajani et al. 2021)
    out = np.zeros((N_SCALES, len(NU)))
    for j in range(N_SCALES):
        nu = coef[j] / sigma[j]
        idx = np.digitize(nu, NU_EDGES) - 1
        ok = (idx >= 0) & (idx < len(NU))
        out[j] = np.bincount(idx[ok], weights=np.abs(nu[ok]), minlength=len(NU))
    return out


def plot_starlet_l1(R, b):
    t = starlet(R.kt[b])
    sigma = t.std(axis=1)
    l1_t = l1_norm(t, sigma)
    rows = ((0, "first guess"), (R.mid, "intermediate"), (R.n_frames - 1, "final MAP"))
    set_style(3.0 * N_SCALES)
    fig, axes = plt.subplots(len(rows), N_SCALES, figsize=(3.0 * N_SCALES, 2.3 * len(rows)), sharex=True, squeeze=False)
    for r, (f, what) in enumerate(rows):
        p = starlet(R.kf[f][b])
        l1_p = l1_norm(p, sigma)
        for j in range(N_SCALES):
            ax = axes[r, j]
            ax.plot(NU, l1_t[j], "k--", lw=1.4)
            ax.plot(NU, l1_p[j], color=R.color, lw=1.6)
            ax.text(
                0.03,
                0.96,
                f"MAP/truth {l1_p[j].sum() / l1_t[j].sum():.2f}\n$r = {np.corrcoef(t[j], p[j])[0, 1]:.2f}$",
                transform=ax.transAxes,
                va="top",
            )
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(f"scale {j + 1}: {SCALE_BANDS[j]}")
            if r == len(rows) - 1:
                ax.set_xlabel("$\\nu = w_j/\\sigma_j^{\\rm truth}$")
        axes[r, 0].set_ylabel(f"{what}\n(step {R.steps[f]})\n$\\ell_1$ norm")
        if f == R.n_frames - 1:
            R.l1_truth[b], R.l1_final[b] = l1_t, l1_p
            R.starlet_r[b] = np.array([np.corrcoef(t[j], p[j])[0, 1] for j in range(N_SCALES)])
    handles = [
        Line2D([], [], color="k", ls="--", lw=1.4, label="truth"),
        Line2D([], [], color=R.color, lw=1.6, label=f"MAP, {R.name}"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(ASSETS / f"fig12-starlet-l1-{R.tag}-bin{DES_BINS[b] + 1}", fig)


# ---------------------------------------------------------------- figures, DES Y3 vs Euclid


def plot_coherence_comparison(runs):
    set_style(12.0)
    fig, axes = plt.subplots(n_bins, 2, figsize=(12.0, 3.4 * n_bins), squeeze=False)
    for R in runs:
        for b in range(n_bins):
            ell_, r_l, w_l = R.kappa_curves[b]
            axes[b, 0].plot(ell_, r_l, color=R.color, lw=1, label=f"MAP, {R.name}")
            axes[b, 0].plot(ell_, w_l, color=R.color, ls=":", lw=1.8, label=f"joint Wiener filter, {R.name}")
            ell_p, r_p = R.proj_curves[b]
            axes[b, 1].plot(ell_p, r_p, color=R.color, lw=1, label=f"MAP, {R.name}")
    for b in range(n_bins):
        axes[b, 0].set(xlabel="multipole $\\ell$", ylabel=f"$\\kappa$ coherence, {BIN[b]}")
        axes[b, 1].set(xlabel="multipole $\\ell$", ylabel=f"projected-IC coherence, {BIN[b]}")
        for ax in axes[b]:
            ax.axhline(1.0, **ONE)
            ax.set_ylim(-0.1, 1.08)
            ax.grid(alpha=0.3)
            ax.legend(loc="lower left")
    fig.tight_layout()
    savefig(ASSETS / "fig13-des-vs-euclid-coherence", fig)


def plot_starlet_comparison(runs):
    set_style(3.0 * N_SCALES)
    fig, axes = plt.subplots(
        2 * n_bins,
        N_SCALES,
        figsize=(3.0 * N_SCALES, 2.0 * 2 * n_bins),
        sharex=True,
        gridspec_kw=dict(height_ratios=[2, 1] * n_bins),
        squeeze=False,
    )
    for b in range(n_bins):
        l1_t = runs[0].l1_truth[b]  # the same truth kappa in both runs
        for j in range(N_SCALES):
            top, bottom = axes[2 * b, j], axes[2 * b + 1, j]
            top.plot(NU, l1_t[j], "k--", lw=1.4)
            ok = l1_t[j] > 0.01 * l1_t[j].max()
            bottom.axhspan(-0.1, 0.1, color="0.9")
            bottom.axhline(0.0, color="0.5", lw=0.8)
            for R in runs:
                top.plot(NU, R.l1_final[b][j], color=R.color, lw=1.5)
                bottom.plot(NU[ok], R.l1_final[b][j][ok] / l1_t[j][ok] - 1, color=R.color, lw=1.2)
            if b == 0:
                top.set_title(f"scale {j + 1}: {SCALE_BANDS[j]}")
            bottom.set_ylim(-1.05, 0.6)
            for ax in (top, bottom):
                ax.grid(alpha=0.3)
        axes[2 * b, 0].set_ylabel(f"$\\ell_1$ norm\n{BIN[b]}")
        axes[2 * b + 1, 0].set_ylabel("MAP / truth $-$ 1")
    for ax in axes[-1]:
        ax.set_xlabel("$\\nu = w_j/\\sigma_j^{\\rm truth}$")
    handles = [Line2D([], [], color="k", ls="--", lw=1.4, label="truth")] + [
        Line2D([], [], color=R.color, lw=1.5, label=f"MAP, {R.name}") for R in runs
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    savefig(ASSETS / "fig14-des-vs-euclid-starlet", fig)


def print_numbers(runs):
    print(f"\n{'':60s}" + "".join(f"{R.name:>26s}" for R in runs))
    row = lambda lab, f: print(f"{lab:<60s}" + "".join(f"{f(R):>26s}" for R in runs))
    row("chi2/N_pix MAP (truth)", lambda R: f"{R.summary['chi2_map']:.3f} ({R.summary['chi2_truth']:.3f})")
    for b in range(n_bins):
        z = f"bin {DES_BINS[b] + 1}"
        row(f"kappa coherence (joint Wiener), {z}", lambda R: f"{R.kappa_rows[b][1]:.3f} ({R.kappa_rows[b][2]:.3f})")
        row(f"kappa transfer, {z}", lambda R: f"{R.kappa_rows[b][3]:.3f}")
        row(f"projected-IC coherence, {z}", lambda R: f"{R.proj_rows[b][1]:.3f}")
        row(
            f"projected-IC coherence per band {[f'{lo}-{hi}' for lo, hi in BANDS]}, {z}",
            lambda R: " ".join(f"{v:.2f}" for v in R.proj_rows[b][3]),
        )
        row(
            f"starlet l1 MAP/truth fine -> coarse, {z}",
            lambda R: " ".join(f"{v:.2f}" for v in R.l1_final[b].sum(axis=1) / R.l1_truth[b].sum(axis=1)),
        )
        row(f"starlet r fine -> coarse, {z}", lambda R: " ".join(f"{v:.2f}" for v in R.starlet_r[b]))
    for i, name in enumerate(("inside the lightcone", "corners")):
        row(f"3D IC RMS MAP/truth, {name}", lambda R: f"{R.rms_rows[i][1]:.3f}")
        row(f"3D IC RMS diff/truth, {name}", lambda R: f"{R.rms_rows[i][2]:.3f}")


def main():
    des = load_run(DES, "DES Y3", "des", "C0", NZ_DES, 0.26)
    euclid = load_run(EUCLID, "Euclid", "euclid", "C3", NZ_EUCLID, 0.30 / np.sqrt(2))
    for R in (des, euclid):
        plot_loss(R)
        plot_kappa_filmstrip(R)
        plot_lightcone(R)
        plot_ic_projection(R)
        plot_ic_cube(R)
        plot_ic_slices(R)
        plot_ic_radial(R)
        plot_kappa(R)
        plot_starlet_l1(R, 0)
        plot_starlet_l1(R, 1)
    plot_coherence_comparison((des, euclid))
    plot_starlet_comparison((des, euclid))
    print_numbers((des, euclid))
    print(f"assets written to {ASSETS}")


if __name__ == "__main__":
    main()
