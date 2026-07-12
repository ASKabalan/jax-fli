# ruff: noqa: E402
"""Experiment 08 — masked shear from a CosmoGrid convergence map.

Weak-lensing surveys observe shear only inside a survey **footprint**, and a Kaiser–Squires
(KS) κ → γ transform on a *cut sky* leaks power across the mask boundary. This script runs the
analysis end-to-end on a real CosmoGrid convergence map and saves every figure as **SVG** into
``./assets`` — the website embeds the SVGs and a paper reuses the same vector files.

Run:
    python 08-masked-shear.py                   # GPU
    JAX_PLATFORMS=cpu python 08-masked-shear.py  # CPU (slower, avoids GPU OOM)

Prints the summary table reproduced in ``README.md``.
"""

import itertools
import sys
import warnings
from pathlib import Path

# float64 is required for the masked spin-2 angular_cl mode-coupling (decoupling) solve;
# it is ill-conditioned in float32 and silently returns NaN. Set this BEFORE importing jax_fli.
import jax

jax.config.update("jax_enable_x64", True)

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from jaxpm.spherical import spherical_visibility_mask

import jax_fli as jfli
from jax_fli.io.catalog import Catalog
from s2fft_lib import _s2fft

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _exputils import savefig, set_style

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
DATA = HERE / "data"

NSIDE = 128
LMAX = 3 * NSIDE - 1
NLB = 16
BIN = 3  # DES Y3 tomographic bin (0-based): highest-redshift bin
APOSIZE_DEG = 2.0
REPO = "ASKabalan/jax-fli-experiments"

OBS_QUAD = (0.0, 0.5, 1.0)  # Case 2: observer on a box edge -> quadrant footprint
OBS_LARGE = (0.1, 0.5, 0.9)  # Case 3: pulled slightly inside -> DES sits entirely inside the footprint

kaiser_squires_method = "jax"
GPU_AVAILABLE = jax.devices("gpu") != []
if GPU_AVAILABLE:
    if  _s2fft.COMPILED_WITH_CUDA:
        print("GPU available and s2fft compiled with CUDA.")
        kaiser_squires_method = "jax_cuda"
    else:
        print("GPU available but s2fft not compiled with CUDA; using CPU fallback.")
else:
    print("No GPU available; using CPU fallback.")


def load_kappa():
    """Load the CosmoGrid convergence (Experiment 0), downgrade, pick one tomographic bin."""
    ds = load_dataset(REPO, "00-cosmogrid-kappa", split="train").with_format("numpy")
    catalog = Catalog.from_dataset(ds)
    kappa_all = catalog.field[0] if isinstance(catalog.field, list) else catalog.field
    print("loaded", type(kappa_all).__name__, kappa_all.array.shape, "nside", kappa_all.nside)

    kappa = kappa_all.ud_sample(NSIDE)[BIN]
    kappa = kappa.replace(array=kappa.array - kappa.array.mean())  # remove the monopole
    print("working kappa:", kappa.array.shape, "nside", kappa.nside)
    return kappa


def visibility(observer):
    """Binary visibility footprint of the box for an observer at a fractional position."""
    return np.asarray(spherical_visibility_mask(NSIDE, observer, threshold=1.0), dtype=int)


def des_center(des_binary):
    """(lon, lat) of the DES footprint centroid, for the gnomview zoom."""
    pix = np.where(des_binary > 0)[0]
    vx, vy, vz = hp.pix2vec(NSIDE, pix)
    lon, lat = hp.vec2ang(np.array([vx.mean(), vy.mean(), vz.mean()]), lonlat=True)
    return float(np.atleast_1d(lon)[0]), float(np.atleast_1d(lat)[0])


def setup_masks(des_binary):
    """DES Y3 footprint + two observer-driven visibility masks (binary + apodized)."""

    def des_coverage(v):
        return float(v[des_binary > 0].mean())

    vis_quad = visibility(OBS_QUAD)
    vis_large = visibility(OBS_LARGE)

    print(f"Case 2 observer {OBS_QUAD}:  fsky={vis_quad.mean():.3f},  DES coverage={des_coverage(vis_quad):.3f}")
    print(f"Case 3 observer {OBS_LARGE}:  fsky={vis_large.mean():.3f},  DES coverage={des_coverage(vis_large):.3f}")

    apo_des = np.asarray(jfli.data.apodize(des_binary, APOSIZE_DEG))
    apo_quad = np.asarray(jfli.data.apodize(vis_quad, APOSIZE_DEG))
    apo_large = np.asarray(jfli.data.apodize(vis_large, APOSIZE_DEG))
    return {
        "binary": {"DES Y3": des_binary, f"visibility {OBS_QUAD}": vis_quad, f"visibility {OBS_LARGE}": vis_large},
        "apo": {"des": apo_des, "quad": apo_quad, "large": apo_large},
        "fsky": {"quad": float(vis_quad.mean()), "large": float(vis_large.mean())},
        "coverage": {"quad": des_coverage(vis_quad), "large": des_coverage(vis_large)},
    }


def fig_masks(masks):
    """fig01 — the three survey/visibility footprints side by side."""
    plt.figure(figsize=(13, 3.2))
    for k, (title, m) in enumerate(masks["binary"].items(), 1):
        hp.mollview(m, sub=(1, 3, k), title=title, cbar=False, bgcolor=(0.0,) * 4)
    savefig(ASSETS / "fig01-masks")


def fig_observer_box(observer, fsky, title, stem):
    """A 3-D wireframe of the unit box [0,1]^3 with the observer position marked."""
    # Temporarily override the styling applied by set_style()
    rc_overrides = {
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        # Optional: Revert to standard Matplotlib sans-serif fonts to fully match the second image
        "text.usetex": False,
        "font.family": "sans-serif"
    }
    
    with plt.rc_context(rc_overrides):
        fig = plt.figure(figsize=(4.8, 4.8))
        ax = fig.add_subplot(111, projection="3d")
        corners = list(itertools.product([0, 1], repeat=3))
        for a, b in itertools.combinations(corners, 2):
            if sum(ai != bi for ai, bi in zip(a, b)) == 1:  # cube edge: differ in exactly one coord
                ax.plot3D(*zip(a, b), color="0.55", lw=1.0)
        ox, oy, oz = observer
        ax.plot([ox, ox], [oy, oy], [0, oz], color="crimson", ls=":", lw=0.9)  # drop line to the base
        ax.scatter([ox], [oy], [oz], color="crimson", s=80, depthshade=False, zorder=6)
        ax.text(ox, oy, oz + 0.07, f"observer\n{observer}", color="crimson", fontsize=8, ha="center")
        ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), xlabel="x", ylabel="y", zlabel="z")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=18, azim=-52)
        ax.set_title(f"{title}\n$f_\\mathrm{{sky}}$ = {fsky:.3f}", fontsize=10)
        savefig(stem, fig)


def masked_estimate(kappa, apo_mask):
    """KS on the apodized footprint → (kappa_masked, g1_masked, g2_masked). The expensive step."""
    k_masked = kappa.replace(array=kappa.array * apo_mask)
    g1m, g2m = np.asarray(k_masked.get_shear(method=kaiser_squires_method).array)
    return np.asarray(k_masked.array), g1m, g2m


def panel(full, est, score_region, title, stem):
    """(K / γ1 / γ2) × (full-sky / masked / residual-on-DES); saves ``stem``, returns RMS ratio."""
    kap, g1_full, g2_full = full
    km, g1m, g2m = est
    w = score_region > 0  # residual is inspected on DES

    def res(a, b):
        r = (a - b).astype(float).copy()
        r[~w] = np.nan
        return r

    rows = [
        (r"$\kappa$", kap, km, res(km, kap)),
        (r"$\gamma_1$", g1_full, g1m, res(g1m, g1_full)),
        (r"$\gamma_2$", g2_full, g2m, res(g2m, g2_full)),
    ]
    plt.figure(figsize=(13, 9))
    for i, (name, full_m, masked, residual) in enumerate(rows):
        for j, (m, col) in enumerate([(full_m, "full sky"), (masked, "masked"), (residual, "residual on DES")]):
            cmap = "RdBu_r" if "residual" in col else "magma"
            hp.mollview(m, sub=(3, 3, 3 * i + j + 1), title=f"{name} — {col}", cbar=True, cmap=cmap, bgcolor=(0.0,) * 4)
    plt.suptitle(title, y=1.02, fontsize=14)
    savefig(ASSETS / stem)

    rms_full = np.sqrt(np.nanmean(g1_full[w] ** 2 + g2_full[w] ** 2))
    rms_res = np.sqrt(np.nanmean((g1m[w] - g1_full[w]) ** 2 + (g2m[w] - g2_full[w]) ** 2))
    ratio = float(rms_res / rms_full)
    print(f"{title}: shear residual RMS / full-sky RMS on DES = {ratio:.3f}")
    return ratio


_CASE_LABELS = ("Case 1 — DES", "Case 2 — edge", "Case 3 — face")
_CASE_KEYS = ("des", "quad", "large")
_CASE_COLORS = ("C3", "C1", "C2")  # red / orange / green: worst -> best


def _gamma1_residual(g1_full, est, w):
    """γ1 residual (truth − recon), NaN outside the DES footprint ``w``."""
    r = (g1_full - est[1]).astype(float)
    r[~w] = np.nan
    return r


def fig_gamma1_residual_pdf(full, ests, des_binary, stem):
    """fig07 — probability density of the γ1 residual (truth − recon) on DES, per mask."""
    g1_full = full[1]
    w = des_binary > 0.5
    resids = [(g1_full - ests[k][1])[w] for k in _CASE_KEYS]
    lim = float(np.nanpercentile(np.abs(np.concatenate(resids)), 99.5))
    bins = np.linspace(-lim, lim, 81)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for lab, r, color in zip(_CASE_LABELS, resids, _CASE_COLORS):
        rms = float(np.sqrt(np.mean(r**2)))
        ax.hist(r, bins=bins, density=True, histtype="step", lw=1.8, color=color, label=f"{lab}  (RMS = {rms:.2e})")
    ax.set_yscale("log")
    ax.set(
        xlabel=r"$\gamma_1$ residual (truth $-$ recon) on DES",
        ylabel="probability density",
        title=r"$\gamma_1$ residual distribution on DES",
    )
    ax.legend()
    savefig(stem, fig)


def fig_gamma1_residual_maps(full, ests, des_binary, clon, clat, stem):
    """fig08 — γ1 residual (truth − recon) zoomed on DES (gnomview), three masks, shared RdBu_r ±vr."""
    g1_full = full[1]
    w = des_binary > 0.5
    vr = float(np.nanpercentile(np.abs(g1_full[w]), 99))  # symmetric scale set by the full-sky signal
    kw = dict(cmap="RdBu_r", min=-vr, max=vr, cbar=True, notext=True)

    fig = plt.figure(figsize=(13, 4.9))
    for k, (lab, key) in enumerate(zip(_CASE_LABELS, _CASE_KEYS)):
        r = _gamma1_residual(g1_full, ests[key], w)
        # margins top bump gives the gnomview title headroom (gnomview fills its cell tighter than mollview)
        hp.gnomview(
            r,
            rot=(clon, clat),
            reso=12.0,
            xsize=420,
            sub=(1, 3, k + 1),
            title=lab,
            margins=(0.01, 0.0, 0.01, 0.05),
            **kw,
        )
    savefig(stem, fig)


def fig_spectra(shear_full, masks):
    """fig09 — mask-decoupled shear EE vs full sky (no E/B purification). Returns median ratios."""
    ps_full = shear_full.angular_cl()  # full sky, ell = 0..lmax
    ell_full = np.asarray(ps_full.wavenumber)
    ee_full = np.asarray(ps_full.array)[0]  # EE component

    def decoupled_ee(apo_mask):
        ps = shear_full.angular_cl(mask=apo_mask, nlb=NLB)
        return np.asarray(ps.wavenumber), np.asarray(ps.array)[0]

    labelled = [
        (masks["apo"]["des"], "DES"),
        (masks["apo"]["quad"], f"vis {OBS_QUAD}"),
        (masks["apo"]["large"], f"vis {OBS_LARGE}"),
    ]

    ratios = {}
    for apo, lab in labelled:
        ll, ee = decoupled_ee(apo)
        ref = np.interp(ll, ell_full, ee_full)
        band = (ll > 40) & (ll < 1.5 * NSIDE)
        ratios[lab] = float(np.median(ee[band] / ref[band]))
        print(f"{lab:>18}: median decoupled-EE / full-sky ratio (40 < l < {1.5 * NSIDE:.0f}) = {ratios[lab]:.3f}")

    plt.figure(figsize=(12, 4.2))
    ax1 = plt.subplot(1, 2, 1)
    ax1.loglog(ell_full[2:], ee_full[2:], "k-", lw=1.5, label="full sky")
    for apo, lab in labelled:
        ll, ee = decoupled_ee(apo)
        ax1.loglog(ll, ee, ".", label=f"decoupled — {lab}")
    ax1.set(xlabel=r"$\ell$", ylabel=r"$C_\ell^{EE}$", title="Shear EE")
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    for apo, lab in labelled:
        ll, ee = decoupled_ee(apo)
        ee_ref = np.interp(ll, ell_full, ee_full)
        ax2.plot(ll, ee / ee_ref, ".-", label=lab)
    ax2.axhline(1.0, color="k", lw=0.8)
    ax2.set(xlabel=r"$\ell$", ylabel="decoupled / full sky", ylim=(0.5, 1.5), title="ratio to full sky")
    ax2.legend()
    plt.tight_layout()
    savefig(ASSETS / "fig09-ee-spectra")
    return ell_full, ee_full, ratios


def main():
    set_style()
    kappa = load_kappa()

    des_binary = jfli.data.get_desy3_mask(NSIDE).astype(float)
    masks = setup_masks(des_binary)
    clon, clat = des_center(des_binary)

    fig_masks(masks)
    fig_observer_box(OBS_QUAD, masks["fsky"]["quad"], f"Case 2 observer {OBS_QUAD}", ASSETS / "fig02-observer-quad")
    fig_observer_box(OBS_LARGE, masks["fsky"]["large"], f"Case 3 observer {OBS_LARGE}", ASSETS / "fig03-observer-large")

    shear_full = kappa.get_shear(method=kaiser_squires_method)  # SphericalShearField, (2, npix)
    g1_full, g2_full = np.asarray(shear_full.array)
    kap = np.asarray(kappa.array)
    full = (kap, g1_full, g2_full)

    # KS get_shear is the expensive step — compute each masked estimate once, reuse in panels + fig07/08.
    ests = {
        "des": masked_estimate(kappa, masks["apo"]["des"]),
        "quad": masked_estimate(kappa, masks["apo"]["quad"]),
        "large": masked_estimate(kappa, masks["apo"]["large"]),
    }

    r1 = panel(full, ests["des"], des_binary, "Case 1 — DES mask", "fig04-case1-des")
    r2 = panel(full, ests["quad"], des_binary, f"Case 2 — visibility mask {OBS_QUAD}", "fig05-case2-vis")
    r3 = panel(full, ests["large"], des_binary, f"Case 3 — visibility mask {OBS_LARGE}", "fig06-case3-vislarge")

    fig_gamma1_residual_pdf(full, ests, des_binary, ASSETS / "fig07-gamma1-residual-pdf")
    fig_gamma1_residual_maps(full, ests, des_binary, clon, clat, ASSETS / "fig08-gamma1-residual-maps")

    ell_full, ee_full, ratios = fig_spectra(shear_full, masks)

    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA / "masked_shear.npz",
        kappa=kap,
        g1_full=g1_full,
        g2_full=g2_full,
        des_binary=des_binary,
        vis_quad=masks["binary"][f"visibility {OBS_QUAD}"],
        vis_large=masks["binary"][f"visibility {OBS_LARGE}"],
        g1m_des=ests["des"][1],
        g1m_quad=ests["quad"][1],
        g1m_large=ests["large"][1],
        ell_full=ell_full,
        ee_full=ee_full,
    )

    print("\n=== summary (for README.md) ===")
    print(f"Case 2 observer {OBS_QUAD}: fsky={masks['fsky']['quad']:.3f}, DES coverage={masks['coverage']['quad']:.3f}")
    print(
        f"Case 3 observer {OBS_LARGE}: fsky={masks['fsky']['large']:.3f}, DES coverage={masks['coverage']['large']:.3f}"
    )
    print(f"residual RMS / full-sky on DES: case1={r1:.3f}, case2={r2:.3f}, case3={r3:.3f}")
    print(f"decoupled-EE / full-sky ratios: {ratios}")
    print(f"DES centroid (lon,lat) = ({clon:.1f}, {clat:.1f})")
    print(f"assets written to {ASSETS}")


if __name__ == "__main__":
    main()
