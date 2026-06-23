import jax

# float64 globally BEFORE jax_cosmo (the enable_x64() context manager breaks jax_cosmo's
# pure_callback comoving-distance cache; the global config flag is the safe route).
jax.config.update("jax_enable_x64", True)

import warnings

import healpy as hp
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import jax_fli as jfli
from jax_fli import compute_theory_cl_for_density
from jax_fli.io import Catalog

warnings.filterwarnings("ignore")  # tophat_z theory emits a DeprecationWarning by design

REPO = "ASKabalan/jax-fli-experiments"
NBINS = 20

C_MEAS = "tab:blue"
C_PW = "k"  # theory x pixwin^2 (matched to the pixelized map)
C_RAW = "tab:red"  # raw continuous-sky theory

SHELL_NEAR = "000"
SHELL_MID = "010"
SHELL_FAR = "030"


# --- load: three single-shell CosmoGrid density maps (nside 2048) + the fiducial cosmology ---
def load_shell(shell):
    cat = Catalog.from_dataset(
        load_dataset(
            REPO,
            data_files=f"00-cosmogrid/density/cosmogrid_density_nside2048_shell_{shell}.parquet",
            split="train",
        ).with_format("numpy")
    )
    return cat.field[0], cat.cosmology[0]


print("Loading near / mid / far density shells from HF ...")
dens_near, cosmo = load_shell(SHELL_NEAR)
dens_mid, _ = load_shell(SHELL_MID)
dens_far, _ = load_shell(SHELL_FAR)

NSIDE = hp.npix2nside(dens_near.array.shape[-1])  # 2048
LMAX = 3 * NSIDE - 1  # the nside HEALPix multipole ceiling
ell = np.arange(LMAX + 1)
pixwin2 = hp.pixwin(NSIDE)[: ell.size] ** 2


# --- measured overdensity C_ell (the map carries pixwin) and continuous-sky theory C_ell, per shell ---
def measure_cl(field):
    """Measured overdensity angular C_ell of one HEALPix shell, via healpy anafast (~27 s)."""
    return np.asarray(field.to(jfli.units.OVERDENSITY).angular_cl(method="healpy", lmax=LMAX).array).squeeze()


def theory_cl(field, radial_weight):
    """Limber number-counts theory C_ell for one shell ('comoving_volume' or 'tophat_z')."""
    return np.asarray(compute_theory_cl_for_density(cosmo, field, jnp.asarray(ell), radial_weight=radial_weight).array).squeeze()


print("Measuring map C_ell for the three shells (healpy anafast) ...")
cl_near = measure_cl(dens_near)
cl_mid = measure_cl(dens_mid)
cl_far = measure_cl(dens_far)


# --- (2l+1)-weighted log bandpowers: bin centers, CV per bin (computed once), and a 1-line binner ---
edges = np.unique(np.geomspace(2, LMAX, NBINS + 1).astype(int))
bins = list(zip(edges[:-1], edges[1:]))  # (lo, hi) integer-ell ranges; ell == array index, so we can slice
w = 2.0 * ell + 1.0
leff = np.array([np.average(ell[a:b], weights=w[a:b]) for a, b in bins])
sigma = np.sqrt(2.0 / np.array([w[a:b].sum() for a, b in bins]))  # full-sky cosmic variance per bin


def bin_cl(cl_1d):
    """(2l+1)-weighted bandpowers of one C_ell(ell)."""
    return np.array([np.average(cl_1d[a:b], weights=w[a:b]) for a, b in bins])


LEGEND = [
    Line2D([], [], color=C_MEAS, marker="o", ls="-", lw=0.9, ms=5, label=r"measured $C_\ell$ (raw + $(2\ell+1)$-binned)"),
    Line2D([], [], color=C_PW, lw=1.8, label=rf"theory $\times$ pixwin$^2$({NSIDE})"),
    Line2D([], [], color=C_RAW, ls="--", lw=1.6, label="theory (continuous sky)"),
    Patch(facecolor="0.8", label=r"$\pm1\sigma$ cosmic variance (full sky)"),
    Patch(facecolor="orange", alpha=0.2, label=r"$\pm5\%$ band"),
    Line2D([], [], color="0.6", ls=":", lw=0.9, label=rf"$\ell = 2\,n_{{\rm side}} = {2 * NSIDE}$"),
]


def _ratio_panel(axr, ratio, color, marker, ls, ylabel):
    """One measured/theory ratio panel: CV band, +-5% band, unity line, the binned ratio."""
    axr.fill_between(leff, 1 - sigma, 1 + sigma, color="0.8", lw=0)
    axr.fill_between(leff, 0.95, 1.05, color="orange", alpha=0.2, lw=0)
    axr.axhline(1.0, color="0.4", ls="--", lw=0.9)
    axr.semilogx(leff, ratio, marker=marker, ls=ls, color=color, ms=3.2, lw=1.0)
    axr.axvline(2 * NSIDE, color="0.6", ls=":", lw=0.9)
    axr.set_ylim(0.0, 1.6)
    axr.set_xlim(2, LMAX)
    axr.grid(alpha=0.2, which="both")
    axr.set_ylabel(ylabel, fontsize=8)


def plot_shell(name, cl_meas, field, radial_weight):
    """One figure: C_ell (top) over TWO ratio panels -- meas/(theory x pixwin^2) and meas/theory."""
    theory = theory_cl(field, radial_weight)
    z = float(np.asarray(field.z_sources).ravel()[0])
    sel = ell >= 2

    cl_b = bin_cl(cl_meas)
    thy_pw_b = bin_cl(theory * pixwin2)  # pixelized footing (matches the map)
    thy_raw_b = bin_cl(theory)  # continuous sky

    fig = plt.figure(figsize=(7.0, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])
    ax = fig.add_subplot(gs[0])
    ax_pw = fig.add_subplot(gs[1], sharex=ax)  # meas / (theory x pixwin^2)
    ax_raw = fig.add_subplot(gs[2], sharex=ax)  # meas / theory (continuous)

    # top: measured (raw + binned) vs theory (pixwin-matched solid, continuous dashed)
    ax.loglog(ell[sel], cl_meas[sel], color=C_MEAS, lw=0.5, alpha=0.3)
    ax.loglog(leff, cl_b, "o", color=C_MEAS, ms=3.6)
    ax.loglog(ell[sel], (theory * pixwin2)[sel], "-", color=C_PW, lw=1.7)
    ax.loglog(ell[sel], theory[sel], "--", color=C_RAW, lw=1.5)
    ax.axvline(2 * NSIDE, color="0.6", ls=":", lw=0.9)
    ax.set_ylabel(r"$C_\ell$")
    ax.set_xlim(2, LMAX)
    ax.grid(alpha=0.2, which="both")
    ax.tick_params(labelbottom=False)

    # two ratio panels
    _ratio_panel(ax_pw, cl_b / thy_pw_b, C_PW, "o", "-", "meas /\n(thy×pixwin²)")
    ax_pw.tick_params(labelbottom=False)
    _ratio_panel(ax_raw, cl_b / thy_raw_b, C_RAW, "s", "--", "meas /\nthy (cont.)")
    ax_raw.set_xlabel(r"$\ell$")

    fig.suptitle(f"{name} shell  z = {z:.3f}   —   radial weight: {radial_weight}", fontsize=12)
    fig.legend(handles=LEGEND, loc="outside lower center", ncol=3, fontsize=8, frameon=False)
    plt.show()


# --- six plots: near / mid / far, each shown for both theory models ---
plot_shell("near", cl_near, dens_near, "comoving_volume")
plot_shell("mid", cl_mid, dens_mid, "comoving_volume")
plot_shell("far", cl_far, dens_far, "comoving_volume")
plot_shell("near", cl_near, dens_near, "tophat_z")
plot_shell("mid", cl_mid, dens_mid, "tophat_z")
plot_shell("far", cl_far, dens_far, "tophat_z")
