"""Born lensing: jax-fli vs glass and vs dorian cross-validation."""

from __future__ import annotations

import os

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from tests.helpers import compare_fields

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
GLASS_MAP_ATOL = 0.005
GLASS_MAP_RTOL = 0.02
GLASS_MAP_MEAN_ATOL = 1e-3
GLASS_CLS_ATOL = 1e-7
GLASS_CLS_RTOL = 0.02
GLASS_CLS_MEAN_ATOL = 1e-10

DORIAN_MAP_ATOL = 0.01
DORIAN_MAP_RTOL = 0.05
DORIAN_MAP_MEAN_ATOL = 1e-3
DORIAN_CLS_ATOL = 1e-7
DORIAN_CLS_RTOL = 0.05
DORIAN_CLS_MEAN_ATOL = 1e-10


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _plot_spectra(ells, jfli_cls_list, ref_cls_list, z_sources, ref_label, path):
    """Log-log spectra comparison plot."""
    fig, ax = plt.subplots()
    for i, z in enumerate(z_sources):
        ax.loglog(ells[i], jfli_cls_list[i], label=f"jfli z={z:.2f}")
        ax.loglog(ells[i], ref_cls_list[i], "--", label=f"{ref_label} z={z:.2f}")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell$")
    ax.legend()
    ax.set_title(f"Born spectra: jfli vs {ref_label}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_ratio(ells, ratio_list, z_sources, ref_label, path):
    """Ratio plot with shaded 2%/5%/10% bands."""
    fig, ax = plt.subplots()
    for pct, alpha in [(0.10, 0.15), (0.05, 0.25), (0.02, 0.4)]:
        ax.axhspan(1 - pct, 1 + pct, color="gray", alpha=alpha, label=f"±{int(pct*100)}%")
    ax.axhline(1.0, color="k", lw=0.5)
    for i, z in enumerate(z_sources):
        ax.plot(ells[i], ratio_list[i], label=f"z={z:.2f}")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell^{\rm jfli} / C_\ell^{\rm " + ref_label + r"}$")
    ax.legend()
    ax.set_title(f"Spectra ratio: jfli / {ref_label}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_born_vs_glass(born_kappa_multi, glass_kappa_maps):
    """Compare jfli Born vs glass: maps and power spectra."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    jfli_arr = np.asarray(born_kappa_multi.array)
    z_sources = np.asarray(born_kappa_multi.z_sources)

    # --- maps ---
    for i, z in enumerate(z_sources):
        z_key = float(z)
        closest_key = min(glass_kappa_maps.keys(), key=lambda k: abs(k - z_key))
        compare_fields(
            jfli_arr[i],
            glass_kappa_maps[closest_key],
            f"glass map z={z_key:.2f}",
            atol=GLASS_MAP_ATOL,
            rtol=GLASS_MAP_RTOL,
            mean_atol=GLASS_MAP_MEAN_ATOL,
        )

    # --- spectra ---
    jfli_ps = born_kappa_multi.angular_cl(method="healpy")
    jfli_cls_all = np.asarray(jfli_ps.spectra)

    ells_list, jfli_list, glass_list = [], [], []
    for i, z in enumerate(z_sources):
        z_key = float(z)
        closest_key = min(glass_kappa_maps.keys(), key=lambda k: abs(k - z_key))
        glass_cls = hp.anafast(glass_kappa_maps[closest_key])
        jfli_cls = jfli_cls_all[i]
        n = min(len(jfli_cls), len(glass_cls))
        ell = np.arange(2, n)

        compare_fields(
            jfli_cls[2:n],
            glass_cls[2:n],
            f"glass Cl z={z_key:.2f}",
            atol=GLASS_CLS_ATOL,
            rtol=GLASS_CLS_RTOL,
            mean_atol=GLASS_CLS_MEAN_ATOL,
        )

        ells_list.append(ell)
        jfli_list.append(np.asarray(jfli_cls[2:n]))
        glass_list.append(glass_cls[2:n])


    _plot_spectra(ells_list, jfli_list, glass_list, z_sources, "glass", os.path.join(PLOT_DIR, "glass_spectra.png"))
    ratio_list = [j / (g + 1e-30) for j, g in zip(jfli_list, glass_list)]
    _plot_ratio(ells_list, ratio_list, z_sources, "glass", os.path.join(PLOT_DIR, "glass_spectra_ratio.png"))


def test_born_vs_dorian(born_kappa_multi, raytrace_born_kappa_multi):
    """Compare jfli Born vs dorian raytrace(born=True): maps and power spectra."""
    pytest.importorskip("dorian")
    os.makedirs(PLOT_DIR, exist_ok=True)

    born_arr = np.asarray(born_kappa_multi.array)
    rt_arr = np.asarray(raytrace_born_kappa_multi.array)
    z_sources = np.asarray(born_kappa_multi.z_sources)

    # --- maps ---
    compare_fields(
        rt_arr,
        born_arr,
        "dorian map",
        atol=DORIAN_MAP_ATOL,
        rtol=DORIAN_MAP_RTOL,
        mean_atol=DORIAN_MAP_MEAN_ATOL,
    )

    # --- spectra ---
    born_ps = born_kappa_multi.angular_cl(method="healpy")
    rt_ps = raytrace_born_kappa_multi.angular_cl(method="healpy")
    born_cls = np.asarray(born_ps.spectra)
    rt_cls = np.asarray(rt_ps.spectra)
    n = min(born_cls.shape[-1], rt_cls.shape[-1])
    ell = np.arange(2, n)

    compare_fields(
        rt_cls[:, 2:n],
        born_cls[:, 2:n],
        "dorian Cl",
        atol=DORIAN_CLS_ATOL,
        rtol=DORIAN_CLS_RTOL,
        mean_atol=DORIAN_CLS_MEAN_ATOL,
    )

    ells_list = [ell] * len(z_sources)
    jfli_list = [np.asarray(born_cls[i, 2:n]) for i in range(len(z_sources))]
    dorian_list = [np.asarray(rt_cls[i, 2:n]) for i in range(len(z_sources))]

    _plot_spectra(ells_list, jfli_list, dorian_list, z_sources, "dorian", os.path.join(PLOT_DIR, "dorian_spectra.png"))
    ratio_list = [j / (d + 1e-30) for j, d in zip(jfli_list, dorian_list)]
    _plot_ratio(ells_list, ratio_list, z_sources, "dorian", os.path.join(PLOT_DIR, "dorian_spectra_ratio.png"))
