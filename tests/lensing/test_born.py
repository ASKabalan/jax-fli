"""Born lensing: jax-fli vs glass and vs dorian cross-validation."""

from __future__ import annotations

import os

import healpy as hp
import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib.pyplot as plt
import numpy as np
import pytest

import jax_fli as jfli
from jax_fli._src.lensing._born import _born_windows
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

# glass's tophat-window accumulation is itself center-evaluated (midpoint-like), so the
# exact-integral (gauss_legendre) born legitimately sits a bit further from glass than the
# midpoint born does on the ~300 Mpc/h test shells (measured max map deviation 0.007 at z=0.5,
# 24/49152 pixels beyond the midpoint atol).
GL_GLASS_MAP_ATOL = 0.01
GL_GLASS_CLS_RTOL = 0.05

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
        ax.axhspan(1 - pct, 1 + pct, color="gray", alpha=alpha, label=f"±{int(pct * 100)}%")
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
def _assert_born_matches_glass(
    born_field, glass_kappa_maps, plot_suffix, map_atol=GLASS_MAP_ATOL, cls_rtol=GLASS_CLS_RTOL
):
    """Shared logic for born-vs-glass comparison (maps + spectra + plots)."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    jfli_arr = np.asarray(born_field.array)
    z_sources = np.asarray(born_field.z_sources)

    # --- maps ---
    for i, z in enumerate(z_sources):
        z_key = float(z)
        closest_key = min(glass_kappa_maps.keys(), key=lambda k: abs(k - z_key))
        compare_fields(
            jfli_arr[i],
            glass_kappa_maps[closest_key],
            f"glass map z={z_key:.2f}",
            atol=map_atol,
            rtol=GLASS_MAP_RTOL,
            mean_atol=GLASS_MAP_MEAN_ATOL,
        )

    # --- spectra ---
    jfli_ps = born_field.angular_cl(method="healpy")
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
            rtol=cls_rtol,
            mean_atol=GLASS_CLS_MEAN_ATOL,
        )

        ells_list.append(ell)
        jfli_list.append(np.asarray(jfli_cls[2:n]))
        glass_list.append(glass_cls[2:n])

    _plot_spectra(
        ells_list, jfli_list, glass_list, z_sources, "glass", os.path.join(PLOT_DIR, f"glass_spectra_{plot_suffix}.png")
    )
    ratio_list = [j / (g + 1e-30) for j, g in zip(jfli_list, glass_list)]
    _plot_ratio(
        ells_list, ratio_list, z_sources, "glass", os.path.join(PLOT_DIR, f"glass_spectra_ratio_{plot_suffix}.png")
    )


def test_born_vs_glass(born_kappa_multi_per_plane, glass_kappa_maps):
    """Compare jfli Born (per-plane normalization, midpoint quadrature) vs glass: maps and power spectra."""
    _assert_born_matches_glass(born_kappa_multi_per_plane, glass_kappa_maps, "midpoint")


def test_born_gl_vs_glass(born_kappa_multi_per_plane_gl, glass_kappa_maps):
    """Compare jfli Born (per-plane normalization, Gauss-Legendre quadrature) vs glass."""
    _assert_born_matches_glass(
        born_kappa_multi_per_plane_gl,
        glass_kappa_maps,
        "gauss_legendre",
        map_atol=GL_GLASS_MAP_ATOL,
        cls_rtol=GL_GLASS_CLS_RTOL,
    )


def _compare_born_vs_dorian(born_field, rt_field, label, plot_suffix):
    """Shared logic for born-vs-dorian comparison (maps + spectra + plots)."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    born_arr = np.asarray(born_field.array)
    rt_arr = np.asarray(rt_field.array)
    z_sources = np.asarray(born_field.z_sources)

    compare_fields(
        rt_arr,
        born_arr,
        f"dorian map ({label})",
        atol=DORIAN_MAP_ATOL,
        rtol=DORIAN_MAP_RTOL,
        mean_atol=DORIAN_MAP_MEAN_ATOL,
    )

    born_ps = born_field.angular_cl(method="healpy")
    rt_ps = rt_field.angular_cl(method="healpy")
    born_cls = np.asarray(born_ps.spectra)
    rt_cls = np.asarray(rt_ps.spectra)
    n = min(born_cls.shape[-1], rt_cls.shape[-1])
    ell = np.arange(2, n)

    compare_fields(
        rt_cls[:, 2:n],
        born_cls[:, 2:n],
        f"dorian Cl ({label})",
        atol=DORIAN_CLS_ATOL,
        rtol=DORIAN_CLS_RTOL,
        mean_atol=DORIAN_CLS_MEAN_ATOL,
    )

    ells_list = [ell] * len(z_sources)
    jfli_list = [np.asarray(born_cls[i, 2:n]) for i in range(len(z_sources))]
    dorian_list = [np.asarray(rt_cls[i, 2:n]) for i in range(len(z_sources))]
    _plot_spectra(
        ells_list,
        jfli_list,
        dorian_list,
        z_sources,
        f"dorian_{label}",
        os.path.join(PLOT_DIR, f"dorian_spectra_{plot_suffix}.png"),
    )
    ratio_list = [j / (d + 1e-30) for j, d in zip(jfli_list, dorian_list)]
    _plot_ratio(
        ells_list,
        ratio_list,
        z_sources,
        f"dorian_{label}",
        os.path.join(PLOT_DIR, f"dorian_spectra_ratio_{plot_suffix}.png"),
    )


def test_born_vs_dorian_global(born_kappa_multi, raytrace_born_kappa_multi):
    """Compare jfli Born vs dorian (both global normalization): maps and power spectra."""
    pytest.importorskip("dorian")
    _compare_born_vs_dorian(born_kappa_multi, raytrace_born_kappa_multi, "global", "global")


def test_born_vs_dorian_per_plane(born_kappa_multi_per_plane, raytrace_born_kappa_multi_per_plane):
    """Compare jfli Born vs dorian (both per-plane normalization): maps and power spectra."""
    pytest.importorskip("dorian")
    _compare_born_vs_dorian(born_kappa_multi_per_plane, raytrace_born_kappa_multi_per_plane, "per_plane", "per_plane")


# ---------------------------------------------------------------------------
# Per-shell window quadrature (_born_windows)
# ---------------------------------------------------------------------------
def test_born_windows_gl_matches_bruteforce(cosmology):
    """GL-16 windows match a 4096-node brute-force integral of the truncated kernel."""
    r = jnp.linspace(150.0, 2850.0, 10)
    d_r = jnp.full(10, 300.0)
    a = jc.background.a_of_chi(cosmology, r)
    z_s = jnp.asarray([0.3, 0.6, 1.0])
    chi_s = jc.background.radial_comoving_distance(cosmology, jc.utils.z2a(z_s)).reshape(-1)

    w_gl = np.asarray(_born_windows(cosmology, r, a, d_r, chi_s, "gauss_legendre"))

    w_brute = np.zeros_like(w_gl)
    for i in range(r.shape[0]):
        lo, hi = float(r[i]) - 150.0, float(r[i]) + 150.0
        for k in range(chi_s.shape[0]):
            span = max(min(hi, float(chi_s[k])) - lo, 0.0)
            chi = lo + (np.arange(4096) + 0.5) * span / 4096
            inv_a = 1.0 / np.asarray(jc.background.a_of_chi(cosmology, jnp.asarray(chi)))
            w_brute[i, k] = span / 4096 * np.sum(chi * inv_a * (1.0 - chi / float(chi_s[k])))

    mask = w_brute > 0
    max_dev = np.max(np.abs(w_gl[mask] / w_brute[mask] - 1.0))
    assert max_dev < 1e-4, f"GL windows deviate from brute force by {max_dev:.2e}"
    assert np.all(w_gl[~mask] == 0.0)


def test_born_windows_gl_equals_midpoint_thin_shells(cosmology):
    """On thin shells away from the chi_s kink, midpoint and GL windows agree."""
    r = jnp.linspace(200.0, 1000.0, 5)
    d_r = jnp.full(5, 1.0)  # 1 Mpc/h shells
    a = jc.background.a_of_chi(cosmology, r)
    chi_s = jc.background.radial_comoving_distance(cosmology, jc.utils.z2a(jnp.asarray([1.0]))).reshape(-1)

    w_mid = np.asarray(_born_windows(cosmology, r, a, d_r, chi_s, "midpoint"))
    w_gl = np.asarray(_born_windows(cosmology, r, a, d_r, chi_s, "gauss_legendre"))
    assert np.max(np.abs(w_gl / w_mid - 1.0)) < 1e-4


def test_born_windows_fat_shell_direction(cosmology):
    """Fat inner shell: the concave kernel makes midpoint overestimate; straddling shell: midpoint
    clips to zero the sub-chi_s mass that the truncated GL integral keeps."""
    chi_s = jc.background.radial_comoving_distance(cosmology, jc.utils.z2a(jnp.asarray([0.6]))).reshape(-1)
    chi_s_val = float(chi_s[0])

    # (a) fat inner ball [0, 600] entirely inside the support
    r_fat = jnp.asarray([300.0])
    a_fat = jc.background.a_of_chi(cosmology, r_fat)
    w_mid = float(_born_windows(cosmology, r_fat, a_fat, jnp.asarray([600.0]), chi_s, "midpoint")[0, 0])
    w_gl = float(_born_windows(cosmology, r_fat, a_fat, jnp.asarray([600.0]), chi_s, "gauss_legendre")[0, 0])
    assert w_mid > w_gl > 0.0, f"expected midpoint > GL > 0 on the fat shell, got {w_mid:.4g} vs {w_gl:.4g}"

    # (b) shell straddling chi_s with its center beyond the source: lo < chi_s < r
    r_str = jnp.asarray([chi_s_val + 200.0])
    a_str = jc.background.a_of_chi(cosmology, r_str)
    w_mid = float(_born_windows(cosmology, r_str, a_str, jnp.asarray([600.0]), chi_s, "midpoint")[0, 0])
    w_gl = float(_born_windows(cosmology, r_str, a_str, jnp.asarray([600.0]), chi_s, "gauss_legendre")[0, 0])
    assert w_mid == 0.0
    assert w_gl > 0.0


def test_born_windows_simpson_matches_gl(cosmology):
    """Composite-Simpson windows agree with GL-16 down to the a_of_chi interpolation floor."""
    r = jnp.linspace(150.0, 2850.0, 10)
    d_r = jnp.full(10, 300.0)
    a = jc.background.a_of_chi(cosmology, r)
    z_s = jnp.asarray([0.3, 0.6, 1.0])
    chi_s = jc.background.radial_comoving_distance(cosmology, jc.utils.z2a(z_s)).reshape(-1)

    w_sim = np.asarray(_born_windows(cosmology, r, a, d_r, chi_s, "simpson"))
    w_gl = np.asarray(_born_windows(cosmology, r, a, d_r, chi_s, "gauss_legendre"))
    assert np.max(np.abs(w_sim - w_gl)) / np.max(np.abs(w_gl)) < 1e-5
    assert np.all(w_sim[w_gl == 0.0] == 0.0)  # shells fully behind the source get exactly zero weight


def test_born_quadrature_invalid_raises(cosmology, lensing_lightcone):
    """An unknown quadrature name raises."""
    with pytest.raises(ValueError, match="quadrature"):
        jfli.born(cosmology, lensing_lightcone, nz_shear=[0.5], quadrature="trapezoid")


def test_born_simpson_default_and_close_to_gl(cosmology, lensing_lightcone):
    """The default quadrature is simpson, and it matches GL at the map level (multi-node rules tie)."""
    sim = jfli.born(cosmology, lensing_lightcone, nz_shear=[0.5, 1.0], normalization="global", quadrature="simpson")
    default = jfli.born(cosmology, lensing_lightcone, nz_shear=[0.5, 1.0], normalization="global")
    gl = jfli.born(
        cosmology, lensing_lightcone, nz_shear=[0.5, 1.0], normalization="global", quadrature="gauss_legendre"
    )
    sim_arr, gl_arr = np.asarray(sim.array), np.asarray(gl.array)
    assert np.array_equal(np.asarray(default.array), sim_arr)
    assert np.all(np.isfinite(sim_arr))
    assert np.max(np.abs(sim_arr - gl_arr)) / np.max(np.abs(gl_arr)) < 1e-3


def test_born_simpson_flat_runs(cosmology, lensing_flat_lightcone):
    """Flat-sky simpson Born runs, is finite, and matches the flat GL maps."""
    sim = jfli.born(
        cosmology, lensing_flat_lightcone, nz_shear=[0.5, 1.0], normalization="global", quadrature="simpson"
    )
    gl = jfli.born(
        cosmology, lensing_flat_lightcone, nz_shear=[0.5, 1.0], normalization="global", quadrature="gauss_legendre"
    )
    sim_arr, gl_arr = np.asarray(sim.array), np.asarray(gl.array)
    assert sim_arr.shape == gl_arr.shape
    assert np.all(np.isfinite(sim_arr))
    assert np.max(np.abs(sim_arr - gl_arr)) / np.max(np.abs(gl_arr)) < 1e-3


def test_born_gl_flat_runs(cosmology, lensing_flat_lightcone, born_flat_kappa_multi):
    """Flat-sky GL Born runs, is finite, and stays close to the flat midpoint maps."""
    gl = jfli.born(
        cosmology, lensing_flat_lightcone, nz_shear=[0.5, 1.0], normalization="global", quadrature="gauss_legendre"
    )
    gl_arr = np.asarray(gl.array)
    mid_arr = np.asarray(born_flat_kappa_multi.array)
    assert gl_arr.shape == mid_arr.shape
    assert np.all(np.isfinite(gl_arr))
    # The chi_s-straddling shell legitimately differs O(10%) between the schemes on ~300 Mpc/h
    # shells (midpoint counts its full width, GL truncates at chi_s); measured 0.134 here.
    max_dev = np.max(np.abs(gl_arr - mid_arr)) / np.max(np.abs(mid_arr))
    assert max_dev < 0.2, f"flat GL vs midpoint relative deviation {max_dev:.3f}"
