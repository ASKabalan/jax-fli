"""Shared utilities, constants, and plot helpers for the Analysis page."""
from __future__ import annotations

import io
from threading import RLock
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

# ---------------------------------------------------------------------------
# Thread safety — matplotlib is not thread-safe
# ---------------------------------------------------------------------------
_plt_lock = RLock()

# ---------------------------------------------------------------------------
# Field type groups
# ---------------------------------------------------------------------------
_SPHERICAL_TYPES = {"SphericalDensity", "SphericalKappaField"}
_FLAT_TYPES = {"FlatDensity", "FlatKappaField"}
_KAPPA_TYPES = {"SphericalKappaField", "FlatKappaField"}
_DENSITY_3D = {"DensityField"}
_PARTICLE_TYPE = {"ParticleField"}
_SPECTRA_TYPES = {"PowerSpectrum"}

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
_PALETTE = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
_COLOR_THEORY = "black"


# ---------------------------------------------------------------------------
# Index parsing
# ---------------------------------------------------------------------------


def parse_slice(s: str) -> Union[slice, int]:
    """
    Parses strings like ':', '1:5', ':-1', '-10:', or '::-1' into a slice object,
    and single values like '0' or '-1' into an integer.
    """
    s = s.strip()

    # Handle empty strings or a generic slice
    if not s or s == ":":
        return slice(None)

    def to_int(val):
        val = val.strip()
        return int(val) if val else None

    # If there are no colons, treat it as a single integer index
    if ":" not in s:
        try:
            return int(s)
        except ValueError:
            # Fallback for invalid strings (matches your original error handling)
            return slice(None)

    # Split by ':' and take up to 3 parts (start, stop, step)
    parts = s.split(":")

    try:
        # Since ':' is in `s`, `parts` will always have at least 2 elements.
        # slice(*[start, stop]) -> slice(start, stop, None)
        # slice(*[start, stop, step]) -> slice(start, stop, step)
        return slice(*[to_int(p) for p in parts[:3]])
    except ValueError:
        # Fallback for invalid strings
        return slice(None)


def indexed_field(entry: dict):
    """Primary field of a catalog entry, sliced by the entry's global index string."""
    fld = entry["catalog"].field[0]
    return fld[parse_slice(entry.get("index", ":"))]


def _spectral_category(entry: dict) -> str | None:
    """Spectral group an entry can be compared within: ``"cl"``, ``"pk"``, or ``None``.

    A raw field maps by type (spherical → Angular Cl, DensityField → 3D P(k)); a
    precomputed ``PowerSpectrum`` maps by its ``.unit`` (ANGULAR_CL → cl, POWER_SPECTRA
    → pk). Flat and particle fields have no spectrum here and return ``None``.
    """
    ft = entry["field_type"]
    if ft in _SPECTRA_TYPES:
        from jax_fli import SpectralUnit

        unit = getattr(entry["catalog"].field[0], "unit", None)
        return "pk" if unit == SpectralUnit.POWER_SPECTRA else "cl"
    if ft in _DENSITY_3D:
        return "pk"
    if ft in _SPHERICAL_TYPES:
        return "cl"
    return None


# ---------------------------------------------------------------------------
# Figure utilities
# ---------------------------------------------------------------------------


def _fig_to_png(fig, dpi: int = 100) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def _fig_to_pdf(fig, dpi: int = 100) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def _make_title(template: str, field, idx: int, *, label: str = "") -> str:
    """Substitute all placeholders in a title template with field metadata."""
    title = template.replace("%l%", label).replace("%i%", str(idx))
    for attr, key, fmt in [
        ("comoving_centers", "%r%", ".3f"),
        ("z_sources", "%z%", ".3f"),
        ("scale_factors", "%a%", ".3f"),
        ("density_width", "%d%", ".3f"),
    ]:
        if key not in title:
            continue
        arr = getattr(field, attr, None)
        if arr is not None:
            try:
                title = title.replace(
                    key, format(float(np.atleast_1d(np.asarray(arr))[idx]), fmt)
                )
            except Exception:
                pass
    return title


def _add_shading(ax, bands: list[float]) -> None:
    """Draw ±pct axhspan bands (sorted descending, widest first) and a dotted 1.0 line."""
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
    pcts = sorted(bands, reverse=True)
    base_alphas = [0.1, 0.25, 0.4]
    for i, frac in enumerate(pcts):
        alpha = base_alphas[i] if i < len(base_alphas) else 0.1
        ax.axhspan(1.0 - frac, 1.0 + frac, color="gray", alpha=alpha, zorder=0)


def _clean_ratio_ax(ax, ylabel: str, bands: list[float], ylim=(0.85, 1.15)) -> None:
    _add_shading(ax, bands)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("")
    ax.set_yscale("linear")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def _apply_shared_log_ylim(fig) -> None:
    """Set one shared, data-bracketing y-range on every log-scale panel of a figure.

    matplotlib's log autoscale can span >10 decades when a handful of near-zero
    points are present (e.g. the C_ell monopole/dipole at ell = 0, 1), squashing the
    real signal out of view. A *single* range across all log panels also keeps the
    hidden y-tick labels on non-leftmost columns meaningful. The range brackets the
    bulk: floor = 2nd-percentile positive value (clamped to at most 6 decades below
    the max), padded ~half a decade each side. Linear axes (ratio panels) are left
    untouched; no-op when there is no positive data on any log panel.
    """
    log_axes = [ax for ax in fig.axes if ax.get_yscale() == "log"]
    vals = [
        np.asarray(ln.get_ydata(), dtype=float)
        for ax in log_axes
        for ln in ax.get_lines()
    ]
    if not vals:
        return
    v = np.concatenate(vals)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < 3:
        return
    hi = float(np.max(v))
    lo = max(float(np.percentile(v, 2.0)), hi * 1e-6)
    if hi > lo > 0:
        for ax in log_axes:
            ax.set_ylim(lo / 2.0, hi * 2.0)


# ---------------------------------------------------------------------------
# Pixel window
# ---------------------------------------------------------------------------


def pixel_window_function(ell, pixel_size_arcmin):
    """Pixel window function W_l = sinc²(l · θ_pix / 2π)."""
    pixel_size_rad = pixel_size_arcmin * (np.pi / (180.0 * 60.0))
    return (np.sinc(ell * pixel_size_rad / (2 * np.pi))) ** 2


# ---------------------------------------------------------------------------
# JCAP publication style
# ---------------------------------------------------------------------------


def set_jcap_style() -> None:
    """Configure matplotlib for JCAP-quality publication figures."""
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 11,
            "figure.figsize": (6.0, 4.5),
            "figure.dpi": 150,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "axes.linewidth": 0.8,
            "xtick.top": True,
            "ytick.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.major.size": 6,
            "xtick.minor.size": 3,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
        }
    )
