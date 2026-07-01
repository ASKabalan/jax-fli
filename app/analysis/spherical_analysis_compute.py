"""Angular Cl figure builders, layout helpers, and compute functions.

Figure builders follow a strict "one function per plot mode" rule so dispatch
is a single dict lookup with no argument-count branching.

Compute functions cover all probe types (density, s3, point sources) and all
field types (spherical density, kappa, flat).
"""
from __future__ import annotations

from math import ceil
from typing import Any, Callable

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.figure import Figure

# Registers jax_fli.data as an attribute of the jax_fli package; below, jfli.data.* is
# otherwise reachable only as a side effect of jax_fli.io's own import graph.
from jax_fli import data  # noqa: F401

from .utils import (
    _COLOR_THEORY,
    _PALETTE,
    _clean_ratio_ax,
    _make_title,
    indexed_field,
    pixel_window_function,
)

# ---------------------------------------------------------------------------
# Private layout helpers
# ---------------------------------------------------------------------------


def _grid_coords(
    n_selected: int, ncols: int
) -> tuple[list[tuple[int, int]], int, int, tuple[int, int]]:
    """Row-major rectangular layout reserving the top-right cell for the legend.

    Returns ``(coords, nrows, ncols, legend_cell)``. ``coords`` is the ``(row, col)``
    of each of the ``n`` panels in row-major order, skipping the reserved legend cell
    at ``(0, ncols - 1)``. Trailing cells of the last row are left empty.
    """
    n = max(1, int(n_selected))
    # Cap to n+1 so picking >= n+1 columns yields a single row (all panels + legend),
    # and never produces absurdly wide grids of empty cells.
    ncols = max(1, min(int(ncols), n + 1))
    legend_cell = (0, ncols - 1)
    nrows = max(1, ceil((n + 1) / ncols))
    coords: list[tuple[int, int]] = []
    for r in range(nrows):
        for c in range(ncols):
            if (r, c) == legend_cell:
                continue
            if len(coords) < n:
                coords.append((r, c))
    return coords, nrows, ncols, legend_cell


def _setup_spectra_grid(n, layout_params, height_ratios):
    """Build the figure + outer GridSpec shared by every Cl builder.

    Returns ``(fig, outer_gs, coords, coords_set, legend_cell)``. The number of
    columns is read from ``layout_params["spec_ncols"]`` (square-ish ``ceil(sqrt(n))``
    fallback when absent).
    """
    ncols_req = int(layout_params.get("spec_ncols") or max(1, ceil(n**0.5)))
    coords, nrows, ncols, legend_cell = _grid_coords(n, ncols_req)
    fig = plt.figure(
        figsize=(
            max(12, float(layout_params["spec_fig_w"]) * ncols),
            sum(height_ratios) * nrows,
        )
    )
    outer_gs = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.08, hspace=0.25)
    return fig, outer_gs, coords, set(coords), legend_cell


def _attach_legend(fig, handles, labels, bands: list[float], anchor=None) -> None:
    """Append shading-band patches to existing handles and attach a figure legend.

    When ``anchor`` (a figure-fraction Bbox) is given the legend is centered inside
    that cell (the reserved top-right grid cell); otherwise it floats at the top-right
    of the figure.
    """
    if bands:
        spacer = mlines.Line2D([], [], color="none")
        handles.append(spacer)
        labels.append("")
        pcts_sorted = sorted(bands, reverse=True)
        base_alphas = [0.1, 0.25, 0.4]
        for i, frac in enumerate(pcts_sorted):
            alpha = base_alphas[i] if i < len(base_alphas) else 0.1
            patch = mpatches.Patch(facecolor="gray", alpha=alpha, edgecolor="none")
            handles.append(patch)
            labels.append(f"±{frac*100:.0f}%")

    if anchor is not None:
        leg = fig.legend(
            handles,
            labels,
            loc="center",
            bbox_to_anchor=anchor,
            bbox_transform=fig.transFigure,
            fontsize=12,
            frameon=True,
            borderpad=1.0,
            labelspacing=0.8,
        )
    else:
        leg = fig.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.9, 0.9),
            fontsize=12,
            frameon=True,
            borderpad=1.0,
            labelspacing=0.8,
        )
    for legobj in leg.legend_handles:
        if isinstance(legobj, mlines.Line2D):
            legobj.set_linewidth(4.0)


def _make_inner_gs(outer_gs, row, col, height_ratios):
    """Create a nested GridSpec for one grid cell."""
    return gridspec.GridSpecFromSubplotSpec(
        len(height_ratios),
        1,
        subplot_spec=outer_gs[row, col],
        height_ratios=height_ratios,
        hspace=0.05,
    )


def _n_shells(spectra_results) -> int:
    arr = spectra_results[0][1].array
    return arr.shape[0] if arr.ndim > 1 else 1


def _cl_slice(cl, shell_idx: int, n_shells: int):
    return cl[shell_idx] if n_shells > 1 else cl


# ---------------------------------------------------------------------------
# Seven figure builders — one per (compare_ref, compare_theory, ratio_only)
# ---------------------------------------------------------------------------
# Uniform signature for all builders:
#   spectra_results  – list of (label, Cl), already sliced to the chosen shells
#   theory_result    – Cl or None
#   layout_params    – {"spec_fig_w", "spec_main_h", "spec_ratio_h", "spec_ncols"}
#   title_template   – str
#   bands            – list[float] (fractional shading bands, e.g. [0.02, 0.05])
#
# Array indexing uses the loop counter i (0, 1, …) because spectra_results is
# pre-sliced.  Titles use spectra_results[0][1] (the sliced Cl) so metadata
# (comoving_centers, scale_factors, etc.) is already correctly aligned.


def _build_cl_main_only(
    spectra_results,
    theory_result,
    layout_params,
    title_template,
    bands,
) -> Figure:
    """Rectangular Cl panels only — no ratio rows."""
    ns = _n_shells(spectra_results)
    height_ratios = [float(layout_params["spec_main_h"])]
    fig, outer_gs, coords, coords_set, legend_cell = _setup_spectra_grid(
        ns, layout_params, height_ratios
    )

    handles_out, labels_out = [], []
    for i, (row, col) in enumerate(coords):
        is_bottom = (row + 1, col) not in coords_set
        inner_gs = _make_inner_gs(outer_gs, row, col, height_ratios)
        ax = fig.add_subplot(inner_gs[0, 0])

        for ci, (lbl, cl) in enumerate(spectra_results):
            color = _PALETTE[ci % len(_PALETTE)]
            cl_s = _cl_slice(cl, i, ns)
            label = f"{lbl} (Ref)" if ci == 0 else lbl
            cl_s.plot(ax=ax, logx=True, logy=True, label=label, color=color)

        ax.grid(True, which="both", ls="--", alpha=0.2)
        ax.set_title(_make_title(title_template, spectra_results[0][1], i))
        if col == 0:
            ax.set_ylabel(r"$C_\ell$")
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        if is_bottom:
            ax.set_xlabel(r"$\ell$")
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

        if i == 0:
            handles_out, labels_out = ax.get_legend_handles_labels()

    _attach_legend(
        fig,
        handles_out,
        labels_out,
        bands,
        anchor=outer_gs[legend_cell].get_position(fig),
    )
    return fig


def _build_cl_with_ref_ratio(
    spectra_results,
    theory_result,
    layout_params,
    title_template,
    bands,
) -> Figure:
    """Rectangular Cl panels + ratio vs reference row."""
    ns = _n_shells(spectra_results)
    height_ratios = [
        float(layout_params["spec_main_h"]),
        float(layout_params["spec_ratio_h"]),
    ]
    fig, outer_gs, coords, coords_set, legend_cell = _setup_spectra_grid(
        ns, layout_params, height_ratios
    )

    handles_out, labels_out = [], []
    for i, (row, col) in enumerate(coords):
        is_bottom = (row + 1, col) not in coords_set
        inner_gs = _make_inner_gs(outer_gs, row, col, height_ratios)

        ax_main = fig.add_subplot(inner_gs[0, 0])
        for ci, (lbl, cl) in enumerate(spectra_results):
            color = _PALETTE[ci % len(_PALETTE)]
            cl_s = _cl_slice(cl, i, ns)
            label = f"{lbl} (Ref)" if ci == 0 else lbl
            cl_s.plot(ax=ax_main, logx=True, logy=True, label=label, color=color)
        ax_main.grid(True, which="both", ls="--", alpha=0.2)
        ax_main.set_title(_make_title(title_template, spectra_results[0][1], i))
        if col == 0:
            ax_main.set_ylabel(r"$C_\ell$")
        else:
            ax_main.set_ylabel("")
            ax_main.tick_params(labelleft=False)
        ax_main.tick_params(labelbottom=False)
        ax_main.set_xlabel("")
        if i == 0:
            handles_out, labels_out = ax_main.get_legend_handles_labels()

        ax_r = fig.add_subplot(inner_gs[1, 0], sharex=ax_main)
        _, ref_cl = spectra_results[0]
        ref_s = _cl_slice(ref_cl, i, ns)
        for ci, (lbl, cl) in enumerate(spectra_results[1:], 1):
            color = _PALETTE[ci % len(_PALETTE)]
            cl_s = _cl_slice(cl, i, ns)
            (cl_s / ref_s).plot(ax=ax_r, logx=True, color=color, legend=False)
        ylabel = "Ratio\n(vs Ref)" if col == 0 else ""
        _clean_ratio_ax(ax_r, ylabel, bands)
        if col != 0:
            ax_r.tick_params(labelleft=False)
        if is_bottom:
            ax_r.set_xlabel(r"$\ell$")
        else:
            ax_r.tick_params(labelbottom=False)

    _attach_legend(
        fig,
        handles_out,
        labels_out,
        bands,
        anchor=outer_gs[legend_cell].get_position(fig),
    )
    return fig


def _build_cl_with_theory_ratio(
    spectra_results,
    theory_result,
    layout_params,
    title_template,
    bands,
) -> Figure:
    """Rectangular Cl panels + ratio vs theory row."""
    ns = _n_shells(spectra_results)
    height_ratios = [
        float(layout_params["spec_main_h"]),
        float(layout_params["spec_ratio_h"]),
    ]
    fig, outer_gs, coords, coords_set, legend_cell = _setup_spectra_grid(
        ns, layout_params, height_ratios
    )

    handles_out, labels_out = [], []
    for i, (row, col) in enumerate(coords):
        is_bottom = (row + 1, col) not in coords_set
        inner_gs = _make_inner_gs(outer_gs, row, col, height_ratios)

        ax_main = fig.add_subplot(inner_gs[0, 0])
        th_s = theory_result[i] if theory_result.array.ndim > 1 else theory_result
        th_s.plot(
            ax=ax_main,
            logx=True,
            logy=True,
            label="Theory",
            color=_COLOR_THEORY,
            linestyle="--",
        )
        for ci, (lbl, cl) in enumerate(spectra_results):
            color = _PALETTE[ci % len(_PALETTE)]
            cl_s = _cl_slice(cl, i, ns)
            label = f"{lbl} (Ref)" if ci == 0 else lbl
            cl_s.plot(ax=ax_main, logx=True, logy=True, label=label, color=color)
            # ylims is 1e-7 5e-2
            ax_main.set_ylim(1e-7, 5e-2)
        ax_main.grid(True, which="both", ls="--", alpha=0.2)
        ax_main.set_title(_make_title(title_template, spectra_results[0][1], i))
        if col == 0:
            ax_main.set_ylabel(r"$C_\ell$")
        else:
            ax_main.set_ylabel("")
            ax_main.tick_params(labelleft=False)
        ax_main.tick_params(labelbottom=False)
        ax_main.set_xlabel("")
        if i == 0:
            handles_out, labels_out = ax_main.get_legend_handles_labels()

        ax_t = fig.add_subplot(inner_gs[1, 0], sharex=ax_main)
        for ci, (lbl, cl) in enumerate(spectra_results):
            color = _PALETTE[ci % len(_PALETTE)]
            cl_s = _cl_slice(cl, i, ns)
            (cl_s / th_s).plot(ax=ax_t, logx=True, color=color, legend=False)
        ylabel = "Ratio\n(vs Theory)" if col == 0 else ""
        _clean_ratio_ax(ax_t, ylabel, bands)
        if col != 0:
            ax_t.tick_params(labelleft=False)
        if is_bottom:
            ax_t.set_xlabel(r"$\ell$")
        else:
            ax_t.tick_params(labelbottom=False)

    _attach_legend(
        fig,
        handles_out,
        labels_out,
        bands,
        anchor=outer_gs[legend_cell].get_position(fig),
    )
    return fig


def _build_cl_with_both_ratios(
    spectra_results,
    theory_result,
    layout_params,
    title_template,
    bands,
) -> Figure:
    """Rectangular Cl panels + ratio vs ref row + ratio vs theory row."""
    ns = _n_shells(spectra_results)
    height_ratios = [
        float(layout_params["spec_main_h"]),
        float(layout_params["spec_ratio_h"]),
        float(layout_params["spec_ratio_h"]),
    ]
    fig, outer_gs, coords, coords_set, legend_cell = _setup_spectra_grid(
        ns, layout_params, height_ratios
    )

    handles_out, labels_out = [], []
    for i, (row, col) in enumerate(coords):
        is_bottom = (row + 1, col) not in coords_set
        inner_gs = _make_inner_gs(outer_gs, row, col, height_ratios)

        ax_main = fig.add_subplot(inner_gs[0, 0])
        th_s = theory_result[i] if theory_result.array.ndim > 1 else theory_result
        th_s.plot(
            ax=ax_main,
            logx=True,
            logy=True,
            label="Theory",
            color=_COLOR_THEORY,
            linestyle="--",
        )
        for ci, (lbl, cl) in enumerate(spectra_results):
            color = _PALETTE[ci % len(_PALETTE)]
            cl_s = _cl_slice(cl, i, ns)
            label = f"{lbl} (Ref)" if ci == 0 else lbl
            cl_s.plot(ax=ax_main, logx=True, logy=True, label=label, color=color)
        ax_main.grid(True, which="both", ls="--", alpha=0.2)
        ax_main.set_title(_make_title(title_template, spectra_results[0][1], i))
        if col == 0:
            ax_main.set_ylabel(r"$C_\ell$")
        else:
            ax_main.set_ylabel("")
            ax_main.tick_params(labelleft=False)
        ax_main.tick_params(labelbottom=False)
        ax_main.set_xlabel("")
        if i == 0:
            handles_out, labels_out = ax_main.get_legend_handles_labels()

        ax_r = fig.add_subplot(inner_gs[1, 0], sharex=ax_main)
        _, ref_cl = spectra_results[0]
        ref_s = _cl_slice(ref_cl, i, ns)
        for ci, (lbl, cl) in enumerate(spectra_results[1:], 1):
            color = _PALETTE[ci % len(_PALETTE)]
            cl_s = _cl_slice(cl, i, ns)
            (cl_s / ref_s).plot(ax=ax_r, logx=True, color=color, legend=False)
        ylabel_r = "Ratio\n(vs Ref)" if col == 0 else ""
        _clean_ratio_ax(ax_r, ylabel_r, bands)
        if col != 0:
            ax_r.tick_params(labelleft=False)
        ax_r.tick_params(labelbottom=False)
        ax_r.set_xlabel("")

        ax_t = fig.add_subplot(inner_gs[2, 0], sharex=ax_main)
        for ci, (lbl, cl) in enumerate(spectra_results):
            color = _PALETTE[ci % len(_PALETTE)]
            cl_s = _cl_slice(cl, i, ns)
            (cl_s / th_s).plot(ax=ax_t, logx=True, color=color, legend=False)
        ylabel_t = "Ratio\n(vs Theory)" if col == 0 else ""
        _clean_ratio_ax(ax_t, ylabel_t, bands)
        if col != 0:
            ax_t.tick_params(labelleft=False)
        if is_bottom:
            ax_t.set_xlabel(r"$\ell$")
        else:
            ax_t.tick_params(labelbottom=False)

    _attach_legend(
        fig,
        handles_out,
        labels_out,
        bands,
        anchor=outer_gs[legend_cell].get_position(fig),
    )
    return fig


def _build_cl_ratio_only_ref(
    spectra_results,
    theory_result,
    layout_params,
    title_template,
    bands,
) -> Figure:
    """Ratio vs reference panels only — no main Cl panel."""
    ns = _n_shells(spectra_results)
    height_ratios = [float(layout_params["spec_ratio_h"])]
    fig, outer_gs, coords, coords_set, legend_cell = _setup_spectra_grid(
        ns, layout_params, height_ratios
    )

    handles_out, labels_out = [], []
    for i, (row, col) in enumerate(coords):
        is_bottom = (row + 1, col) not in coords_set
        inner_gs = _make_inner_gs(outer_gs, row, col, height_ratios)

        ax = fig.add_subplot(inner_gs[0, 0])
        _, ref_cl = spectra_results[0]
        ref_s = _cl_slice(ref_cl, i, ns)
        for ci, (lbl, cl) in enumerate(spectra_results[1:], 1):
            color = _PALETTE[ci % len(_PALETTE)]
            cl_s = _cl_slice(cl, i, ns)
            (cl_s / ref_s).plot(ax=ax, logx=True, color=color, legend=False, label=lbl)

        ax.set_title(_make_title(title_template, spectra_results[0][1], i))
        ylabel = "Ratio\n(vs Ref)" if col == 0 else ""
        _clean_ratio_ax(ax, ylabel, bands)
        if col != 0:
            ax.tick_params(labelleft=False)
        if is_bottom:
            ax.set_xlabel(r"$\ell$")
        else:
            ax.tick_params(labelbottom=False)

        if i == 0:
            handles_out, labels_out = ax.get_legend_handles_labels()

    _attach_legend(
        fig,
        handles_out,
        labels_out,
        bands,
        anchor=outer_gs[legend_cell].get_position(fig),
    )
    return fig


def _build_cl_ratio_only_theory(
    spectra_results,
    theory_result,
    layout_params,
    title_template,
    bands,
) -> Figure:
    """Ratio vs theory panels only — no main Cl panel."""
    ns = _n_shells(spectra_results)
    height_ratios = [float(layout_params["spec_ratio_h"])]
    fig, outer_gs, coords, coords_set, legend_cell = _setup_spectra_grid(
        ns, layout_params, height_ratios
    )

    handles_out, labels_out = [], []
    for i, (row, col) in enumerate(coords):
        is_bottom = (row + 1, col) not in coords_set
        inner_gs = _make_inner_gs(outer_gs, row, col, height_ratios)

        ax = fig.add_subplot(inner_gs[0, 0])
        th_s = theory_result[i] if theory_result.array.ndim > 1 else theory_result
        for ci, (lbl, cl) in enumerate(spectra_results):
            color = _PALETTE[ci % len(_PALETTE)]
            cl_s = _cl_slice(cl, i, ns)
            (cl_s / th_s).plot(ax=ax, logx=True, color=color, legend=False, label=lbl)

        ax.set_title(_make_title(title_template, spectra_results[0][1], i))
        ylabel = "Ratio\n(vs Theory)" if col == 0 else ""
        _clean_ratio_ax(ax, ylabel, bands)
        if col != 0:
            ax.tick_params(labelleft=False)
        if is_bottom:
            ax.set_xlabel(r"$\ell$")
        else:
            ax.tick_params(labelbottom=False)

        if i == 0:
            handles_out, labels_out = ax.get_legend_handles_labels()

    _attach_legend(
        fig,
        handles_out,
        labels_out,
        bands,
        anchor=outer_gs[legend_cell].get_position(fig),
    )
    return fig


def _build_cl_ratio_only_both(
    spectra_results,
    theory_result,
    layout_params,
    title_template,
    bands,
) -> Figure:
    """Both ratio rows (vs ref and vs theory) — no main Cl panel."""
    ns = _n_shells(spectra_results)
    rh = float(layout_params["spec_ratio_h"])
    height_ratios = [rh, rh]
    fig, outer_gs, coords, coords_set, legend_cell = _setup_spectra_grid(
        ns, layout_params, height_ratios
    )

    handles_out, labels_out = [], []
    for i, (row, col) in enumerate(coords):
        is_bottom = (row + 1, col) not in coords_set
        inner_gs = _make_inner_gs(outer_gs, row, col, height_ratios)

        th_s = theory_result[i] if theory_result.array.ndim > 1 else theory_result
        _, ref_cl = spectra_results[0]
        ref_s = _cl_slice(ref_cl, i, ns)

        ax_r = fig.add_subplot(inner_gs[0, 0])
        for ci, (lbl, cl) in enumerate(spectra_results[1:], 1):
            color = _PALETTE[ci % len(_PALETTE)]
            cl_s = _cl_slice(cl, i, ns)
            (cl_s / ref_s).plot(
                ax=ax_r, logx=True, color=color, legend=False, label=lbl
            )
        ax_r.set_title(_make_title(title_template, spectra_results[0][1], i))
        ylabel_r = "Ratio\n(vs Ref)" if col == 0 else ""
        _clean_ratio_ax(ax_r, ylabel_r, bands)
        if col != 0:
            ax_r.tick_params(labelleft=False)
        ax_r.tick_params(labelbottom=False)
        ax_r.set_xlabel("")
        if i == 0:
            handles_out, labels_out = ax_r.get_legend_handles_labels()

        ax_t = fig.add_subplot(inner_gs[1, 0], sharex=ax_r)
        for ci, (lbl, cl) in enumerate(spectra_results):
            color = _PALETTE[ci % len(_PALETTE)]
            cl_s = _cl_slice(cl, i, ns)
            (cl_s / th_s).plot(ax=ax_t, logx=True, color=color, legend=False)
        ylabel_t = "Ratio\n(vs Theory)" if col == 0 else ""
        _clean_ratio_ax(ax_t, ylabel_t, bands)
        if col != 0:
            ax_t.tick_params(labelleft=False)
        if is_bottom:
            ax_t.set_xlabel(r"$\ell$")
        else:
            ax_t.tick_params(labelbottom=False)

    _attach_legend(
        fig,
        handles_out,
        labels_out,
        bands,
        anchor=outer_gs[legend_cell].get_position(fig),
    )
    return fig


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_CL_BUILDERS: dict[tuple[bool, bool, bool], Callable[..., Any]] = {
    (False, False, False): _build_cl_main_only,
    (True, False, False): _build_cl_with_ref_ratio,
    (False, True, False): _build_cl_with_theory_ratio,
    (True, True, False): _build_cl_with_both_ratios,
    (True, False, True): _build_cl_ratio_only_ref,
    (False, True, True): _build_cl_ratio_only_theory,
    (True, True, True): _build_cl_ratio_only_both,
}


# ---------------------------------------------------------------------------
# Pixel window helper
# ---------------------------------------------------------------------------


def _apply_pixwin(
    theory_result,
    ref_field_obj,
    lmin: int,
    lmax: int,
) -> object:
    """Multiply theory Cl by the pixel window squared.

    Branch selection:
    - ``nside`` attribute set (not None) → spherical: ``healpy.pixwin``
    - ``flatsky_npix`` attribute present → flat: sinc² from pixel scale
    - neither → return unchanged
    """
    nside_val = getattr(ref_field_obj, "nside", None)
    if nside_val is not None:
        import healpy as hp

        pw_full = np.asarray(hp.pixwin(int(nside_val), lmax=int(lmax)))
        pw = pw_full[int(lmin) :]
        return theory_result.replace(array=theory_result.array * pw**2)

    flatsky_npix = getattr(ref_field_obj, "flatsky_npix", None)
    if flatsky_npix is not None:
        ny, nx = flatsky_npix
        fs = ref_field_obj.field_size
        if hasattr(fs, "__len__"):
            size_y, size_x = float(fs[0]), float(fs[1])
        else:
            size_y = size_x = float(fs)
        pixel_scale = 0.5 * (size_y * 60.0 / ny + size_x * 60.0 / nx)
        pw = pixel_window_function(np.asarray(theory_result.wavenumber), pixel_scale)
        return theory_result.replace(array=theory_result.array * pw**2)

    return theory_result


def _slice_results_to_shells(
    spectra_results: list[tuple],
    selected_shells: slice,
) -> list[tuple]:
    """Return a copy of spectra_results keeping only the selected shells.

    Uses native PowerSpectrum indexing so all metadata (comoving_centers,
    scale_factors, etc.) is correctly sliced.  Single-shell Cls are returned
    unchanged (1-D array has no leading shell axis).
    """
    if not spectra_results:
        return spectra_results
    if _n_shells(spectra_results) == 1:
        return spectra_results
    return [(label, cl[selected_shells]) for label, cl in spectra_results]


# ---------------------------------------------------------------------------
# Cl computation
# ---------------------------------------------------------------------------


def _observer_visibility_mask(field):
    """Apodized observer-visibility footprint inferred from the field's observer
    position. Returns None for a centered observer (whole sky) — i.e. no footprint
    restriction. Apodization scale fixed at 1.0 deg (matches fli-simulate /
    fli-summary-stats); not user-configurable by design."""
    import jax_fli as jfli

    nside = getattr(field, "nside", None)
    if nside is None:
        return None
    mask = jfli.data.build_observer_visibility_mask(
        tuple(field.observer_position), nside, 1.0
    )
    return None if np.ndim(mask) == 0 else mask


def compute_cls(
    active_entries: list[dict],
    lmin: int,
    lmax: int,
) -> list[tuple[str, Any]]:
    """Return one Angular Cl ``PowerSpectrum`` per entry, all sliced to ``[lmin, lmax]``.

    Entries are dispatched per type so a selection may mix precomputed spectra with
    raw maps:
      - ``PowerSpectrum`` (already an Angular Cl) — used as-is.
      - ``SphericalDensity`` — contrast-normalised (``to(OVERDENSITY, per_plane)``) then
        ``angular_cl(lmax=…, method='healpy')``.
      - ``SphericalKappaField`` — ``angular_cl(...)`` directly (no normalisation).

    Off-centre observers yield a mask, so ``angular_cl`` returns decoupled bandpowers;
    such a Cl is only comparable with stored Cls on the same ell grid (the caller's
    ell-length check guards this).
    """
    from jax_fli.units import OVERDENSITY

    spectra_results = []
    for entry in active_entries:
        fld = indexed_field(entry)
        ft = entry["field_type"]
        if ft == "PowerSpectrum":
            ps = fld  # already an Angular Cl spectrum
        elif ft == "SphericalDensity":
            fld = fld.to(OVERDENSITY, normalization="per_plane")
            ps = fld.angular_cl(
                lmax=int(lmax), method="healpy", mask=_observer_visibility_mask(fld)
            )
        elif ft == "SphericalKappaField":
            ps = fld.angular_cl(
                lmax=int(lmax), method="healpy", mask=_observer_visibility_mask(fld)
            )
        else:
            raise ValueError(f"Unsupported field type for Cl computation: {ft}")
        ells = np.asarray(ps.wavenumber)
        lo = int(np.searchsorted(ells, int(lmin)))
        hi = int(np.searchsorted(ells, int(lmax), side="right"))
        cl = ps.replace(array=ps.array[..., lo:hi], wavenumber=ps.wavenumber[lo:hi])
        spectra_results.append((entry["label"], cl))
    return spectra_results


# ---------------------------------------------------------------------------
# Theory computation
# ---------------------------------------------------------------------------


def compute_theory_cl(
    ref_obj,
    ells,
    nl_fn_name: str,
    probe_type: str,
    z_sources: list[float],
    nz_zmax: float = 2.0,
    apply_pixwin: bool = False,
    lmin: int = 0,
    lmax: int = 500,
    selected_shells=None,
) -> object | None:
    """Compute theory Cl for any probe type.

    Parameters
    ----------
    probe_type:
        ``"density"``       — density-density Cl via Limber.
        ``"s3"``            — weak-lensing Cl using Stage-III n(z).
        ``"point sources"`` — weak-lensing Cl using explicit ``z_sources``.
    z_sources:
        Source redshifts for ``probe_type="point sources"``.
    nz_zmax:
        Maximum redshift for the density probe (ignored for lensing probes).
    apply_pixwin:
        Multiply theory by the pixel window squared if True.
    lmin, lmax:
        Ell range, used for pixel-window slicing in the spherical branch.
    selected_shells:
        Shell selection (slice or int) applied to the reference field and to
        n(z) arrays for lensing probes so theory is computed only for the
        shells actually displayed.
    """
    import jax
    import jax_cosmo as jc
    import jax_fli as jfli

    ENABLE_X64 = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)

    ref_field_obj = ref_obj.field[0]
    if selected_shells is not None:
        ref_field_obj = ref_field_obj[selected_shells]
        if ref_field_obj.scale_factors.shape == (0,):
            st.error("No valid shells selected for the indexing range")
            st.stop()
    ref_cosmo = ref_obj.cosmology[0]
    nl_fn = jc.power.halofit if nl_fn_name == "halofit" else "linear"

    if probe_type == "density":
        theory_result = jfli.compute_theory_cl_for_density(
            ref_cosmo,
            ref_field_obj,
            ells,
            nonlinear_fn=nl_fn,
            nz_zmax=float(nz_zmax),
        )
    else:
        from jax_fli.data import get_des_y3_nz_shear, get_stage3_nz_shear

        if probe_type == "point sources":
            z_src = list(z_sources)
        elif probe_type == "s3":
            z_src = get_stage3_nz_shear()
            if selected_shells is not None:
                z_src = z_src[selected_shells]
        elif probe_type == "desY3":
            z_src = get_des_y3_nz_shear()
            if selected_shells is not None:
                z_src = z_src[selected_shells]
        else:
            raise ValueError(f"Unsupported probe type: {probe_type}")

        theory_result = jfli.compute_theory_cl(
            ref_cosmo,
            ells,
            z_src,
            probe_type="weak_lensing",
            nonlinear_fn=nl_fn,
        )

    if apply_pixwin and theory_result is not None:
        theory_result = _apply_pixwin(theory_result, ref_field_obj, lmin, lmax)

    jax.config.update("jax_enable_x64", ENABLE_X64)
    return theory_result
