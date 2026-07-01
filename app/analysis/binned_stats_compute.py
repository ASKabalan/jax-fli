"""Binned summary-statistic (Peak counts / PDF) figure builders and compute.

These statistics return ``BinnedStatistic`` objects (``PDF`` / ``PeakCounts``) whose
``.plot(ax=, logx=, logy=, label=, color=)`` and arithmetic (``a / b``) behave exactly
like the Cl ``PowerSpectrum`` objects, so the grid / legend / ratio helpers from
``spherical_analysis_compute`` are reused verbatim.  Unlike Cl there is no theory curve;
ratios are computed against the reference entry only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from .spherical_analysis_compute import (
    _attach_legend,
    _cl_slice,
    _make_inner_gs,
    _n_shells,
    _setup_spectra_grid,
)
from .utils import _PALETTE, _clean_ratio_ax, _make_title, indexed_field

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------


def compute_binned(
    active_entries: list[dict],
    stat_kind: str,
    *,
    bins: int,
    rng: tuple[float, float] | None,
    normalize: bool = False,
    density: bool = True,
) -> list[tuple[str, Any]]:
    """Compute Peak counts or PDF for every active entry over a *shared* bin grid.

    ``rng`` is the shared ``range`` passed to every entry so all results share identical
    bin edges — required for a meaningful overlay and ratio (see the module that builds it).
    Each entry's field is taken via :func:`indexed_field` (the global per-file index).
    """
    results: list[tuple[str, Any]] = []
    for entry in active_entries:
        fld = indexed_field(entry)
        if stat_kind == "peak_counts":
            stat = fld.compute_peak_counts(bins=int(bins), range=rng, normalize=normalize)
        elif stat_kind == "pdf":
            stat = fld.compute_pdf(bins=int(bins), range=rng, density=density)
        else:
            raise ValueError(f"Unknown binned stat kind: {stat_kind}")
        results.append((entry["label"], stat))
    return results


# ---------------------------------------------------------------------------
# Figure builders — one per (compare_ref, ratio_only)
# ---------------------------------------------------------------------------
# Uniform signature:
#   results        – list of (label, BinnedStatistic), batched over shells
#   layout_params  – {"spec_fig_w", "spec_main_h", "spec_ratio_h", "spec_ncols"}
#   title_template – str (%r%/%z%/%a% placeholders)
#   bands          – list[float] fractional shading bands
#   axis_labels    – {"xlabel", "ylabel", "ratio_ylabel"}
#   logx, logy     – bool, applied to the main panel (ratio panel stays linear-y)


def _ratio_ylim(results) -> tuple[float, float] | None:
    """Robust shared y-limits for ratio panels.

    Binned ratios (peak/PDF) routinely span 0.5–2× and blow up to ``inf`` in empty-bin
    tails, so a fixed Cl-style window clips while pure autoscale is swamped by the
    divide-by-near-zero spikes. Only the *well-sampled* bins (reference above 1% of its
    peak) contribute to the 2nd–98th percentile window (padded), which always keeps the
    0.8–1.2 neighbourhood visible.
    """
    ns = _n_shells(results)
    ref = results[0][1]
    chunks = []
    for i in range(ns):
        ref_arr = np.asarray(_cl_slice(ref, i, ns).array)
        sampled = np.abs(ref_arr) > 1e-2 * np.nanmax(np.abs(ref_arr))
        for _, stat in results[1:]:
            r = np.asarray(_cl_slice(stat, i, ns).array) / ref_arr
            r = r[sampled & np.isfinite(r) & (r > 0)]
            if r.size:
                chunks.append(r)
    if not chunks:
        return None
    allv = np.concatenate(chunks)
    lo = min(float(np.percentile(allv, 2)) * 0.9, 0.8)
    hi = max(float(np.percentile(allv, 98)) * 1.1, 1.2)
    return (max(0.0, lo), hi)


def _build_binned_main_only(results, layout_params, title_template, bands, axis_labels, logx, logy) -> Figure:
    """Rectangular stat panels only — no ratio rows."""
    ns = _n_shells(results)
    height_ratios = [float(layout_params["spec_main_h"])]
    fig, outer_gs, coords, coords_set, legend_cell = _setup_spectra_grid(ns, layout_params, height_ratios)

    handles_out, labels_out = [], []
    for i, (row, col) in enumerate(coords):
        is_bottom = (row + 1, col) not in coords_set
        inner_gs = _make_inner_gs(outer_gs, row, col, height_ratios)
        ax = fig.add_subplot(inner_gs[0, 0])

        for ci, (lbl, stat) in enumerate(results):
            color = _PALETTE[ci % len(_PALETTE)]
            s = _cl_slice(stat, i, ns)
            label = f"{lbl} (Ref)" if ci == 0 else lbl
            s.plot(ax=ax, logx=logx, logy=logy, label=label, color=color)

        ax.grid(True, which="both", ls="--", alpha=0.2)
        ax.set_title(_make_title(title_template, results[0][1], i))
        if col == 0:
            ax.set_ylabel(axis_labels["ylabel"])
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        if is_bottom:
            ax.set_xlabel(axis_labels["xlabel"])
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


def _build_binned_with_ref_ratio(results, layout_params, title_template, bands, axis_labels, logx, logy) -> Figure:
    """Rectangular stat panels + ratio vs reference row."""
    ns = _n_shells(results)
    height_ratios = [
        float(layout_params["spec_main_h"]),
        float(layout_params["spec_ratio_h"]),
    ]
    fig, outer_gs, coords, coords_set, legend_cell = _setup_spectra_grid(ns, layout_params, height_ratios)
    ratio_ylim = _ratio_ylim(results)

    handles_out, labels_out = [], []
    for i, (row, col) in enumerate(coords):
        is_bottom = (row + 1, col) not in coords_set
        inner_gs = _make_inner_gs(outer_gs, row, col, height_ratios)

        ax_main = fig.add_subplot(inner_gs[0, 0])
        for ci, (lbl, stat) in enumerate(results):
            color = _PALETTE[ci % len(_PALETTE)]
            s = _cl_slice(stat, i, ns)
            label = f"{lbl} (Ref)" if ci == 0 else lbl
            s.plot(ax=ax_main, logx=logx, logy=logy, label=label, color=color)
        ax_main.grid(True, which="both", ls="--", alpha=0.2)
        ax_main.set_title(_make_title(title_template, results[0][1], i))
        if col == 0:
            ax_main.set_ylabel(axis_labels["ylabel"])
        else:
            ax_main.set_ylabel("")
            ax_main.tick_params(labelleft=False)
        ax_main.tick_params(labelbottom=False)
        ax_main.set_xlabel("")
        if i == 0:
            handles_out, labels_out = ax_main.get_legend_handles_labels()

        ax_r = fig.add_subplot(inner_gs[1, 0], sharex=ax_main)
        _, ref_stat = results[0]
        ref_s = _cl_slice(ref_stat, i, ns)
        for ci, (lbl, stat) in enumerate(results[1:], 1):
            color = _PALETTE[ci % len(_PALETTE)]
            s = _cl_slice(stat, i, ns)
            (s / ref_s).plot(ax=ax_r, logx=logx, color=color, legend=False)
        ylabel = axis_labels["ratio_ylabel"] if col == 0 else ""
        _clean_ratio_ax(ax_r, ylabel, bands, ylim=ratio_ylim)
        if col != 0:
            ax_r.tick_params(labelleft=False)
        if is_bottom:
            ax_r.set_xlabel(axis_labels["xlabel"])
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


def _build_binned_ratio_only_ref(results, layout_params, title_template, bands, axis_labels, logx, logy) -> Figure:
    """Ratio vs reference panels only — no main stat panel."""
    ns = _n_shells(results)
    height_ratios = [float(layout_params["spec_ratio_h"])]
    fig, outer_gs, coords, coords_set, legend_cell = _setup_spectra_grid(ns, layout_params, height_ratios)
    ratio_ylim = _ratio_ylim(results)

    handles_out, labels_out = [], []
    for i, (row, col) in enumerate(coords):
        is_bottom = (row + 1, col) not in coords_set
        inner_gs = _make_inner_gs(outer_gs, row, col, height_ratios)

        ax = fig.add_subplot(inner_gs[0, 0])
        _, ref_stat = results[0]
        ref_s = _cl_slice(ref_stat, i, ns)
        for ci, (lbl, stat) in enumerate(results[1:], 1):
            color = _PALETTE[ci % len(_PALETTE)]
            s = _cl_slice(stat, i, ns)
            (s / ref_s).plot(ax=ax, logx=logx, color=color, legend=False, label=lbl)

        ax.set_title(_make_title(title_template, results[0][1], i))
        ylabel = axis_labels["ratio_ylabel"] if col == 0 else ""
        _clean_ratio_ax(ax, ylabel, bands, ylim=ratio_ylim)
        if col != 0:
            ax.tick_params(labelleft=False)
        if is_bottom:
            ax.set_xlabel(axis_labels["xlabel"])
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


_BINNED_BUILDERS: dict[tuple[bool, bool], Callable[..., Any]] = {
    (False, False): _build_binned_main_only,
    (False, True): _build_binned_main_only,
    (True, False): _build_binned_with_ref_ratio,
    (True, True): _build_binned_ratio_only_ref,
}
