"""DensityField and ParticleField pure compute: P(k) builders and field map renderers.

No streamlit imports — all functions return plain data (arrays, figures, bytes).
Figure builders follow a strict "one function per plot mode" rule — no nested
ifs for routing. Three modes: main only, with theory ratio, ratio only.
"""

from __future__ import annotations

from collections.abc import Callable
from math import ceil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .utils import (
    _COLOR_THEORY,
    _PALETTE,
    _clean_ratio_ax,
    _fig_to_png,
    _make_title,
    _plt_lock,
    indexed_field,
)

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _pk_ratio_ylim(ax, bands: list[float]) -> None:
    """Set ylim for a P(k) ratio panel — wider than Cl because scatter is larger."""
    margin = max(bands) * 1.5 if bands else 0.5
    ax.set_ylim(1.0 - margin, 1.0 + margin)


def _snap_title(ref_fld, snap_idx: int) -> str:
    """Return '$a=X.XX$  ($z=Y.YY$)' title string for a given snapshot index."""
    import jax_cosmo as jc

    scale_factors = None
    if ref_fld.scale_factors is not None:
        scale_factors = np.asarray(ref_fld.scale_factors)
    if scale_factors is not None:
        a = float(scale_factors[snap_idx]) if scale_factors.ndim > 0 else float(scale_factors)
    else:
        a = 1.0
    z = float(jc.utils.a2z(a))
    return f"$a={a:.2f}$  ($z={z:.2f}$)"


def _pk_arr_for_snap(pk_sim, snap_idx: int):
    arr = np.asarray(pk_sim.array)
    return arr[snap_idx] if arr.ndim > 1 else arr


# ---------------------------------------------------------------------------
# Three P(k) figure builders — one per plot mode
# ---------------------------------------------------------------------------
# Uniform signature:
#   pk_results    – list of (label, PK), already sliced to chosen snapshots
#   theory_pks    – list of arrays (one per snap) or None
#   ref_fld       – reference field object, already sliced (for scale factors)
#   layout_params – {"fig_w", "main_h", "ratio_h"}
#   bands         – list[float] fractional shading bands
#
# Iterate sequentially (col = 0, 1, …) — data is pre-sliced so sequential
# indices align with ref_fld metadata.


def _build_pk_main_only(
    pk_results,
    theory_pks,
    ref_fld,
    title_template,
    layout_params,
    bands,
) -> Figure:
    """P(k) top panels only for each selected snapshot — no ratio row."""
    ref_arr = np.asarray(pk_results[0][1].array)
    n = ref_arr.shape[0] if ref_arr.ndim > 1 else 1
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(float(layout_params["fig_w"]) * n, float(layout_params["main_h"])),
        squeeze=False,
    )

    for col in range(n):
        ax = axes[0, col]
        k = np.asarray(pk_results[0][1].wavenumber)

        for ci, (lbl, pk_sim) in enumerate(pk_results):
            color = _PALETTE[ci % len(_PALETTE)]
            pk_arr = _pk_arr_for_snap(pk_sim, col)
            ax.loglog(k, pk_arr, color=color, linewidth=2, label=lbl)

        ax.set_title(_snap_title(ref_fld, col))
        if col == 0:
            ax.set_ylabel(r"$P(k)$ [$h^{-3}\,\mathrm{Mpc}^3$]")
        ax.set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")
        ax.legend(fontsize="small")
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("3D Matter Power Spectrum", fontsize=14)
    fig.tight_layout()
    return fig


def _build_pk_with_theory_ratio(
    pk_results,
    theory_pks,
    ref_fld,
    title_template,
    layout_params,
    bands,
) -> Figure:
    """P(k) top panels + sim/theory ratio row for each selected snapshot."""
    ref_arr = np.asarray(pk_results[0][1].array)
    n = ref_arr.shape[0] if ref_arr.ndim > 1 else 1
    fig, axes = plt.subplots(
        2,
        n,
        figsize=(
            float(layout_params["fig_w"]) * n,
            float(layout_params["main_h"]) + float(layout_params["ratio_h"]),
        ),
        sharex="col",
        gridspec_kw={
            "height_ratios": [
                float(layout_params["main_h"]),
                float(layout_params["ratio_h"]),
            ],
            "hspace": 0.05,
        },
        squeeze=False,
    )

    for col in range(n):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        k = np.asarray(pk_results[0][1].wavenumber)

        # Theory
        th = np.asarray(theory_pks[col]) if theory_pks else None
        if th is not None:
            ax_top.loglog(
                k,
                th,
                color=_COLOR_THEORY,
                linestyle="--",
                linewidth=2,
                label="Theory (Halofit)",
            )

        for ci, (lbl, pk_sim) in enumerate(pk_results):
            color = _PALETTE[ci % len(_PALETTE)]
            pk_arr = _pk_arr_for_snap(pk_sim, col)
            ax_top.loglog(k, pk_arr, color=color, linewidth=2, label=lbl)

        ax_top.set_title(_snap_title(ref_fld, col))
        if col == 0:
            ax_top.set_ylabel(r"$P(k)$ [$h^{-3}\,\mathrm{Mpc}^3$]")
        ax_top.legend(fontsize="small")
        ax_top.grid(True, which="both", alpha=0.3)
        ax_top.tick_params(labelbottom=False)

        # Ratio
        for ci, (lbl, pk_sim) in enumerate(pk_results):
            if th is None:
                break
            color = _PALETTE[ci % len(_PALETTE)]
            pk_arr = _pk_arr_for_snap(pk_sim, col)
            ax_bot.semilogx(k, pk_arr / th, color=color, linewidth=2)

        ylabel = "Sim / Theory" if col == 0 else ""
        _clean_ratio_ax(ax_bot, ylabel, bands)
        _pk_ratio_ylim(ax_bot, bands)
        if col != 0:
            ax_bot.tick_params(labelleft=False)
        ax_bot.set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")

        if col == n - 1 and bands:
            ax_bot.legend(loc="lower left", fontsize="x-small")

    fig.suptitle("3D Matter Power Spectrum vs Theory", fontsize=14)
    fig.tight_layout()
    return fig


def _build_pk_ratio_only_theory(
    pk_results,
    theory_pks,
    ref_fld,
    title_template,
    layout_params,
    bands,
) -> Figure:
    """Sim/theory ratio panels only — no main P(k) panel."""
    ref_arr = np.asarray(pk_results[0][1].array)
    n = ref_arr.shape[0] if ref_arr.ndim > 1 else 1
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(float(layout_params["fig_w"]) * n, float(layout_params["ratio_h"])),
        squeeze=False,
    )

    for col in range(n):
        ax = axes[0, col]
        k = np.asarray(pk_results[0][1].wavenumber)
        th = np.asarray(theory_pks[col]) if theory_pks else None

        for ci, (lbl, pk_sim) in enumerate(pk_results):
            if th is None:
                break
            color = _PALETTE[ci % len(_PALETTE)]
            pk_arr = _pk_arr_for_snap(pk_sim, col)
            ax.semilogx(k, pk_arr / th, color=color, linewidth=2, label=lbl)

        ax.set_title(_snap_title(ref_fld, col))
        ylabel = "Sim / Theory" if col == 0 else ""
        _clean_ratio_ax(ax, ylabel, bands)
        _pk_ratio_ylim(ax, bands)
        if col != 0:
            ax.tick_params(labelleft=False)
        ax.set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")

    fig.suptitle("P(k) / Theory", fontsize=14)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

PK_BUILDERS: dict[tuple[bool, bool], Callable[..., Any]] = {
    (False, False): _build_pk_main_only,
    (True, False): _build_pk_with_theory_ratio,
    (True, True): _build_pk_ratio_only_theory,
}


# ---------------------------------------------------------------------------
# Compute functions
# ---------------------------------------------------------------------------


def compute_pk(
    active_entries: list[dict],
) -> tuple[list[tuple[str, object]], object, object]:
    """Return one 3D P(k) ``PowerSpectrum`` per entry, mixing precomputed and raw.

    Per entry: a precomputed ``PowerSpectrum`` (3D P(k)) is used as-is; a raw
    ``DensityField`` is transformed via ``.power()``.

    Returns
    -------
    (pk_results, ref_fld, ref_cosmo)  — ref_* taken from the first (reference) entry.
    """
    pk_results = []
    for entry in active_entries:
        fld = indexed_field(entry)
        pk_sim = fld if entry["field_type"] == "PowerSpectrum" else fld.power()
        pk_results.append((entry["label"], pk_sim))
    ref_fld = indexed_field(active_entries[0])
    ref_cosmo = active_entries[0]["catalog"].cosmology[0]
    return pk_results, ref_fld, ref_cosmo


def compute_theory_pk(
    ref_fld,
    ref_cosmo,
    pk_results: list[tuple[str, Any]],
    nl_fn_name: str,
) -> list | None:
    """Compute theory P(k) for each snapshot present in the reference field.

    Returns a list of arrays, one per snapshot, aligned with the wavenumber
    grid of ``pk_results[0][1].wavenumber``.
    """
    import jax
    import jax_cosmo as jc

    ref_pk = pk_results[0][1]
    ref_pk_arr = np.asarray(ref_pk.array)
    n_snaps = ref_pk_arr.shape[0] if ref_pk_arr.ndim > 1 else 1

    sfa = None
    if ref_fld.scale_factors is not None:
        sfa = np.asarray(ref_fld.scale_factors)

    nl_fn = jc.power.halofit if nl_fn_name == "halofit" else jc.power.linear

    theory_pks = []
    for i in range(n_snaps):
        a_i = float(sfa[i]) if (sfa is not None and sfa.ndim > 0) else (float(sfa) if sfa is not None else 1.0)
        th = jax.jit(
            jc.power.nonlinear_matter_power,
            static_argnames=["nonlinear_fn"],
        )(ref_cosmo, ref_pk.wavenumber, a=a_i, nonlinear_fn=nl_fn)
        theory_pks.append(th)
    return theory_pks


# ---------------------------------------------------------------------------
# Field map renderers
# ---------------------------------------------------------------------------
# All renderers return (png_bytes, fig) on success or (None, None) on failure.
# The caller is responsible for closing the figure (e.g. after storing it in
# session state for deferred PDF export via render_save_figure).


def render_density_field_map(
    selected_entry: dict,
    plot_field,
    map_params: dict,
    d_params: dict,
) -> tuple[bytes, object] | tuple[None, None]:
    """Render a 3D DensityField map; return (png_bytes, fig).

    Parameters
    ----------
    d_params:
        Dict with keys: ``elev``, ``azim``, ``zoom``, ``edges``, ``levels``,
        ``project_slices``, ``crop`` (3-tuple of slices), ``do_project``,
        ``nz_slices``.
    """
    n_plots = plot_field.shape[0] if plot_field.is_batched() else 1
    ncols = int(map_params["ncols"])
    nrows = max(1, ceil(n_plots / ncols))

    titles = [
        _make_title(map_params["title_template"], plot_field, i, label=selected_entry["label"]) for i in range(n_plots)
    ]

    fig = None
    with _plt_lock:
        try:
            if d_params["do_project"]:
                projected = plot_field.project(nz_slices=int(d_params["nz_slices"]))
                fig, _ = projected.plot(
                    ncols=ncols,
                    figsize=(
                        float(map_params["fig_w"]) * ncols,
                        float(map_params["fig_h"]) * nrows,
                    ),
                    cmap=map_params["cmap"],
                    colorbar=map_params["colorbar"],
                    vmin=map_params["vmin"],
                    vmax=map_params["vmax"],
                    titles=titles,
                )
            else:
                fig, _ = plot_field.plot(
                    ncols=ncols,
                    figsize=(
                        float(map_params["fig_w"]) * ncols,
                        float(map_params["fig_h"]) * nrows,
                    ),
                    cmap=map_params["cmap"],
                    colorbar=map_params["colorbar"],
                    vmin=map_params["vmin"],
                    vmax=map_params["vmax"],
                    titles=titles,
                    elev=float(d_params["elev"]),
                    azim=float(d_params["azim"]),
                    zoom=float(d_params["zoom"]),
                    edges=bool(d_params["edges"]),
                    levels=int(d_params["levels"]),
                    project_slices=int(d_params["project_slices"]),
                    crop=d_params["crop"],
                )
            png = _fig_to_png(fig, dpi=int(map_params["dpi"]))
            return png, fig
        except Exception as e:
            print(f"[density_analysis] DensityField map rendering failed: {e}")
            if fig is not None:
                plt.close(fig)
            return None, None


def render_particle_field_map(
    selected_entry: dict,
    plot_field,
    map_params: dict,
    p_params: dict,
) -> tuple[bytes, object] | tuple[None, None]:
    """Render a ParticleField scatter plot; return (png_bytes, fig).

    Parameters
    ----------
    p_params:
        Dict with keys: ``thinning``, ``point_size``, ``alpha``, ``elev``,
        ``azim``, ``zoom``, ``weights``, ``weights_title``.
    """
    n_plots = plot_field.shape[0] if plot_field.is_batched() else 1
    ncols = int(map_params["ncols"])
    nrows = max(1, ceil(n_plots / ncols))

    titles = [
        _make_title(map_params["title_template"], plot_field, i, label=selected_entry["label"]) for i in range(n_plots)
    ]

    fig = None
    with _plt_lock:
        try:
            fig, _ = plot_field.plot(
                ncols=ncols,
                figsize=(
                    float(map_params["fig_w"]) * ncols,
                    float(map_params["fig_h"]) * nrows,
                ),
                cmap=map_params["cmap"],
                colorbar=map_params["colorbar"],
                vmin=map_params["vmin"],
                vmax=map_params["vmax"],
                titles=titles,
                weights=p_params["weights"],
                weights_title=p_params["weights_title"],
                thinning=int(p_params["thinning"]),
                point_size=float(p_params["point_size"]),
                alpha=float(p_params["alpha"]),
                elev=float(p_params["elev"]),
                azim=float(p_params["azim"]),
                zoom=float(p_params["zoom"]),
            )
            png = _fig_to_png(fig, dpi=int(map_params["dpi"]))
            return png, fig
        except Exception as e:
            print(f"[density_analysis] ParticleField map rendering failed: {e}")
            if fig is not None:
                plt.close(fig)
            return None, None
