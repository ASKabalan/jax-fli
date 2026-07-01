"""Peak counts / PDF tab — one generic ``binned_tab`` for both statistics.

Mirrors ``spherical_analysis_form.cl_tab`` (reference selector, plot layout, shading
bands, compare-vs-reference ratio, ratio-only) but without theory.  Spherical maps only.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .binned_stats_compute import (
    _BINNED_BUILDERS,
    _build_binned_main_only,
    compute_binned,
)
from .utils import (
    _SPHERICAL_TYPES,
    _apply_shared_log_ylim,
    _fig_to_png,
    _plt_lock,
    indexed_field,
)

# Per-statistic display configuration.
_STAT_CFG = {
    "peak_counts": {
        "label": "peak counts",
        "ylabel": "Peak counts",
        "default_logy": True,
    },
    "pdf": {
        "label": "PDF",
        "ylabel": "PDF",
        "default_logy": True,
    },
}


def _shared_range(
    active_entries: list[dict],
    normalize: bool,
    auto: bool,
    user_lo: float,
    user_hi: float,
) -> tuple[float, float]:
    """Shared bin ``range`` passed to *every* entry so bin edges match (valid ratio).

    Manual range wins when valid; otherwise normalized peaks use the fixed (-2, 6) S/N
    span and everything else uses the global ``[min, max]`` over all entries' map values.
    """
    if not auto:
        lo, hi = float(user_lo), float(user_hi)
        if hi > lo:
            return (lo, hi)
    if normalize:
        return (-2.0, 6.0)
    los, his = [], []
    for e in active_entries:
        a = np.asarray(indexed_field(e).array)
        los.append(float(a.min()))
        his.append(float(a.max()))
    return (min(los), max(his))


def binned_tab(
    active_entries: list[dict],
    ref_field_type: str,
    stat_kind: str,
) -> None:
    """Render the Peak counts (``stat_kind="peak_counts"``) or PDF (``"pdf"``) tab."""
    cfg = _STAT_CFG[stat_kind]
    if ref_field_type not in _SPHERICAL_TYPES:
        st.error(
            f"{cfg['label'].capitalize()} is only supported for spherical maps "
            f"(SphericalDensity / SphericalKappaField). Current field type: **{ref_field_type}**."
        )
        return

    kp = f"analysis_{stat_kind}"
    results_key = f"{kp}_results"
    png_key = f"{kp}_png"
    fig_key = f"{kp}_fig"

    spec_params_col, spec_plot_col = st.columns([1, 3])

    with spec_params_col:
        with st.container(border=True):
            st.markdown("**Parameters**")

            # --- Reference entry selector (reorder so ref is index 0) ---
            _labels = [e["label"] for e in active_entries]
            _ref_label = st.selectbox("Reference entry", _labels, index=0, key=f"{kp}_ref_entry")
            _ref_idx = _labels.index(_ref_label)
            active_entries = [active_entries[_ref_idx]] + [e for i, e in enumerate(active_entries) if i != _ref_idx]

            # --- Binning ---
            st.markdown("**Binning**")
            bins = st.number_input("Bins", min_value=4, max_value=500, value=50, key=f"{kp}_bins")
            if stat_kind == "peak_counts":
                normalize = st.checkbox(
                    "Normalize (S/N)",
                    value=True,
                    key=f"{kp}_normalize",
                    help="Bin in (m - mean) / std units; fixed (-2, 6) σ span by default.",
                )
                density = True
            else:
                density = st.checkbox("Density (unit integral)", value=True, key=f"{kp}_density")
                normalize = False

            auto_range = st.checkbox(
                "Auto range",
                value=True,
                key=f"{kp}_auto_range",
                help="Shared range across all entries: global [min, max] of the data "
                "(or fixed (-2, 6) for normalized peaks). Uncheck to set it manually.",
            )
            rc1, rc2 = st.columns(2)
            with rc1:
                user_lo = st.number_input(
                    "Range min",
                    value=-2.0 if normalize else 0.0,
                    format="%.4f",
                    key=f"{kp}_rmin",
                    disabled=auto_range,
                )
            with rc2:
                user_hi = st.number_input(
                    "Range max",
                    value=6.0 if normalize else 1.0,
                    format="%.4f",
                    key=f"{kp}_rmax",
                    disabled=auto_range,
                )

            lc1, lc2 = st.columns(2)
            with lc1:
                logx = st.checkbox("Log x", value=False, key=f"{kp}_logx")
            with lc2:
                logy = st.checkbox("Log y", value=cfg["default_logy"], key=f"{kp}_logy")

            # --- Plot layout ---
            st.markdown("**Plot layout**")
            spec_ncols = st.number_input("Columns", min_value=1, max_value=10, value=2, key=f"{kp}_ncols")
            spec_fig_w = st.number_input(
                "Width/col",
                min_value=2.0,
                max_value=16.0,
                value=5.0,
                step=0.5,
                key=f"{kp}_fig_w",
            )
            spec_main_h = st.number_input(
                "Main height",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.5,
                key=f"{kp}_main_h",
            )
            spec_ratio_h = st.number_input(
                "Ratio height",
                min_value=0.5,
                max_value=5.0,
                value=1.0,
                step=0.25,
                key=f"{kp}_ratio_h",
            )
            spec_dpi = st.number_input(
                "Render DPI",
                min_value=50,
                max_value=2000,
                value=100,
                step=25,
                key=f"{kp}_dpi",
            )
            title_template = st.text_input(
                "Panel title template",
                value="χ %r% Mpc/h",
                key=f"{kp}_title_template",
                help="%r% = comoving distance  |  %z% = redshift  |  %a% = scale factor",
            )

            # --- Shading bands ---
            st.markdown("**Shading bands** (set to 0 to disable)")
            bc1, bc2 = st.columns(2)
            with bc1:
                band_a = st.number_input(
                    "Band 1 %",
                    min_value=0.0,
                    max_value=100.0,
                    value=10.0,
                    step=1.0,
                    key=f"{kp}_band_a",
                )
            with bc2:
                band_b = st.number_input(
                    "Band 2 %",
                    min_value=0.0,
                    max_value=100.0,
                    value=20.0,
                    step=1.0,
                    key=f"{kp}_band_b",
                )
            bands = [v / 100 for v in [band_a, band_b] if v > 0]

            # --- Comparison ---
            compare_ref = st.checkbox(
                "Compare against reference",
                value=False,
                key=f"{kp}_compare_ref",
                disabled=(len(active_entries) < 2),
            )
            ratio_only = st.checkbox(
                "Ratio only (hide main panel)",
                value=False,
                key=f"{kp}_ratio_only",
                disabled=not compare_ref,
            )

            # --- Buttons ---
            _has = bool(st.session_state.get(results_key))
            cb1, cb2 = st.columns(2)
            with cb1:
                compute_btn = st.button("Compute", key=f"{kp}_compute_btn", type="primary")
            with cb2:
                redraw_btn = st.button(
                    "Redraw",
                    key=f"{kp}_redraw_btn",
                    disabled=not _has,
                    help="Re-render from cached values without recomputing",
                )

    with spec_plot_col:
        if compute_btn:
            rng = _shared_range(active_entries, normalize, auto_range, user_lo, user_hi)
            try:
                with st.spinner(f"Computing {cfg['label']}..."):
                    results = compute_binned(
                        active_entries,
                        stat_kind,
                        bins=int(bins),
                        rng=rng,
                        normalize=normalize,
                        density=density,
                    )
                st.session_state[results_key] = results
                st.caption(f"Shared bin range: [{rng[0]:.4g}, {rng[1]:.4g}]")
            except Exception as e:
                st.error(f"{cfg['label'].capitalize()} computation failed: {e}")

        results = st.session_state.get(results_key)
        if (compute_btn or redraw_btn) and results:
            layout_params = {
                "spec_fig_w": spec_fig_w,
                "spec_main_h": spec_main_h,
                "spec_ratio_h": spec_ratio_h,
                "spec_ncols": spec_ncols,
            }
            xlabel = "S/N" if (stat_kind == "peak_counts" and normalize) else "pixel value"
            axis_labels = {
                "xlabel": xlabel,
                "ylabel": cfg["ylabel"],
                "ratio_ylabel": "Ratio\n(vs Ref)",
            }
            eff_compare_ref = compare_ref and len(results) > 1
            eff_ratio_only = ratio_only and eff_compare_ref
            builder = _BINNED_BUILDERS.get((eff_compare_ref, eff_ratio_only), _build_binned_main_only)
            with st.spinner("Rendering..."):
                with _plt_lock:
                    fig = builder(
                        results,
                        layout_params,
                        title_template,
                        bands,
                        axis_labels,
                        logx,
                        logy,
                    )
                    _apply_shared_log_ylim(fig)
                    old_fig = st.session_state.pop(fig_key, None)
                    if old_fig is not None:
                        plt.close(old_fig)
                    st.session_state[png_key] = _fig_to_png(fig, dpi=int(spec_dpi))
                    st.session_state[fig_key] = fig

        png = st.session_state.get(png_key)
        fig = st.session_state.get(fig_key)
        if png:
            st.image(png)
            if fig is not None:
                from .save_figure import render_save_figure

                render_save_figure(fig, key_prefix=f"{kp}_save", filename=stat_kind)
        else:
            st.info(f"Click **Compute** to generate the {cfg['label']}.")
