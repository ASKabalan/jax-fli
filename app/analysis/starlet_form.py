"""Starlet (spherical wavelet) tab — per-scale HEALPix maps + RdBu difference column.

The starlet transform operates on a single map, so a shell picker selects one shell from
the globally-indexed field.  Layout: rows = wavelet scales, columns = entries; when
"compare against reference" is on, an extra ``(entry − ref)`` diff column per scale is
drawn with ``cmap="RdBu"`` and a per-scale symmetric colour scale.

Requires the optional CosmoStat backend (``pip install jax-fli[starlet]``); when absent,
``starlet_coefficients`` raises a loud ImportError that is surfaced to the user.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .utils import _SPHERICAL_TYPES, _fig_to_png, _plt_lock, indexed_field

_CMAPS = [
    "magma",
    "viridis",
    "inferno",
    "plasma",
    "cividis",
    "coolwarm",
    "RdBu_r",
    "hot",
    "bone",
    "gray",
]


def _single_map(entry: dict, shell: int):
    """Index one shell of an entry's globally-indexed field, clamped to its shell count."""
    fld = indexed_field(entry)
    if fld.is_batched():
        si = min(int(shell), fld.array.shape[0] - 1)
        return fld[si]
    return fld


def starlet_tab(active_entries: list[dict], ref_field_type: str) -> None:
    """Render the Starlet tab (spherical maps only)."""
    if ref_field_type not in _SPHERICAL_TYPES:
        st.error(
            "Starlet is only supported for spherical maps "
            f"(SphericalDensity / SphericalKappaField). Current field type: **{ref_field_type}**."
        )
        return

    kp = "analysis_starlet"
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

            # --- Shell picker (single map) ---
            _ref_fld = indexed_field(active_entries[0])
            _ns = _ref_fld.array.shape[0] if _ref_fld.is_batched() else 1
            shell = st.number_input(
                "Shell index",
                min_value=0,
                max_value=max(0, _ns - 1),
                value=0,
                key=f"{kp}_shell",
                disabled=(_ns == 1),
                help="Starlet transforms one map; pick which shell of the indexed field.",
            )

            nscales = st.number_input("Scales", min_value=2, max_value=10, value=5, key=f"{kp}_nscales")
            normalize = st.checkbox(
                "Normalize scales",
                value=False,
                key=f"{kp}_normalize",
                help="Divide each scale by its CosmoStat TabNorm.",
            )
            map_cmap = st.selectbox("Map colormap", _CMAPS, key=f"{kp}_cmap")

            st.markdown("**Panel size**")
            pc1, pc2 = st.columns(2)
            with pc1:
                panel_w = st.number_input(
                    "Width/panel",
                    min_value=2.0,
                    max_value=12.0,
                    value=4.0,
                    step=0.5,
                    key=f"{kp}_panel_w",
                )
            with pc2:
                panel_h = st.number_input(
                    "Height/panel",
                    min_value=2.0,
                    max_value=12.0,
                    value=3.0,
                    step=0.5,
                    key=f"{kp}_panel_h",
                )
            spec_dpi = st.number_input(
                "Render DPI",
                min_value=50,
                max_value=2000,
                value=100,
                step=25,
                key=f"{kp}_dpi",
            )

            compare_ref = st.checkbox(
                "Compare against reference (diff column, RdBu)",
                value=False,
                key=f"{kp}_compare_ref",
                disabled=(len(active_entries) < 2),
            )

            _has = bool(st.session_state.get(results_key))
            cb1, cb2 = st.columns(2)
            with cb1:
                compute_btn = st.button("Compute", key=f"{kp}_compute_btn", type="primary")
            with cb2:
                redraw_btn = st.button(
                    "Redraw",
                    key=f"{kp}_redraw_btn",
                    disabled=not _has,
                    help="Re-render from cached coefficients without recomputing",
                )

    with spec_plot_col:
        if compute_btn:
            try:
                with st.spinner("Computing starlet coefficients..."):
                    results = [
                        (
                            e["label"],
                            _single_map(e, shell).starlet_coefficients(nscales=int(nscales), normalize=normalize),
                        )
                        for e in active_entries
                    ]
                st.session_state[results_key] = results
            except ImportError as e:
                st.error(
                    "Starlet requires the CosmoStat backend. Install it with "
                    "`pip install jax-fli[starlet]`.\n\n"
                    f"Original error: {e}"
                )
            except Exception as e:
                st.error(f"Starlet computation failed: {e}")

        results = st.session_state.get(results_key)
        if (compute_btn or redraw_btn) and results:
            n_entries = len(results)
            ns_scales = int(results[0][1].array.shape[0])

            # nside/npix alignment guard before any (entry − ref) subtraction.
            npix_set = {int(sc.array.shape[-1]) for _, sc in results}
            do_diff = compare_ref and n_entries > 1
            if do_diff and len(npix_set) > 1:
                st.error(
                    "Entries have different nside/npix — cannot compute starlet diffs. Showing per-scale maps only."
                )
                do_diff = False

            n_diff = (n_entries - 1) if do_diff else 0
            ncols_total = n_entries + n_diff

            with st.spinner("Rendering..."):
                with _plt_lock:
                    fig, axes = plt.subplots(
                        ns_scales,
                        ncols_total,
                        figsize=(panel_w * ncols_total, panel_h * ns_scales),
                        squeeze=False,
                    )

                    # Entry map columns (per-panel auto colour scale).
                    for col, (label, sc) in enumerate(results):
                        titles = [f"{label}\nscale {s + 1}" for s in range(ns_scales)]
                        sc.plot(
                            ax=axes[:, col],
                            titles=titles,
                            cmap=map_cmap,
                            colorbar=True,
                        )

                    # Diff columns: (entry − ref), per-scale symmetric RdBu.
                    if do_diff:
                        _, ref_sc = results[0]
                        for j, (label, sc) in enumerate(results[1:], start=1):
                            diff = sc - ref_sc
                            diff_arr = np.asarray(diff.array)
                            col = n_entries + (j - 1)
                            for s in range(ns_scales):
                                v = float(np.abs(diff_arr[s]).max())
                                vmin, vmax = (-v, v) if v > 0 else (None, None)
                                diff[s].plot(
                                    ax=axes[s, col],
                                    titles=[f"{label} − ref\nscale {s + 1}"],
                                    cmap="RdBu",
                                    vmin=vmin,
                                    vmax=vmax,
                                    colorbar=True,
                                )

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

                render_save_figure(fig, key_prefix=f"{kp}_save", filename="starlet")
        else:
            st.info("Click **Compute** to generate the starlet coefficients.")
