"""DensityField and ParticleField Streamlit UI: 3D P(k) tab."""
from __future__ import annotations

import os

import numpy as np
import streamlit as st

from .density_analysis_compute import (
    PK_BUILDERS,
    compute_pk,
    compute_theory_pk,
)
from .utils import (
    _apply_shared_log_ylim,
    _fig_to_png,
    _plt_lock,
)


def pk_tab(active_entries: list[dict]) -> None:
    """Render the full 3D P(k) tab.

    Entries may be mixed: a precomputed ``PowerSpectrum`` (3D P(k)) is used as-is and a
    raw ``DensityField`` is transformed via ``.power()`` (handled in ``compute_pk``).
    The caller (form.py) routes only P(k)-category entries here.
    """
    spec_params_pk, spec_plot_pk = st.columns([1, 3])

    with spec_params_pk:
        with st.container(border=True):
            st.markdown("**Parameters**")

            _labels = [e["label"] for e in active_entries]
            _ref_label = st.selectbox(
                "Reference entry",
                _labels,
                index=0,
                key="analysis_pk_ref_entry",
            )
            _ref_idx = _labels.index(_ref_label)
            active_entries = [active_entries[_ref_idx]] + [
                e for i, e in enumerate(active_entries) if i != _ref_idx
            ]

            pk_nl_fn = st.selectbox(
                "Nonlinear fn", ["halofit", "linear"], key="analysis_pk_nl_fn"
            )

            compare_theory_pk = st.checkbox(
                "Compare against theory", value=False, key="analysis_pk_compare_theory"
            )
            ratio_only_pk = st.checkbox(
                "Ratio only (hide main panel)",
                value=False,
                key="analysis_ratio_only_pk",
                disabled=not compare_theory_pk,
                help="Show only the ratio panel without the main P(k) panel.",
            )

            pk_fig_w = st.number_input(
                "Width/snapshot",
                min_value=2.0,
                max_value=16.0,
                value=5.0,
                step=0.5,
                key="analysis_pk_fig_w",
            )
            pk_main_h = st.number_input(
                "Main panel height",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.5,
                key="analysis_pk_main_h",
            )
            pk_ratio_h = st.number_input(
                "Ratio panel height",
                min_value=0.5,
                max_value=5.0,
                value=1.0,
                step=0.25,
                key="analysis_pk_ratio_h",
            )
            pk_dpi = st.number_input(
                "Render DPI",
                min_value=50,
                max_value=2000,
                value=100,
                step=25,
                key="analysis_pk_dpi",
            )
            title_template = st.text_input(
                "Panel title template",
                value="χ %r% Mpc/h",
                key="analysis_pk_title_template",
                help="%r% = comoving distance  |  %z% = redshift  |  %a% = scale factor",
            )

            st.markdown("**Shading bands** (set to 0 to disable)")
            pb1, pb2 = st.columns(2)
            with pb1:
                pk_band_10 = st.number_input(
                    "±10% band",
                    min_value=0.0,
                    max_value=100.0,
                    value=10.0,
                    step=1.0,
                    key="analysis_pk_band_10",
                )
            with pb2:
                pk_band_20 = st.number_input(
                    "±20% band",
                    min_value=0.0,
                    max_value=100.0,
                    value=20.0,
                    step=1.0,
                    key="analysis_pk_band_20",
                )
            pk_bands = [v / 100 for v in [pk_band_10, pk_band_20] if v > 0]

            has_pk = bool(st.session_state.get("analysis_pk_results"))
            pk_cb1, pk_cb2 = st.columns(2)
            with pk_cb1:
                pk_compute_btn = st.button(
                    "Compute",
                    key="analysis_pk_compute_btn",
                    type="primary",
                )
            with pk_cb2:
                pk_redraw_btn = st.button(
                    "Redraw",
                    key="analysis_pk_redraw_btn",
                    disabled=not has_pk,
                    help="Re-render from cached P(k) values without recomputing",
                )

    with spec_plot_pk:
        if pk_compute_btn:
            try:
                with st.spinner("Computing 3D power spectra..."):
                    pk_results, ref_fld_pk, ref_cosmo_pk = compute_pk(active_entries)

                if pk_results:
                    # HARD: same number of snapshots and same k-grid length.
                    if len({s[1].spectra.shape[0] for s in pk_results}) > 1:  # type: ignore
                        st.error(
                            "Power spectra have different number of snapshots — cannot compare."
                        )
                        st.stop()
                    if len({s[1].wavenumber.size for s in pk_results}) > 1:  # pyright: ignore[reportGeneralTypeIssues]
                        st.error(
                            "Power spectra have different k-grid lengths — cannot compare. "
                            "This happens when mixing spectra computed on different k binnings."
                        )
                        st.stop()
                    # SOFT: warn on differing scale factors (guarded against None).
                    _sf = [getattr(s[1], "scale_factors", None) for s in pk_results]
                    if all(v is not None for v in _sf):
                        _rtol = float(os.getenv("JAX_FLI_COMPARE_RTOL", "1e-1"))
                        _atol = float(os.getenv("JAX_FLI_COMPARE_ATOL", "1e-1"))
                        _arr = np.array([np.asarray(v) for v in _sf])
                        if not np.all(
                            np.isclose(_arr, _arr[0], rtol=_rtol, atol=_atol)
                        ):
                            st.warning("Power spectra have different scale factors.")
                    theory_pks = None
                    if compare_theory_pk:
                        with st.spinner("Computing theory P(k)..."):
                            theory_pks = compute_theory_pk(
                                ref_fld_pk, ref_cosmo_pk, pk_results, pk_nl_fn
                            )
                    st.session_state["analysis_pk_results"] = (
                        pk_results,
                        theory_pks,
                        ref_fld_pk,
                    )
            except Exception as e:
                print(f"[density_analysis] DensityField pk compute failed: {e}")

        pk_cached = st.session_state.get("analysis_pk_results")
        if (pk_compute_btn or pk_redraw_btn) and pk_cached:
            _pk_results, _theory_pks, _ref_fld_pk = pk_cached

            layout_params = {
                "fig_w": pk_fig_w,
                "main_h": pk_main_h,
                "ratio_h": pk_ratio_h,
            }
            eff_theory = compare_theory_pk and _theory_pks is not None
            eff_ratio_only = ratio_only_pk and eff_theory
            builder = PK_BUILDERS.get(
                (eff_theory, eff_ratio_only), PK_BUILDERS[(False, False)]
            )

            with st.spinner("Rendering..."):
                with _plt_lock:
                    fig_pk = builder(
                        _pk_results,
                        _theory_pks,
                        _ref_fld_pk,
                        title_template,
                        layout_params,
                        pk_bands,
                    )
                    _apply_shared_log_ylim(fig_pk)
                    st.session_state["analysis_pk_png"] = _fig_to_png(
                        fig_pk, dpi=int(pk_dpi)
                    )
                    st.session_state["analysis_pk_fig"] = fig_pk

        pk_png = st.session_state.get("analysis_pk_png")
        pk_fig = st.session_state.get("analysis_pk_fig")
        if pk_png:
            st.image(pk_png)
            if pk_fig is not None:
                from .save_figure import render_save_figure

                render_save_figure(
                    pk_fig, key_prefix="pk", filename="power_spectrum_3d"
                )
        else:
            st.info("Click **Compute** to generate the 3D matter power spectrum.")
