"""Reusable save-figure dialog: format, DPI, transparency, download."""
from __future__ import annotations

import io

import streamlit as st


def render_save_figure(fig, key_prefix: str = "save", filename: str = "figure") -> None:
    """Render save-figure controls and download button.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    key_prefix : str
        Unique prefix for widget keys (avoids collisions when used multiple times).
    filename : str
        Base filename (without extension) for the download.
    """
    c_fmt, c_dpi, c_transp, c_btn = st.columns([1, 1, 1, 1])
    with c_fmt:
        fmt = st.selectbox("Format", ["png", "pdf"], key=f"{key_prefix}_fmt")
    with c_dpi:
        dpi = st.number_input(
            "DPI",
            min_value=50,
            max_value=600,
            value=150,
            step=50,
            key=f"{key_prefix}_dpi",
        )
    with c_transp:
        transparent = st.checkbox(
            "Transparent", value=False, key=f"{key_prefix}_transp"
        )
    with c_btn:
        buf = io.BytesIO()
        fig.savefig(
            buf, format=fmt, dpi=dpi, transparent=transparent, bbox_inches="tight"
        )
        buf.seek(0)
        st.download_button(
            label=f"Download .{fmt}",
            data=buf,
            file_name=f"{filename}.{fmt}",
            mime=f"image/{fmt}" if fmt == "png" else "application/pdf",
            key=f"{key_prefix}_dl",
        )
