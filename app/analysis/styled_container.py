"""CSS injection for styled form containers."""
from __future__ import annotations

import streamlit as st


def inject_custom_css() -> None:
    """Inject custom CSS for bordered containers. Call once per page after set_page_config."""
    st.markdown(
        """
        <style>
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 12px !important;
            background-color: #e8f0fe !important;
            border: 1px solid #c5d9f0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
