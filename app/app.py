"""jax-fli Analysis app — Streamlit entry point (local dev + future HF Space)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import streamlit as st
from analysis import run
from analysis.styled_container import inject_custom_css

st.set_page_config(page_title="jax-fli Analysis", layout="wide")
inject_custom_css()
st.title("Analysis")
st.markdown(
    "Dashboard for analyzing `jax-fli` simulation & inference outputs.\n\n"
    "- **Run your own simulation** with `fli-simulate` or `fli-launcher` — see the "
    "[Scripts & utilities docs](https://github.com/DifferentiableUniverseInitiative/jax-fli/tree/main/docs/4-scripts-and-utilities).\n"
    "- **Or pick from experiments already run** — load a catalog from the "
    "[`ASKabalan/jax-fli-experiments`](https://huggingface.co/datasets/ASKabalan/jax-fli-experiments) "
    "HuggingFace dataset below to compare and analyze."
    "The experiments mirror the ones explain in the docs [ADD LINK]"
)

run()
