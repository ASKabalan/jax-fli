# ruff: noqa: E402
"""Experiment 12 — IC-gradient (forward + backward) strong & weak scaling figures.

Reads the ``perf_pm.csv`` produced on the cluster (profiler schema, one row per run) from the
``ASKabalan/jax-fli-experiments`` HuggingFace dataset and renders the four scaling figures with the
``jax-hpc-profiler`` package: strong / weak scaling × min-time / peak-temp-memory, one line per adjoint
variant. CPU-only — no ``jax_fli`` / GPU needed.

The raw ``function`` column is a *unique* per-run string; the profiler groups lines by that column, so we
rewrite it to the adjoint variant (``reverse`` / ``ckpt-4`` / ``ckpt-8`` / ``ckpt-16`` / ``ckpt-30``) — that
becomes the five lines — and split the rows into a weak CSV and a strong CSV (strong g64 = 1024³/64 = 256³
would otherwise collide with weak's local 256³). All runs are float64.

    /home/wassim/Projects/NBody/jax-fli/.venv/bin/python build.py    # CPU, loads CSV from HF
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from huggingface_hub import snapshot_download
from jax_hpc_profiler.plotting import plot_by_data_size

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
DATA = HERE / "data"

REPO = "ASKabalan/jax-fli-experiments"
CSV = "12-gradient-scaling/perf/perf_pm.csv"

# five adjoint variants, in the order they should appear in the legend (memory ↔ compute trade)
VARIANTS = ["reverse", "ckpt-4", "ckpt-8", "ckpt-16", "ckpt-30"]

# --- load the perf CSV from HF (only this ~5 KB file, not the multi-GB density maps) ----------------
root = snapshot_download(REPO, repo_type="dataset", allow_patterns=[CSV])
raw = pd.read_csv(f"{root}/{CSV}", header=None)

# --- normalize the series label + split weak / strong into profiler-ready CSVs ----------------------
# function name e.g. pm30_exp12_{weak_g4|strong_M1024_g64}_{rev|ckpt4|ckpt8|ckpt16|ckpt30}_s0.
tokens = raw[0].str.split("_")
kind = tokens.str[2]  # weak / strong
label = {"rev": "reverse", "ckpt4": "ckpt-4", "ckpt8": "ckpt-8", "ckpt16": "ckpt-16", "ckpt30": "ckpt-30"}
raw[0] = tokens.str[-2].map(label)  # the token before s0 is the variant → the five lines
raw = raw[raw[0].notna()]  # the CSV also holds ckpt20 runs; unmapped → dropped here
kind = kind[raw.index]
DATA.mkdir(parents=True, exist_ok=True)
WEAK = DATA / "_weak.csv"
STRONG = DATA / "_strong.csv"
raw[kind == "weak"].to_csv(WEAK, header=False, index=False)
raw[kind == "strong"].to_csv(STRONG, header=False, index=False)


def plot_strong_time():
    """Strong scaling (fixed 1024³ grid, more GPUs), min wall-time. Five adjoint variants."""
    plot_by_data_size(
        csv_files=[str(STRONG)],
        data_size_queries=["global_1024x1024x1024"],
        functions=VARIANTS,
        plot_columns=["min_time"],
        label_text="%f%",
        xlabel="Number of GPUs",
        title="Strong scaling —",
        figure_size=(6, 4.5),
        xscale="log2",
        time_units="s",
        output=str(ASSETS / "fig01-strong-time.svg"),
    )


def plot_strong_memory():
    """Strong scaling, peak per-device temporary (scratch) memory."""
    plot_by_data_size(
        csv_files=[str(STRONG)],
        data_size_queries=["global_1024x1024x1024"],
        functions=VARIANTS,
        plot_columns=["temp_size"],
        memory_units="GB",
        label_text="%f%",
        xlabel="Number of GPUs",
        title="Strong scaling —",
        figure_size=(6, 4.5),
        xscale="log2",
        output=str(ASSETS / "fig02-strong-memory.svg"),
    )


def plot_weak_time():
    """Weak scaling (fixed 256³/GPU), min wall-time. Five adjoint variants."""
    plot_by_data_size(
        csv_files=[str(WEAK)],
        data_size_queries=["local_256x256x256"],
        functions=VARIANTS,
        plot_columns=["min_time"],
        label_text="%f%",
        xlabel="Number of GPUs",
        title="Weak scaling —",
        figure_size=(6, 4.5),
        xscale="log2",
        time_units="s",
        output=str(ASSETS / "fig03-weak-time.svg"),
    )


def plot_weak_memory():
    """Weak scaling, peak per-device temporary memory (constant per-GPU work)."""
    plot_by_data_size(
        csv_files=[str(WEAK)],
        data_size_queries=["local_256x256x256"],
        functions=VARIANTS,
        plot_columns=["temp_size"],
        memory_units="GB",
        label_text="%f%",
        xlabel="Number of GPUs",
        title="Weak scaling —",
        figure_size=(6, 4.5),
        xscale="log2",
        output=str(ASSETS / "fig04-weak-memory.svg"),
    )


def main():
    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none", "savefig.bbox": "tight"})
    ASSETS.mkdir(parents=True, exist_ok=True)
    plot_strong_time()
    plot_strong_memory()
    plot_weak_time()
    plot_weak_memory()
    plt.close("all")
    print(f"assets written to {ASSETS}")


if __name__ == "__main__":
    main()
