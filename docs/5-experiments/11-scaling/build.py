# ruff: noqa: E402
"""Experiment 11 — PM forward-model strong & weak scaling figures.

Reads the ``perf_pm.csv`` produced on the cluster (profiler schema, one row per run) from the
``ASKabalan/jax-fli-experiments`` HuggingFace dataset and renders the four scaling figures with the
``jax-hpc-profiler`` package: strong / weak scaling × min-time / peak-temp-memory. CPU-only — no
``jax_fli`` / GPU needed.

The raw ``function`` column is a *unique* per-run string (it embeds the GPU count / mesh), which the
profiler cannot group into scaling lines. So we rewrite it to a single constant ``pm-forward`` (the two
lines then come from the ``precision`` column: float32 / float64) and split the rows into a weak CSV and
a strong CSV — weak (local 256³) and strong g64 (1024³/64 = 256³) would otherwise collide on a
``local_256³`` query. The profiler reads those two derived CSVs by path.

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
CSV = "11-scaling/perf/perf_pm.csv"

# --- load the perf CSV from HF (only this ~2 KB file, not the multi-GB density maps) ----------------
root = snapshot_download(REPO, repo_type="dataset", allow_patterns=[CSV])
raw = pd.read_csv(f"{root}/{CSV}", header=None)

# --- normalize the series label + split weak / strong into profiler-ready CSVs ----------------------
# function name e.g. pm50_exp11_{weak_g4|strong_M1024_g32}_{f32|f64}_s0 → tokens[2] is weak/strong.
kind = raw[0].str.split("_").str[2]
raw[0] = "pm-forward"  # single series; the two lines are float32 vs float64 (the precision column)
DATA.mkdir(parents=True, exist_ok=True)
WEAK = DATA / "_weak.csv"
STRONG = DATA / "_strong.csv"
raw[kind == "weak"].to_csv(WEAK, header=False, index=False)
raw[kind == "strong"].to_csv(STRONG, header=False, index=False)


def plot_strong_time():
    """Strong scaling (fixed grid, more GPUs), min wall-time. Subplots 1024³ and 2048³."""
    plot_by_data_size(
        csv_files=[str(STRONG)],
        data_size_queries=["global_1024x1024x1024", "global_2048x2048x2048"],
        precisions=["float32", "float64"],
        plot_columns=["min_time"],
        label_text="%pr%",
        xlabel="Number of GPUs",
        title="Strong scaling —",
        figure_size=(6.5, 8),
        xscale="log2",
        time_units="s",
        output=str(ASSETS / "fig01-strong-time.svg"),
    )


def plot_strong_memory():
    """Strong scaling, peak per-device temporary (scratch) memory (each GPU holds ~1/N of the mesh)."""
    plot_by_data_size(
        csv_files=[str(STRONG)],
        data_size_queries=["global_1024x1024x1024", "global_2048x2048x2048"],
        precisions=["float32", "float64"],
        plot_columns=["temp_size"],
        memory_units="GB",
        label_text="%pr%",
        xlabel="Number of GPUs",
        title="Strong scaling —",
        figure_size=(6.5, 8),
        xscale="log2",
        output=str(ASSETS / "fig02-strong-memory.svg"),
    )


def plot_weak_time():
    """Weak scaling (fixed 256³/GPU), min wall-time. One subplot."""
    plot_by_data_size(
        csv_files=[str(WEAK)],
        data_size_queries=["local_256x256x256"],
        precisions=["float32", "float64"],
        plot_columns=["min_time"],
        label_text="%pr%",
        xlabel="Number of GPUs",
        title="Weak scaling —",
        figure_size=(6, 4),
        xscale="log2",
        time_units="s",
        output=str(ASSETS / "fig03-weak-time.svg"),
    )


def plot_weak_memory():
    """Weak scaling, peak per-device temporary memory (constant per-GPU work)."""
    plot_by_data_size(
        csv_files=[str(WEAK)],
        data_size_queries=["local_256x256x256"],
        precisions=["float32", "float64"],
        plot_columns=["temp_size"],
        memory_units="GB",
        label_text="%pr%",
        xlabel="Number of GPUs",
        title="Weak scaling —",
        figure_size=(6, 4),
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
