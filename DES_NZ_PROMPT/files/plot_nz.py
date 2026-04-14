"""
Visualize DES Y3 n(z) source distributions and compare with Stage 3.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json


def plot_nz(data_dir, survey_name, ax=None, colors=None, linestyle="-"):
    """Plot n(z) bins from a data directory containing txt files and meta.json."""
    meta_file = list(Path(data_dir).glob("*_meta.json"))
    if meta_file:
        with open(meta_file[0]) as f:
            meta = json.load(f)
        n_bins = meta["n_bins"]
        files = [meta["bins"][f"bin{i+1}"]["file"] for i in range(n_bins)]
        zmeans = [meta["bins"][f"bin{i+1}"]["z_mean"] for i in range(n_bins)]
    else:
        files = sorted(Path(data_dir).glob("nz_*.txt"))
        n_bins = len(files)
        zmeans = [None] * n_bins

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    if colors is None:
        cmap = plt.cm.viridis
        colors = [cmap(i / max(n_bins - 1, 1)) for i in range(n_bins)]

    for i in range(n_bins):
        if isinstance(files[i], str):
            fpath = Path(data_dir) / files[i]
        else:
            fpath = files[i]
        z, nz = np.loadtxt(fpath, unpack=True)
        label = f"{survey_name} Bin {i+1}"
        if zmeans[i]:
            label += f" (<z>={zmeans[i]:.2f})"
        ax.plot(z, nz, label=label, color=colors[i], linestyle=linestyle, linewidth=1.8)

    return ax


def main():
    stage3_dir = "/mnt/user-data/uploads"
    y3_dir = "/home/claude/des_y3_data"

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=False)

    # ── Left: Stage 3 (CosmoGrid) ──
    ax = axes[0]
    stage3_files = sorted(Path(stage3_dir).glob("nz_stage3_*.txt"))
    colors_s3 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, f in enumerate(stage3_files):
        z, nz = np.loadtxt(f, unpack=True)
        ax.plot(z, nz, label=f"Shell {i+1}", color=colors_s3[i], linewidth=1.8)
    ax.set_xlabel("Redshift", fontsize=13)
    ax.set_ylabel("n(z)", fontsize=13)
    ax.set_title("Stage 3 (CosmoGrid)", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 3.0)

    # ── Right: DES Y3 ──
    ax = axes[1]
    with open(Path(y3_dir) / "des_y3_meta.json") as f:
        meta = json.load(f)
    colors_y3 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i in range(4):
        fname = meta["bins"][f"bin{i+1}"]["file"]
        zmean = meta["bins"][f"bin{i+1}"]["z_mean"]
        neff = meta["bins"][f"bin{i+1}"]["gals_per_arcmin2"]
        z, nz = np.loadtxt(Path(y3_dir) / fname, unpack=True)
        ax.plot(
            z, nz,
            label=f"Bin {i+1} (<z>={zmean:.2f}, n$_{{eff}}$={neff})",
            color=colors_y3[i], linewidth=1.8,
        )
    ax.set_xlabel("Redshift", fontsize=13)
    ax.set_ylabel("n(z)", fontsize=13)
    ax.set_title("DES Y3 (Metacalibration)", fontsize=14)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 3.0)

    plt.tight_layout()
    outpath = "/home/claude/nz_comparison_stage3_desy3.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
