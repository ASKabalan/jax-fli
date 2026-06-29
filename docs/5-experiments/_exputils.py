"""Shared figure helpers for the experiments in ``docs/5-experiments``.

Each experiment is a runnable Python script that *saves* its figures (rather than showing
them) so the website and a paper can reuse the same assets. Figures are exported as **SVG**:
vector, web-native (embeds in markdown via ``![](fig.svg)``), and the text stays editable.
Figures carry **no overarching title** — the README paragraph under each figure is its caption.

Note on HEALPix maps: a ``healpy.mollview`` / ``gnomview`` is inherently a raster, so its SVG
embeds a bitmap. Render maps with a high ``xsize`` so that embedded raster is sharp; the axes,
graticule and colorbar stay vector. Pure line plots (spectra, ratios) and the 3-D wireframe
boxes are true vector in SVG.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def _deprecated_set_style() -> None:
    """Uniform, paper-ready matplotlib defaults for every experiment figure."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "svg.fonttype": "none",  # keep SVG text as selectable/editable text
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.grid": False,
        }
    )


def set_style() -> None:
    """Uniform, JCAP-ready matplotlib defaults for experiment figures."""

    # JCAP text width is roughly 6 inches.
    # Using the golden ratio (~0.618) for height is a standard aesthetic choice.
    fig_width = 6.0
    fig_height = fig_width * 0.618

    plt.rcParams.update(
        {
            # --- Figure Size and Resolution ---
            "figure.figsize": (fig_width, fig_height),
            "figure.dpi": 150,  # Screen drafting resolution
            "savefig.dpi": 600,  # Minimum standard for JCAP print
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,  # Minimize whitespace padding
            "svg.fonttype": "none",
            # --- Fonts and Text (Matching LaTeX) ---
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],  # The default LaTeX font
            "font.size": 11,  # JCAP standard body text size
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            # --- Axes and Ticks (Physics Standard) ---
            "axes.grid": False,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",  # Ticks point inward (physics convention)
            "ytick.direction": "in",
            "xtick.top": True,  # Mirrored ticks on top/right axes
            "ytick.right": True,
            "xtick.major.size": 5,
            "ytick.major.width": 1.0,
            "ytick.major.size": 5,
            "xtick.minor.visible": True,  # Minor ticks enabled
            "ytick.minor.visible": True,
        }
    )


def savefig(stem: str | PathLike[str], fig: Figure | None = None, *, formats=("svg",)) -> None:
    """Save ``fig`` (default: the current figure) to ``stem.<fmt>`` for each format.

    ``stem`` is a path *without* extension; parent directories are created. The figure is
    closed afterwards so a long script doesn't accumulate open figures. Default is SVG only.
    """
    fig = fig or plt.gcf()
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), transparent=True)
    plt.close(fig)
