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

import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

THESIS_WIDTH_IN = 4.98  # the manuscript's \textwidth: figures authored at this width carry literal sizes
DISPLAY_WIDTH_IN = 9.375  # ~900 px: the width a README figure is displayed at


def set_style(width_in: float | None = None) -> None:
    """Uniform, JCAP-ready matplotlib defaults for experiment figures.

    Styling matches the thesis figures (Computer Modern serif, inward mirrored ticks). Two
    sizing regimes:

    * ``width_in=None`` — the exact thesis sizes: 8.5 pt text authored at the printed
      4.98 in width, so every size is the size that reaches paper. Use it for figures
      meant to be read at that physical width (the lensing-reference replicas).
    * ``width_in=<figure width>`` — every font size and line width is scaled by
      ``width_in / DISPLAY_WIDTH_IN``, which normalises the figure to the ~900 px width a
      README displays it at: the text then reaches the screen at the same *apparent* size
      for every figure, large multi-panel strips included. Pass the width of the figure
      about to be drawn.
    """

    # JCAP text width is roughly 6 inches.
    # Using the golden ratio (~0.618) for height is a standard aesthetic choice.
    fig_width = 6.0
    fig_height = fig_width * 0.618

    s = 1.0 if width_in is None else width_in / DISPLAY_WIDTH_IN
    plt.rcParams.update(
        {
            # --- Figure Size and Resolution ---
            "figure.figsize": (fig_width, fig_height),
            "figure.dpi": 150,  # Screen drafting resolution
            "savefig.dpi": 600,  # Minimum standard for JCAP print
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,  # Minimize whitespace padding
            "svg.fonttype": "none",
            # --- Fonts and Text (Matching LaTeX / the thesis figures) ---
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],  # The default LaTeX font
            "font.size": 8.5 * s,
            "axes.titlesize": 9 * s,
            "axes.labelsize": 8.5 * s,
            "legend.fontsize": 8.2 * s,
            "xtick.labelsize": 8.2 * s,
            "ytick.labelsize": 8.2 * s,
            # --- Axes and Ticks (Physics Standard) ---
            "axes.grid": False,
            "axes.linewidth": 0.8 * s,
            "grid.linewidth": 0.5 * s,
            "lines.linewidth": 1.2 * s,
            "xtick.direction": "in",  # Ticks point inward (physics convention)
            "ytick.direction": "in",
            "xtick.top": True,  # Mirrored ticks on top/right axes
            "ytick.right": True,
            "xtick.major.size": 3.2 * s,
            "xtick.major.width": 0.8 * s,
            "ytick.major.size": 3.2 * s,
            "ytick.major.width": 0.8 * s,
            "xtick.minor.size": 1.8 * s,
            "ytick.minor.size": 1.8 * s,
            "xtick.minor.visible": True,  # Minor ticks enabled
            "ytick.minor.visible": True,
        }
    )


def savefig(stem: str | os.PathLike[str], fig: Figure | None = None, *, formats=("svg",)) -> None:
    """Save ``fig`` (default: the current figure) to ``stem.<fmt>`` for each format.

    ``stem`` is a path *without* extension; parent directories are created. The figure is
    closed afterwards so a long script doesn't accumulate open figures. Default is SVG only.
    """
    PAPER_PATH = os.environ.get("FLI_PAPER_PATH", None)

    fig = fig or plt.gcf()
    if PAPER_PATH is not None:
        stem = str(stem)
        if "assets" in stem:
            stem = stem.split("assets")[-1].lstrip("/")
        stem = Path(PAPER_PATH) / stem
        formats = ("pdf",)

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), transparent=True)
    plt.close(fig)
    print(f"✅ Saved figure: {stem}.[{', '.join(formats)}]")
