"""Shared figure helpers for the experiments in ``docs/5-experiments``.

Each experiment is a runnable Python script that *saves* its figures (rather than showing
them) so the website and a paper can reuse the same assets. Figures are exported as **SVG**:
vector, web-native (embeds in markdown via ``![](fig.svg)``), and the text stays editable.

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


def set_style() -> None:
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
