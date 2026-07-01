"""2D field map rendering for flat field types (FlatDensity, FlatKappaField).

Flat fields use a simpler plot API than spherical ones — no ``border_linewidth``
or ``projection_type`` parameters. For spherical types (SphericalDensity,
SphericalKappaField) use ``spherical_analysis_form.render_field_map`` instead.

Angular Cl is NOT supported for flat types — the Cl tab shows an error for them.
"""
from __future__ import annotations

from math import ceil

import matplotlib.pyplot as plt

from .utils import _fig_to_png, _make_title, _plt_lock


def render_flat_field_map(
    selected_entry: dict,
    plot_field,
    map_params: dict,
) -> tuple[bytes, object] | tuple[None, None]:
    """Render a flat (FlatDensity / FlatKappaField) field map; return (png_bytes, fig).

    Parameters
    ----------
    selected_entry:
        Catalog entry dict with at least ``"label"`` key.
    plot_field:
        The (possibly sliced) field object with a ``.plot()`` method.
    map_params:
        Dict of rendering settings::

            ncols, cmap, fig_w, fig_h, colorbar, ticks,
            vmin, vmax, dpi, title_template, label

        Note: ``border`` and ``projection`` are ignored for flat types.

    Returns
    -------
    (png_bytes, fig) on success; (None, None) on failure.
    The caller is responsible for closing the figure after use.
    """
    n_maps = plot_field.shape[0] if plot_field.is_batched() else 1
    ncols = int(map_params["ncols"])
    nrows = max(1, ceil(n_maps / ncols))
    titles = []
    for i in range(n_maps):
        t = _make_title(
            map_params["title_template"],
            plot_field,
            i,
            label=map_params.get("label", ""),
        )
        titles.append(t)

    fig = None
    with _plt_lock:
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(map_params["fig_w"] * ncols, map_params["fig_h"] * nrows),
        )
        try:
            plot_field.plot(
                ax=axes,
                titles=titles,
                cmap=map_params["cmap"],
                colorbar=map_params["colorbar"],
                show_ticks=map_params["ticks"],
                vmin=map_params["vmin"],
                vmax=map_params["vmax"],
            )
            png = _fig_to_png(fig, dpi=int(map_params["dpi"]))
            return png, fig
        except Exception as e:
            print(f"[flat_analysis] Field map rendering failed: {e}")
            if fig is not None:
                plt.close(fig)
            return None, None
