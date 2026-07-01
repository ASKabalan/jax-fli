"""Analysis page UI: file loading, field map visualization, and spectra routing.

This module owns all session-state management and top-level UI scaffolding.
Actual analysis logic is delegated to the analysis-specific modules.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .utils import (
    _DENSITY_3D,
    _FLAT_TYPES,
    _PARTICLE_TYPE,
    _SPECTRA_TYPES,
    _SPHERICAL_TYPES,
    _fig_to_png,
    _plt_lock,
    _spectral_category,
    indexed_field,
)

# ---------------------------------------------------------------------------
# Session state constants
# ---------------------------------------------------------------------------

CATALOGS_KEY = "analysis_catalogs"
HF_REPO_ID = "ASKabalan/jax-fli-experiments"


def _init_session_state():
    if CATALOGS_KEY not in st.session_state:
        st.session_state[CATALOGS_KEY] = []
    if "analysis_nz_point_sources" not in st.session_state:
        st.session_state["analysis_nz_point_sources"] = [0.01]


# ---------------------------------------------------------------------------
# Session state callbacks
# ---------------------------------------------------------------------------


def _build_entry(local_path: str, source: str = "local", hf_path: str | None = None):
    """Build a catalog entry dict from a local parquet file. Returns None on failure."""
    import jax_fli as jfli

    try:
        catalog = jfli.io.Catalog.from_parquet(local_path)
    except Exception as e:
        st.error(f"Failed to load {Path(local_path).name}: {e}")
        return None

    field = catalog.field[0]
    field_name = field.name if field.name else Path(local_path).stem
    field_type = type(field).__name__
    return {
        "id": uuid.uuid4().hex,
        "path": local_path,
        "source": source,
        "hf_path": hf_path,
        "label": field_name,
        "catalog": catalog,
        "field_type": field_type,
        "is_spectra": field_type in _SPECTRA_TYPES,
        "index": ":",
        "active": True,
    }


def _load_catalog(path: str):
    import glob as _glob

    path = path.strip()
    if not path:
        return

    # Glob pattern (contains *, ?, or [)
    if any(c in path for c in ("*", "?", "[")):
        matched = sorted(_glob.glob(path, recursive=True))
        if not matched:
            st.error(f"No files matched pattern: {path}")
            return
        for pf in matched:
            if Path(pf).is_file():
                _load_catalog(pf)
        return

    p = Path(path)
    if p.is_dir():
        parquet_files = sorted(p.rglob("*.parquet"))
        if not parquet_files:
            st.error(f"No .parquet files found under {path}")
            return
        for pf in parquet_files:
            _load_catalog(str(pf))
        return

    for entry in st.session_state[CATALOGS_KEY]:
        if entry["path"] == path:
            st.toast(f"Already loaded: {Path(path).name}")
            return

    entry = _build_entry(path, source="local")
    if entry is not None:
        st.session_state[CATALOGS_KEY].append(entry)


@st.cache_data(show_spinner="Fetching HuggingFace file list…")
def _discover_hf_files(repo_id: str) -> list[str]:
    """Return a sorted list of repo-relative parquet paths for a HF dataset repo."""
    from huggingface_hub import list_repo_files

    return sorted(f for f in list_repo_files(repo_id, repo_type="dataset") if f.endswith(".parquet"))


def _build_hf_tree(paths: list[str]) -> dict:
    """Nest a flat list of repo-relative paths into a directory tree.

    Each node maps a sub-folder name -> child node; a node's own files live under the empty-string
    key "" (path components are never empty, so this never collides with a real folder name).
    """
    root: dict = {}
    for p in paths:
        parts = p.split("/")
        node = root
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node.setdefault("", []).append(p)
    return root


def _count_parquet(node: dict) -> int:
    """Total parquet files at and below a tree node."""
    return len(node.get("", [])) + sum(_count_parquet(c) for k, c in node.items() if k != "")


def _render_hf_node(node: dict, repo_id: str, prefix: str, depth: int) -> None:
    """Recursively render one directory node: its own files, then each sub-folder as an expander."""
    files = node.get("", [])
    if files:
        sel_key = f"analysis_hf_sel_{prefix}"
        fc1, fc2 = st.columns([5, 1])
        with fc1:
            selected = st.multiselect(
                "Files",
                files,
                format_func=lambda p: Path(p).name,
                key=sel_key,
                label_visibility="collapsed",
            )
        with fc2:
            st.button(
                "Load",
                key=f"analysis_hf_load_{prefix}",
                on_click=_load_hf_files,
                args=(repo_id, sel_key),
                disabled=not selected,
            )

    for name in sorted(k for k in node if k != ""):
        child = node[name]
        child_prefix = f"{prefix}/{name}" if prefix else name
        with st.expander(
            f"{name} · {_count_parquet(child)} files",
            expanded=False,
            type="compact" if depth > 0 else "default",
        ):
            _render_hf_node(child, repo_id, child_prefix, depth + 1)


def _load_hf_files(repo_id: str, sel_key: str):
    from huggingface_hub import hf_hub_download

    for rel_path in st.session_state.get(sel_key, []):
        hf_path = f"{repo_id}/{rel_path}"
        if any(e.get("hf_path") == hf_path for e in st.session_state[CATALOGS_KEY]):
            st.toast(f"Already loaded: {Path(rel_path).name}")
            continue
        try:
            local_path = hf_hub_download(repo_id, rel_path, repo_type="dataset")
        except Exception as e:
            st.error(f"Failed to download {rel_path}: {e}")
            continue
        entry = _build_entry(local_path, source="hf", hf_path=hf_path)
        if entry is not None:
            st.session_state[CATALOGS_KEY].append(entry)


def _toggle_active(entry_id: str, key: str):
    for e in st.session_state[CATALOGS_KEY]:
        if e["id"] == entry_id:
            e["active"] = st.session_state[key]
            break


def _update_label(entry_id: str, key: str):
    for e in st.session_state[CATALOGS_KEY]:
        if e["id"] == entry_id:
            e["label"] = st.session_state[key]
            break


def _update_index(entry_id: str, key: str):
    for e in st.session_state[CATALOGS_KEY]:
        if e["id"] == entry_id:
            e["index"] = st.session_state[key]
            break


def _remove_entry(entry_id: str):
    st.session_state[CATALOGS_KEY] = [e for e in st.session_state[CATALOGS_KEY] if e["id"] != entry_id]


# ---------------------------------------------------------------------------
# Section 1: File Loading
# ---------------------------------------------------------------------------


def _render_hf_browser() -> None:
    with st.expander("\U0001f917 Load from HuggingFace Hub", expanded=False):
        repo_id = st.text_input("Dataset repo", value=HF_REPO_ID, key="analysis_hf_repo")
        try:
            paths = _discover_hf_files(repo_id)
        except Exception as e:
            st.error(f"Could not list files for '{repo_id}': {e}")
            return
        if not paths:
            st.info("No parquet files found in this repo.")
            return

        _render_hf_node(_build_hf_tree(paths), repo_id, prefix="", depth=0)


def _render_file_loading(entries: list[dict]) -> None:
    with st.container(border=True):
        st.subheader("Load Parquet Files")
        col_path, col_btn = st.columns([5, 1])
        with col_path:
            new_path = st.text_input(
                "Parquet file path or folder",
                key="analysis_new_path",
                placeholder="/path/to/file.parquet  or  /folder/  or  /folder/**/spectra*.parquet",
                label_visibility="collapsed",
            )
        with col_btn:
            st.button(
                "Load",
                key="analysis_load_btn",
                on_click=_load_catalog,
                args=(new_path,),
            )

        _render_hf_browser()

        if entries:
            hb, hl, hp, hi, ht, hr = st.columns([0.5, 2, 3, 1, 1.5, 0.5])
            hb.caption("Active")
            hl.caption("Label")
            hp.caption("Path")
            hi.caption("Index")
            ht.caption("Type")

        for i, entry in enumerate(entries):
            eid = entry["id"]
            is_hf = entry.get("source") == "hf"
            cb, cl, cp, ci, ct, cr = st.columns([0.5, 2, 3, 1, 1.5, 0.5])
            with cb:
                st.checkbox(
                    f"**#{i + 1}**",
                    value=entry.get("active", True),
                    key=f"analysis_active_{eid}",
                    on_change=_toggle_active,
                    args=(eid, f"analysis_active_{eid}"),
                )
            with cl:
                st.text_input(
                    "Label",
                    value=entry["label"],
                    key=f"analysis_label_{eid}",
                    on_change=_update_label,
                    args=(eid, f"analysis_label_{eid}"),
                    label_visibility="collapsed",
                )
            with cp:
                path_display = f"\U0001f917 {entry.get('hf_path', entry['path'])}" if is_hf else entry["path"]
                st.text_input(
                    "Path",
                    value=path_display,
                    key=f"analysis_path_{eid}",
                    disabled=True,
                    label_visibility="collapsed",
                )
            with ci:
                st.text_input(
                    "Index",
                    value=entry.get("index", ":"),
                    key=f"analysis_index_{eid}",
                    on_change=_update_index,
                    args=(eid, f"analysis_index_{eid}"),
                    disabled=not entry["catalog"].field[0].is_batched(),
                    label_visibility="collapsed",
                    placeholder=":",
                    help="numpy-like index applied everywhere, e.g. ':', '0:6', '-3:'.",
                )
            with ct:
                st.caption(entry["field_type"])
            with cr:
                st.button(
                    "\u2716",
                    key=f"analysis_rm_{eid}",
                    on_click=_remove_entry,
                    args=(eid,),
                )

        if not entries:
            st.info("No files loaded. Enter a parquet file path and click Load.")


# ---------------------------------------------------------------------------
# Section 2: Field Map Visualization
# ---------------------------------------------------------------------------


def _render_field_map_section(entries: list[dict]) -> None:
    st.divider()
    st.subheader("Visualization")

    file_labels = [f"[{i + 1}] {e['label']}" for i, e in enumerate(entries)]
    selected_idx = st.selectbox(
        "Select file to display",
        range(len(entries)),
        format_func=lambda i: file_labels[i],
        key="analysis_vis_select",
    )

    selected_entry = entries[selected_idx]
    field = indexed_field(selected_entry)
    field_type_str = selected_entry["field_type"]

    # Invalidate cached PNG when the selected file or its index changes
    _fkey = (selected_entry["path"], selected_entry.get("index", ":"))
    if st.session_state.get("_field_cache_path") != _fkey:
        st.session_state.pop("analysis_field_png", None)
        st.session_state["_field_cache_path"] = _fkey

    is_spectra = selected_entry.get("is_spectra", False)

    if is_spectra:
        st.info("Precomputed spectra — field rendering disabled.")
    else:
        with st.container(border=True):
            st.markdown("**Field Map Settings**")
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            with mc1:
                map_ncols = st.number_input(
                    "Columns",
                    min_value=1,
                    max_value=10,
                    value=2,
                    key="analysis_map_ncols",
                )
                _proj_disabled = field_type_str in (_DENSITY_3D | _PARTICLE_TYPE | _FLAT_TYPES)
                map_projection = st.selectbox(
                    "Projection",
                    ["mollweide", "cart", "polar", "aitoff", "hammer", "lambert"],
                    key="analysis_map_proj",
                    disabled=_proj_disabled,
                )
            with mc2:
                map_cmap = st.selectbox(
                    "Colormap",
                    [
                        "magma",
                        "viridis",
                        "inferno",
                        "plasma",
                        "cividis",
                        "coolwarm",
                        "RdBu_r",
                        "hot",
                        "bone",
                        "gray",
                    ],
                    key="analysis_map_cmap",
                )
                map_border = st.number_input(
                    "Border width",
                    min_value=0.0,
                    max_value=5.0,
                    value=1.0,
                    step=0.5,
                    key="analysis_map_border",
                    disabled=_proj_disabled,
                )
            with mc3:
                map_fig_w = st.number_input(
                    "Width/map",
                    min_value=2.0,
                    max_value=12.0,
                    value=4.0,
                    step=0.5,
                    key="analysis_map_fig_w",
                )
                map_fig_h = st.number_input(
                    "Height/map",
                    min_value=2.0,
                    max_value=12.0,
                    value=4.0,
                    step=0.5,
                    key="analysis_map_fig_h",
                )
            with mc4:
                map_colorbar = st.checkbox("Colorbar", value=True, key="analysis_map_cbar")
                map_ticks = st.checkbox(
                    "Graticule",
                    value=False,
                    key="analysis_map_ticks",
                    disabled=_proj_disabled,
                )
            with mc5:
                map_use_vmin = st.checkbox("Custom vmin", value=False, key="analysis_map_use_vmin")
                map_vmin = st.number_input(
                    "vmin",
                    value=0.0,
                    format="%.4f",
                    key="analysis_map_vmin",
                    disabled=not map_use_vmin,
                )
                map_use_vmax = st.checkbox("Custom vmax", value=False, key="analysis_map_use_vmax")
                map_vmax = st.number_input(
                    "vmax",
                    value=1.0,
                    format="%.4f",
                    key="analysis_map_vmax",
                    disabled=not map_use_vmax,
                )
                apply_fn = st.text_input(
                    "Apply function",
                    value="np.log(x + 5e-2)",
                    key="analysis_map_apply_fn",
                    help="Optional numpy expression, e.g. 'np.log10(x + 1e-5)'. Use 'x' as variable.",
                )

            tc1, tc2 = st.columns([3, 1])
            with tc1:
                map_title_template = st.text_input(
                    "Panel title template",
                    value="%l% - %i%",
                    key="analysis_map_title_template",
                    help="%l% = Label | %i% = Index | %r% = comoving distance  |  %z% = redshift  |  %a% = scale factor",
                )
            with tc2:
                map_dpi = st.number_input(
                    "Render DPI",
                    min_value=50,
                    max_value=2000,
                    value=100,
                    step=25,
                    key="analysis_map_dpi",
                )

            # Type-specific expandable options
            d_params = {}
            p_params = {}
            if field_type_str in _DENSITY_3D:
                with st.expander("3D Plot Options"):
                    dc1, dc2, dc3 = st.columns(3)
                    with dc1:
                        d_params["elev"] = st.number_input("Elevation", value=40.0, step=5.0, key="analysis_d_elev")
                        d_params["levels"] = st.number_input(
                            "Levels",
                            min_value=4,
                            max_value=256,
                            value=64,
                            key="analysis_d_levels",
                        )
                    with dc2:
                        d_params["azim"] = st.number_input("Azimuth", value=-30.0, step=5.0, key="analysis_d_azim")
                        d_params["project_slices"] = st.number_input(
                            "Project slices",
                            min_value=1,
                            max_value=128,
                            value=10,
                            key="analysis_d_project_slices",
                        )
                    with dc3:
                        d_params["zoom"] = st.number_input(
                            "Zoom",
                            min_value=0.1,
                            max_value=5.0,
                            value=0.8,
                            step=0.1,
                            key="analysis_d_zoom",
                        )
                        d_params["edges"] = st.checkbox("Edges", value=True, key="analysis_d_edges")

                    st.markdown("**Crop** (e.g. `:` or `10:50`)")
                    cc1, cc2, cc3 = st.columns(3)
                    with cc1:
                        d_crop_x = st.text_input("Crop X", value=":", key="analysis_d_crop_x")
                    with cc2:
                        d_crop_y = st.text_input("Crop Y", value=":", key="analysis_d_crop_y")
                    with cc3:
                        d_crop_z = st.text_input("Crop Z", value=":", key="analysis_d_crop_z")

                    pc1, pc2 = st.columns([1, 1])
                    with pc1:
                        d_params["do_project"] = st.checkbox("Project to 2D", value=False, key="analysis_d_do_project")
                    with pc2:
                        d_params["nz_slices"] = st.number_input(
                            "nz_slices",
                            min_value=1,
                            max_value=128,
                            value=10,
                            key="analysis_d_nz_slices",
                            disabled=not d_params["do_project"],
                        )

                    for dim, raw in [("x", d_crop_x), ("y", d_crop_y), ("z", d_crop_z)]:
                        try:
                            d_params[f"crop_{dim}"] = eval(f"np.s_[{raw}]", {"np": np})
                        except Exception:
                            d_params[f"crop_{dim}"] = slice(None)
                    d_params["crop"] = (
                        d_params.pop("crop_x"),
                        d_params.pop("crop_y"),
                        d_params.pop("crop_z"),
                    )

            elif field_type_str in _PARTICLE_TYPE:
                with st.expander("Particle Plot Options"):
                    pc1, pc2, pc3 = st.columns(3)
                    with pc1:
                        p_params["thinning"] = st.number_input(
                            "Thinning",
                            min_value=1,
                            max_value=64,
                            value=4,
                            key="analysis_p_thinning",
                        )
                        p_params["elev"] = st.number_input("Elevation", value=40.0, step=5.0, key="analysis_p_elev")
                    with pc2:
                        p_params["point_size"] = st.number_input(
                            "Point size",
                            min_value=0.5,
                            max_value=50.0,
                            value=5.0,
                            step=0.5,
                            key="analysis_p_point_size",
                        )
                        p_params["azim"] = st.number_input("Azimuth", value=-30.0, step=5.0, key="analysis_p_azim")
                    with pc3:
                        p_params["alpha"] = st.slider(
                            "Alpha",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.6,
                            step=0.05,
                            key="analysis_p_alpha",
                        )
                        p_params["zoom"] = st.number_input(
                            "Zoom",
                            min_value=0.1,
                            max_value=5.0,
                            value=0.8,
                            step=0.1,
                            key="analysis_p_zoom",
                        )
                    pw1, pw2 = st.columns(2)
                    with pw1:
                        weights_raw = st.text_input(
                            "Weights",
                            value="",
                            key="analysis_p_weights",
                            help="Leave empty for none, or enter: 'redshift', 'z', 'scale', 'a', 'comoving', 'r', or a float.",
                        )
                    with pw2:
                        p_params["weights_title"] = (
                            st.text_input(
                                "Weights title",
                                value="",
                                key="analysis_p_weights_title",
                                help="Optional colorbar label.",
                            )
                            or None
                        )
                    if not weights_raw:
                        p_params["weights"] = None
                    else:
                        try:
                            p_params["weights"] = float(weights_raw)
                        except ValueError:
                            p_params["weights"] = weights_raw

            plot_btn = st.button("Plot", key="analysis_plot_btn", type="primary")

        with st.container(border=True):
            st.markdown("**Field Map**")

            if plot_btn:
                # Field is already sliced by the global per-file index.
                plot_field = field

                # Apply function
                if apply_fn.strip():
                    try:
                        parsed_fn = eval(f"lambda x: {apply_fn}", {"np": np})
                        plot_field = plot_field.apply_fn(parsed_fn)
                    except Exception as e:
                        st.error(f"Apply function error: {e}")

                map_params = {
                    "ncols": int(map_ncols),
                    "cmap": map_cmap,
                    "fig_w": float(map_fig_w),
                    "fig_h": float(map_fig_h),
                    "colorbar": map_colorbar,
                    "ticks": map_ticks,
                    "vmin": map_vmin if map_use_vmin else None,
                    "vmax": map_vmax if map_use_vmax else None,
                    "border": float(map_border),
                    "projection": map_projection,
                    "dpi": int(map_dpi),
                    "title_template": map_title_template,
                    "label": selected_entry["label"],
                }

                png = None
                fig = None
                if field_type_str in _DENSITY_3D:
                    from . import density_analysis_compute

                    png, fig = density_analysis_compute.render_density_field_map(
                        selected_entry, plot_field, map_params, d_params
                    )
                elif field_type_str in _PARTICLE_TYPE:
                    from . import density_analysis_compute

                    png, fig = density_analysis_compute.render_particle_field_map(
                        selected_entry, plot_field, map_params, p_params
                    )
                elif field_type_str in _SPHERICAL_TYPES:
                    from . import spherical_analysis_form

                    png, fig = spherical_analysis_form.render_field_map(selected_entry, plot_field, map_params)
                else:  # flat types
                    from . import flat_analysis

                    png, fig = flat_analysis.render_flat_field_map(selected_entry, plot_field, map_params)
                    print(f"[form] Flat field map rendered: png size = {len(png) if png else 'None'}, fig = {fig}")

                if png is not None:
                    st.session_state["analysis_field_png"] = png
                    st.session_state["analysis_field_fig"] = fig
                else:
                    st.error("Field map rendering failed — check the console for details.")

            field_png = st.session_state.get("analysis_field_png")
            field_fig = st.session_state.get("analysis_field_fig")
            if field_png:
                st.image(field_png)
                if field_fig is not None:
                    from .save_figure import render_save_figure

                    render_save_figure(field_fig, key_prefix="field_map", filename="field_map")
            else:
                st.info("Adjust settings above, then click **Plot**.")

    # Metadata mini-plots (fast, no heavy compute)
    meta_attrs = [
        (a, lbl, u)
        for a, lbl, u in [
            ("comoving_centers", "Comoving Centers", "Mpc/h"),
            ("scale_factors", "Scale Factors", "a"),
            ("z_sources", "z Sources", "z"),
            ("density_width", "Density Width", "Mpc/h"),
        ]
        if getattr(field, a, None) is not None
    ]
    if meta_attrs:
        # 1. Logic to render the plots into a single PNG buffer
        meta_png = None
        with _plt_lock:
            # Use a slightly wider figsize for the 2x2 grid
            fig_m, axes = plt.subplots(1, 4, figsize=(18, 4))
            axes_flat = axes.flatten()

            for i, (attr, lbl, unit) in enumerate(meta_attrs):
                ax = axes_flat[i]
                arr = np.asarray(getattr(field, attr))

                if arr.ndim == 0:
                    ax.axhline(float(arr), color="C0", linestyle="--")
                    ax.text(
                        0.5,
                        float(arr),
                        f"{float(arr):.2f}",
                        transform=ax.get_yaxis_transform(),
                        ha="center",
                        va="bottom",
                    )
                else:
                    ax.plot(arr, marker="o", markersize=4, linewidth=1.5)
                    ax.set_xticks(np.arange(len(arr)))
                    ax.grid(True, linestyle="--", alpha=0.6)

                ax.set_title(lbl, fontsize=11, fontweight="bold")
                ax.set_xlabel("Shell" if arr.ndim > 0 else "", fontsize=9)
                ax.set_ylabel(unit, fontsize=9)
                ax.tick_params(labelsize=8)

            # Hide unused quadrants
            for j in range(len(meta_attrs), 4):
                axes_flat[j].axis("off")

            fig_m.tight_layout()

            # Use your utility function from utils.py
            meta_png = _fig_to_png(fig_m, dpi=600)
            plt.close(fig_m)

        # 2. Display the rendered PNG in a clean container
        if meta_png:
            st.caption("Field Metadata Analysis")
            st.image(meta_png)

    with st.expander("Field info"):
        st.code(repr(field), language=None)

    # print a markdown table of all metadata attributes
    with st.expander("Metadata table"):
        attributes = ["comoving_centers", "scale_factors", "z_sources", "density_width"]

        # Build the table header and separator
        table_md = "| Comoving | Scale factor | Redshift | Density |\n"
        table_md += "|---|---|---|---|\n"

        # 1. Gather all the data into a list of lists/arrays
        columns_data = []
        for attr in attributes:
            val = getattr(field, attr, None)
            if val is None:
                columns_data.append([])
            elif isinstance(val, (float, int)):
                columns_data.append([val])  # Wrap single numbers in a list
            else:
                columns_data.append(val)  # Keep as iterable/numpy array

        # 2. Find the maximum length among all the columns
        max_rows = max([len(col) for col in columns_data] + [0])

        # 3. Build the table row by row
        for i in range(max_rows):
            row_vals = []
            for col in columns_data:
                # If this column has data for the current row, format it
                if i < len(col):
                    row_vals.append(f"{col[i]:.4f}")
                else:
                    # Otherwise, leave the cell empty
                    row_vals.append("")

            # Combine the formatted values into a markdown row
            table_md += f"| {' | '.join(row_vals)} |\n"

        if max_rows > 0:
            st.markdown(table_md)
        else:
            st.info("No metadata attributes available to display.")


# ---------------------------------------------------------------------------
# Section 3: Summary Stat routing
# ---------------------------------------------------------------------------


def _render_spectra_section(active_entries: list[dict]) -> None:
    st.divider()
    st.subheader("Summary Stat")

    # Route each entry by spectral category so a selection may mix a precomputed
    # PowerSpectrum with the raw field that produces the same spectrum.
    cl_entries = [e for e in active_entries if _spectral_category(e) == "cl"]
    pk_entries = [e for e in active_entries if _spectral_category(e) == "pk"]
    sph_raw = [e for e in active_entries if e["field_type"] in _SPHERICAL_TYPES]

    tab_cl, tab_pk, tab_peak, tab_pdf, tab_star = st.tabs(["Angular Cl", "3D P(k)", "Peak counts", "PDF", "Starlet"])

    with tab_cl:
        if cl_entries:
            from . import spherical_analysis_form

            spherical_analysis_form.cl_tab(cl_entries)
        else:
            st.info(
                "No Angular Cl spectra selected — load a SphericalDensity / "
                "SphericalKappaField map or a precomputed Angular Cl PowerSpectrum."
            )

    with tab_pk:
        if pk_entries:
            from . import density_analysis_form

            density_analysis_form.pk_tab(pk_entries)
        else:
            st.info("No 3D P(k) spectra selected — load a DensityField or a precomputed 3D P(k) PowerSpectrum.")

    with tab_peak:
        if sph_raw:
            from . import binned_stats_form

            binned_stats_form.binned_tab(sph_raw, sph_raw[0]["field_type"], "peak_counts")
        else:
            st.info("Peak counts requires a raw spherical map (SphericalDensity / SphericalKappaField).")

    with tab_pdf:
        if sph_raw:
            from . import binned_stats_form

            binned_stats_form.binned_tab(sph_raw, sph_raw[0]["field_type"], "pdf")
        else:
            st.info("PDF requires a raw spherical map (SphericalDensity / SphericalKappaField).")

    with tab_star:
        if sph_raw:
            from . import starlet_form

            starlet_form.starlet_tab(sph_raw, sph_raw[0]["field_type"])
        else:
            st.info("Starlet requires a raw spherical map (SphericalDensity / SphericalKappaField).")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Render the full Analysis page. Called by the Streamlit entry point."""
    _init_session_state()
    entries = st.session_state[CATALOGS_KEY]

    _render_file_loading(entries)

    if not entries:
        st.stop()

    _render_field_map_section(entries)

    active_entries = [e for e in entries if e.get("active", True)]
    if not active_entries:
        st.warning("No active entries selected.")
        st.stop()

    _render_spectra_section(active_entries)
