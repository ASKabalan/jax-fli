"""Render the panel PNGs consumed by animate_map_scene.py (Manim) from a MAP run directory.

Runs offline in the project venv (CPU is enough, no model re-run):

    uv run --no-sync python animate_map_prep.py --out-dir ../data/13-LPT-MassMapping/mesh_416_DES --title "DES Y3"

The IC is shown the way lensing constrains it, as in the BORG-WL papers: projected along every line of sight with the
lensing efficiency of the source bins (``DensityField.sky_projection``, averaged over the bins, from the lightcone's
inner edge). Each face of the box shows that projection along the ray from the observer (the box centre) through it,
the view of notebook 14, section 11. Nothing is smoothed.

Writes `<out-dir>/anim_frames/` with
    truth_cube.png             truth IC, projected, on the faces of the box
    cube_<r>.png               guess IC of frame r, same projection and colour scale
    diff_cube_<r>.png          truth - guess, projected (symmetric RdBu on the truth's scale)
    truth_kappa_<p>.png        truth kappa, source bin p
    kappa_<r>_<p>.png          guess kappa, frame r, bin p (the truth's colour scale)
    diff_kappa_<r>_<p>.png     truth - guess kappa, frame r, bin p (symmetric RdBu on the truth's scale)
plus `frame_index.json` with the per-frame readout: step, loss, and the truth-guess correlation of kappa (per bin)
and of the projected IC.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")

import healpy as hp
import jax
import jax_cosmo as jc
import matplotlib.pyplot as plt
import numpy as np

import jax_fli as jfli
from jax_fli.data.nz import get_des_y3_nz_shear
from jax_fli.fields.density import plot_3d_density
from jax_fli.fields.lightcone import plot_spherical_density

jax.config.update("jax_enable_x64", True)


def cube_faces(sky, nside, n):
    """The sky map along the ray from the box centre through each cell of the three faces plot_3d_density draws."""
    g = np.arange(n) - 0.5 * n
    u, v = np.meshgrid(g, g, indexing="ij")
    near, far = np.full_like(u, g[0]), np.full_like(u, g[-1])
    vol = np.zeros((n,) * 3, np.float32)
    for face, (x, y, z) in (
        ((slice(None), slice(None), 0), (u, v, near)),
        ((slice(None), 0, slice(None)), (u, near, v)),
        ((n - 1, slice(None), slice(None)), (far, u, v)),
    ):
        vol[face] = sky[hp.vec2pix(nside, x.ravel(), y.ravel(), z.ravel())].reshape(n, n)
    return vol


def render_cube_png(path, vol, lim, cmap):
    fig = plt.figure(figsize=(4.6, 4.6), dpi=220)
    ax = fig.add_subplot(projection="3d")
    plot_3d_density(
        ax,
        vol,
        project_slices=1,
        labels=("", "", ""),
        ticks=([], [], []),
        vmin=-lim,
        vmax=lim,
        cmap=cmap,
        colorbar=False,
        zoom=0.95,
        levels=128,
    )
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=220, transparent=True)
    plt.close(fig)


def render_kappa_png(path, plane, vmin, vmax, cmap="magma"):
    fig, ax = plt.subplots(figsize=(6.2, 3.72), dpi=220)
    plot_spherical_density(
        ax,
        plane,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        show_colorbar=False,
        show_ticks=False,
        title="",
        border_linewidth=0.8,
    )
    fig.tight_layout(pad=0.1)
    fig.savefig(path, dpi=220, transparent=True)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--out-dir",
        required=True,
        help="MAP run directory (data/13-LPT-MassMapping/mesh_416_DES or mesh_416_EUCLID, fetched with hf download)",
    )
    ap.add_argument("--title", default="", help="survey name shown in the animation title")
    ap.add_argument("--des-bins", type=int, nargs="+", default=(1, 2), help="DES Y3 source bins of the run (0-based)")
    ap.add_argument("--max-z", type=float, default=1.0)
    ap.add_argument("--cube-cells", type=int, default=256, help="cells per face of the rendered box")
    args = ap.parse_args()

    out = Path(args.out_dir)
    frame_dir = out / "anim_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    load = lambda p: jfli.io.Catalog.from_parquet(str(p)).field[0]
    by_frame = lambda d, pre: sorted((out / d).glob(f"{pre}_*.parquet"), key=lambda p: int(p.stem.split("_")[-1]))
    cosmo = jc.Planck18()
    nz_shear = [get_des_y3_nz_shear(zmax=args.max_z)[i] for i in args.des_bins]

    truth_ic = load(out / "true_ic.parquet")
    truth_kappa = load(out / "truth_kappa.parquet")
    lc = load(out / "truth_density_lightcone.parquet")
    r_min = float(np.min(np.asarray(lc.comoving_centers) - 0.5 * np.asarray(lc.density_width)))
    nside = int(truth_kappa.nside)
    project = lambda ic: np.asarray(
        ic.sky_projection(cosmo, nz_shear, nside=nside, r_min=r_min).array, dtype=np.float64
    ).mean(axis=0)
    rows = sorted(json.loads((out / "metrics.json").read_text()), key=lambda r: r["frame"])

    kt = np.asarray(truth_kappa.array)
    pt = project(truth_ic)
    lim_p = float(np.percentile(np.abs(pt), 99.8))
    kmin, kmax = (float(x) for x in np.percentile(kt, [0.1, 99.9]))
    lim_k = float(np.percentile(np.abs(kt), 99.9))
    render_cube_png(frame_dir / "truth_cube.png", cube_faces(pt, nside, args.cube_cells), lim_p, "magma")
    for p in range(kt.shape[0]):
        render_kappa_png(frame_dir / f"truth_kappa_{p}.png", kt[p], kmin, kmax)

    r_kappa, r_proj = [], []
    for r, (ic_file, k_file) in enumerate(zip(by_frame("ic_evolution", "ic"), by_frame("kappa_evolution", "kappa"))):
        pp = project(load(ic_file))
        kp = np.asarray(load(k_file).array, dtype=np.float64)
        render_cube_png(frame_dir / f"cube_{r}.png", cube_faces(pp, nside, args.cube_cells), lim_p, "magma")
        render_cube_png(frame_dir / f"diff_cube_{r}.png", cube_faces(pt - pp, nside, args.cube_cells), lim_p, "RdBu_r")
        for p in range(kp.shape[0]):
            render_kappa_png(frame_dir / f"kappa_{r}_{p}.png", kp[p], kmin, kmax)
            render_kappa_png(frame_dir / f"diff_kappa_{r}_{p}.png", kt[p] - kp[p], -lim_k, lim_k, cmap="RdBu_r")
        r_kappa.append([float(np.corrcoef(kt[p], kp[p])[0, 1]) for p in range(kp.shape[0])])
        r_proj.append(float(np.corrcoef(pt, pp)[0, 1]))
        print(
            f"frame {r} (step {rows[r]['step']}): r(kappa) {np.round(r_kappa[-1], 3)}  r(projected IC) {r_proj[-1]:.3f}",
            flush=True,
        )

    meta = {
        "title": args.title,
        "n_frames": len(r_proj),
        "steps": [r["step"] for r in rows],
        "loss": [r["loss"] for r in rows],
        "r_kappa": r_kappa,
        "r_proj": r_proj,
        "planes": [f"z = {float(z):.2f}" for z in np.asarray(truth_kappa.z_sources)],
        "r_min": r_min,
    }
    (frame_dir / "frame_index.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(list(frame_dir.glob('*.png')))} PNGs + frame_index.json to {frame_dir}")


if __name__ == "__main__":
    main()
