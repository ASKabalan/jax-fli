#!/usr/bin/env python
"""Experiment 06 — geometry prep for the CosmoGrid-matched density shells.

Works out, *before* any cluster run, the box / mesh / shell geometry needed to simulate the
**same density shells as CosmoGrid at nside 2048** (cf. Experiment 00) deep enough to lens the
**DES Y3** source bins, for two observer placements:

  * **full sky**     observer at the box centre ``(0.5, 0.5, 0.5)``  -> isotropic box ``2r``
  * **big quadrant** observer near a corner ``(0.1, 0.5, 0.9)``      -> box ``(1.2r, 2.0r, 1.2r)``

and for two source depths (a **2-bin** set = DES Y3 bins 1+2, a **3-bin** set = bins 1+2+3;
bin 4's tail reaches too far). Each of that **2x2** is run under both a 1-D **slab** (``pdim 128 1``)
and a 2-D **pencil** (``pdim 32 4``) device decomposition -> **8 sims**.

What it does (all on CPU):
  1. Load DES Y3 ``n(z)`` and measure each bin's *effective end* — the redshift past which the
     density has dropped to ~zero (``n(z) >= THRESH_FRAC * peak``), calibrated so bin 3 ~ z 1.0.
  2. Size each box from the observer factors 1 + 2*min(f, 1-f) (cf.
     ``jax_fli.compute_box_size_from_redshift``): round 2r(z_max) up to a tidy side L; the quadrant is
     (0.6L, L, 0.6L), i.e. (1.2r, 2.0r, 1.2r) for observer (0.1, 0.5, 0.9).
  3. Pull the published CosmoGrid nside-2048 density shells from HuggingFace (one parquet per shell)
     and read each shell's scale factor + comoving edges from its metadata (the map itself is not
     needed, so we never materialise it past the two scalars).
  4. Select the shells that fall inside each box and emit the ``--ts-near`` / ``--ts-far`` edge
     lists (CosmoGrid scale-factor edges) that reproduce them.
  5. Size the mesh + GPU layout: full sky 2048^3; quadrant packs the SAME ~2048^3 cell budget into its
     smaller volume at isotropic, finer dx, with a mesh chosen so it is valid under BOTH the slab and
     pencil device grids. Both on 128 GPUs.
  6. Save the geometry + visibility-mask figures (assets/exp06-*.svg) and write geometry.sh for run.sh.

    python prep_geometry.py
"""

from __future__ import annotations

import os

# Pure host-side analysis (n(z), distances, nside-4 maps) — pin to CPU before importing jax.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import gc  # noqa: E402
import sys  # noqa: E402
from math import lcm  # noqa: E402
from pathlib import Path  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import jax_cosmo as jc  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import jax_fli as jfli  # noqa: E402
from jax_fli.data import get_des_y3_nz_shear  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _exputils import savefig, set_style  # noqa: E402

# --------------------------------------------------------------------------------------------------
# Choices (calibrated; see the README)
# --------------------------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
REPO = "ASKabalan/jax-fli-experiments"
DENSITY_SHELLS = "00-cosmogrid/density/cosmogrid_density_nside2048_shell_*.parquet"  # one parquet per shell

# n(z) "effective end" = last z where n(z) >= THRESH_FRAC * peak. The DES Y3 bins carry a thin
# (~0.5-1% of peak) high-z noise floor, so a 1-2% cut clips to the grid edge and inverts the bin
# order; 10% of peak is robust, monotonic in bin number, and lands bin 3 at z~1.06 (the user's
# visual "bin 3 reaches z~1"). Tails past this carry <~few% of the sources.
THRESH_FRAC = 0.10

OBS_FULL = (0.5, 0.5, 0.5)  # full-sky observer (box centre) -> isotropic 2r cube
# Big-quadrant observer: EXACTLY the Experiment 08 corner geometry (one centred axis, two corner axes),
# so the lightcone visibility footprint is identical to Exp 08's and a later masking analysis recovers the
# same mask. The factors 1 + 2*min(f, 1-f) = (1.2, 2.0, 1.2) give box (1.2r, 2.0r, 1.2r): the long
# (centred) axis is SECOND, so the short sharded axis is first. We therefore size the quadrant mesh so it
# is valid under BOTH the slab and the pencil device grid (see quadrant_mesh). Earlier revisions
# X/Y-reoriented this to (0.5,0.1,0.9) to keep the long axis first; that reflected the footprint across
# x=y and no longer matched Exp 08 — fixed here.
OBS_QUAD = (0.1, 0.5, 0.9)
MESH_FULL = 2048  # full-sky cubic mesh side, float64 on 128 GPUs
N_GPUS = 128  # h100: 4 GPU/node -> 32 nodes
GPUS_PER_NODE = 4
BOX_STEP = 200  # round 2r up to a multiple of this for tidy box sides (4200, 5000)
PDIM_SLAB = (128, 1)  # 1-D slab decomposition (shard the first axis)
PDIM_PENCIL = (32, 4)  # 2-D pencil decomposition (shard the first two axes)
PDIMS = (PDIM_SLAB, PDIM_PENCIL)  # the meshes must be valid under both

ZGRID = np.linspace(0.005, 2.995, 600)  # DES Y3 n(z) support


# --------------------------------------------------------------------------------------------------
# 1. DES Y3 n(z): per-bin effective end + mean, and a couple of CDF percentiles for transparency
# --------------------------------------------------------------------------------------------------
def _z_at_frac(v, frac):
    """Last z where n(z) >= frac * peak (the effective high-z end of the distribution)."""
    idx = np.where(v >= frac * float(v.max()))[0]
    return float(ZGRID[idx[-1]]) if len(idx) else 0.0


def nz_summary(nz_list):
    rows = []
    vals_all = []
    for i, nz in enumerate(nz_list):
        v = np.asarray(nz(jnp.asarray(ZGRID)))
        vals_all.append(v)
        z_mean = float(np.trapezoid(ZGRID * v, ZGRID) / np.trapezoid(v, ZGRID))
        rows.append(
            dict(
                bin=i + 1,
                z_mean=z_mean,
                z_end=_z_at_frac(v, THRESH_FRAC),  # operative box depth
                z_end5=_z_at_frac(v, 0.05),  # sensitivity bracket (looser cut)
                z_end20=_z_at_frac(v, 0.20),  # sensitivity bracket (tighter cut)
            )
        )
    return rows, np.array(vals_all)


# --------------------------------------------------------------------------------------------------
# 2. CosmoGrid shells from HuggingFace, downsampled to nside 4 -> per-shell scale factors / edges
# --------------------------------------------------------------------------------------------------
def load_cosmogrid_shells():
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    # One parquet per shell under ``00-cosmogrid/density/``. We only need each shell's comoving edges +
    # scale factor (its metadata), so we read those two scalars per shell and free the map immediately —
    # no downsampling, no field concatenation (each shell's field carries a distinct ``name``, which would
    # break a tree.map concat anyway).
    root = snapshot_download(REPO, repo_type="dataset", allow_patterns=DENSITY_SHELLS)
    files = sorted(Path(root).glob(DENSITY_SHELLS))
    if not files:
        raise RuntimeError(f"no CosmoGrid density shells matched {DENSITY_SHELLS} on {REPO}")
    print(f"[shells] reading per-shell metadata from {len(files)} CosmoGrid density shells")
    com, width, cosmo = [], [], None
    for fp in files:
        cat = jfli.io.Catalog.from_dataset(
            load_dataset("parquet", data_files=str(fp), split="train").with_format("numpy")
        )
        f = cat.field[0]
        com.append(float(np.asarray(f.comoving_centers).ravel()[0]))
        width.append(float(np.asarray(f.density_width).ravel()[0]))
        cosmo = cosmo or cat.cosmology[0]
        del cat, f
        gc.collect()
    com, width = np.array(com), np.array(width)
    near, far = com - width / 2.0, com + width / 2.0
    order = np.argsort(com)  # ascending comoving distance
    com, near, far = com[order], near[order], far[order]
    a_near = np.asarray(jc.background.a_of_chi(cosmo, jnp.asarray(near)))
    a_far = np.asarray(jc.background.a_of_chi(cosmo, jnp.asarray(far)))
    z_near, z_far = 1.0 / a_near - 1.0, 1.0 / a_far - 1.0
    return cosmo, dict(com=com, near=near, far=far, a_near=a_near, a_far=a_far, z_near=z_near, z_far=z_far)


# --------------------------------------------------------------------------------------------------
# 3. Box + mesh + GPU sizing
# --------------------------------------------------------------------------------------------------
def r_of_z(cosmo, z):
    return float(jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(float(z))).squeeze())


def _axis_ok(n, p, *, hm=0.5, detail=False):
    """A mesh axis of size ``n`` sharded over ``p`` processes is valid iff ``n/p`` is an integer.

    Odd halos (``int((n/p)*hm)``) are now fine: jaxpm "allow odd halo extent" (higher-order-paint
    >= b56d7e9) fixed the ``slice_unpad_impl`` off-by-one (``x[-halo//2:]`` was parsed as
    ``x[(-halo)//2:]``). BEFORE that commit the halo also had to be EVEN — re-add ``ok and halo % 2
    == 0`` below if pinned to an older jaxpm. Returns the bool, or ``(bool, halo)`` if ``detail``."""
    ok = n % p == 0
    halo = int((n // p) * hm) if ok else -1
    return (ok, halo) if detail else ok


def _round_valid(target, procs):
    """Nearest mesh-axis size to ``target`` that is valid (integer local under every process count in
    ``procs``) — searched over multiples of ``lcm(procs)`` so divisibility is guaranteed."""
    procs = sorted({int(p) for p in procs})
    step = lcm(*procs) if len(procs) > 1 else procs[0]
    base = max(step, int(round(target / step)) * step)
    for k in range(1024):
        for cand in (base + k * step, base - k * step):
            if cand >= step and all(_axis_ok(cand, p) for p in procs):
                return cand
    raise RuntimeError(f"no valid mesh axis near {target:.0f} for procs {procs}")


def quadrant_mesh(box_quad, budget_side, pdims):
    """Mesh ∝ box_quad with isotropic dx and ~``budget_side**3`` cells, valid under EVERY process grid in
    ``pdims`` so the SAME mesh runs as both a 1-D slab and a 2-D pencil. ``box_quad`` has its longest
    (centred, factor-2.0) axis SECOND (observer (0.1,0.5,0.9)): axis 0 is sharded over pdim_x, axis 1 over
    pdim_y, axis 2 is nominally replicated. But the 2-D pencil FFT all-to-all transposes EVERY axis across
    both process-grid dimensions, so ALL THREE axes are rounded valid under EVERY pdim factor (not just
    their initial owner) — each must be divisible by both pdim_x and pdim_y, else jaxpm raises
    'all_to_all split_axis (n) has to be divisible by the size of the named axis x'. With pdims
    {(128,1),(32,4)} that lcm is 128, so all three become multiples of 128 (e.g. n1 3600 -> 3584)."""
    box = np.asarray(box_quad, dtype=float)
    assert box[1] == box.max(), "box_quad must have its longest (centred) axis second"
    f = box / box.max()  # (s, 1.0, s), s = 0.6
    long = (budget_side**3 / float(np.prod(f))) ** (1.0 / 3.0)  # cells along the long (2nd) axis
    all_p = [p for pd in pdims for p in pd]  # every process count an axis is (re)sharded over across transposes
    n0 = _round_valid(long * f[0], all_p)  # short axis 0
    n1 = _round_valid(long, all_p)  # long axis (2nd)
    n2 = _round_valid(long * f[2], all_p)  # short axis 2 (also transposed across the pencil FFT)
    return (n0, n1, n2)


def select_shells(shells, r_max):
    """Shells fully inside the box (far edge <= r_max)."""
    m = shells["far"] <= r_max + 1e-6
    return m


def fmt(a):
    return " ".join(f"{x:.6f}" for x in a)


# --------------------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------------------
def make_figure(cosmo, nz_list, rows, nz_vals, shells, sets):
    set_style()
    cmap = plt.get_cmap("YlOrRd")
    colors = [cmap(x) for x in np.linspace(0.35, 0.95, len(nz_list))]
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, sharex=True, figsize=(8.2, 8.4), gridspec_kw={"height_ratios": [1.0, 0.8, 0.55], "hspace": 0.08}
    )

    # --- n(z) ---
    for i, v in enumerate(nz_vals):
        ax1.plot(ZGRID, v, color=colors[i], lw=2, label=f"Bin {i + 1} (z_mean={rows[i]['z_mean']:.2f})")
        ax1.axvline(rows[i]["z_end"], color=colors[i], lw=1, ls=":")
    ax1.set_ylabel(r"$n(z)$")
    ax1.set_ylim(bottom=0.0)
    ax1.legend(frameon=False, loc="upper right")

    def _z2chi(z):
        return np.asarray(jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(np.atleast_1d(z))))

    def _chi2z(chi):
        return np.asarray(jc.utils.a2z(jc.background.a_of_chi(cosmo, np.atleast_1d(chi))))

    secax = ax1.secondary_xaxis("top", functions=(_z2chi, _chi2z))
    secax.set_xlabel(r"Comoving distance $\chi$ [$h^{-1}\mathrm{Mpc}$]")

    # --- q(z) lensing efficiency ---
    qz = np.asarray(jc.probes.WeakLensing(nz_list).kernel(cosmo, jnp.asarray(ZGRID), 1000.0))
    for i in range(len(nz_list)):
        ax2.plot(ZGRID, qz[i], color=colors[i], lw=2)
    ax2.set_ylabel(r"$q(z)$")
    ax2.set_ylim(bottom=0.0)

    # --- CosmoGrid shell tessellation + box depths ---
    in2 = select_shells(shells, sets["2bin"]["r"])
    in3 = select_shells(shells, sets["3bin"]["r"])
    for j in range(len(shells["z_near"])):
        if in2[j]:
            c = "#2c7fb8"
        elif in3[j]:
            c = "#f0a830"
        else:
            c = "0.85"
        ax3.axvspan(shells["z_near"][j], shells["z_far"][j], 0.15, 0.85, facecolor=c, edgecolor="white", lw=0.4)
    ax3.set_yticks([])
    ax3.set_ylabel("CosmoGrid\nshells", rotation=0, ha="right", va="center")
    ax3.set_xlabel(r"Redshift $z$")
    ax3.set_xlim(0.0, 1.9)
    shell_leg = [
        Patch(facecolor="#2c7fb8", label=f"in 2-bin set ({sets['2bin']['nsel']})"),
        Patch(facecolor="#f0a830", label=f"added for 3-bin set ({sets['3bin']['nsel'] - sets['2bin']['nsel']})"),
        Patch(facecolor="0.85", label="excluded"),
    ]
    ax3.legend(handles=shell_leg, frameon=True, framealpha=0.9, loc="upper center", ncol=3, fontsize=8)

    # z_max cut lines on all panels
    for ax in (ax1, ax2, ax3):
        ax.axvline(sets["2bin"]["z_max"], color="#2c7fb8", lw=1.6, ls="--")
        ax.axvline(sets["3bin"]["z_max"], color="#d95f0e", lw=1.6, ls="--")

    leg = [
        Line2D([0], [0], color="#2c7fb8", ls="--", lw=1.6, label=f"2-bin depth z={sets['2bin']['z_max']:.2f}"),
        Line2D([0], [0], color="#d95f0e", ls="--", lw=1.6, label=f"3-bin depth z={sets['3bin']['z_max']:.2f}"),
        Line2D([0], [0], color="k", ls=":", lw=1, label=f"per-bin n(z) end ({int(THRESH_FRAC * 100)}% of peak)"),
    ]
    ax2.legend(handles=leg, frameon=False, loc="upper right")
    savefig(ASSETS / "exp06-geometry", fig)


def make_mask_figure():
    """The big-quadrant visibility footprint (``jaxpm.spherical.spherical_visibility_mask``) at the
    corrected observer — documents that it IS the Exp 08 corner geometry (a centred cap, not the
    X/Y-reflected one), so a later masking analysis recovers the same mask. ``R_min`` is set to a
    representative deepest-shell fraction to expose the cap (the orientation, not the exact size, is the
    point); the full-sky observer sees the whole sphere."""
    import healpy as hp
    from jaxpm.spherical import spherical_visibility_mask

    nside = 128
    mask = np.asarray(spherical_visibility_mask(nside, jnp.asarray(OBS_QUAD), box_size=1.0, R_min=0.45, R_max=np.inf))
    set_style()
    fig = plt.figure(figsize=(6.0, 4.4))
    hp.mollview(
        mask,
        fig=fig,
        cmap="viridis",
        cbar=False,
        bgcolor=(0.0,) * 4,
        title="",
    )
    savefig(ASSETS / "exp06-mask", fig)


# --------------------------------------------------------------------------------------------------
def main():
    nz_list = get_des_y3_nz_shear()
    rows, nz_vals = nz_summary(nz_list)
    print("\nDES Y3 source bins (n(z) on z in [0.005, 2.995]); z_end = last z with n(z) >= 10% of peak:")
    print(f"  {'bin':>3} {'z_mean':>7} {'z_end':>7} {'(5%)':>7} {'(20%)':>7}")
    for r in rows:
        print(f"  {r['bin']:>3} {r['z_mean']:>7.3f} {r['z_end']:>7.3f} {r['z_end5']:>7.3f} {r['z_end20']:>7.3f}")

    cosmo, shells = load_cosmogrid_shells()
    nshell = len(shells["z_near"])
    print(
        f"\n[shells] {nshell} shells, z in [{shells['z_near'].min():.3f}, {shells['z_far'].max():.3f}], "
        f"comoving in [{shells['near'].min():.1f}, {shells['far'].max():.1f}] Mpc/h"
    )

    # 2-bin set -> end(bin2), 3-bin set -> end(bin3); bin 4 reported for reference only
    z2, z3, z4 = rows[1]["z_end"], rows[2]["z_end"], rows[3]["z_end"]
    sets = {
        "2bin": dict(z_max=z2, r=r_of_z(cosmo, z2)),
        "3bin": dict(z_max=z3, r=r_of_z(cosmo, z3)),
    }
    r4 = r_of_z(cosmo, z4)
    print(
        f"\n[depth] bin4 end z={z4:.3f}  r={r4:.0f} Mpc/h  full-sky box {2 * r4:.0f}^3 -> dx={2 * r4 / MESH_FULL:.2f} "
        f"Mpc/h (deeper + coarser at fixed {MESH_FULL}^3) -> excluded by the 2-/3-bin design"
    )

    print("\n=== 2x2 simulation geometry ===")
    for key in ("2bin", "3bin"):
        z_max, r = sets[key]["z_max"], sets[key]["r"]
        m = select_shells(shells, r)
        ts_near, ts_far = shells["a_near"][m], shells["a_far"][m]
        sets[key]["ts_near"], sets[key]["ts_far"], sets[key]["nsel"] = ts_near, ts_far, int(m.sum())

        # Round 2r up to a tidy multiple of BOX_STEP that still contains the shells; the quadrant keeps the
        # observer's (1.2, 2.0, 1.2) factor ratio off the SAME rounded full-sky side L (long axis SECOND).
        # The natural box is jfli.compute_box_size_from_redshift(cosmo, z_max, obs); we round it for tidiness.
        L = int(np.ceil(2.0 * r / BOX_STEP) * BOX_STEP)
        assert L / 2.0 >= r, f"rounded box side {L} too small for r={r:.0f}"
        box_full = np.array([L, L, L], dtype=float)
        box_quad = np.array([0.6 * L, float(L), 0.6 * L], dtype=float)  # (1.2r', 2r', 1.2r'), r'=L/2
        dx_full = box_full[0] / MESH_FULL  # full-sky dx (cubic mesh)
        mesh_quad = quadrant_mesh(box_quad, MESH_FULL, PDIMS)  # same cell budget, valid for slab+pencil
        dx_quad = box_quad / np.asarray(mesh_quad)  # per axis (isotropic by construction)
        sets[key].update(box_full=box_full, box_quad=box_quad, dx_full=dx_full, dx_quad=dx_quad, mesh_quad=mesh_quad)

        # Per-shell mesh resolution: a mode at the Nyquist wavelength 2*dx at comoving distance d subtends
        # multipole l_max ~ pi*d/dx. The geometry comparison is mesh-limited above this; near shells bind it.
        d_sel = shells["com"][m]
        ell_full = np.pi * d_sel / dx_full
        ell_quad = np.pi * d_sel / dx_quad[0]
        sets[key]["ell_full"] = (float(ell_full.min()), float(ell_full.max()))
        sets[key]["ell_quad"] = (float(ell_quad.min()), float(ell_quad.max()))

        print(f"\n  [{key}]  z_max={z_max:.3f}  r={r:.0f} Mpc/h  shells_selected={int(m.sum())}/{nshell}")
        print(f"     full sky  : box {box_full[0]:.0f}^3   mesh {MESH_FULL}^3   dx={dx_full:.3f} Mpc/h   {N_GPUS} GPUs")
        print(
            f"     quadrant  : box ({box_quad[0]:.0f}, {box_quad[1]:.0f}, {box_quad[2]:.0f})  mesh {mesh_quad}  "
            f"dx=({dx_quad[0]:.3f}, {dx_quad[1]:.3f}, {dx_quad[2]:.3f})   {N_GPUS} GPUs"
        )
        # halo / divisibility sanity: EVERY (mesh, pdim) pairing run.sh launches must give an even halo.
        for mname, mesh in (("full", (MESH_FULL,) * 3), ("quad", tuple(mesh_quad))):
            for pd in PDIMS:
                ok0, h0 = _axis_ok(mesh[0], pd[0], detail=True)
                ok1, h1 = _axis_ok(mesh[1], pd[1], detail=True)
                tag = "slab" if pd == PDIM_SLAB else "pencil"
                print(f"     {mname:>4} {tag:>6} pdim={pd}: halo=({h0},{h1})  {'OK' if ok0 and ok1 else 'BAD'}")
                assert ok0 and ok1, f"{mname} mesh {mesh} invalid under pdim {pd}"
        cells_full = MESH_FULL**3 / N_GPUS
        cells_quad = float(np.prod(mesh_quad)) / N_GPUS
        print(f"     per-GPU cells ({N_GPUS}): full {cells_full:.2e}  quad {cells_quad:.2e}  (float64 ceiling ~1.34e8)")
        print(
            f"     mesh l_max ~ pi*d/dx (innermost..outermost shell): full {ell_full.min():.0f}..{ell_full.max():.0f}"
            f"   quad {ell_quad.min():.0f}..{ell_quad.max():.0f}   (nside 2048 carries l~6000)"
        )

    # geometry.sh — every value run.sh needs, so the four launches carry no hand-typed numbers.
    def vec(a):
        return " ".join(f"{x:.1f}" for x in a)

    mesh_quad = sets["2bin"]["mesh_quad"]  # identical for both sets (ratio is exact)
    out = HERE / "geometry.sh"
    with out.open("w") as fh:
        fh.write("# generated by prep_geometry.py — DO NOT EDIT. Sourced by run.sh.\n")
        fh.write("# 2-bin set = DES Y3 bins 1+2 ; 3-bin set = bins 1+2+3. CosmoGrid shell a-edges (a_near > a_far).\n")
        fh.write(
            f"# {N_GPUS} GPUs ({N_GPUS // GPUS_PER_NODE} nodes x {GPUS_PER_NODE}); each mesh runs as a slab AND a pencil.\n\n"
        )
        fh.write(f'MESH_FULL="{MESH_FULL} {MESH_FULL} {MESH_FULL}"\n')
        fh.write(f'MESH_QUAD="{mesh_quad[0]} {mesh_quad[1]} {mesh_quad[2]}"\n')
        fh.write(f'OBS_FULL="{" ".join(str(x) for x in OBS_FULL)}"\n')
        fh.write(f'OBS_QUAD="{" ".join(str(x) for x in OBS_QUAD)}"\n')
        fh.write(f"NODES={N_GPUS // GPUS_PER_NODE}\n")
        fh.write(f'PDIM_SLAB="{PDIM_SLAB[0]} {PDIM_SLAB[1]}"      # 1-D slab decomposition\n')
        fh.write(f'PDIM_PENCIL="{PDIM_PENCIL[0]} {PDIM_PENCIL[1]}"   # 2-D pencil decomposition\n\n')
        for key in ("2bin", "3bin"):
            tag = key.upper()
            fh.write(f'BOX_{tag}_FULL="{vec(sets[key]["box_full"])}"\n')
            fh.write(f'BOX_{tag}_QUAD="{vec(sets[key]["box_quad"])}"\n')
            fh.write(f'TS_NEAR_{tag}="{fmt(sets[key]["ts_near"])}"\n')
            fh.write(f'TS_FAR_{tag}="{fmt(sets[key]["ts_far"])}"\n')
    print(f"\n[write] {out}")

    make_figure(cosmo, nz_list, rows, nz_vals, shells, sets)
    print(f"[write] {ASSETS / 'exp06-geometry.svg'}")
    make_mask_figure()
    print(f"[write] {ASSETS / 'exp06-mask.svg'}")


if __name__ == "__main__":
    main()
