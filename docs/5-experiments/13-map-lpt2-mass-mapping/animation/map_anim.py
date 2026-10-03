"""The 1200^3 MAP reconstruction of experiment 13 (DES Y3), animated from the cache that map_data.py writes
(.cache/map_mesh_1200_DES.npz; see its docstring for the run). Needs ffmpeg:

    uv run --no-sync python map_anim.py

map_run.mp4: the truth on the left, the reconstruction on the right, for the IC projected with the lensing
efficiency (drawn on the faces of the box, the observer at its centre) and the convergence of source bins 2 and 3
(Mollweide). The reconstruction moves through the 21 saved Adam steps (0 to 400), cross-faded from one to the next
for display; the step counter and the correlation with the truth are those of the saved step. While it runs, the
box is drawn as the 16 x 2 pencils of the run's decomposition over 32 GPUs (split in x and y, full height in z),
which join again at the end.

compare_run.mp4: over the same saved steps, the cross-correlation coefficient C_l^{k^ k} / sqrt(C_l^{k^ k^} C_l^{kk})
and the power ratio C_l^{k^ k^} / C_l^{kk} of the reconstructed kappa k^ against the truth k, for both bins, each
against its joint Wiener-filter value (dotted: r_W and r_W^2), up to l = 1000 with the scale cut of the likelihood
(l_max = 700, above which the forward model has no kappa) marked; and the starlet l1 norm of kappa
in bin 3 at the five scales against the truth.

Outputs (this directory): map_run.mp4, map_run.gif, map_first.png (the entry state), map_last.png,
compare_run.mp4, compare_run.gif, compare_last.png. Each PNG is a frame of its video, drawn by the same code.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import healpy as hp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402

HERE = Path(__file__).resolve().parent
D = np.load(HERE / ".cache" / "map_mesh_1200_DES.npz")
STEPS = D["steps"]
NK = len(STEPS)
FPS = 20
BG = "#faf7f0"

# the palette and type of the defence slides this animation is drawn for
KW, INK, GREY, BLUE = "#C2560A", "#2E2E2E", "#6E6E6E", "#3B6FB6"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "axes.facecolor": "none",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
        "lines.linewidth": 2.2,
        "savefig.bbox": None,
        "savefig.transparent": False,
    }
)


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))


def timeline(hold0, split, per, gather, hold1):
    """(keyframe position, pencil gap in [0, 1]) per video frame."""
    out = [(0.0, 0.0)] * hold0
    out += [(0.0, ease(i / (split - 1))) for i in range(split)]
    for k in range(NK - 1):
        out += [(k + ease((i + 1) / per), 1.0) for i in range(per)]
    out += [(NK - 1.0, 1 - ease(i / (gather - 1))) for i in range(gather)]
    out += [(NK - 1.0, 0.0)] * hold1
    return out


def at(stack, pos):
    """Keyframe stack blended at a fractional position."""
    k = min(int(np.floor(pos)), NK - 2)
    w = pos - k
    return (1 - w) * stack[k] + w * stack[k + 1]


def shown(pos):
    return int(np.clip(np.floor(pos + 0.5), 0, NK - 1))


def encode(frames_dir, stem):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(FPS),
            "-i",
            str(frames_dir / "f_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "16",
            "-movflags",
            "+faststart",
            str(HERE / f"{stem}.mp4"),
        ],
        check=True,
    )
    palette = "fps=15,scale=1040:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(HERE / f"{stem}.mp4"),
            "-vf",
            palette,
            str(HERE / f"{stem}.gif"),
        ],
        check=True,
    )


# ------------------------------------------------------------------ the IC on the faces of the box
NSIDE = hp.npix2nside(D["proj_truth"].size)
N = 256
G = (np.arange(N) + 0.5) / N - 0.5
# the pencils of the run (--pdims 16 2) as (i, j) = (x index, y index), in painter's order: the back row first
PX, PY = 16, 2
PENCILS = [(i, j) for j in reversed(range(PY)) for i in range(PX)]
X_CUT = [slice(i * N // PX, (i + 1) * N // PX) for i in range(PX)]
Y_CUT = [slice(j * N // PY, (j + 1) * N // PY) for j in range(PY)]
SPREAD = 0.025  # gap between neighbouring pencils in x, split open


def faces(sky):
    """Sky map -> values on the outer faces of the box: top (z = 0.5), front (y = -0.5), right (x = 0.5)."""
    u, v = np.meshgrid(G, G)  # u along columns, v along rows
    half = np.full_like(u, 0.5)
    look = {"top": (u, v, half), "front": (u, -half, v), "right": (half, u, v)}
    return {k: sky[hp.vec2pix(NSIDE, x.ravel(), y.ravel(), z.ravel())].reshape(N, N) for k, (x, y, z) in look.items()}


F_TRUTH = faces(D["proj_truth"])
F_GUESS = {k: np.array([faces(p)[k] for p in D["proj"]]) for k in F_TRUTH}
LIM_P = float(np.percentile(np.abs(D["proj_truth"]), 99.8))
NORM_P = Normalize(-LIM_P, LIM_P)
DEPTH = 0.5 * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])


def P(x, y, z):
    """Cabinet projection: x to the right, z up, depth y along the diagonal at half length."""
    return np.array([x + DEPTH[0] * (y + 0.5), z + DEPTH[1] * (y + 0.5)])


def draw_face(ax, img, o, e1, e2, z, edge=1.0):
    """One face of a pencil: the image mapped affinely onto its parallelogram, or a flat cut face.
    z is the painter's order (matplotlib would otherwise put every patch above every image)."""
    po, p1, p2 = P(*o), P(*(np.add(o, e1))) - P(*o), P(*(np.add(o, e2))) - P(*o)
    corners = np.array([po, po + p1, po + p1 + p2, po + p2, po])
    if img is None:
        ax.fill(corners[:, 0], corners[:, 1], color="#b9bfca", lw=0, zorder=z)
    else:
        tr = Affine2D(np.array([[p1[0], p2[0], po[0]], [p1[1], p2[1], po[1]], [0, 0, 1]]))
        ax.imshow(
            img,
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap="magma",
            norm=NORM_P,
            interpolation="bilinear",
            transform=tr + ax.transData,
            zorder=z,
        )
    if edge > 0:
        ax.plot(corners[:, 0], corners[:, 1], color="#3b3b3b", lw=0.6, alpha=edge, zorder=z + 0.5)


def draw_box(ax, fc, gap, label_alpha):
    """The box as PX x PY pencils: pencil (i, j) moves by gap * SPREAD (i - (PX - 1) / 2) in x and the back row by
    gap * 0.4 in y, so the front row stays in place. Assembled (gap = 0) the faces are drawn whole, so no seam shows
    where the pieces meet. A cut face inside the box is drawn flat: the ray-traced projection has no meaning on a
    plane through the observer."""
    if gap == 0:
        draw_face(ax, fc["front"], (-0.5, -0.5, -0.5), (1, 0, 0), (0, 0, 1), 1, 0)
        draw_face(ax, fc["top"], (-0.5, -0.5, 0.5), (1, 0, 0), (0, 1, 0), 2, 0)
        draw_face(ax, fc["right"], (0.5, -0.5, -0.5), (0, 1, 0), (0, 0, 1), 3, 0)
    for n, (i, j) in enumerate(PENCILS if gap > 0 else []):
        x0, x1, y0 = -0.5 + i / PX, -0.5 + (i + 1) / PX, -0.5 + j / PY
        sx, sy = gap * SPREAD * (i - (PX - 1) / 2), gap * 0.4 * j
        cols, rows = X_CUT[i], Y_CUT[j]
        draw_face(
            ax,
            fc["front"][:, cols] if j == 0 else None,
            (x0 + sx, y0 + sy, -0.5),
            (1 / PX, 0, 0),
            (0, 0, 1),
            3 * n + 1,
            gap,
        )
        draw_face(ax, fc["top"][rows, cols], (x0 + sx, y0 + sy, 0.5), (1 / PX, 0, 0), (0, 1 / PY, 0), 3 * n + 2, gap)
        draw_face(
            ax,
            fc["right"][:, rows] if i == PX - 1 else None,
            (x1 + sx, y0 + sy, -0.5),
            (0, 1 / PY, 0),
            (0, 0, 1),
            3 * n + 3,
            gap,
        )
    if label_alpha > 0:  # one label for the whole decomposition, on the top face
        c = P(0.0, 0.2 * gap, 0.5)
        ax.text(
            c[0],
            c[1],
            f"{PX * PY} GPUs",
            ha="center",
            va="center",
            fontsize=11,
            color=INK,
            alpha=label_alpha,
            zorder=3 * PX * PY + 5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85 * label_alpha),
        )
    if gap < 1:  # the outline of the whole box
        for o, e1, e2 in (
            ((-0.5, -0.5, -0.5), (1, 0, 0), (0, 0, 1)),
            ((-0.5, -0.5, 0.5), (1, 0, 0), (0, 1, 0)),
            ((0.5, -0.5, -0.5), (0, 1, 0), (0, 0, 1)),
        ):
            po, p1, p2 = P(*o), P(*np.add(o, e1)) - P(*o), P(*np.add(o, e2)) - P(*o)
            c = np.array([po, po + p1, po + p1 + p2, po + p2, po])
            ax.plot(c[:, 0], c[:, 1], color="#3b3b3b", lw=0.8, alpha=1 - gap, zorder=3 * PX * PY + 4)
    ax.set_xlim(-0.8, 1.2)
    ax.set_ylim(-0.56, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")


# ------------------------------------------------------------------ kappa in Mollweide
MOLL = hp.projector.MollweideProj(xsize=420)


def mollweide(m):
    p = MOLL.projmap(np.asarray(m, dtype=np.float64), lambda x, y, z: hp.vec2pix(hp.npix2nside(np.size(m)), x, y, z))
    return np.ma.masked_invalid(np.where(np.isfinite(p), p, np.nan))


K_TRUTH = [mollweide(D["kappa_truth"][b]) for b in range(2)]
K_GUESS = np.array([[np.ma.filled(mollweide(k[b]), np.nan) for b in range(2)] for k in D["kappa"]])
NORM_K = Normalize(*np.percentile(D["kappa_truth"], [0.1, 99.9]))
MAGMA = plt.get_cmap("magma").copy()
MAGMA.set_bad(alpha=0)


def draw_moll(ax, img):
    ax.imshow(img, origin="lower", cmap=MAGMA, norm=NORM_K, interpolation="bilinear")
    t = np.linspace(0, 2 * np.pi, 400)
    h, w = img.shape
    ax.plot(w / 2 + (w / 2 - 1) * np.cos(t), h / 2 + (h / 2 - 1) * np.sin(t), color="#3b3b3b", lw=0.8)
    ax.axis("off")


def map_frame(pos, gap):
    fig = plt.figure(figsize=(10.4, 5.4), dpi=150, facecolor=BG)
    k = shown(pos)
    fig.text(0.37, 0.955, "truth", ha="center", fontsize=15, color=INK)
    fig.text(0.72, 0.955, "reconstruction", ha="center", fontsize=15, color=INK)
    fig.text(0.015, 0.955, f"Adam step {STEPS[k]:d} / {STEPS[-1]:d}", ha="left", fontsize=13, color=KW)
    fig.text(0.935, 0.955, "correlation", ha="center", fontsize=12.5, color=GREY)
    rows = [
        ("initial conditions\nprojected with\nthe lensing kernel", 0.59, 0.34),
        (r"$\kappa$, DES Y3 bin 2", 0.31, 0.25),
        (r"$\kappa$, DES Y3 bin 3", 0.03, 0.25),
    ]
    for label, y, h in rows:
        fig.text(0.015, y + h / 2, label, ha="left", va="center", fontsize=12.5, color=INK, linespacing=1.25)
    # row 1: the box
    draw_box(fig.add_axes([0.235, 0.59, 0.27, 0.34]), F_TRUTH, 0.0, 0.0)
    draw_box(fig.add_axes([0.585, 0.59, 0.27, 0.34]), {key: at(F_GUESS[key], pos) for key in F_TRUTH}, gap, gap)
    fig.text(0.9, 0.745, rf"$r = {D['r_proj'][k]:.2f}$", fontsize=13, color=INK, va="center")
    # rows 2 and 3: kappa
    for b, (_, y, h) in enumerate(rows[1:]):
        draw_moll(fig.add_axes([0.25, y, 0.24, h]), K_TRUTH[b])
        draw_moll(fig.add_axes([0.6, y, 0.24, h]), np.ma.masked_invalid(at(K_GUESS[:, b], pos)))
        fig.text(0.9, y + h / 2, rf"$r = {D['r_kappa'][k, b]:.2f}$", fontsize=13, color=INK, va="center")
    return fig


# ------------------------------------------------------------------ cross-correlation, power ratio, starlet l1
ELL = D["ell"]
ELL_MAX, ELL_SHOW = int(D["ell_max"]), 1000  # the scale cut of the likelihood, and the l range drawn
NU = D["nu"]
EDGE = np.arange(0, len(ELL) + 1, 20)  # bands of 20 multipoles, for display
ELL_B = np.array([ELL[a:b].mean() for a, b in zip(EDGE[:-1], EDGE[1:], strict=True)])
L_HI = 3 * int(D["st_nside"]) / 2 ** np.arange(5)  # the top multipole of each starlet scale
SCALES = [f"ℓ ≈ {h / 2:.0f}–{h:.0f}" for h in L_HI[:-1]] + [f"ℓ ≤ {L_HI[-1]:.0f}"]
BINS = [(KW, "bin 2"), (BLUE, "bin 3")]
L1T = D["l1_truth"][1] / 1e3
L1_TOP = 1.12 * max(L1T.max(), (D["l1"][:, 1] / 1e3).max())


def band(y):
    return np.stack([np.asarray(y)[..., a:b].mean(-1) for a, b in zip(EDGE[:-1], EDGE[1:], strict=True)], -1)


# a Wiener estimate has C^{k^ k} = C^{k^ k^} = [C (C + N)^-1 C]: its power ratio is r_W^2
WIENER_B = band(D["wiener"])
WIENER2_B = band(D["wiener"] ** 2)
RATIO = D["trans"] ** 2  # C_l^{k^ k^} / C_l^{kk}


def compare_frame(pos):
    fig = plt.figure(figsize=(10.4, 5.0), dpi=150, facecolor=BG)
    k = shown(pos)
    fig.text(0.07, 0.96, f"Adam step {STEPS[k]:d} / {STEPS[-1]:d}", ha="left", fontsize=13, color=KW)
    fig.text(0.975, 0.96, r"$\hat\kappa$: reconstruction, $\kappa$: truth", ha="right", fontsize=12, color=INK)
    coh, ratio = band(at(D["coh"], pos)), band(at(RATIO, pos))
    ax_c = fig.add_axes([0.12, 0.6, 0.35, 0.33])
    ax_t = fig.add_axes([0.625, 0.6, 0.35, 0.33])
    for b, (c, lab) in enumerate(BINS):
        ax_c.plot(ELL_B, WIENER_B[b], color=c, ls=":", lw=2.0)
        ax_c.plot(ELL_B, coh[b], color=c, lw=2.0, label=lab)
        ax_t.plot(ELL_B, WIENER2_B[b], color=c, ls=":", lw=2.0)
        ax_t.plot(ELL_B, ratio[b], color=c, lw=2.0, label=lab)
    for ax, ylab in (
        (ax_c, r"$\dfrac{C_\ell^{\hat\kappa\kappa}}{\sqrt{C_\ell^{\hat\kappa\hat\kappa}\,C_\ell^{\kappa\kappa}}}$"),
        (ax_t, r"$\dfrac{C_\ell^{\hat\kappa\hat\kappa}}{C_\ell^{\kappa\kappa}}$"),
    ):
        ax.axhline(1, color=GREY, ls="--", lw=1)
        ax.axvspan(ELL_MAX, ELL_SHOW, color="#e4e0d6", lw=0)
        ax.axvline(ELL_MAX, color=INK, ls="--", lw=1.2)
        ax.text(
            0.5 * (ELL_MAX + ELL_SHOW),
            0.5,
            f"scale cut\n$\\ell_{{\\max}} = {ELL_MAX}$",
            ha="center",
            va="center",
            fontsize=10.5,
            color=GREY,
        )
        ax.set_xlim(2, ELL_SHOW)
        ax.set_ylim(-0.1, 1.15)
        ax.set_xlabel(r"$\ell$", labelpad=0)
        ax.set_ylabel(ylab, fontsize=15, labelpad=2)
    handles = ax_c.get_legend_handles_labels()[0] + [Line2D([], [], color=GREY, ls=":", lw=2.0)]
    ax_c.legend(
        handles,
        ["bin 2", "bin 3", "joint Wiener filter"],
        loc="upper right",
        bbox_to_anchor=(0.69, 0.86),
        fontsize=10.5,
        handlelength=1.6,
        borderaxespad=0.2,
        labelspacing=0.2,
    )
    l1 = at(D["l1"][:, 1], pos) / 1e3
    for j in range(5):
        ax = fig.add_axes([0.07 + j * 0.184, 0.1, 0.16, 0.28])
        ax.plot(NU, L1T[j], color=INK, ls="--", lw=1.6, label="truth")
        ax.plot(NU, l1[j], color=BLUE, lw=2.0, label="reconstruction")
        ax.set_xlim(-5, 5)
        ax.set_ylim(0, L1_TOP)
        ax.set_xticks([-4, 0, 4])
        ax.set_title(f"scale {j + 1}, {SCALES[j]}", fontsize=11, color=INK, pad=3)
        ax.set_xlabel(r"$\nu$", labelpad=0)
        if j:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(r"$\ell_1$ norm [$10^3$]", fontsize=12)
            ax.legend(loc="upper left", fontsize=9.5, handlelength=1.3, borderaxespad=0.1, labelspacing=0.15)
    fig.text(0.07, 0.445, r"starlet $\ell_1$ norm of $\kappa$, bin 3", fontsize=13, color=INK)
    return fig


def render(make, schedule, stem):
    tmp = Path(tempfile.mkdtemp(prefix=f"{stem}_"))
    try:
        for i, args in enumerate(schedule):
            fig = make(*args)
            fig.savefig(tmp / f"f_{i:04d}.png", facecolor=BG)
            plt.close(fig)
        if stem == "map":  # the comparison video needs no still entry state
            shutil.copy(tmp / "f_0000.png", HERE / f"{stem}_first.png")
        shutil.copy(tmp / f"f_{len(schedule) - 1:04d}.png", HERE / f"{stem}_last.png")
        encode(tmp, f"{stem}_run")
    finally:
        shutil.rmtree(tmp)
    print(f"wrote {stem}_run.mp4 and .gif ({len(schedule)} frames, {len(schedule) / FPS:.1f} s) and its still frames")


render(map_frame, timeline(12, 16, 6, 16, 20), "map")
sched = [(0.0,)] * 10 + [(k + ease((i + 1) / 5),) for k in range(NK - 1) for i in range(5)] + [(NK - 1.0,)] * 30
render(compare_frame, sched, "compare")
