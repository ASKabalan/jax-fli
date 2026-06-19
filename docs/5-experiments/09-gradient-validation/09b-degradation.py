# ruff: noqa: E402
"""Experiment 09 (gradient stability) — accuracy *and* memory of the two adjoints across three knobs (float64).

Companion to ``09-gradient-validation.py``. Each figure is a **single panel** with a twin y-axis showing the
two things you trade off at every setting of one control:

  - **accuracy** — markers, left log axis — the per-voxel finite-difference check: for the 16 largest-|grad|
    voxels, the median relative error ``|g_i − FD_i| / |FD_i|`` between the adjoint gradient ``g`` and a
    central finite difference of the scalar loss ``L = ½ Σ array²``. Across the **step-checkpoint** count it
    is dead flat (fig03) — that knob is a pure memory↔compute trade and never touches the gradient; across the
    **integration-step** count it creeps up (fig02), since more steps is a genuinely different, more nonlinear
    integration, not a memory trade. reverse and checkpointed always coincide (same gradient).
  - **memory** — bars, right axis — the XLA scratch buffer ``temp_size_in_bytes`` of the compiled
    reverse/checkpointed gradient.

Four series per x = solver x adjoint (DoubleKickDrift / BullFrog) x (reverse / checkpointed). A reverse
gradient that blows up (NaN) still ran and used memory, so its bar is drawn and its accuracy marked with an
× at the top — never silently dropped.

Runs at **16³** on GPU. Memory is the XLA temp buffer (``temp_size_in_bytes``); on GPU this fully captures
the FFT scratch (verified: reverse temp scales ~mesh³, 6.95× from 16³→32³), but cuFFT is leaner in absolute
terms than a CPU run, so the numbers are backend-specific — the *trend* is the point, and production-scale
magnitudes are Exp 10. The per-voxel finite difference is 32 forward passes/config.

    uv run python 09b-degradation.py     # 16³, float64, GPU
"""

import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # float64 throughout — finite differences need it

import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm.auto import tqdm

import jax_fli as jfli

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _exputils import savefig, set_style

print(f"jax-fli {jfli.__version__}  with devices {jax.devices()}  (jax {jax.__version__})")


def _log(msg):
    """Timestamped milestone line, via tqdm.write so it doesn't clobber the progress bar."""
    tqdm.write(f"[{time.strftime('%H:%M:%S')}] {msg}")


HERE = Path(__file__).resolve().parent
DATA = HERE / "data_f64"
cosmo = jc.Planck18()

# Config — 16³ mesh, nside matched. (fig02 steps) + (fig03 checkpoints): single output.
RES = (16, 16, 16)  # 16³ on GPU — accuracy + temp-scaling illustration (production magnitudes → Exp 10)
BOX = 1000.0
NSIDE = 16
A0 = 0.001
A1 = 1.0
STEPS = [5, 10, 15, 20, 30, 50, 80]
CKPT_STEPS = 50
CKPTS = [1, 2, 5, 10, 20, 30, 50]
# (fig04 shells) — spherical lightcone. SHELLS runs to 64; nb_shells ≤ n_steps is required (a shell needs
# ≥1 step), so this sweep uses 80 steps to admit 64 shells.
NSTEPS_SH = 80
SHELLS = [4, 8, 16, 32, 64]

SOLVER_FULL = {"kdk": "DoubleKickDrift", "bf": "BullFrog"}
# 4 series = solver x adjoint. reverse = warm, checkpointed = cool; DoubleKickDrift saturated, BullFrog light.
SERIES = [
    ("kdk", "rev", "DoubleKickDrift · reverse adjoint", "#D55E00"),  # vermillion
    ("bf", "rev", "BullFrog · reverse adjoint", "#E69F00"),  # orange
    ("kdk", "chk", "DoubleKickDrift · checkpointed adjoint", "#0072B2"),  # blue
    ("bf", "chk", "BullFrog · checkpointed adjoint", "#56B4E9"),  # sky blue
]
_ADJ = {"rev": "reverse", "chk": "checkpointed"}

# Accuracy = per-voxel FD median rel error; in float64 it sits at the central-difference floor (~1e-8).
ACC_LO, ACC_HI = 1e-9, 1e-6
ACC_LABEL = "median |g_i − FD_i| / |FD_i|   (top-16 |grad| voxels)"


def _ic(res, box, nside):
    return jfli.gaussian_initial_conditions(
        jax.random.key(0), res, (box, box, box), cosmo=cosmo, nside=nside
    )


def _solver(name, n_steps, nside):
    interp = jfli.NoInterp(painting=jfli.PaintingOptions(target="spherical", paint_nside=nside, scheme="bilinear"))
    if name == "kdk":
        return jfli.DoubleKickDrift(interp_kernel=interp, gradient_order=0, t0=A0, t1=A1, n_steps=n_steps)
    # BullFrog needs time_stepping="a" for the reversible (reverse) adjoint.
    return jfli.BullFrog(interp_kernel=interp, gradient_order=0, time_stepping="a", t0=A0, t1=A1, n_steps=n_steps)


def _make_obs(ic0, solver, nb_shells):
    """Scalar observable: the CLI loss ``L = ½ Σ array²``. ``adjoint=None`` is the forward ``jax.lax``
    loop (to *evaluate* L for the finite difference); 'reverse'/'checkpointed' are the reverse adjoints."""

    def obs(x, adjoint, step_checkpoints=None):
        ic = ic0.replace(array=x)
        dx, p = jfli.lpt(cosmo, ic, ts=A0, order=2, gradient_order=0, dealiased=False, exact_growth=False)
        out = jfli.nbody(
            cosmo,
            dx,
            p,
            solver=solver,
            nb_shells=nb_shells,
            adjoint=adjoint,
            step_checkpoints=step_checkpoints,
            min_width=1.0,
        )
        return 0.5 * jnp.sum(jnp.square(out.array))

    return obs


def compute_finite_diff(loss_fwd, x0, idxs, eps):
    """Per-voxel central finite differences of the scalar loss at the spot-check voxels ``idxs``:
    ``∂L/∂δ_i ≈ [L(δ + ε e_i) − L(δ − ε e_i)] / 2ε``. Returns the length-``len(idxs)`` finite-diff vector."""
    fd = np.empty(len(idxs))
    for i, ix in enumerate(idxs):
        lp = float(loss_fwd(x0.at[ix].add(eps)))
        lm = float(loss_fwd(x0.at[ix].add(-eps)))
        fd[i] = (lp - lm) / (2 * eps)
    return fd


def _compiled_grad(obs, x0, adjoint, step_checkpoints):
    """Compile ∇L once for one adjoint; return (g, temp_bytes). The same compiled executable yields both
    the gradient (for accuracy) and its scratch memory. A blown-up backsolve returns NaN but still compiles
    and reports memory."""

    def grad_at(x):
        return jax.grad(lambda xx: obs(xx, adjoint, step_checkpoints))(x)

    comp = jax.jit(grad_at).lower(x0).compile()
    g = np.asarray(comp(x0))
    analysis = comp.memory_analysis()  # Optional[...] in the XLA API; present on CPU here
    mem = int(analysis.temp_size_in_bytes) if analysis is not None else 0
    return g, mem


def _compute():
    out = DATA / "degradation.npz"
    if out.exists():
        return
    payload = {}
    total = 2 * (len(STEPS) * 2 + (1 + len(CKPTS)) + len(SHELLS) * 2)  # 2 solvers × (steps + ckpts + shells)
    _log(f"computing {total} configs at {RES[0]}³ (each compiles a gradient — the first is slow) ...")
    bar = tqdm(total=total, desc="09b FD+mem")

    def put(key, med, mx, mem):
        payload[key + "__med"] = np.array(med)
        payload[key + "__max"] = np.array(mx)
        payload[key + "__mem"] = np.array([mem])
        bar.update(1)
        bar.set_postfix_str(f"{key}  med={med:.1e}  {mem / 1e6:.1f}MB")

    def metrics(obs, x0, eps, adjoints):
        """adjoints: list of (key, adjoint, step_checkpoints). One reverse gradient defines the spot-check
        voxels + FD reference (shared); one compiled gradient per adjoint gives accuracy + memory."""
        loss_fwd = jax.jit(lambda x: obs(x, None))
        g_rev, mem_rev = _compiled_grad(obs, x0, "reverse", None)
        flat = np.argsort(np.abs(g_rev).ravel())[-16:]
        idxs = list(zip(*np.unravel_index(flat, g_rev.shape)))
        fd_vec = compute_finite_diff(loss_fwd, x0, idxs, eps)
        for key, adjoint, sc in adjoints:
            g, mem = (g_rev, mem_rev) if (adjoint == "reverse" and sc is None) else _compiled_grad(obs, x0, adjoint, sc)
            comp_rel = np.array(
                [abs(float(g[ix]) - fd_vec[i]) / (abs(fd_vec[i]) + 1e-300) for i, ix in enumerate(idxs)]
            )
            med = float(np.median(comp_rel)) if np.all(np.isfinite(g.ravel()[flat])) else np.nan
            mx = float(np.max(comp_rel)) if np.isfinite(med) else np.nan
            put(key, med, mx, mem)
            _log(f"{key}: med={med:.2e}  mem={mem / 1e6:.1f}MB")

    # ---- fig02 (steps) + fig03 (checkpoints): single spherical output ----
    ic0 = _ic(RES, BOX, NSIDE)
    x0 = ic0.array
    eps = float(np.finfo(x0.dtype).eps) ** (1.0 / 3.0)  # FD step ~ machine_eps**(1/3)
    for name in ("kdk", "bf"):
        for s in STEPS:  # fig02
            obs = _make_obs(ic0, _solver(name, s, NSIDE), 1)
            metrics(
                obs,
                x0,
                eps,
                [
                    (f"A__{name}__s{s}__rev", "reverse", None),
                    (f"A__{name}__s{s}__chk", "checkpointed", max(1, int(np.ceil(np.log2(s))))),
                ],
            )
        _log(f"fig02 (steps) {SOLVER_FULL[name]} done")
        obs = _make_obs(ic0, _solver(name, CKPT_STEPS, NSIDE), 1)  # fig03
        ckpt_adj = [(f"B__{name}__rev", "reverse", None)] + [
            (f"B__{name}__c{k}__chk", "checkpointed", k) for k in CKPTS
        ]
        metrics(obs, x0, eps, ckpt_adj)
        _log(f"fig03 (step-checkpoints) {SOLVER_FULL[name]} done")

    # ---- fig04 (shells): spherical lightcone ----
    # checkpointed inner step_checkpoints = log2(integration steps between two consecutive shells)
    # ≈ log2(NSTEPS_SH / nb_shells):  n=4→5, 8→4, 16→3, 32→2, 64→1.  reverse is O(1) (stores no trajectory).
    for name in ("kdk", "bf"):
        for n in SHELLS:
            sc = max(1, int(np.ceil(np.log2(NSTEPS_SH / n))))
            obs = _make_obs(ic0, _solver(name, NSTEPS_SH, NSIDE), n)
            metrics(
                obs,
                x0,
                eps,
                [(f"S__{name}__n{n}__rev", "reverse", None), (f"S__{name}__n{n}__chk", "checkpointed", sc)],
            )
            _log(f"fig04 shells n={n}: checkpointed inner step_checkpoints={sc}")
        _log(f"fig04 (shells) {SOLVER_FULL[name]} done")

    bar.close()
    np.savez(out, **payload)
    _log(f"cached → {out.name}")


# --------------------------------------------------------------------------------------------------
# Plotting — single dual-axis panel (memory bars + accuracy markers) per figure.
# --------------------------------------------------------------------------------------------------


def _mb(d, key):
    k = key + "__mem"
    return float(d[k][0]) / 1e6 if k in d else np.nan


def _acc(d, key):
    k = key + "__med"
    return float(d[k]) if k in d else np.nan


def _failed(d, key):
    k = key + "__med"
    return k in d and not np.isfinite(float(d[k]))


def _panel(ax, axr, ticklabels, mem_MB, acc, fail, *, xlabel):
    pos = np.arange(len(ticklabels))
    s_n = len(SERIES)
    w = 0.8 / s_n
    cap = ACC_HI * 0.45
    for s, (_sv, _adj, _lab, color) in enumerate(SERIES):
        off = (s - (s_n - 1) / 2) * w
        xs = pos + off
        m = np.asarray(mem_MB[s], float)
        a = np.asarray(acc[s], float)
        f = np.asarray(fail[s], bool)
        ran = np.isfinite(m)
        axr.bar(xs[ran], m[ran], width=w * 0.9, color=color, alpha=0.45, edgecolor="none", zorder=1)
        good = np.isfinite(a)
        ax.plot(xs[good], np.clip(a[good], ACC_LO, ACC_HI), color=color, lw=1.0, marker="o", ms=5, zorder=5)
        if f.any():
            ax.plot(xs[f], np.full(int(f.sum()), cap), color=color, marker="x", ms=9, mew=2.2, ls="none", zorder=6)
    ax.set_yscale("log")
    ax.set_ylim(ACC_LO, ACC_HI)
    ax.set_xticks(pos)
    ax.set_xticklabels(ticklabels)
    ax.set_xlabel(xlabel)
    ax.grid(True, which="major", axis="y", ls=":", alpha=0.35)
    ax.set_axisbelow(True)


def _make_fig(ticklabels, mem, acc, fail, *, xlabel, suptitle, stem, mem_note=None):
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    axr = ax.twinx()
    _panel(ax, axr, ticklabels, mem, acc, fail, xlabel=xlabel)
    all_mem = np.concatenate([np.ravel(np.asarray(m, float)) for m in mem])
    axr.set_ylim(0, 1.12 * np.nanmax(all_mem))  # zero-based so the memory trend reads directly
    ax.set_ylabel(ACC_LABEL)
    axr.set_ylabel("XLA temp buffer (MB)")
    if mem_note:
        axr.text(0.98, 0.02, mem_note, transform=axr.transAxes, ha="right", va="bottom", fontsize=8, color="0.35")

    series_handles = [Patch(facecolor=c, alpha=0.55, label=lab) for _s, _a, lab, c in SERIES]
    enc_handles = [
        Patch(facecolor="0.6", alpha=0.55, label="bar = XLA temp buffer (memory)"),
        Line2D([0], [0], color="0.3", marker="o", ls="-", lw=1, label="marker = per-voxel FD-vs-adjoint error"),
        Line2D([0], [0], color="0.3", marker="x", ms=9, mew=2.2, ls="none", label="× = gradient NaN"),
    ]
    fig.legend(handles=series_handles, loc="upper left", bbox_to_anchor=(0.07, 0.99), ncol=2, frameon=False)
    fig.legend(handles=enc_handles, loc="upper right", bbox_to_anchor=(0.99, 0.99), ncol=1, frameon=False)
    fig.suptitle(suptitle, y=1.05, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    savefig(HERE / "assets" / stem, fig)
    print(f"Saved figure → {HERE / 'assets' / (stem + '.svg')}")


def _fig_steps(d):
    mem = [[_mb(d, f"A__{sv}__s{s}__{adj}") for s in STEPS] for sv, adj, _l, _c in SERIES]
    acc = [[_acc(d, f"A__{sv}__s{s}__{adj}") for s in STEPS] for sv, adj, _l, _c in SERIES]
    fail = [[_failed(d, f"A__{sv}__s{s}__{adj}") for s in STEPS] for sv, adj, _l, _c in SERIES]
    _make_fig(
        [str(s) for s in STEPS],
        mem,
        acc,
        fail,
        xlabel="integration steps",
        suptitle=f"Exp 09 — accuracy & memory vs number of integration steps (single spherical output, {RES[0]}³)",
        stem="fig02-steps",
    )


def _fig_checkpoints(d):
    ticks = ["rev\n(O(1))", *[str(k) for k in CKPTS]]
    nan_ck = [np.nan] * len(CKPTS)
    m_rows, a_rows, f_rows = [], [], []
    for sv, adj, _l, _c in SERIES:
        if adj == "rev":
            m_rows.append([_mb(d, f"B__{sv}__rev"), *nan_ck])
            a_rows.append([_acc(d, f"B__{sv}__rev"), *nan_ck])
            f_rows.append([_failed(d, f"B__{sv}__rev"), *[False] * len(CKPTS)])
        else:
            m_rows.append([np.nan, *[_mb(d, f"B__{sv}__c{k}__chk") for k in CKPTS]])
            a_rows.append([np.nan, *[_acc(d, f"B__{sv}__c{k}__chk") for k in CKPTS]])
            f_rows.append([False, *[_failed(d, f"B__{sv}__c{k}__chk") for k in CKPTS]])
    _make_fig(
        ticks,
        m_rows,
        a_rows,
        f_rows,
        xlabel=f"step-checkpoints stored  (rev = reverse adjoint, O(1) memory; {CKPT_STEPS} steps)",
        suptitle=f"Exp 09 — accuracy & memory vs number of step-checkpoints (single spherical output, {RES[0]}³)",
        stem="fig03-checkpoints",
    )


def _fig_shells(d):
    mem = [[_mb(d, f"S__{sv}__n{n}__{adj}") for n in SHELLS] for sv, adj, _l, _c in SERIES]
    acc = [[_acc(d, f"S__{sv}__n{n}__{adj}") for n in SHELLS] for sv, adj, _l, _c in SERIES]
    fail = [[_failed(d, f"S__{sv}__n{n}__{adj}") for n in SHELLS] for sv, adj, _l, _c in SERIES]
    _make_fig(
        [str(n) for n in SHELLS],
        mem,
        acc,
        fail,
        xlabel="# shells saved (lightcone snapshots)",
        suptitle=f"Exp 09 — accuracy & memory vs number of saved shells (spherical lightcone, {NSTEPS_SH} steps, {RES[0]}³)",
        stem="fig04-shells",
        mem_note="reverse ≈ +1 HEALPix map / shell  (npix · 8 B)",
    )


def _fmt(v):
    return "  nan" if not np.isfinite(v) else f"{v:5.0e}"


def _print_summary(d):
    print(f"\n=== fig02 steps ===   (steps = {STEPS})")
    for name in ("kdk", "bf"):
        tr = [_fmt(_acc(d, f"A__{name}__s{s}__rev")).strip() for s in STEPS]
        tc = [_fmt(_acc(d, f"A__{name}__s{s}__chk")).strip() for s in STEPS]
        print(f"  {SOLVER_FULL[name]:16}: FD-vs-adjoint rev={tr}  chk={tc}")

    print(f"\n=== fig03 checkpoints ({CKPT_STEPS} steps) ===   (checkpoints = {CKPTS})")
    for name in ("kdk", "bf"):
        tc = [_fmt(_acc(d, f"B__{name}__c{k}__chk")).strip() for k in CKPTS]
        memc = [f"{_mb(d, f'B__{name}__c{k}__chk'):.0f}" for k in CKPTS]
        print(f"  {SOLVER_FULL[name]:16}: rev MB={_mb(d, f'B__{name}__rev'):.0f}  chk acc={tc}  chk MB={memc}")

    print(f"\n=== fig04 shells ({NSTEPS_SH} steps) ===   (shells = {SHELLS})")
    for name in ("kdk", "bf"):
        memr = [f"{_mb(d, f'S__{name}__n{n}__rev'):.0f}" for n in SHELLS]
        memc = [f"{_mb(d, f'S__{name}__n{n}__chk'):.0f}" for n in SHELLS]
        print(f"  {SOLVER_FULL[name]:16}: rev MB={memr}  chk MB={memc}")


def _plot(d):
    set_style()
    _print_summary(d)
    _fig_steps(d)
    _fig_checkpoints(d)
    _fig_shells(d)


def main():
    _compute()
    path = DATA / "degradation.npz"
    if not path.exists():
        print("Need data_f64/degradation.npz to draw figures.")
        return
    _plot(np.load(path))


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    main()
