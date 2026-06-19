# ruff: noqa: E402
"""Experiment 09 — gradient correctness via finite differences (float64).

Validates the initial-condition gradient of the `jax-fli` forward model (the machinery behind
`fli-simulate --grad`) against an independent **per-voxel finite difference**. The observable is the
**scalar** loss ``L = ½ Σ observable.array²`` (exactly what `fli-simulate --grad` differentiates), where
``observable`` is a **spherical HEALPix** painting — one shell (``nb_shells=1``) or a multi-shell lightcone.

For the 16 voxels with the largest ``|∂L/∂δ|`` we compare the adjoint's gradient component ``g_i`` to a
central finite difference ``FD_i = [L(δ + ε e_i) − L(δ − ε e_i)] / 2ε`` and report the **median** relative
error ``|g_i − FD_i| / |FD_i|``. Picking high-signal voxels and taking the median keeps the FD denominator
away from zero, so this lands at the float64 central-difference floor — which is set by the FD's own
truncation (``∝ ε²·L‴``) and so is solver-dependent: BullFrog ~1e-8, DoubleKickDrift ~5e-7 (both far below 1,
both backend-independent; the gap is the FD reference, not the adjoint — ``reverse ≡ checkpointed`` exactly).

The finite difference is the looser, *independent* cross-check. The sharp, FD-free correctness proof is the
AD-vs-AD transpose test (forward-mode ``⟨w,Jv⟩`` vs the adjoint ``⟨Jᵀw,v⟩``), which holds to ~1e-12 in
float64 at any resolution and is asserted by ``tests/nbody/test_adjoints.py::test_adjoint_transpose``.

The figure is a single panel: x = {single output, lightcone}, four solver×adjoint series; reverse and
checkpointed overlap (they compute the same gradient).

    uv run python 09-gradient-validation.py     # 16³, float64, GPU
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

# --- config: modest 16³ mesh, nside matched -------------------------------------------------------
RES = (16, 16, 16)
BOX = (1000.0, 1000.0, 1000.0)  # Mpc/h
A0 = 0.001  # PM start at a=0.001 (z≈999) — the project convention (solver default t0=0.001)
A1 = 1.0  # final output at z=0
NSTEPS = 20  # 20 steps from a=0.001 to a=1.0
NSIDE = 16  # nside for spherical painting
NB_SHELLS = 4  # number of shells for the lightcone output

cosmo = jc.Planck18()

# (key, title, nb_shells) — both spherical, painted at NSIDE
CASES = [
    ("single", "single spherical output\n(nb_shells=1)", 1),
    ("lightcone", "spherical lightcone\n(nb_shells=4)", NB_SHELLS),
]
# 4 series = solver x adjoint (same colours as the sweep figures).
SERIES = [
    ("kdk", "rev", "DoubleKickDrift · reverse adjoint", "#D55E00"),
    ("bf", "rev", "BullFrog · reverse adjoint", "#E69F00"),
    ("kdk", "chk", "DoubleKickDrift · checkpointed adjoint", "#0072B2"),
    ("bf", "chk", "BullFrog · checkpointed adjoint", "#56B4E9"),
]
_ADJ = {"rev": "reverse", "chk": "checkpointed"}


def _ic():
    return jfli.gaussian_initial_conditions(jax.random.key(0), RES, BOX, cosmo=cosmo, nside=NSIDE)


def _solver(name, nb_shells):
    interp = jfli.NoInterp(painting=jfli.PaintingOptions(target="spherical", paint_nside=NSIDE, scheme="bilinear"))
    if name == "kdk":
        return jfli.DoubleKickDrift(interp_kernel=interp, gradient_order=0, t0=A0, t1=A1, n_steps=NSTEPS)
    return jfli.BullFrog(interp_kernel=interp, gradient_order=0, time_stepping="a", t0=A0, t1=A1, n_steps=NSTEPS)


def _make_obs(ic0, solver, nb_shells):
    """Scalar observable: the CLI loss ``L = ½ Σ array²``. ``adjoint=None`` selects the forward-mode
    ``jax.lax`` loop (used here only to *evaluate* L for the finite difference); 'reverse'/'checkpointed'
    select the reverse-mode adjoints (for ``grad``)."""

    def obs(x, adjoint):
        ic = ic0.replace(array=x)
        dx, p = jfli.lpt(cosmo, ic, ts=A0, order=2, gradient_order=0, dealiased=False, exact_growth=False)
        out = jfli.nbody(cosmo, dx, p, solver=solver, nb_shells=nb_shells, adjoint=adjoint, min_width=1.0)
        return 0.5 * jnp.sum(jnp.square(out.array))

    return obs


def compute_finite_diff(loss_fwd, x0, idxs, eps):
    """Per-voxel central finite differences of the scalar loss at the spot-check voxels ``idxs``:
    ``∂L/∂δ_i ≈ [L(δ + ε e_i) − L(δ − ε e_i)] / 2ε``. ``loss_fwd(x)`` evaluates the scalar loss (forward
    value). Returns the length-``len(idxs)`` finite-difference vector."""
    fd = np.empty(len(idxs))
    for i, ix in enumerate(idxs):
        lp = float(loss_fwd(x0.at[ix].add(eps)))
        lm = float(loss_fwd(x0.at[ix].add(-eps)))
        fd[i] = (lp - lm) / (2 * eps)
    return fd


def _grad(obs, x0, adjoint):
    """∇L for one reverse-mode adjoint (cotangent 1.0); returns the IC-shaped gradient as a numpy array."""
    return np.asarray(jax.jit(jax.grad(lambda x: obs(x, _ADJ[adjoint])))(x0))


def _compute():
    """Per-voxel FD-vs-adjoint median rel-error for every (case, solver, adjoint); cache to data_f64/."""
    out = DATA / "grad_validation.npz"
    if out.exists():
        return
    ic0 = _ic()
    x0 = ic0.array
    eps = float(np.finfo(x0.dtype).eps) ** (1.0 / 3.0)  # FD step ~ machine_eps**(1/3)
    payload = {"__eps": np.array(eps)}
    configs = [(key, name, nb) for key, _title, nb in CASES for name in ("kdk", "bf")]
    _log(f"computing {len(configs)} (case × solver) configs (eps={eps:.2e}) ...")
    bar = tqdm(total=len(configs), desc="09 per-voxel FD")
    for key, name, nb in configs:
        obs = _make_obs(ic0, _solver(name, nb), nb)
        loss_fwd = jax.jit(lambda x, o=obs: o(x, None))  # forward-only eval; jit so the FD calls are cheap

        # Reverse gradient defines the spot-check voxels (top-16 |grad|, denominator-safe).
        g_rev = _grad(obs, x0, "rev")
        flat = np.argsort(np.abs(g_rev).ravel())[-16:]
        idxs = list(zip(*np.unravel_index(flat, g_rev.shape)))
        fd_vec = compute_finite_diff(loss_fwd, x0, idxs, eps)  # forward FD reference (shared by adjoints)

        for _sv, adj, _lab, _c in [s for s in SERIES if s[0] == name]:
            g = g_rev if adj == "rev" else _grad(obs, x0, adj)
            comp_rel = np.array(
                [abs(float(g[ix]) - fd_vec[i]) / (abs(fd_vec[i]) + 1e-300) for i, ix in enumerate(idxs)]
            )
            med, mx = float(np.median(comp_rel)), float(np.max(comp_rel))
            payload[f"{key}__{name}__{adj}__med"] = np.array(med)
            payload[f"{key}__{name}__{adj}__max"] = np.array(mx)
            payload[f"{key}__{name}__{adj}__fd"] = fd_vec  # cache raw values so replotting needs no re-run
            payload[f"{key}__{name}__{adj}__g"] = np.array([float(g[ix]) for ix in idxs])
            _log(f"{key:9s} {name}·{adj}: per-voxel FD rel median={med:.2e} max={mx:.2e}")
        bar.update(1)
        bar.set_postfix_str(f"{key} {name}")
    bar.close()
    np.savez(out, **payload)
    _log(f"cached → {out.name}")


def _plot():
    path = DATA / "grad_validation.npz"
    if not path.exists():
        print("Need data_f64/grad_validation.npz — run _compute first.")
        return
    d = np.load(path)

    set_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    YLO, YHI = 1e-9, 1e-6  # float64 per-voxel FD floor (~1e-8) with headroom

    xpos = {key: i for i, (key, _t, _nb) in enumerate(CASES)}
    for s, (name, adj, _lab, color) in enumerate(SERIES):
        off = (s - (len(SERIES) - 1) / 2) * 0.16
        for key, _title, _nb in CASES:
            med = float(d[f"{key}__{name}__{adj}__med"])
            ax.plot(
                xpos[key] + off,
                min(max(med, YLO), YHI),
                marker="o" if adj == "chk" else "D",
                ms=9,
                color=color,
                mec="0.2",
                mew=0.5,
                zorder=5,
            )
    ax.set_yscale("log")
    ax.set_ylim(YLO, YHI)
    ax.set_xlim(-0.5, len(CASES) - 0.5)
    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels([t for _k, t, _nb in CASES])
    ax.set_ylabel("median |g_i − FD_i| / |FD_i|   (top-16 |grad| voxels)")
    ax.grid(True, which="major", axis="y", ls=":", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    series_handles = [
        Line2D([0], [0], marker="D" if adj == "rev" else "o", ls="none", color=c, mec="0.2", mew=0.5, label=lab)
        for _n, adj, lab, c in SERIES
    ]
    fig.legend(
        handles=series_handles, loc="upper center", ncol=2, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 1.0)
    )
    fig.suptitle(
        f"Exp 09 — IC-gradient adjoints vs per-voxel finite differences (float64, {RES[0]}³)", y=1.08, fontsize=12
    )
    caption = (
        "Reverse-mode adjoint gradient vs central per-voxel finite differences of the scalar loss "
        "L=½Σ array² (median over the 16 largest-|grad| voxels, ε=ε_machine^⅓). reverse and checkpointed "
        "overlap — they compute the same gradient. The sharp FD-free proof is the AD-vs-AD transpose test "
        "(~1e-12; tests/nbody/test_adjoints.py::test_adjoint_transpose); see the README."
    )
    fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=8, wrap=True, color="0.25")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig(HERE / "assets" / "fig01-transpose-test", fig)
    print(f"\nSaved figure → {HERE / 'assets' / 'fig01-transpose-test.svg'}")


def main():
    _compute()
    _plot()


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    main()
