"""Experiment 00 — density convergence across the 56 CosmoGrid shells.

For every shell of the native CosmoGrid nside-2048 density lightcone (56 shells, z ≈ 0.006 → 1.6,
streamed from HuggingFace ``00-cosmogrid/density/cosmogrid_density_nside2048_shell_NNN.parquet``) this
measures the overdensity angular C_ell (healpy anafast, ℓ_max = 2000) and compares it to the analytic
``comoving_volume`` Limber number-counts theory (``jax_fli.compute_theory_cl_for_density``) — the model
shown correct in experiment 01 (the legacy ``tophat_z`` weighting is biased; not used here).

The figures plot, per shell, the (2ℓ+1)-weighted measured/theory ratio in a narrow band centred on a
target multipole (the "convergence point"), as a function of the shell's comoving distance χ. Three
targets (ℓ = 200, 300, 400), rendered twice: against the raw continuous-sky theory and against the
pixel-window-matched theory (× pixwin²(2048)). At these scales pixwin²(2048) ≳ 0.996, so the two are
nearly identical — i.e. the HEALPix pixel window is negligible below ℓ ≈ 400 at nside 2048.

Streaming one shell at a time keeps peak RAM at ~one map (the full stack is ~11 GB). The measured and
theory spectra are cached to a local ``.npz`` so figure tweaks re-run instantly.

Run from the repo root (CPU is fine; first run ≈ 6–8 min incl. HF downloads, then cached):
    JAX_PLATFORMS=cpu uv run --no-sync python \
        docs/5-experiments/00-cosmogrid-reference/density_convergence.py
"""

from __future__ import annotations

import jax

# float64 globally BEFORE jax_cosmo (the enable_x64() context manager breaks jax_cosmo's
# pure_callback comoving-distance cache; the global config flag is the safe route). Also required for
# the thin z≈0.006 near shell, where the float32 theory goes NaN.
jax.config.update("jax_enable_x64", True)

import sys
from pathlib import Path

import healpy as hp
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

import jax_fli as jfli
from jax_fli import compute_theory_cl_for_density
from jax_fli.io import Catalog

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _exputils import savefig, set_style  # noqa: E402

ASSETS = HERE / "assets"
REPO = "ASKabalan/jax-fli-experiments"
SHELLS = "00-cosmogrid/density_spectra/spectra_cosmogrid_density_nside2048.parquet"
N_SHELLS = 56
NSIDE = 2048

TARGETS = [200, 300, 400]  # convergence-point multipoles


# --------------------------------------------------------------------------------------------
# Load the measured spectra and 
# --------------------------------------------------------------------------------------------


cat = Catalog.from_dataset(load_dataset(REPO, data_files=SHELLS, split="train").with_format("numpy"))
field, cosmo = cat.field[0], cat.cosmology[0]
theory = compute_theory_cl_for_density(cosmo, field, jnp.arange(LMAX + 1), radial_weight="comoving_volume")

pixwin2 = hp.pixwin(NSIDE)[: ell.size] ** 2
theory_pw = theory * pixwin2[None, :]  # pixel-window-matched theory (same footing as the map)


@CLAUDE DO NOT CHANGE ABOVE
# --------------------------------------------------------------------------------------------
# (2ℓ+1)-weighted measured/theory ratio in a narrow band around a target ℓ, per shell, + the CV sigma.
# --------------------------------------------------------------------------------------------

@ take LMAX FROM THE PRECOMPUTED SPECTRA
def band_ratio(meas_2d, theory_2d, lo, hi):
    """Per-shell (2ℓ+1)-weighted measured/theory over ℓ∈[lo, hi], plus the full-sky CV sigma."""
    sel = (ell >= lo) & (ell <= hi)
    w = 2.0 * ell[sel] + 1.0
    mb = (meas_2d[:, sel] * w).sum(axis=1) / w.sum()
    tb = (theory_2d[:, sel] * w).sum(axis=1) / w.sum()
    return mb / tb, float(np.sqrt(2.0 / w.sum()))


# --------------------------------------------------------------------------------------------
# One figure = a 1×3 row of convergence panels (ℓ = 200, 300, 400) for one theory footing.
# --------------------------------------------------------------------------------------------
def convergence_figure(theory_2d, title, fname):
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(5.0 * len(TARGETS), 4.6), sharey=True)
    for ax, ltarget in zip(axes, TARGETS):
        lo, hi = round(0.9 * ltarget), round(1.1 * ltarget)
        ratio, cv = band_ratio(meas, theory_2d, lo, hi)
        ax.axhspan(1 - 0.05, 1 + 0.05, color="orange", alpha=0.18, lw=0, label="±5%")
        ax.axhspan(1 - cv, 1 + cv, color="0.8", lw=0, label=f"±1σ CV ({cv:.1%})")
        ax.axhline(1.0, color="0.4", ls="--", lw=1.0)
        ax.plot(chi, ratio, "o-", color="tab:blue", ms=4, lw=1.2)
        ax.set_title(f"ℓ ≈ {ltarget}  (band {lo}–{hi})", fontsize=11)
        ax.set_xlabel(r"comoving distance $\chi$  [Mpc/$h$]")
        ax.set_ylim(0.8, 1.2)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel(r"measured / theory  ($(2\ell+1)$-weighted)")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    savefig(ASSETS / fname, fig)
    print(f"  wrote {ASSETS / fname}.svg")


def main():
    set_style()
    convergence_figure(
        theory,
        "CosmoGrid density convergence to comoving-volume Limber theory — continuous-sky theory",
        "fig01-convergence-continuous",
    )
    convergence_figure(
        theory_pw,
        r"CosmoGrid density convergence — theory × pixwin$^2$(2048) (pixel-window-matched)",
        "fig02-convergence-pixwin",
    )
    print(f"\nFigures in {ASSETS}")


if __name__ == "__main__":
    main()
