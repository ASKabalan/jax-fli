"""Experiment 04 — step convergence (solver × step count): figures.

Renders the SVG figures for ``docs/5-experiments/04-step-convergence/README.md`` from the per-shell
spherical angular power spectra of the four solver/stepping variants — BullFrog run in the scale factor
(``bfa``) and the growth factor (``bfd``), ``dkd`` DriftKickDrift and ``kdk`` DoubleKickDrift (both in the
scale factor) — run at ``--nb-steps ∈ {10, 20, 30, 40, 50}`` (steps 5/6 fail — the step count must exceed
the 10 shells). Everything else is fixed (2048³, nside 2048, 10 shells, float64).

Figures (-> ``assets/``):
  fig01 / fig02   BullFrog scale-factor a (bfa), shells 0–4 / 5–9: C_ell at each step count + ratio to 50-step.
  fig03 / fig04   the same for BullFrog growth-factor D (bfd).
  fig05 / fig06   the same for DriftKickDrift (dkd).
  fig07 / fig08   the same for DoubleKickDrift (kdk).
  fig09           20- vs 30-step convergence of all four variants at the near / mid / far shell:
                  C_ell + ratio to the Limber number-counts theory (× pixwin²(2048)).

Run from the repo root (CPU is fine; ~a minute, loads 20 precomputed spectra):
    JAX_PLATFORMS=cpu uv run --no-sync python docs/5-experiments/04-step-convergence/build.py
"""

from __future__ import annotations

import jax

# float64 globally BEFORE jax_cosmo (the enable_x64() context manager breaks jax_cosmo's
# pure_callback comoving-distance cache; the global config flag is the safe route).
jax.config.update("jax_enable_x64", True)

import sys
from pathlib import Path

import healpy as hp
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from huggingface_hub import snapshot_download
from matplotlib import cm
from matplotlib.lines import Line2D

from jax_fli import compute_theory_cl_for_density
from jax_fli.io import Catalog

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _exputils import savefig, set_style  # noqa: E402

ASSETS = HERE / "assets"
REPO = "ASKabalan/jax-fli-experiments"

NSIDE = 2048
STEPS = [10, 20, 30, 40, 50]  # published step counts (5/6 fail: steps must exceed the 10 shells)
# 4 solver/stepping variants: BullFrog run both ways (bfa = scale-factor a, bfd = growth-factor D),
# plus DriftKickDrift and DoubleKickDrift (both in a). bfa/bfd kept adjacent to contrast the stepping.
SOLVERS = {
    "bfa": "BullFrog, scale-factor a",
    "bfd": "BullFrog, growth-factor D",
    "dkd": "DriftKickDrift",
    "kdk": "DoubleKickDrift",
}
NEAR_SHELL, MID_SHELL, FAR_SHELL = 1, 5, 9
NLB = 16  # multipoles per bandpower bin
STEP_COLORS = {s: c for s, c in zip(STEPS, cm.viridis(np.linspace(0.0, 0.88, len(STEPS))))}
SOLVER_COLORS = {"bfa": "tab:blue", "bfd": "tab:purple", "dkd": "tab:orange", "kdk": "tab:green"}
STEP7_STYLES = {20: "-", 30: "--"}  # the comparison fig distinguishes the two step counts by line style

root = snapshot_download(REPO, repo_type="dataset", local_files_only=True)

# -----------------------------------------------------------------------------
# Explicit per-file loads — every HF parquet (4 solver/stepping variants × 5 step counts) on its own line.
# -----------------------------------------------------------------------------
bfa_s10_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/04-step-size/density_spectra/spectra_bfa_s10.parquet", split="train")
)
bfa_s20_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/04-step-size/density_spectra/spectra_bfa_s20.parquet", split="train")
)
bfa_s30_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/04-step-size/density_spectra/spectra_bfa_s30.parquet", split="train")
)
bfa_s40_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/04-step-size/density_spectra/spectra_bfa_s40.parquet", split="train")
)
bfa_s50_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/04-step-size/density_spectra/spectra_bfa_s50.parquet", split="train")
)

bfd_s10_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/04-step-size/density_spectra/spectra_bfd_s10.parquet", split="train")
)
bfd_s20_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/04-step-size/density_spectra/spectra_bfd_s20.parquet", split="train")
)
bfd_s30_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/04-step-size/density_spectra/spectra_bfd_s30.parquet", split="train")
)
bfd_s40_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/04-step-size/density_spectra/spectra_bfd_s40.parquet", split="train")
)
bfd_s50_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/04-step-size/density_spectra/spectra_bfd_s50.parquet", split="train")
)

dkd_s10_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/04-step-size/density_spectra/spectra_dkd_s10.parquet", split="train")
)
dkd_s20_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/04-step-size/density_spectra/spectra_dkd_s20.parquet", split="train")
)
dkd_s30_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/04-step-size/density_spectra/spectra_dkd_s30.parquet", split="train")
)
dkd_s40_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/04-step-size/density_spectra/spectra_dkd_s40.parquet", split="train")
)
dkd_s50_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/04-step-size/density_spectra/spectra_dkd_s50.parquet", split="train")
)

kdk_s10_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/04-step-size/density_spectra/spectra_kdk_s10.parquet", split="train")
)
kdk_s20_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/04-step-size/density_spectra/spectra_kdk_s20.parquet", split="train")
)
kdk_s30_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/04-step-size/density_spectra/spectra_kdk_s30.parquet", split="train")
)
kdk_s40_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/04-step-size/density_spectra/spectra_kdk_s40.parquet", split="train")
)
kdk_s50_cat = Catalog.from_dataset(
    load_dataset("parquet", data_files=f"{root}/04-step-size/density_spectra/spectra_kdk_s50.parquet", split="train")
)

cosmo = bfa_s50_cat.cosmology[0]  # one fiducial cosmology shared by every run

# Solver -> step -> raw C_ell PowerSpectrum, assembled by hand from the explicit loads above.
spectra = {
    "bfa": {
        10: bfa_s10_cat.field[0],
        20: bfa_s20_cat.field[0],
        30: bfa_s30_cat.field[0],
        40: bfa_s40_cat.field[0],
        50: bfa_s50_cat.field[0],
    },
    "bfd": {
        10: bfd_s10_cat.field[0],
        20: bfd_s20_cat.field[0],
        30: bfd_s30_cat.field[0],
        40: bfd_s40_cat.field[0],
        50: bfd_s50_cat.field[0],
    },
    "dkd": {
        10: dkd_s10_cat.field[0],
        20: dkd_s20_cat.field[0],
        30: dkd_s30_cat.field[0],
        40: dkd_s40_cat.field[0],
        50: dkd_s50_cat.field[0],
    },
    "kdk": {
        10: kdk_s10_cat.field[0],
        20: kdk_s20_cat.field[0],
        30: kdk_s30_cat.field[0],
        40: kdk_s40_cat.field[0],
        50: kdk_s50_cat.field[0],
    },
}

ref = spectra["bfa"][50]
LMAX = int(np.asarray(ref.wavenumber).max())
ell = jnp.arange(LMAX + 1)
ell_full = np.asarray(ref.wavenumber)
z_shells = np.asarray(ref.z_sources)
chi_shells = np.asarray(ref.comoving_centers)
width_shells = np.asarray(ref.density_width)
n_shells = np.asarray(ref.array).shape[0]

# Silently assert the shell geometry (centres + widths) matches across every solver and step count, and
# that every parquet carries the same cosmology — these are what make the per-shell comparisons valid.
for _sol in SOLVERS:
    for _st in STEPS:
        assert np.allclose(np.asarray(spectra[_sol][_st].comoving_centers), chi_shells)
        assert np.allclose(np.asarray(spectra[_sol][_st].density_width), width_shells)
for _c in (bfa_s10_cat, bfd_s50_cat, dkd_s50_cat, kdk_s50_cat):
    assert np.isclose(float(_c.cosmology[0].Omega_c), float(cosmo.Omega_c))
    assert np.isclose(float(_c.cosmology[0].sigma8), float(cosmo.sigma8))

# Theory (comoving-volume Limber number counts) on the HEALPix pixel-window footing; bin on fixed edges.
theory_pw = compute_theory_cl_for_density(cosmo, ref, ell) * (hp.pixwin(NSIDE, lmax=LMAX) ** 2)
theory_pw_arr = np.asarray(theory_pw.array)
theory_b = theory_pw.bin(nlb=NLB, lmin=2)
leff = np.asarray(theory_b.wavenumber)
theory_b_arr = np.asarray(theory_b.array)

# Bandpower-binned measured spectra (solver -> step -> binned PowerSpectrum), identical edges to theory.
meas_b = {sol: {st: spectra[sol][st].bin(nlb=NLB, lmin=2) for st in STEPS} for sol in SOLVERS}
meas_b_arr = {sol: {st: np.asarray(meas_b[sol][st].array) for st in STEPS} for sol in SOLVERS}


# =============================================================================
# fig01–fig08 — per-solver step convergence: C_ell at each step count (top) + ratio to the 50-step run
# =============================================================================
def plot_step_convergence(solver, shell_idxs, stem):
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
    for col, sh in enumerate(shell_idxs):
        ax_s, ax_r = axes[0, col], axes[1, col]
        for st in STEPS:
            ax_s.plot(leff, meas_b_arr[solver][st][sh], color=STEP_COLORS[st], lw=1.5)
        ax_s.set_xscale("log")
        ax_s.set_yscale("log")
        ax_s.set_xlim(max(2.0, leff.min() * 0.8), LMAX)
        ax_s.set_title(f"shell {sh}:  z = {z_shells[sh]:.3f}", fontsize=11)
        ax_s.grid(True, which="both", ls=":", alpha=0.4)
        if col == 0:
            ax_s.set_ylabel(r"$C_\ell$")
        # bottom (3:1): each step's binned C_ell over the 50-step binned C_ell
        ax_r.axhspan(0.98, 1.02, color="0.7", alpha=0.3)
        ax_r.axhline(1.0, color="0.4", ls="--", lw=0.9)
        for st in STEPS:
            ax_r.plot(leff, meas_b_arr[solver][st][sh] / meas_b_arr[solver][50][sh], color=STEP_COLORS[st], lw=1.3)
        ax_r.set_xscale("log")
        ax_r.set_xlim(max(2.0, leff.min() * 0.8), LMAX)
        ax_r.set_ylim(0.97, 1.03)
        ax_r.set_xlabel(r"multipole $\ell$")
        ax_r.grid(True, which="both", ls=":", alpha=0.4)
        if col == 0:
            ax_r.set_ylabel("meas / 50-step")
    handles = [Line2D([], [], color=STEP_COLORS[st], lw=1.6, label=f"{st} steps") for st in STEPS]
    handles[-1].set_label("50 steps (reference)")
    handles += [Line2D([], [], color="0.7", lw=6, alpha=0.5, label=r"$\pm2\%$")]
    fig.legend(handles=handles, loc="upper center", ncol=7, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    savefig(ASSETS / stem, fig)


# =============================================================================
# fig09 — 20 vs 30 steps, all four solver/stepping variants, near / mid / far shell: C_ell + ratio to theory
# =============================================================================
def plot_solvers_near_mid_far(stem):
    cols = [("near", NEAR_SHELL), ("mid", MID_SHELL), ("far", FAR_SHELL)]
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16.5, 6.4), gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
    for col, (label, sh) in enumerate(cols):
        ax_s, ax_r = axes[0, col], axes[1, col]
        ax_s.plot(ell_full[2:], theory_pw_arr[sh][2:], color="k", ls=":", lw=1.4, zorder=5)
        for sol in SOLVERS:
            for st in (20, 30):
                ax_s.plot(leff, meas_b_arr[sol][st][sh], color=SOLVER_COLORS[sol], ls=STEP7_STYLES[st], lw=1.5)
        ax_s.set_xscale("log")
        ax_s.set_yscale("log")
        ax_s.set_xlim(max(2.0, leff.min() * 0.8), LMAX)
        ax_s.set_title(f"{label} shell {sh}:  z = {z_shells[sh]:.3f},  χ = {chi_shells[sh]:.0f} Mpc/h", fontsize=11)
        ax_s.grid(True, which="both", ls=":", alpha=0.4)
        if col == 0:
            ax_s.set_ylabel(r"$C_\ell$")
        # bottom (3:1): binned measured / binned theory
        ax_r.axhspan(0.95, 1.05, color="0.7", alpha=0.3)
        ax_r.axhline(1.0, color="0.4", ls="--", lw=0.9)
        for sol in SOLVERS:
            for st in (20, 30):
                ax_r.plot(
                    leff,
                    meas_b_arr[sol][st][sh] / theory_b_arr[sh],
                    color=SOLVER_COLORS[sol],
                    ls=STEP7_STYLES[st],
                    lw=1.3,
                )
        ax_r.set_xscale("log")
        ax_r.set_xlim(max(2.0, leff.min() * 0.8), LMAX)
        ax_r.set_ylim(0.7, 1.2)
        ax_r.set_xlabel(r"multipole $\ell$")
        ax_r.grid(True, which="both", ls=":", alpha=0.4)
        if col == 0:
            ax_r.set_ylabel("meas / theory")
    handles = [Line2D([], [], color=SOLVER_COLORS[s], lw=1.8, label=f"{s} ({SOLVERS[s]})") for s in SOLVERS]
    handles += [
        Line2D([], [], color="0.3", ls="-", lw=1.6, label="20 steps"),
        Line2D([], [], color="0.3", ls="--", lw=1.6, label="30 steps"),
        Line2D([], [], color="k", ls=":", lw=1.4, label=r"Limber number-counts theory $\times\,w_\ell^2$"),
        Line2D([], [], color="0.7", lw=6, alpha=0.5, label=r"$\pm5\%$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=8, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    savefig(ASSETS / stem, fig)


def main():
    set_style()
    plot_step_convergence("bfa", range(0, 5), "fig01-bfa-shells-0-4")
    plot_step_convergence("bfa", range(5, 10), "fig02-bfa-shells-5-9")
    plot_step_convergence("bfd", range(0, 5), "fig03-bfd-shells-0-4")
    plot_step_convergence("bfd", range(5, 10), "fig04-bfd-shells-5-9")
    plot_step_convergence("dkd", range(0, 5), "fig05-dkd-shells-0-4")
    plot_step_convergence("dkd", range(5, 10), "fig06-dkd-shells-5-9")
    plot_step_convergence("kdk", range(0, 5), "fig07-kdk-shells-0-4")
    plot_step_convergence("kdk", range(5, 10), "fig08-kdk-shells-5-9")
    plot_solvers_near_mid_far("fig09-solvers-near-mid-far")
    print(f"assets written to {ASSETS}")


if __name__ == "__main__":
    main()
