"""Experiment 10 figures.

fig01 — the 832^3 CosmoGrid initial condition against its 1024^3 spectral UPsample.
fig02 — the same, DOWNsampled to 512^3 (the resampler runs both ways; downsampling is the
        validation direction, for cheap runs off the same realization).

Spectra come from ``prep_ic_spectra.py`` (heavy: the 832^3 white field is 2.3 GB); this
script only plots the small committed CSVs it writes, and rebuilds the linear-theory curve
from the cosmology recorded in each file's header.
"""

from __future__ import annotations

import jax

# float64 globally BEFORE jax_cosmo (the enable_x64() context manager breaks jax_cosmo's
# pure_callback comoving-distance cache; the global config flag is the safe route).
jax.config.update("jax_enable_x64", True)

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import jax_cosmo as jc  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _exputils import savefig, set_style  # noqa: E402

ASSETS = HERE / "assets"

C_SRC = "#1f4e79"  # the 832^3 source
C_TGT = "#c1440e"  # the resampled field
C_TH = "#777777"  # linear theory
C_FRAC = "#2a7f4f"  # inherited-mode fraction


def _load(path: Path) -> tuple[dict, dict]:
    """Return (header scalars, columns) from a prep_ic_spectra.py CSV.

    Header: every `key=value` token in the leading comment block whose value is a number. The
    prose comments also contain bare "=" signs, so a token only counts when both halves are
    non-empty and the right half parses.
    """
    lines = path.read_text().splitlines()
    ncomment = sum(1 for line in lines if line.startswith("#"))
    head = {}
    for line in lines[:ncomment]:
        for tok in line.lstrip("# ").split():
            key, sep, val = tok.partition("=")
            if sep and key and val:
                try:
                    head[key] = float(val)
                except ValueError:
                    pass
    names = lines[ncomment].strip().split(",")
    cols = dict(zip(names, np.genfromtxt(path, delimiter=",", skip_header=ncomment + 1, unpack=True)))
    return head, cols


def plot_ic_resample(csv_name: str, stem: str):
    """P(k), transfer and coherence between the source IC and one resampled copy.

    Direction-agnostic: `ns > nt` is a downsample and the panels read the same way, with the
    roles of the two Nyquist guides swapped.
    """
    head, d = _load(HERE / csv_name)
    cosmo = jc.Cosmology(
        **{p: head[p] for p in ("Omega_c", "Omega_b", "h", "n_s", "sigma8", "w0", "wa", "Omega_k", "Omega_nu")}
    )
    kgrid = np.logspace(-4, 1, 512)
    pkgrid = np.asarray(jc.power.linear_matter_power(cosmo, kgrid))

    k, kf = d["kcen"], 2 * np.pi / head["box"]
    ns, nt = int(head["ns"]), int(head["nt"])
    up = nt > ns
    k_blk = kf * (min(ns, nt) // 2 - 1)  # largest sphere fully inside the shared block
    k_src, k_tgt = kf * (ns // 2), kf * (nt // 2)
    kind = "upsample" if up else "downsample"

    mb = d["cb"] > 0  # shared modes: ONE set for P11, P22, P12
    P11 = d["s11b"][mb] / d["cb"][mb]
    P22 = d["s22b"][mb] / d["cb"][mb]
    P12 = d["s12b"][mb] / d["cb"][mb]
    T = np.sqrt(P22 / P11)
    r = P12 / np.sqrt(P11 * P22)

    m1, m2 = d["c11a"] > 0, d["c22a"] > 0
    Pd1 = d["pk1a"][m1] / d["c11a"][m1]  # colored spectrum on the source grid
    Pd2 = d["pk2a"][m2] / d["c22a"][m2]  # colored spectrum on the target grid
    frac = d["cb"][m2] / d["c22a"][m2]

    fig, ax = plt.subplots(
        3, 1, figsize=(6.0, 8.4), sharex=True, gridspec_kw={"height_ratios": [1.45, 1.15, 1.0], "hspace": 0.07}
    )
    for a in ax:
        a.grid(True, which="major", alpha=0.25, lw=0.6)
        a.grid(True, which="minor", alpha=0.10, lw=0.4)
        a.axvline(k_src, color="0.35", ls="--", lw=1.0, zorder=0)
        a.axvline(k_tgt, color="0.35", ls=":", lw=1.0, zorder=0)

    ax[0].loglog(
        kgrid,
        pkgrid,
        color=C_TH,
        lw=2.4,
        alpha=0.55,
        label=r"linear theory $P_{\rm lin}(k)$ (Eisenstein--Hu, CosmoGrid cosmology)",
    )
    ax[0].loglog(k[m1], Pd1, color=C_SRC, lw=1.3, label=rf"${ns}^3$ CosmoGrid IC, coloured (own grid)")
    ax[0].loglog(k[m2], Pd2, color=C_TGT, lw=1.3, ls="--", label=rf"${nt}^3$ spectral {kind}, coloured (own grid)")
    ax[0].set_ylabel(r"$P(k)\;\;[(h^{-1}\mathrm{Mpc})^3]$")
    ax[0].set_ylim(3e-1, 1e5)
    ax[0].legend(loc="lower left", frameon=False, fontsize=8.3)

    ax[1].plot(k[mb], T, color=C_TGT, lw=1.5, label=rf"transfer $T(k)=\sqrt{{P_{{{nt}}}/P_{{{ns}}}}}$, shared modes")
    ax[1].plot(
        k[mb],
        r,
        color=C_SRC,
        lw=1.5,
        ls="--",
        label=rf"coherence $r(k)=P_{{{ns}\times{nt}}}/\sqrt{{P_{{{ns}}}P_{{{nt}}}}}$, shared modes",
    )
    ax[1].plot(
        k[m2],
        frac,
        color=C_FRAC,
        lw=1.5,
        ls="-.",
        label=rf"inherited fraction: shared modes $/$ all ${nt}^3$ modes in the shell",
    )
    ax[1].axhline(1.0, color="0.5", lw=0.7, zorder=0)
    ax[1].set_ylim(-0.06, 1.20)
    ax[1].set_ylabel(r"$T(k)$,\; $r(k)$,\; inherited fraction")
    ax[1].legend(loc="lower left", frameon=False, fontsize=8.3)

    # |T-1| is IDENTICALLY zero -- the two spectra are sums over bitwise-equal float32 values --
    # so it draws nothing on a log axis. Say so in the legend rather than leave a phantom entry.
    ax[2].loglog(k[mb], np.abs(1.0 - r), color=C_SRC, lw=1.2, ls="--", label=r"$|1-r(k)|$")
    ax[2].plot([], [], color=C_TGT, lw=1.2, label=r"$|T(k)-1| \equiv 0$ (bitwise-equal modes; no line on a log axis)")
    ax[2].axhline(
        np.finfo(np.float32).eps,
        color="0.35",
        lw=1.0,
        ls=":",
        label=r"float32 $\varepsilon=1.2\times10^{-7}$ (the IC is stored float32)",
    )
    ax[2].set_ylim(1e-11, 1e-5)
    ax[2].set_ylabel("departure from unity")
    ax[2].set_xlabel(r"$k\;\;[h\,\mathrm{Mpc}^{-1}]$")
    # opaque frame: the float32-eps guide runs the full width at 1e-7, under the legend
    ax[2].legend(loc="upper left", fontsize=8.3, frameon=True, facecolor="white", framealpha=0.92, edgecolor="none")
    ax[2].set_xlim(kf * 0.9, kf * (max(ns, nt) // 2) * 1.85)

    for kk, lbl, ha in ((k_src, rf"${ns}^3$ Nyquist", "right"), (k_tgt, rf"${nt}^3$ Nyquist", "left")):
        ax[0].text(
            kk * (0.93 if ha == "right" else 1.08), 1.5e4, lbl, rotation=90, ha=ha, va="top", fontsize=8, color="0.35"
        )

    savefig(ASSETS / stem, fig)

    lo = k[mb] < k_blk
    inherited = frac[k[m2] < k_blk]
    print(
        f"  {ns}^3 -> {nt}^3 ({kind}): below the shared-block edge k={k_blk:.2f}, "
        f"max|T-1| = {np.abs(T[lo] - 1).max():.1e}, max|1-r| = {np.abs(1 - r[lo]).max():.1e}, "
        f"inherited >= {inherited.min():.3f}"
    )


def main():
    set_style()
    plot_ic_resample("ic_spectra_832_1024.csv", "fig01-ic-upsample-832-1024")
    plot_ic_resample("ic_spectra_832_512.csv", "fig02-ic-downsample-832-512")
    print(f"assets written to {ASSETS}")


if __name__ == "__main__":
    main()
