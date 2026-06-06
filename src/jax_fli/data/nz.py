from __future__ import annotations

import json

import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np

__all__ = ["get_stage3_nz_shear", "get_des_y3_nz_shear", "plot_nz"]


def get_stage3_nz_shear(
    gals_per_arcmin2: list[float] | None = None,
    bw: float = 0.01,
    zmax: float | None = None,
) -> list:
    """Load Stage 3 weak lensing redshift distributions.

    Parameters
    ----------
    gals_per_arcmin2 : list of float, optional
        Galaxy densities for each of the 4 tomographic bins.
        Default: [7, 8.5, 7.5, 6]
    bw : float, default=0.01
        KDE bandwidth parameter.
    zmax : float, optional
        Maximum redshift. If None, uses max from data files.

    Returns
    -------
    list of jc.redshift.kde_nz
        List of 4 redshift distribution objects.
    """
    from importlib.resources import files

    if gals_per_arcmin2 is None:
        gals_per_arcmin2 = [7.0, 8.5, 7.5, 6.0]

    data_dir = files("jax_fli.data").joinpath("stage3_nz")
    nz_files = sorted(data_dir.glob("nz_stage3_*.txt"))

    nz_shear = []
    for nz_file, g in zip(nz_files, gals_per_arcmin2):
        z, nz = np.loadtxt(nz_file, unpack=True)
        nz_shear.append(
            jc.redshift.kde_nz(
                jnp.asarray(z),
                jnp.asarray(nz),
                bw=bw,
                zmax=zmax if zmax is not None else float(z.max()),
                gals_per_arcmin2=g,
            )
        )

    return nz_shear


def get_des_y3_nz_shear(
    gals_per_arcmin2: list[float] | None = None,
    bw: float = 0.01,
    zmax: float | None = None,
) -> list:
    """Load DES Y3 weak lensing source redshift distributions.

    Parameters
    ----------
    gals_per_arcmin2 : list of float, optional
        Galaxy densities for each of the 4 tomographic bins.
        Default uses DES Y3 fiducial values: [1.476, 1.479, 1.484, 1.461]
    bw : float, default=0.01
        KDE bandwidth parameter.
    zmax : float, optional
        Maximum redshift. If None, uses max from data files.

    Returns
    -------
    list of jc.redshift.kde_nz
        List of 4 redshift distribution objects.

    References
    ----------
    Abbott et al. 2022 (arXiv:2105.13549), Myles et al. 2021 (arXiv:2012.08566)
    """
    from importlib.resources import files

    data_dir = files("jax_fli.data").joinpath("des_y3_nz")
    meta_file = data_dir.joinpath("des_y3_meta.json")

    with meta_file.open() as f:
        meta = json.load(f)

    if gals_per_arcmin2 is None:
        gals_per_arcmin2 = [meta["bins"][f"bin{i + 1}"]["gals_per_arcmin2"] for i in range(4)]

    nz_shear = []
    for i, g in enumerate(gals_per_arcmin2):
        fname = meta["bins"][f"bin{i + 1}"]["file"]
        nz_file = data_dir.joinpath(fname)
        z, nz = np.loadtxt(nz_file, unpack=True)
        nz_shear.append(
            jc.redshift.kde_nz(
                jnp.asarray(z),
                jnp.asarray(nz),
                bw=bw,
                zmax=zmax if zmax is not None else float(z.max()),
                gals_per_arcmin2=g,
            )
        )

    return nz_shear


def plot_nz(nz_sources, cosmo=None, z_bins=None, labels=None, ell=1000.0, cmap="YlOrRd"):
    """Plot tomographic n(z) distributions with their lensing efficiency q(z).

    Produces a two-panel figure sharing the redshift axis:

    - Top: the source redshift distributions n(z), one curve per tomographic
      bin, with a secondary top axis showing comoving distance chi [h^-1 Mpc].
    - Bottom: the weak-lensing efficiency kernel q(z) for each bin, computed
      with ``jax_cosmo.probes.WeakLensing``.

    Parameters
    ----------
    nz_sources : list of callable
        Redshift distribution objects (e.g. from get_stage3_nz_shear).
    cosmo : jax_cosmo.Cosmology, optional
        Cosmology used for the comoving-distance axis and q(z). Default:
        ``jc.Planck18()``.
    z_bins : array-like, optional
        Redshift grid to evaluate on. Default: jnp.linspace(0.0, 2.0, 300).
    labels : list of str, optional
        Per-bin labels. Default: "Bin 1", "Bin 2", ...
    ell : float, default=1000.0
        Multipole at which the lensing kernel is evaluated. Only rescales the
        q(z) y-axis (the ell dependence is a z-independent scalar); a large
        value makes the ell factor approximately 1.
    cmap : str, default="YlOrRd"
        Matplotlib colormap, sampled light->dark across the bins.

    Returns
    -------
    (matplotlib.figure.Figure, (matplotlib.axes.Axes, matplotlib.axes.Axes))
        The figure and its (top n(z), bottom q(z)) axes.
    """
    import matplotlib.pyplot as plt

    if cosmo is None:
        cosmo = jc.Planck18()
    if z_bins is None:
        z_bins = jnp.linspace(0.0, 2.0, 300)
    if labels is None:
        labels = [f"Bin {i + 1}" for i in range(len(nz_sources))]

    # Warm colormap sampled light->dark, bin 1 -> bin N.
    cm = plt.get_cmap(cmap)
    colors = [cm(x) for x in np.linspace(0.35, 0.95, len(nz_sources))]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={"hspace": 0.0})

    # --- top: n(z) ---
    for i, nz in enumerate(nz_sources):
        ax_top.plot(z_bins, nz(z_bins), color=colors[i], lw=2, label=labels[i])
    ax_top.set_ylabel(r"$n(z)$")
    ax_top.legend(frameon=False)
    ax_top.set_ylim(bottom=0.0)

    # --- secondary top axis: comoving distance chi [h^-1 Mpc] ---
    def _z2chi(z):
        return np.asarray(jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(np.atleast_1d(z))))

    def _chi2z(chi):
        return np.asarray(jc.utils.a2z(jc.background.a_of_chi(cosmo, np.atleast_1d(chi))))

    secax = ax_top.secondary_xaxis("top", functions=(_z2chi, _chi2z))
    secax.set_xlabel(r"Comoving distance $\chi$ [$h^{-1}\mathrm{Mpc}$]")

    # --- bottom: lensing efficiency q(z) ---
    tracer = jc.probes.WeakLensing(nz_sources)
    qz = np.asarray(tracer.kernel(cosmo, z_bins, ell))  # (nbins, nz)
    for i in range(len(nz_sources)):
        ax_bot.plot(z_bins, qz[i], color=colors[i], lw=2)
    ax_bot.set_xlabel(r"Redshift $z$")
    ax_bot.set_ylabel(r"$q(z)$")
    ax_bot.set_ylim(bottom=0.0)
    ax_bot.set_xlim(float(z_bins[0]), float(z_bins[-1]))

    return fig, (ax_top, ax_bot)
