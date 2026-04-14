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


def plot_nz(nz_sources, z_bins=None, labels=None, ax=None):
    """Plot a list of n(z) redshift distributions.

    Parameters
    ----------
    nz_sources : list of callable
        Redshift distribution objects (e.g. from get_stage3_nz_shear).
    z_bins : array-like, optional
        Redshift grid to evaluate on. Default: jnp.linspace(0.0, 3.0, 200).
    labels : list of str, optional
        Per-bin labels. Default: "Bin 1", "Bin 2", ...
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Default: current axes (plt.gca()).

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if z_bins is None:
        z_bins = jnp.linspace(0.0, 3.0, 200)
    if labels is None:
        labels = [f"Bin {i + 1}" for i in range(len(nz_sources))]

    if ax is None:
        ax = plt.gca()

    for i, nz in enumerate(nz_sources):
        ax.plot(z_bins, nz(z_bins), label=labels[i])

    ax.set_xlabel("Redshift")
    ax.set_ylabel("n(z)")
    ax.legend()
    return ax
