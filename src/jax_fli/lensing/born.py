from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_cosmo as jc

from .._src.base import ConvergenceUnit
from .._src.lensing import _attach_source_metadata, _born_flat, _born_spherical
from .._src.lensing._born import _born_windows, _simps_weights
from .._src.lensing._normalize_nz import _normalize_sources
from ..fields import FieldStatus, FlatKappaField, SphericalDensity, SphericalKappaField

__all__ = ["born", "plot_born_windows"]


@jax.jit(static_argnames=["n_integrate", "normalization", "quadrature"])
def born(
    cosmo,
    lightcone,
    nz_shear,
    min_z=0.01,
    max_z=1.5,
    n_integrate=32,
    normalization="global",
    quadrature="midpoint",
):
    if nz_shear is None:
        raise ValueError("nz_shear must be provided for lensing")

    if lightcone.status != FieldStatus.LIGHTCONE:
        raise ValueError(f"Expected lightcone with status=LIGHTCONE, got {lightcone.status}")

    scale_factors = jnp.atleast_1d(lightcone.scale_factors)

    if lightcone.array.ndim not in [2, 3]:
        raise ValueError(f"Lightcone array must be 2D (spherical) or 3D (flat), got {lightcone.array.ndim}D")

    r_center = jc.background.radial_comoving_distance(cosmo, scale_factors)

    density_plane_width = lightcone.density_width

    is_spherical = isinstance(lightcone, SphericalDensity)

    if is_spherical:
        source_map = _born_spherical(
            cosmo,
            lightcone,
            r_center,
            scale_factors,
            nz_shear,
            density_plane_width,
            min_z,
            max_z,
            n_integrate,
            normalization,
            quadrature,
        )
    else:
        source_map = _born_flat(
            cosmo,
            lightcone,
            r_center,
            scale_factors,
            nz_shear,
            density_plane_width,
            min_z,
            max_z,
            n_integrate,
            normalization,
            quadrature,
        )

    base_field = lightcone.replace(status=FieldStatus.KAPPA)

    if is_spherical:
        kappas = SphericalKappaField.FromDensityMetadata(
            array=source_map,
            field=base_field,
            status=FieldStatus.KAPPA,
            unit=ConvergenceUnit.DIMENSIONLESS,
            z_sources=nz_shear,
        )
    else:
        kappas = FlatKappaField.FromDensityMetadata(
            array=source_map,
            field=base_field,
            status=FieldStatus.KAPPA,
            unit=ConvergenceUnit.DIMENSIONLESS,
            z_sources=nz_shear,
        )

    # Replace shell-level metadata (inherited from lightcone) with per-source-bin metadata
    source_kind, sources = _normalize_sources(nz_shear)
    kappas = _attach_source_metadata(kappas, cosmo, source_kind, sources, min_z, max_z, n_integrate)

    # Re-shard the spherical convergence into the lensing layout (BINS/N, NPIX/M); warns/raises per
    # the mesh case. Flat convergence keeps the volumetric layout.
    if is_spherical:
        kappas = kappas.apply_sharding()

    return kappas


def plot_born_windows(
    nz_shear,
    cosmo=None,
    comoving_centers=None,
    density_width=None,
    *,
    min_z=0.01,
    max_z=1.5,
    n_integrate=32,
    z_kernel=0.6,
):
    """Plot the Born lensing kernel against the per-shell windows of one or more radial schemes.

    Produces a two-panel figure sharing the same shell weights ``born`` itself uses
    (:func:`jax_fli._src.lensing._born._born_windows`):

    - Left: the continuous lensing kernel ``w(chi) = chi (1+z) (1 - chi/chi_s)`` for a source at
      ``z_kernel`` (exact area shaded), overlaid with each scheme's shell windows as
      midpoint-rule rectangles (height = kernel at the shell center, width = shell width).
      A rectangle poking above the shaded area is weight the midpoint quadrature invents.
    - Right: the per-shell ratio of the midpoint weight to the exact Gauss-Legendre weight,
      collapsed over the source n(z) (or evaluated per source redshift for scalar sources),
      one line per tomographic bin. Thin shells sit at 1; wide near shells overshoot.

    Parameters
    ----------
    nz_shear : list of jc.redshift.redshift_distribution or array of float
        Source distributions (e.g. from ``get_stage3_nz_shear``) or scalar source redshifts.
    cosmo : jax_cosmo.Cosmology, optional
        Cosmology for distances and ``a_of_chi``. Default: ``jc.Planck18()``.
    comoving_centers : array or dict of {label: array}
        Shell centers chi [Mpc/h] — a dict compares several radial schemes on one figure.
    density_width : array or dict of {label: array}
        Shell widths Delta chi [Mpc/h], matching ``comoving_centers`` (same labels for dicts).
    min_z, max_z, n_integrate : float, float, int
        The n(z) integration grid — keep identical to the ``born`` call being diagnosed.
    z_kernel : float, default=0.6
        Source redshift of the continuous kernel curve in the left panel (display only).

    Returns
    -------
    (matplotlib.figure.Figure, (matplotlib.axes.Axes, matplotlib.axes.Axes))
        The figure and its (kernel + windows, weight ratio) axes.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if cosmo is None:
        cosmo = jc.Planck18()
    if comoving_centers is None or density_width is None:
        raise ValueError("comoving_centers and density_width are required (arrays, or {label: array} dicts)")
    if not isinstance(comoving_centers, dict):
        comoving_centers = {None: comoving_centers}
        density_width = {None: density_width}
    if set(comoving_centers) != set(density_width):
        raise ValueError("comoving_centers and density_width must carry the same scheme labels")

    source_kind, sources = _normalize_sources(nz_shear)
    if source_kind == "distribution":
        z_grid = jnp.linspace(min_z, max_z, n_integrate + 1)
        # (K, Z): n(z) x Simpson weights — the same collapse born applies to the kappa grid.
        collapse = jnp.stack([nz(z_grid) for nz in sources]) * _simps_weights(min_z, max_z, n_integrate)
    else:
        z_grid = jnp.atleast_1d(sources)
        collapse = jnp.eye(z_grid.shape[0])  # scalar sources: one column per source redshift
    chi_source = jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(z_grid)).reshape(-1)

    fig, (ax_kernel, ax_ratio) = plt.subplots(1, 2, figsize=(13.5, 4.6))

    chi_k = float(
        np.asarray(jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(jnp.asarray(z_kernel)))).ravel()[0]
    )
    chi_line = jnp.linspace(0.0, chi_k, 512)
    w_line = chi_line / jc.background.a_of_chi(cosmo, chi_line) * (1.0 - chi_line / chi_k)
    ax_kernel.plot(
        np.asarray(chi_line), np.asarray(w_line), color="k", lw=1.8, label=rf"$w(\chi)$ for $z_s={z_kernel}$"
    )
    ax_kernel.fill_between(np.asarray(chi_line), np.asarray(w_line), color="0.85", zorder=0)

    cm = plt.get_cmap("tab10")
    for j, label in enumerate(comoving_centers):
        centers = jnp.asarray(comoving_centers[label])
        widths = jnp.asarray(density_width[label])
        a_shell = jc.background.a_of_chi(cosmo, centers)
        w_mid = _born_windows(cosmo, centers, a_shell, widths, chi_source, "midpoint")  # (S, Z)
        w_gl = _born_windows(cosmo, centers, a_shell, widths, chi_source, "gauss_legendre")
        color = cm(j)

        heights = np.asarray(centers / a_shell * jnp.clip(1.0 - centers / chi_k, 0.0, None))
        for i, (c, dw) in enumerate(zip(np.asarray(centers), np.asarray(widths))):
            lo = c - dw / 2
            if lo > chi_k:
                break
            ax_kernel.add_patch(plt.Rectangle((lo, 0), dw, heights[i], fill=False, edgecolor=color, lw=1.4, ls="--"))
            ax_kernel.axvline(lo + dw, color="0.6", lw=0.5)

        ks_mid = np.asarray(jnp.einsum("kz,sz->ks", collapse, w_mid))
        ks_gl = np.asarray(jnp.einsum("kz,sz->ks", collapse, w_gl))
        for b in range(ks_mid.shape[0]):
            ok = ks_gl[b] > 1e-12 * ks_gl[b].max()
            name = f"bin {b + 1}" if label is None else f"{label} bin {b + 1}"
            ls = ["-", "--", ":", "-."][b % 4]
            ax_ratio.plot(np.asarray(centers)[ok], ks_mid[b][ok] / ks_gl[b][ok], color=color, ls=ls, lw=1.6, label=name)

    ax_kernel.set_xlim(0, chi_k * 1.02)
    ax_kernel.set_ylim(bottom=0)
    ax_kernel.set_xlabel(r"$\chi$ [$h^{-1}\mathrm{Mpc}$]")
    ax_kernel.set_ylabel(r"$w(\chi) = \chi\,(1+z)\,(1-\chi/\chi_s)$")
    ax_kernel.legend(loc="upper right", fontsize=9)

    ax_ratio.axhline(1.0, color="0.4", ls="--", lw=0.9)
    ax_ratio.set_xlabel(r"shell center $\chi$ [$h^{-1}\mathrm{Mpc}$]")
    ax_ratio.set_ylabel("midpoint weight / exact weight")
    ax_ratio.grid(True, ls=":", alpha=0.4)
    ax_ratio.legend(fontsize=8, ncol=max(1, len(comoving_centers)))

    fig.tight_layout()
    return fig, (ax_kernel, ax_ratio)
