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
    quadrature="both",
):
    """Diagnose the shell quadrature of the Born lensing integral for one or more radial schemes.

    A single-panel figure whose content is chosen by ``quadrature`` (all three share the exact shell
    weights ``born`` itself uses, :func:`jax_fli._src.lensing._born._born_windows`):

    - ``"midpoint"`` / ``"gauss_legendre"``: the continuous lensing kernel
      ``w(chi) = chi (1+z) (1 - chi/chi_s)`` for a source at ``z_kernel`` (black curve, exact area
      shaded), overlaid with that quadrature's shell **windows** drawn as boxes. Each box spans its
      shell width ``Delta chi`` and its **area is the shell's Born weight** (so box height =
      weight / Delta chi). Midpoint boxes are flat at the kernel's shell-center value and poke above
      the exact area on wide near shells; Gauss-Legendre boxes carry the exact per-shell integral, so
      they tile the shaded area. The title reports the net ``sum of windows`` vs the exact total.
    - ``"both"`` (default): the per-shell ratio ``midpoint / exact``, collapsed over the source n(z)
      (or per source redshift for scalar sources), one line per tomographic bin. Thin shells sit at
      1; wide near shells overshoot. ``z_kernel`` is ignored here (the ratio uses ``nz_shear``).

    Parameters
    ----------
    nz_shear : list of jc.redshift.redshift_distribution or array of float
        Source distributions (e.g. from ``get_stage3_nz_shear``) or scalar source redshifts. Only used
        by ``quadrature="both"``; the single-quadrature views use ``z_kernel`` instead.
    cosmo : jax_cosmo.Cosmology, optional
        Cosmology for distances and ``a_of_chi``. Default: ``jc.Planck18()``.
    comoving_centers : array or dict of {label: array}
        Shell centers chi [Mpc/h] — a dict compares several radial schemes on one figure.
    density_width : array or dict of {label: array}
        Shell widths Delta chi [Mpc/h], matching ``comoving_centers`` (same labels for dicts).
    min_z, max_z, n_integrate : float, float, int
        The n(z) integration grid for ``quadrature="both"`` — keep identical to the ``born`` call.
    z_kernel : float, default=0.6
        Source redshift of the kernel curve + windows in the single-quadrature views.
    quadrature : {"both", "midpoint", "gauss_legendre"}, default="both"
        Which view to draw (see above).

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and its single axis.
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
    if quadrature not in ("both", "midpoint", "gauss_legendre"):
        raise ValueError(f"quadrature must be 'both', 'midpoint', or 'gauss_legendre', got {quadrature!r}")

    cm = plt.get_cmap("tab10")

    # ---- quadrature="both": the per-shell midpoint/exact ratio, folded through the source n(z) ----
    if quadrature == "both":
        source_kind, sources = _normalize_sources(nz_shear)
        if source_kind == "distribution":
            z_grid = jnp.linspace(min_z, max_z, n_integrate + 1)
            # (K, Z): n(z) x Simpson weights — the same collapse born applies to the kappa grid.
            collapse = jnp.stack([nz(z_grid) for nz in sources]) * _simps_weights(min_z, max_z, n_integrate)
        else:
            z_grid = jnp.atleast_1d(sources)
            collapse = jnp.eye(z_grid.shape[0])  # scalar sources: one column per source redshift
        chi_source = jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(z_grid)).reshape(-1)

        fig, ax = plt.subplots(figsize=(7.6, 4.7))
        single = len(comoving_centers) == 1  # one scheme → colour by bin; several → colour by scheme
        for j, label in enumerate(comoving_centers):
            centers = jnp.asarray(comoving_centers[label])
            widths = jnp.asarray(density_width[label])
            a_shell = jc.background.a_of_chi(cosmo, centers)
            w_mid = _born_windows(cosmo, centers, a_shell, widths, chi_source, "midpoint")  # (S, Z)
            w_gl = _born_windows(cosmo, centers, a_shell, widths, chi_source, "gauss_legendre")
            ks_mid = np.asarray(jnp.einsum("kz,sz->ks", collapse, w_mid))
            ks_gl = np.asarray(jnp.einsum("kz,sz->ks", collapse, w_gl))
            for b in range(ks_mid.shape[0]):
                ok = ks_gl[b] > 1e-12 * ks_gl[b].max()
                name = f"bin {b + 1}" if label is None else f"{label} bin {b + 1}"
                color, ls = (cm(b), "-") if single else (cm(j), ["-", "--", ":", "-."][b % 4])
                ax.plot(np.asarray(centers)[ok], ks_mid[b][ok] / ks_gl[b][ok], color=color, ls=ls, lw=1.6, label=name)

        ax.axhline(1.0, color="0.4", ls="--", lw=0.9)
        ax.set_xlabel(r"shell center $\chi$ [$h^{-1}\mathrm{Mpc}$]")
        ax.set_ylabel("midpoint weight / exact (GL) weight")
        ax.set_title("Born shell quadrature error: midpoint / Gauss–Legendre")
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(fontsize=8, ncol=max(1, len(comoving_centers)))
        fig.tight_layout()
        return fig, ax

    # ---- quadrature="midpoint"/"gauss_legendre": w(chi) with that scheme's shell windows as boxes ----
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    chi_k = float(
        np.asarray(jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(jnp.asarray(z_kernel)))).ravel()[0]
    )
    chi_line = jnp.linspace(0.0, chi_k, 512)
    w_line = chi_line / jc.background.a_of_chi(cosmo, chi_line) * (1.0 - chi_line / chi_k)
    ax.fill_between(np.asarray(chi_line), np.asarray(w_line), color="0.9", zorder=0, label="exact kernel area")
    ax.plot(np.asarray(chi_line), np.asarray(w_line), color="k", lw=1.9, zorder=4, label=rf"$w(\chi)$, $z_s={z_kernel}$")

    chi_s_arr = jnp.atleast_1d(jnp.asarray(chi_k))
    single = len(comoving_centers) == 1
    for j, label in enumerate(comoving_centers):
        centers = np.asarray(comoving_centers[label])
        widths = np.asarray(density_width[label])
        a_shell = jc.background.a_of_chi(cosmo, jnp.asarray(centers))
        # box area = shell Born weight under this quadrature → box height = weight / width
        w_shell = np.asarray(_born_windows(cosmo, jnp.asarray(centers), a_shell, jnp.asarray(widths), chi_s_arr, quadrature)[:, 0])
        w_exact = np.asarray(_born_windows(cosmo, jnp.asarray(centers), a_shell, jnp.asarray(widths), chi_s_arr, "gauss_legendre")[:, 0])
        heights = np.where(widths > 0, w_shell / widths, 0.0)
        contrib = (centers - widths / 2) < chi_k  # shells in front of the source

        for i in np.nonzero(contrib)[0]:
            lo = centers[i] - widths[i] / 2
            if single:
                face, edge, alpha = (("#4C72B0", "#AFC7E3")[i % 2], "0.3", 0.6)  # alternating shades → shell widths pop
            else:
                face, edge, alpha = (cm(j), cm(j), 0.28)
            ax.add_patch(plt.Rectangle((lo, 0), widths[i], heights[i], facecolor=face, edgecolor=edge, lw=0.7, alpha=alpha, zorder=1))
        if quadrature == "midpoint":  # midpoint samples the kernel at each shell center — mark it on the curve
            ax.plot(centers[contrib], heights[contrib], "o", ms=3.5, color=("#14315e" if single else cm(j)), zorder=5)

        tot, tot_ex = w_shell[contrib].sum(), w_exact[contrib].sum()
        bias = 100.0 * (tot / tot_ex - 1.0) if tot_ex else 0.0
        pre = "" if label is None else f"{label}: "
        ax.plot([], [], " ", label=f"{pre}Σ windows {tot:.3g} ({bias:+.1f}% vs exact)")

    qname = "midpoint (rectangle rule)" if quadrature == "midpoint" else "Gauss–Legendre (exact)"
    ax.set_xlim(0, chi_k * 1.02)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$\chi$ [$h^{-1}\mathrm{Mpc}$]")
    ax.set_ylabel(r"$w(\chi) = \chi\,(1+z)\,(1-\chi/\chi_s)$")
    ax.set_title(rf"Born shell windows — {qname};  box width $=\Delta\chi$, box area $=$ shell weight")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig, ax
