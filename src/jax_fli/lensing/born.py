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
    quadrature="simpson",
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
    quadrature=("midpoint", "simpson", "gauss_legendre"),
    legend_loc=None,
):
    """Diagnose the shell quadrature of the Born lensing integral for one or more radial schemes.

    Always produces **two** figures for the quadrature schemes in ``quadrature`` (all sharing the exact
    shell weights ``born`` itself uses, :func:`jax_fli._src.lensing._born._born_windows`):

    1. **Window plot** — the continuous lensing kernel ``w(chi) = chi (1+z) (1 - chi/chi_s)`` for a
       source at ``z_kernel`` (black curve, exact area shaded), overlaid with each scheme's shell
       **windows** drawn as boxes. Each box spans its shell width ``Delta chi`` and its **area is the
       shell's Born weight** (so box height = weight / Delta chi). Midpoint boxes are flat at the
       kernel's shell-center value and poke above the exact area on wide near shells; Simpson and
       Gauss-Legendre boxes carry the (near-)exact per-shell integral, so they tile the shaded area.
       The legend reports each scheme's ``sum of windows`` and its % bias vs Gauss-Legendre. When several
       radial schemes are overlaid, boxes are coloured by quadrature and styled by scheme (solid, dashed).
    2. **Ratio plot** — the per-shell ratio ``scheme / Gauss-Legendre`` for every non-GL scheme,
       collapsed over the source n(z) (or per source redshift for scalar sources), one line per
       tomographic bin. Thin shells sit at 1; wide near shells overshoot. ``z_kernel`` is ignored
       here (the ratio uses ``nz_shear``).

    Neither figure carries an on-canvas title (paper style — title it in the caption).

    Parameters
    ----------
    nz_shear : list of jc.redshift.redshift_distribution or array of float
        Source distributions (e.g. from ``get_stage3_nz_shear``) or scalar source redshifts. Only used
        by the ratio plot; the window plot uses ``z_kernel`` instead.
    cosmo : jax_cosmo.Cosmology, optional
        Cosmology for distances and ``a_of_chi``. Default: ``jc.Planck18()``.
    comoving_centers : array or dict of {label: array}
        Shell centers chi [Mpc/h] — a dict compares several radial schemes on one figure.
    density_width : array or dict of {label: array}
        Shell widths Delta chi [Mpc/h], matching ``comoving_centers`` (same labels for dicts).
    min_z, max_z, n_integrate : float, float, int
        The n(z) integration grid for the ratio plot — keep identical to the ``born`` call.
    z_kernel : float, default=0.6
        Source redshift of the kernel curve + windows in the window plot.
    quadrature : str or sequence of str, default=("midpoint", "simpson", "gauss_legendre")
        The quadrature scheme(s) to overlay, each one of ``"midpoint"``, ``"simpson"``,
        ``"gauss_legendre"``. A single string is accepted and wrapped in a list.
    legend_loc : str, optional
        Window-plot legend placement: any Matplotlib ``loc`` string, or ``"outside"`` to put the legend
        just outside the axes on the right. Default (``None``) uses ``"upper right"``.

    Returns
    -------
    (fig_windows, fig_ratio) : tuple of matplotlib.figure.Figure
        The window-point figure and the scheme/Gauss-Legendre ratio figure.
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
    if isinstance(quadrature, str):
        quadrature = [quadrature]
    quadrature = list(quadrature)
    allowed = ("midpoint", "simpson", "gauss_legendre")
    bad = [q for q in quadrature if q not in allowed]
    if bad:
        raise ValueError(f"quadrature entries must each be one of {allowed}, got {bad}")

    qcolor = {"midpoint": "#C44E52", "simpson": "#55A868", "gauss_legendre": "#4C72B0"}
    qshort = {"midpoint": "midpoint", "simpson": "Simpson", "gauss_legendre": "GL"}
    draw_order = [q for q in ("gauss_legendre", "simpson", "midpoint") if q in quadrature]
    scheme_ls = {label: ls for label, ls in zip(comoving_centers, ("-", "--", "-.", ":"))}
    scheme_marker = {label: m for label, m in zip(comoving_centers, ("o", "s", "^", "D"))}

    # ---- window plot: w(chi) with each scheme's shell windows drawn as boxes (box area = shell weight).
    # Boxes are coloured by quadrature (midpoint vs exact) and styled by radial scheme (solid, dashed, ...). ----
    fig_w, ax = plt.subplots(figsize=(8.6, 5.1))
    chi_k = float(
        np.asarray(jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(jnp.asarray(z_kernel)))).ravel()[0]
    )
    chi_line = jnp.linspace(0.0, chi_k, 512)
    w_line = chi_line / jc.background.a_of_chi(cosmo, chi_line) * (1.0 - chi_line / chi_k)
    ax.fill_between(np.asarray(chi_line), np.asarray(w_line), color="0.9", zorder=0, label="exact kernel area")
    ax.plot(np.asarray(chi_line), np.asarray(w_line), color="k", lw=1.9, zorder=6, label=rf"$w(\chi)$, $z_s={z_kernel}$")

    chi_s_arr = jnp.atleast_1d(jnp.asarray(chi_k))
    for label in comoving_centers:
        centers = np.asarray(comoving_centers[label])
        widths = np.asarray(density_width[label])
        a_shell = jc.background.a_of_chi(cosmo, jnp.asarray(centers))
        contrib = (centers - widths / 2) < chi_k  # shells in front of the source
        w_exact = np.asarray(
            _born_windows(cosmo, jnp.asarray(centers), a_shell, jnp.asarray(widths), chi_s_arr, "gauss_legendre")[:, 0]
        )
        pre = "" if label is None else f"{label}: "
        for q in draw_order:
            # box area = shell Born weight under this quadrature → box height = weight / width
            w_shell = np.asarray(
                _born_windows(cosmo, jnp.asarray(centers), a_shell, jnp.asarray(widths), chi_s_arr, q)[:, 0]
            )
            heights = np.where(widths > 0, w_shell / widths, 0.0)
            for i in np.nonzero(contrib)[0]:
                lo = centers[i] - widths[i] / 2
                ax.add_patch(
                    plt.Rectangle(
                        (lo, 0),
                        widths[i],
                        heights[i],
                        facecolor="none",
                        edgecolor=qcolor[q],
                        ls=scheme_ls[label],
                        lw=1.3,
                        zorder=3,
                    )
                )
            if q == "midpoint":  # midpoint samples the kernel at each shell center — mark it on the curve
                ax.plot(centers[contrib], heights[contrib], scheme_marker[label], ms=3.5, color=qcolor[q], zorder=5)
            tot, tot_ex = w_shell[contrib].sum(), w_exact[contrib].sum()
            bias = 100.0 * (tot / tot_ex - 1.0) if tot_ex else 0.0
            ax.plot(
                [],
                [],
                color=qcolor[q],
                ls=scheme_ls[label],
                lw=2.0,
                label=f"{pre}{qshort[q]}: Σ {tot:.3g} ({bias:+.1f}% vs GL)",
            )

    ax.set_xlim(0, chi_k * 1.02)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$\chi$ [$h^{-1}\mathrm{Mpc}$]")
    ax.set_ylabel(r"$w(\chi) = \chi\,(1+z)\,(1-\chi/\chi_s)$")
    if legend_loc == "outside":
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, borderaxespad=0.0)
    else:
        ax.legend(loc=legend_loc or "upper right", fontsize=8)
    fig_w.tight_layout()

    # ---- ratio plot: per-shell scheme/GL weight, folded through the source n(z), one line per bin ----
    source_kind, sources = _normalize_sources(nz_shear)
    if source_kind == "distribution":
        z_grid = jnp.linspace(min_z, max_z, n_integrate + 1)
        # (K, Z): n(z) x Simpson weights — the same collapse born applies to the kappa grid.
        collapse = (
            jnp.stack([nz(z_grid) for nz in sources]) * _simps_weights(n_integrate) * ((max_z - min_z) / n_integrate)
        )
    else:
        z_grid = jnp.atleast_1d(sources)
        collapse = jnp.eye(z_grid.shape[0])  # scalar sources: one column per source redshift
    chi_source = jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(z_grid)).reshape(-1)
    ratio_schemes = [q for q in quadrature if q != "gauss_legendre"]

    fig_r, axr = plt.subplots(figsize=(7.6, 4.7))
    for label in comoving_centers:
        centers = jnp.asarray(comoving_centers[label])
        widths = jnp.asarray(density_width[label])
        a_shell = jc.background.a_of_chi(cosmo, centers)
        w_gl = _born_windows(cosmo, centers, a_shell, widths, chi_source, "gauss_legendre")
        ks_gl = np.asarray(jnp.einsum("kz,sz->ks", collapse, w_gl))
        for q in ratio_schemes:
            w_q = _born_windows(cosmo, centers, a_shell, widths, chi_source, q)  # (S, Z)
            ks_q = np.asarray(jnp.einsum("kz,sz->ks", collapse, w_q))
            for b in range(ks_gl.shape[0]):
                ok = ks_gl[b] > 1e-12 * ks_gl[b].max()
                pre = "" if label is None else f"{label} "
                axr.plot(
                    np.asarray(centers)[ok],
                    ks_q[b][ok] / ks_gl[b][ok],
                    color=qcolor[q],
                    ls=["-", "--", ":", "-."][b % 4],
                    lw=1.6,
                    label=f"{pre}{qshort[q]} bin {b + 1}",
                )

    axr.axhline(1.0, color="0.4", ls="--", lw=0.9)
    axr.set_xlabel(r"shell center $\chi$ [$h^{-1}\mathrm{Mpc}$]")
    axr.set_ylabel("scheme weight / Gauss–Legendre weight")
    axr.grid(True, ls=":", alpha=0.4)
    axr.legend(fontsize=8, ncol=max(1, len(ratio_schemes)))
    fig_r.tight_layout()

    return fig_w, fig_r
