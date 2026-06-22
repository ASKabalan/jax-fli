from __future__ import annotations

import numbers
import warnings
from collections.abc import Callable, Iterable
from typing import Literal, Union

import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax.tree_util import register_pytree_node_class
from jax_cosmo.redshift import redshift_distribution
from jax_cosmo.scipy.integrate import simps
from jaxtyping import Array

from jax_fli.data.metadata import _max_z_source

from .power_spec import PowerSpectrum

__all__ = ["comoving_tophat", "compute_theory_cl", "compute_theory_cl_for_density", "tophat_z"]

# Type alias for z_source parameter
ZSourceType = Union[
    float,
    jc.redshift.redshift_distribution,
    list[float | jc.redshift.redshift_distribution],
]


@register_pytree_node_class
class tophat_z(redshift_distribution):
    """Top-hat redshift distribution between zmin and zmax.

    Parameters
    ----------
    zmin : float
        Minimum redshift of the top-hat window.
    zmax : float
        Maximum redshift of the top-hat window.
    gals_per_arcmin2 : float, optional
        Galaxy density per square arcminute (default 1.0).

    Examples
    --------
    >>> nz = tophat_z(0.8, 1.2, gals_per_arcmin2=1.0)
    """

    def pz_fn(self, z):
        zmin, zmax = self.params
        return jnp.where((z >= zmin) & (z <= zmax), 1.0, 0.0)


@register_pytree_node_class
class comoving_tophat(redshift_distribution):
    """Source selection uniform in comoving VOLUME between two comoving distances.

    A painted density shell (HEALPix or flat-sky) is a comoving-volume projection of the 3D
    overdensity: its per-pixel value weights ``delta_3D`` by ``q(chi) ∝ chi**2`` over the shell
    ``[chi_near, chi_far]`` (the comoving volume element ``chi**2 dchi``).  The Limber prediction
    that reproduces this projection is a number-counts spectrum whose source distribution is
    uniform in comoving volume.

    jax_cosmo's number-counts radial kernel is ``n(z) * b(z) * H(z)`` (``jax_cosmo/probes.py``),
    so choosing ``n(z) ∝ chi(z)**2 / H(z)`` makes the kernel ``∝ chi**2`` — exactly the painted
    selection.  With jax_cosmo's ``∫ n(z) dz = 1`` normalisation this yields the effective radial
    weight ``chi**2 / ∫ chi**2 dchi``, i.e. the measured comoving-volume tophat.

    Unlike :class:`tophat_z`, the normalisation is taken over the shell's **own** redshift support
    ``[z_near, z_far]`` (fixed 256-point Simpson), so it is independent of the (deprecated) global
    ``nz_zmax`` and stays accurate for narrow, low-redshift shells.

    Parameters
    ----------
    cosmo : jc.Cosmology
        Carried as a pytree param so the distribution stays jit-able and differentiable.
    chi_near, chi_far : float
        Comoving-distance edges of the shell (Mpc/h), ``chi_near < chi_far``.

    Examples
    --------
    >>> nz = comoving_tophat(cosmo, 850.0, 950.0)
    """

    def pz_fn(self, z):
        cosmo, chi_near, chi_far = self.params
        a = jc.utils.z2a(z)
        chi = jc.background.radial_comoving_distance(cosmo, a)
        h_z = jc.background.H(cosmo, a)
        return jnp.where((chi >= chi_near) & (chi <= chi_far), chi**2 / h_z, 0.0)

    def __call__(self, z):
        cosmo, chi_near, chi_far = self.params
        z_near = jc.utils.a2z(jc.background.a_of_chi(cosmo, jnp.atleast_1d(chi_near)))[0]
        z_far = jc.utils.a2z(jc.background.a_of_chi(cosmo, jnp.atleast_1d(chi_far)))[0]
        norm = simps(lambda t: self.pz_fn(t), z_near, z_far, 256)
        return self.pz_fn(z) / norm


def _normalize_z_source(
    z_source: ZSourceType,
) -> list[jc.redshift.redshift_distribution]:
    """Normalize z_source to a list of redshift distributions."""
    # Handle single values
    if isinstance(z_source, numbers.Real):
        return [jc.redshift.delta_nz(z_source)]
    # 0-d JAX/numpy array (e.g. a Python float converted by jit tracing)
    if hasattr(z_source, "shape") and z_source.shape == ():
        return [jc.redshift.delta_nz(z_source)]
    if isinstance(z_source, jc.redshift.redshift_distribution):
        return [z_source]

    # Handle list
    if isinstance(z_source, Iterable):
        result = []
        for zs in z_source:
            if isinstance(zs, Array):
                result.append(jc.redshift.delta_nz(zs))
            elif isinstance(zs, jc.redshift.redshift_distribution):
                result.append(zs)
            else:
                raise TypeError(f"Each z_source must be a scalar or redshift_distribution, got {type(zs)}")
        return result
    else:
        raise TypeError("z_source must be a scalar, redshift_distribution, or list thereof.")


def _resolve_nonlinear_fn(nonlinear_fn: str | Callable) -> Callable:
    """Resolve nonlinear_fn parameter to a callable."""
    if nonlinear_fn == "linear":
        return jc.power.linear
    elif nonlinear_fn == "halofit":
        return jc.power.halofit
    elif callable(nonlinear_fn):
        return nonlinear_fn
    raise ValueError(f"nonlinear_fn must be 'linear', 'halofit', or a callable, got {nonlinear_fn}")


def _create_probe(
    nz_list: list[jc.redshift.redshift_distribution],
    probe_type: str,
    sigma_e: float = 0.0,
):
    """Create jax_cosmo probe from redshift distributions."""
    if probe_type == "weak_lensing":
        return jc.probes.WeakLensing(nz_list, sigma_e=sigma_e)
    elif probe_type == "number_counts":
        bias = jc.bias.constant_linear_bias(1.0)
        return jc.probes.NumberCounts(nz_list, bias, has_rsd=False)
    else:
        raise ValueError(f"probe_type must be 'weak_lensing' or 'number_counts', got {probe_type}")


def _get_auto_indices(n_bins: int) -> jnp.ndarray:
    """Get indices of auto-spectra (diagonal) from upper triangular indexing.

    jax_cosmo returns spectra in upper triangular order:
    (0,0), (0,1), ..., (0,n-1), (1,1), (1,2), ..., (n-1,n-1)

    This function returns the indices corresponding to auto-spectra (i,i).
    """
    rows, cols = jnp.triu_indices(n_bins)
    return jnp.where(rows == cols, size=n_bins)[0]


@jax.jit(static_argnames=["probe_type", "nonlinear_fn", "cross"])
def compute_theory_cl(
    cosmo: jc.Cosmology,
    ell: jnp.ndarray,
    z_source: ZSourceType,
    *,
    probe_type: str = "weak_lensing",
    sigma_e: float = 0.0,
    nonlinear_fn: str | Callable = jc.power.halofit,
    cross: bool = False,
) -> PowerSpectrum:
    """
    Compute theoretical angular power spectrum C_ell.

    Parameters
    ----------
    cosmo : jc.Cosmology
        Cosmology object from jax_cosmo.
    ell : jnp.ndarray
        Array of multipole moments.
    z_source : float, redshift_distribution, or list thereof
        Source redshift(s). Floats are converted to delta_nz distributions.
        Can be a single value or a list for tomographic analysis.
    probe_type : str, optional
        Either "weak_lensing" (default) or "number_counts".
    sigma_e : float, optional
        Intrinsic ellipticity dispersion for weak lensing (default 0.0).
        Ignored for number_counts probe.
    nonlinear_fn : str or callable, optional
        Nonlinear power spectrum function. Default is jc.power.halofit.
        Use "linear" for jc.power.linear, or pass a custom callable.
    cross : bool, optional
        If False (default), return only auto-spectra (diagonal).
        If True, return all auto and cross-spectra.

    Returns
    -------
    PowerSpectrum
        Angular power spectrum with:
        - array: shape (n_ell,) for single z_source, (n_cls, n_ell) otherwise
        - wavenumber: ell values
        - name: "Cl"
        - scale_factors: tuple of (i,j) index pairs indicating which spectra

    Examples
    --------
    >>> import jax_cosmo as jc
    >>> from jax_fli.power import compute_theory_cl, tophat_z

    Single source redshift:
    >>> cosmo = jc.Planck15()
    >>> ell = jnp.arange(2, 100)
    >>> cl = compute_theory_cl(cosmo, ell, z_source=1.0)

    Multiple sources, auto-spectra only:
    >>> cl_auto = compute_theory_cl(cosmo, ell, z_source=[0.5, 1.0, 1.5])
    >>> cl_auto.array.shape  # (3, 98)

    All spectra including cross:
    >>> cl_all = compute_theory_cl(cosmo, ell, z_source=[0.5, 1.0], cross=True)
    >>> cl_all.array.shape  # (3, 98) for (0,0), (0,1), (1,1)

    Number counts probe:
    >>> cl_nc = compute_theory_cl(cosmo, ell, z_source=1.0, probe_type="number_counts")

    Top-hat redshift distribution:
    >>> nz = tophat_z(0.8, 1.2, gals_per_arcmin2=1.0)
    >>> cl_tophat = compute_theory_cl(cosmo, ell, z_source=nz)

    Linear power spectrum:
    >>> cl_linear = compute_theory_cl(cosmo, ell, z_source=1.0, nonlinear_fn="linear")
    """
    if not jax.config.x64_enabled:
        warnings.warn(
            "compute_theory_cl is more accurate with jax_enable_x64=True; "
            "results may be inaccurate for thin / low-z / high-z source distributions.",
            UserWarning,
            stacklevel=2,
        )

    ell = jnp.asarray(ell)
    nz_list = _normalize_z_source(z_source)
    n_bins = len(nz_list)

    # Resolve nonlinear function
    nl_fn = _resolve_nonlinear_fn(nonlinear_fn)

    # Create probe
    probe = _create_probe(nz_list, probe_type, sigma_e=sigma_e)

    # Compute angular Cls
    cl_matrix = jc.angular_cl.angular_cl(cosmo, ell, [probe], nonlinear_fn=nl_fn)

    # cl_matrix shape: (n_cls, n_ell) where n_cls = n_bins*(n_bins+1)//2
    # Ordering: (0,0), (0,1), ..., (1,1), (1,2), ..., (n-1,n-1)

    if not cross and n_bins > 1:
        # Extract only auto-spectra
        auto_indices = _get_auto_indices(n_bins)
        cl_matrix = cl_matrix[auto_indices]

    # Handle single z_source case - squeeze to 1D
    if n_bins == 1:
        cl_matrix = cl_matrix.squeeze()

    # Compute per-bin effective source redshift.
    # delta_nz (point source): use the z value directly.
    # Any other distribution: compute weighted mean z via numerical integration.
    z_sources_list = []
    for nz in nz_list:
        if isinstance(nz, jc.redshift.delta_nz):
            z_sources_list.append(jnp.asarray(nz.params[0]))
        else:
            z_sources_list.append(_max_z_source(nz, min_z=0.0, max_z=float(nz.zmax)))
    z_sources = jnp.stack(z_sources_list)
    scale_factors = jc.utils.z2a(z_sources)
    comoving_centers = jc.background.radial_comoving_distance(cosmo, scale_factors)
    density_width = jnp.zeros(n_bins)

    return PowerSpectrum(
        array=cl_matrix,
        wavenumber=ell,
        name="Cl",
        scale_factors=scale_factors,
        comoving_centers=comoving_centers,
        density_width=density_width,
        z_sources=z_sources,
    )


def _limber_density_pair_cl(cosmo, chi_near_i, chi_far_i, chi_near_j, chi_far_j, ells, nl_fn, n_chi):
    """Edge-exact Limber C_ell for a comoving-volume shell pair (auto if i == j).

    A painted shell selects ``delta_3D`` with the comoving-volume weight ``q(chi) ∝ chi**2`` over
    ``[chi_near, chi_far]``.  The Limber cross-spectrum of shells i and j is therefore

        C_ell^{ij} = (1 / (I_i I_j)) * ∫_overlap chi**2 P((ell+0.5)/chi, a(chi)) dchi ,
        I = (chi_far**3 - chi_near**3) / 3 ,  overlap = [max(near), min(far)] .

    This is algebraically identical to a jax_cosmo number-counts spectrum with
    :class:`comoving_tophat` sources (the ``c**2`` and ``H`` factors cancel), but integrates over
    each shell's own support with a fixed ``n_chi`` grid (edges aligned), so thin / low-z / high-z
    shells are not under-sampled the way jax_cosmo's fixed 512-point ``[0, zmax]`` Simpson is.
    Disjoint shells give ``overlap`` width 0 -> ``C_ell = 0``.
    """
    lo = jnp.maximum(chi_near_i, chi_near_j)
    hi = jnp.minimum(chi_far_i, chi_far_j)
    width = jnp.maximum(hi - lo, 0.0)
    chi = lo + width * jnp.linspace(0.0, 1.0, n_chi)
    a = jc.background.a_of_chi(cosmo, chi)
    inv_norm = 9.0 / ((chi_far_i**3 - chi_near_i**3) * (chi_far_j**3 - chi_near_j**3))

    def pk_at(single_ell):
        k = (single_ell + 0.5) / jnp.clip(chi, 1.0)
        return jc.power.nonlinear_matter_power(cosmo, k, a, nonlinear_fn=nl_fn)

    pk = jax.vmap(pk_at)(ells)  # (n_ell, n_chi)
    integrand = chi[None, :] ** 2 * pk
    dx = width / (n_chi - 1)
    return dx * (integrand.sum(-1) - 0.5 * (integrand[..., 0] + integrand[..., -1])) * inv_norm


@jax.jit(static_argnames=["nonlinear_fn", "cross", "nz_zmax", "radial_weight", "n_chi"])
def compute_theory_cl_for_density(
    cosmo: jc.Cosmology,
    lightcone,
    ells: jnp.ndarray,
    nonlinear_fn: str | Callable = jc.power.halofit,
    cross: bool = False,
    nz_zmax: float = 2.0,
    radial_weight: Literal["comoving_volume", "tophat_z"] = "comoving_volume",
    n_chi: int = 128,
) -> PowerSpectrum:
    """Compute theory number-counts Cls directly from a lightcone density field.

    A painted density shell is a comoving-VOLUME projection of the 3D overdensity: its per-pixel
    value weights ``delta_3D`` by ``q(chi) ∝ chi**2`` over the shell ``[chi_near, chi_far]``.  The
    matching Limber prediction (``radial_weight="comoving_volume"``, default) is therefore

        C_ell^{ij} = (1 / (I_i I_j)) ∫_overlap chi**2 P((ell+0.5)/chi, a(chi)) dchi ,
        I = (chi_far**3 - chi_near**3) / 3 ,

    evaluated **edge-exactly** over each shell's own support with an ``n_chi``-point grid (see
    :func:`_limber_density_pair_cl`).  This is the same model as :class:`comoving_tophat`
    (``n(z) ∝ chi(z)**2 / H(z)``) but integrated directly, avoiding jax_cosmo's fixed 512-point
    ``[0, zmax]`` Limber sampling — which under-resolves thin / low-z / high-z shells and makes the
    result spuriously ``nz_zmax``-dependent.  Shell geometry (``comoving_centers``, ``density_width``)
    supplies the comoving edges; the result is independent of ``nz_zmax``.

    Parameters
    ----------
    cosmo : jc.Cosmology
    lightcone : FlatDensity, SphericalDensity, or PowerSpectrum
        Must carry .comoving_centers and .density_width of shape (n_shells,).
        A PowerSpectrum with status=FieldStatus.SPECTRA is also accepted — its
        stored AbstractField metadata is used for the theory computation.
    ells : jnp.ndarray
        Multipole moments.
    nonlinear_fn : str or callable, optional
        Nonlinear power spectrum function (default jc.power.halofit).
    cross : bool, optional
        If True, return all auto + cross spectra. Default False (auto only).
    radial_weight : {"comoving_volume", "tophat_z"}, optional
        ``"comoving_volume"`` (default): correct comoving-shell selection ``n(z) ∝ chi**2 / H(z)``,
        normalised over each shell's redshift support (``nz_zmax``-independent).
        ``"tophat_z"``: legacy behaviour — uniform in redshift with jax_cosmo's ``[0, nz_zmax]``
        normalisation; kept only for comparison/reproducibility (emits a ``DeprecationWarning``).
    nz_zmax : float, optional
        Only used by the legacy ``radial_weight="tophat_z"`` path (default 2.0).  Ignored by the
        default comoving-volume path, whose normalisation is over each shell's own support.
    n_chi : int, optional
        Number of radial integration points per shell for the comoving-volume path (default 128;
        the integrand is smooth so this converges to <0.1%).  Cost scales as
        ``n_shells * n_ell * n_chi`` power-spectrum evaluations.

    Returns
    -------
    PowerSpectrum
        shape (n_shells, n_ell) for auto-spectra, or (n_cls, n_ell) with cross.

    Notes
    -----
    This is a flat-sky **Limber** prediction (jax_cosmo).  For the nearest, lowest-redshift shells
    at large scales the Limber approximation itself deviates from the exact spherical ``C_ell``;
    that residual is expected and is not a bug.  To compare against a painted map, take the spectrum
    of the OVERDENSITY (``field.to(OVERDENSITY).angular_cl(...)``) and, at high ``ell``, account for
    the HEALPix pixel window (``pixwin(nside)**2``).
    """
    if not jax.config.x64_enabled:
        warnings.warn(
            "compute_theory_cl_for_density is more accurate with jax_enable_x64=True; "
            "results may be inaccurate for thin / low-z / high-z shells due to Limber sampling.",
            UserWarning,
            stacklevel=2,
        )

    r_centers = lightcone.comoving_centers
    r_widths = lightcone.density_width

    r_near = jnp.atleast_1d(r_centers - r_widths / 2)
    r_far = jnp.atleast_1d(r_centers + r_widths / 2)

    if radial_weight == "comoving_volume":
        ells = jnp.asarray(ells)
        nl_fn = _resolve_nonlinear_fn(nonlinear_fn)
        n_shells = r_near.shape[0]
        if cross:
            rows, cols = jnp.triu_indices(n_shells)
            cl_array = jax.vmap(
                lambda i, j: _limber_density_pair_cl(
                    cosmo, r_near[i], r_far[i], r_near[j], r_far[j], ells, nl_fn, n_chi
                )
            )(rows, cols)
        else:
            cl_array = jax.vmap(lambda cn, cf: _limber_density_pair_cl(cosmo, cn, cf, cn, cf, ells, nl_fn, n_chi))(
                r_near, r_far
            )
        if cl_array.shape[0] == 1:  # single shell -> match the squeezed (n_ell,) convention
            cl_array = cl_array[0]
        return PowerSpectrum(
            array=cl_array,
            wavenumber=ells,
            name="Cl",
            scale_factors=lightcone.scale_factors,
            comoving_centers=lightcone.comoving_centers,
            density_width=lightcone.density_width,
            z_sources=lightcone.z_sources,
        )

    if radial_weight != "tophat_z":
        raise ValueError(f"radial_weight must be 'comoving_volume' or 'tophat_z', got {radial_weight!r}")

    warnings.warn(
        "radial_weight='tophat_z' is the legacy density-shell model (uniform in redshift with "
        "nz_zmax-sensitive normalisation); it is kept only for comparison and reproducibility. "
        "Use the default 'comoving_volume' for a correct comoving-shell prediction.",
        DeprecationWarning,
        stacklevel=2,
    )
    z_near = jc.utils.a2z(jc.background.a_of_chi(cosmo, r_near))
    z_far = jc.utils.a2z(jc.background.a_of_chi(cosmo, r_far))
    nz_list = [tophat_z(zn, zf, gals_per_arcmin2=1.0, zmax=nz_zmax) for zn, zf in zip(z_near, z_far)]

    theory_cl = compute_theory_cl(
        cosmo,
        ell=ells,
        z_source=nz_list,
        probe_type="number_counts",
        nonlinear_fn=nonlinear_fn,
        cross=cross,
    )
    return theory_cl.replace(
        scale_factors=lightcone.scale_factors,
        comoving_centers=lightcone.comoving_centers,
        density_width=lightcone.density_width,
        z_sources=lightcone.z_sources,
    )
