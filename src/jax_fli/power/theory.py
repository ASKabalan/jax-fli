from __future__ import annotations

import numbers
from collections.abc import Callable, Iterable
from functools import partial
from typing import Union

import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax.tree_util import register_pytree_node_class
from jax_cosmo.redshift import redshift_distribution
from jaxtyping import Array

from jax_fli.data.metadata import _max_z_source

from .power_spec import PowerSpectrum

__all__ = ["compute_theory_cl", "compute_theory_cl_for_density", "tophat_z"]

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


@partial(jax.jit, static_argnames=["probe_type", "nonlinear_fn", "cross"])
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


@partial(jax.jit, static_argnames=["nonlinear_fn", "cross", "nz_zmax"])
def compute_theory_cl_for_density(
    cosmo: jc.Cosmology,
    lightcone,
    ells: jnp.ndarray,
    nonlinear_fn: str | Callable = jc.power.halofit,
    cross: bool = False,
    nz_zmax: float = 2.0,
) -> PowerSpectrum:
    """Compute theory number-counts Cls directly from a lightcone density field.

    Converts shell geometry (comoving_centers, density_width) into tophat_z
    redshift distributions and delegates to compute_theory_cl with
    probe_type="number_counts".

    Parameters
    ----------
    cosmo : jc.Cosmology
    lightcone : FlatDensity, SphericalDensity, or PowerSpectrum
        Must have .comoving_centers and .density_width of shape (n_shells,).
        A PowerSpectrum with status=FieldStatus.SPECTRA is also accepted — its
        stored AbstractField metadata is used for the theory computation.
    ells : jnp.ndarray
        Multipole moments.
    nonlinear_fn : str or callable, optional
        Nonlinear power spectrum function (default jc.power.halofit).
    cross : bool, optional
        If True, return all auto + cross spectra. Default False (auto only).
    nz_zmax : float, optional
        Integration upper bound for the tophat_z normalization (default 2.0).
        jax_cosmo normalises n(z) with 256 Simpson points over [0, nz_zmax];
        spacing = nz_zmax/255.  Must satisfy two conditions:
        (1) nz_zmax > max shell redshift (otherwise normalization is wrong),
        (2) nz_zmax/255 < min shell width in z (otherwise some shells get
            no integration point → _norm=0 → NaN).
        Reduce this value if you have very narrow shells at low redshift.

    Returns
    -------
    PowerSpectrum
        shape (n_shells, n_ell) for auto-spectra, or (n_cls, n_ell) with cross.
    """
    r_centers = lightcone.comoving_centers
    r_widths = lightcone.density_width

    r_near = r_centers - r_widths / 2
    r_far = r_centers + r_widths / 2

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

    theory_cl = theory_cl.replace(
        scale_factors=lightcone.scale_factors,
        comoving_centers=lightcone.comoving_centers,
        density_width=lightcone.density_width,
        z_sources=lightcone.z_sources,
    )
    return theory_cl
