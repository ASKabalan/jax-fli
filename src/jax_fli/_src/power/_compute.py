import healpy as hp
import jax.core
import jax.numpy as jnp
import jax_healpy as jhp
import numpy as np
from jaxpm.utils import power_spectrum


def _power(
    mesh,
    mesh2=None,
    *,
    box_shape=None,
    kedges=None,
    dk=None,
    kmax=None,
    multipoles=0,
    los=jnp.array([0.0, 0.0, 1.0]),
    compensate_order=None,
    shotnoise=None,
):
    """Auto/cross 3D power spectrum via ``jaxpm.utils.power_spectrum``.

    Thin wrapper that forwards jax-fli's binning knobs (``kedges``/``dk``/``kmax``)
    and the optional grid corrections.

    Parameters
    ----------
    compensate_order : int, str, or None
        Deconvolve the mass-assignment window of the given order (NGP=1, CIC=2,
        TSC=3, PCS=4): multiply ``|delta_k|**2`` by ``compensation_kernel**2``.
    shotnoise : (order, nbar) or None
        Subtract the aliased shot noise ``(1 / nbar) * C_order(k)`` before
        deconvolving (auto-spectrum only, i.e. ``mesh2 is None``). ``nbar`` is the
        mean number density in the same units as ``box_shape``
        (``nbar = N / box.prod()``; one particle per cell -> ``mesh.prod() / box.prod()``).
    """
    los_arg = [0.0, 0.0, 1.0] if los is None else los

    # jax-fli's contract returns a 1D spectrum for a single multipole, whether
    # given as a scalar or a length-1 sequence. jaxpm only returns 1D for a
    # scalar (a length-1 sequence yields a (1, n_k) array), so unwrap to match.
    if isinstance(multipoles, (list | tuple)) and len(multipoles) == 1:
        multipoles = multipoles[0]

    return power_spectrum(
        mesh,
        mesh2,
        box_shape=box_shape,
        kedges=kedges,
        dk=dk,
        kmax=kmax,
        multipoles=multipoles,
        los=los_arg,
        compensate_order=compensate_order,
        shotnoise=shotnoise,
    )


def _flat_cl(map2d, map2=None, *, pixel_size=None, field_size=None, ell_edges=None):
    nx, ny = map2d.shape

    # pixel_size [radians per pixel] (scalar or (px, py)) or derived from field_size
    if pixel_size is None:
        if field_size is None:
            raise ValueError("pixel_size or field_size must be provided for flat-sky Cl")
        field_x, field_y = field_size
        px, py = field_x / nx, field_y / ny
    else:
        px, py = pixel_size

    if map2 is None:
        map2 = map2d
    elif map2.shape != map2d.shape:
        raise ValueError("map2 must have the same shape as map2d for cross Cl")

    # FFTs
    map_fft = jnp.fft.fft2(map2d)
    map2_fft = jnp.fft.fft2(map2)

    # flat-sky normalization: A_pix / N_pix = (px * py) / (nx * ny)
    norm = (px * py) / (nx * ny)
    pk_2d = (map_fft * map2_fft.conj()) * norm

    # ℓ-grid
    # ℓx, ℓy in 1/rad
    lx = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=px)
    ly = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=py)
    LX, LY = jnp.meshgrid(lx, ly, indexing="ij")
    ell_grid = jnp.sqrt(LX**2 + LY**2)

    # bin edges
    if ell_edges is None:
        ell_max = ell_grid.max()
        ell_edges = jnp.linspace(0.0, ell_max, 50)
    ell_edges = jnp.asarray(ell_edges)

    # radial binning
    dig = jnp.digitize(ell_grid.reshape(-1), ell_edges)
    nbins = ell_edges.shape[0] + 1

    ell_count = jnp.bincount(dig, length=nbins)
    ell_sum = jnp.bincount(dig, weights=ell_grid.reshape(-1), length=nbins)
    cl_sum = jnp.bincount(dig, weights=pk_2d.reshape(-1).real, length=nbins)

    denom = jnp.where(ell_count > 0, ell_count, 1)
    ell_avg = (ell_sum / denom)[1:-1]  # drop underflow/overflow bins
    cl = (cl_sum / denom)[1:-1]

    return ell_avg, cl


def _spherical_cl(map_sphere, map_sphere2=None, *, lmax=None, method="jax"):
    """Spherical (HEALPix) angular power spectrum using jax_healpy.anafast."""
    if method == "healpy":
        if not jax.core.is_concrete(map_sphere):
            raise ValueError("method='healpy' requires concrete (non-jax) arrays")
        if map_sphere2 is not None and not jax.core.is_concrete(map_sphere2):
            raise ValueError("method='healpy' requires concrete (non-jax) arrays")

        map_sphere_np = np.asarray(map_sphere)
        map_sphere2_np = None if map_sphere2 is None else np.asarray(map_sphere2)
        cl = hp.anafast(map_sphere_np, map_sphere2_np, lmax=lmax, pol=False)
        ell_out = np.arange(cl.shape[0]) * 1.0  # Convert to float for consistency
        return jnp.asarray(ell_out), jnp.asarray(cl)

    cl = jhp.anafast(map_sphere, map_sphere2, lmax=lmax, pol=False, method=method)
    ell_out = jnp.arange(cl.shape[-1]) * 1.0
    return ell_out, jnp.asarray(cl)


def _cross_spherical_cl(maps, *, lmax=None, method="healpy"):
    """
    Compute all cross-angular power spectra for batched HEALPix maps.

    For B maps, computes K = B*(B+1)/2 cross-spectra in upper triangular order.

    Parameters
    ----------
    maps : array_like
        Batched HEALPix maps with shape (B, npix)
    lmax : int, optional
        Maximum multipole moment. Defaults to 3*nside-1.
    method : str, default="healpy"
        Must be "healpy". JAX method not supported for cross-spectra.

    Returns
    -------
    ell : jnp.ndarray
        Array of multipole moments
    cls : jnp.ndarray
        Cross-spectra with shape (K, n_ell) where K = B*(B+1)/2
        Ordering: (0,0), (0,1), ..., (0,B-1), (1,1), ..., (B-1,B-1)
    """
    if method != "healpy":
        raise ValueError(
            f"cross_angular_cl_spherical only supports method='healpy', got method='{method}'. "
            "JAX method is not implemented for cross-spectra computation."
        )

    if not jax.core.is_concrete(maps):
        raise ValueError("method='healpy' requires concrete (non-traced) arrays")

    # Convert to numpy for healpy
    maps_np = np.asarray(maps)

    # healpy.anafast with multiple maps returns all cross-spectra in upper triangular order
    # Shape: (K, n_ell) where K = B*(B+1)/2
    cls = hp.anafast(maps_np, lmax=lmax, pol=False)

    # Generate ell array
    ell_out = np.arange(cls.shape[-1]) * 1.0  # Convert to float for consistency

    return jnp.asarray(ell_out), jnp.asarray(cls)


def _transfer(mesh0, mesh1, *, box_shape, kedges=None, dk=None, kmax=None, compensate_order=None, shotnoise=None):
    """Monopole transfer function sqrt(P1/P0).

    ``compensate_order`` and ``shotnoise`` are applied to both auto-spectra.
    """
    k, pk0 = _power(
        mesh0,
        None,
        box_shape=box_shape,
        kedges=kedges,
        dk=dk,
        kmax=kmax,
        multipoles=0,
        compensate_order=compensate_order,
        shotnoise=shotnoise,
    )
    _, pk1 = _power(
        mesh1,
        None,
        box_shape=box_shape,
        kedges=kedges,
        dk=dk,
        kmax=kmax,
        multipoles=0,
        compensate_order=compensate_order,
        shotnoise=shotnoise,
    )
    return k, (pk1 / pk0) ** 0.5


def _coherence(mesh0, mesh1, *, box_shape, kedges=None, dk=None, kmax=None, compensate_order=None, shotnoise=None):
    """Monopole coherence pk01 / sqrt(pk0 pk1).

    ``compensate_order`` deconvolves every spectrum; ``shotnoise`` is subtracted
    from the two auto-spectra only (it does not apply to the cross term pk01).
    """
    k, pk01 = _power(
        mesh0,
        mesh1,
        box_shape=box_shape,
        kedges=kedges,
        dk=dk,
        kmax=kmax,
        multipoles=0,
        compensate_order=compensate_order,
    )
    _, pk0 = _power(
        mesh0,
        None,
        box_shape=box_shape,
        kedges=kedges,
        dk=dk,
        kmax=kmax,
        multipoles=0,
        compensate_order=compensate_order,
        shotnoise=shotnoise,
    )
    _, pk1 = _power(
        mesh1,
        None,
        box_shape=box_shape,
        kedges=kedges,
        dk=dk,
        kmax=kmax,
        multipoles=0,
        compensate_order=compensate_order,
        shotnoise=shotnoise,
    )
    return k, pk01 / (pk0 * pk1) ** 0.5
