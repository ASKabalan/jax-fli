import healpy as hp
import jax.core
import jax.numpy as jnp
import jax_healpy as jhp
import numpy as np
from jaxpm.distributed import fft3d
from jaxpm.kernels import compensation_kernel, fftk, gridding_shotnoise_kernel
from jaxpm.spherical import deconvolve_map
from scipy.special import legendre


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
    """Auto/cross 3D power spectrum, sharded in, replicated spectrum out.

    The numbers are those of ``jaxpm.utils.power_spectrum`` (same bins, ``norm='ortho'``, corrections, cross
    term ``|sum P_01|``), but every N^3 step runs on the devices that hold the field: ``fft3d`` (jaxdecomp,
    distributed when ``mesh`` is sharded), ``|k|`` and the bin indices from ``fftk`` in fft3d's own
    layout, and the per-bin sums as a scatter-add into a small replicated array (local sums, then an
    all-reduce). jaxpm builds ``|k|`` and the bin index of every cell with host numpy and bakes them into the
    program as N^3 constants, and its ``jnp.fft.fftn`` gathers a sharded field: at 1600^3 that host-OOM-killed
    the multi-host MAP runs.

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
    # jax-fli's contract returns a 1D spectrum for a single multipole, whether
    # given as a scalar or a length-1 sequence.
    if isinstance(multipoles, (list | tuple)) and len(multipoles) == 1:
        multipoles = multipoles[0]
    poles = np.atleast_1d(multipoles)
    mesh_shape = np.array(mesh.shape)
    box_shape = mesh_shape if box_shape is None else np.asarray(box_shape)

    # bin edges: jaxpm's rule (host, tiny)
    if kmax is None:
        kmax = np.pi * np.min(mesh_shape / box_shape)  # Nyquist
    if kedges is None or isinstance(kedges, int | float):
        if kedges is None:
            dk = 2 * np.pi / np.min(box_shape) * 2 if dk is None else dk  # twice the fundamental
        elif isinstance(kedges, int):
            dk = kmax / (kedges + 1)
        else:
            dk = kedges
        if dk <= 0:
            raise ValueError("dk must be positive and non-zero")
        kedges = np.arange(dk, kmax, dk) + dk / 2
    n_bins = len(kedges) + 1

    # Fourier modes, in fft3d's (possibly transposed) layout; fftk gives radians per cell in X, Y, Z order
    meshk = fft3d(mesh)
    kvec_cell = fftk(meshk)
    # physical k per axis computed exactly as jaxpm does (host, N values each): with the default dk the bin
    # edges sit at odd multiples of the fundamental, so on-axis modes lie exactly on an edge and the last bit
    # of |k| decides their bin
    kvec = [
        jnp.asarray((2 * np.pi * m / b) * np.fft.fftfreq(m)).reshape(k.shape)
        for k, m, b in zip(kvec_cell, mesh_shape, box_shape)
    ]
    kmesh = jnp.sqrt(sum(k**2 for k in kvec))
    norm = 1.0 / np.prod(mesh_shape)  # norm='ortho' on both transforms
    if mesh2 is None:
        mmk = (meshk.real**2 + meshk.imag**2) * norm
    else:
        mmk = meshk * fft3d(mesh2).conj() * norm

    # grid corrections, per mode before binning (both are anisotropic)
    if compensate_order is not None or shotnoise is not None:
        if shotnoise is not None and mesh2 is None:
            sn_order, nbar = shotnoise
            mmk = mmk - gridding_shotnoise_kernel(kvec_cell, sn_order) / (nbar * np.prod(box_shape / mesh_shape))
        if compensate_order is not None:
            mmk = mmk * compensation_kernel(kvec_cell, compensate_order) ** 2

    # per-bin sums: each device scatters its own cells, XLA all-reduces the (n_bins,) partial sums
    dig = jnp.digitize(kmesh, jnp.asarray(kedges))

    def bin_sum(weights):
        return jnp.zeros(n_bins, dtype=weights.dtype).at[dig].add(weights)

    kcount = bin_sum(jnp.ones_like(kmesh))
    kavg = (bin_sum(kmesh) / kcount)[1:-1]

    if np.any(poles != 0):
        los_vec = np.asarray([0.0, 0.0, 1.0] if los is None else los, dtype=float)
        los_vec = los_vec / np.linalg.norm(los_vec)
        mu = sum(k * los_i for k, los_i in zip(kvec, los_vec))
        mu = jnp.where(kmesh == 0, 0.0, mu / jnp.where(kmesh == 0, 1.0, kmesh))

    pk = []
    for ell in poles:
        weights = mmk * (2 * ell + 1) * (1.0 if ell == 0 else jnp.polyval(jnp.asarray(legendre(ell).coeffs), mu))
        if mesh2 is None:
            psum = bin_sum(weights)
        else:
            psum = (bin_sum(weights.real) ** 2 + bin_sum(weights.imag) ** 2) ** 0.5
        pk.append(psum)
    pk = (jnp.stack(pk) / kcount)[:, 1:-1] * np.prod(box_shape / mesh_shape)  # cell units -> (Mpc/h)^3

    return kavg, pk[0] if np.ndim(multipoles) == 0 else pk


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
        if isinstance(map_sphere, jax.core.Tracer):
            raise ValueError("method='healpy' requires concrete (non-jax) arrays")
        if map_sphere2 is not None and isinstance(map_sphere2, jax.core.Tracer):
            raise ValueError("method='healpy' requires concrete (non-jax) arrays")

        map_sphere_np = np.asarray(map_sphere)
        map_sphere2_np = None if map_sphere2 is None else np.asarray(map_sphere2)
        cl = hp.anafast(map_sphere_np, map_sphere2_np, lmax=lmax, pol=False)
        ell_out = np.arange(cl.shape[0]) * 1.0  # Convert to float for consistency
        return jnp.asarray(ell_out), jnp.asarray(cl)

    cl = jhp.anafast(map_sphere, map_sphere2, lmax=lmax, pol=False, method=method)
    ell_out = jnp.arange(cl.shape[-1]) * 1.0
    return ell_out, jnp.asarray(cl)


def _deconvolve_spherical(
    hmap,
    *,
    method,
    nside,
    lmax=None,
    lcut=None,
    kernel_width_arcmin=None,
    kernel_width_pixels=None,
    smoothing_interpretation="fwhm",
    iter=0,
    w_floor=1e-8,
):
    """Deconvolve the HEALPix mass-assignment window from a single painted map."""
    return deconvolve_map(
        hmap,
        method,
        nside,
        lmax=lmax,
        lcut=lcut,
        kernel_width_arcmin=kernel_width_arcmin,
        kernel_width_pixels=kernel_width_pixels,
        smoothing_interpretation=smoothing_interpretation,
        iter=iter,
        w_floor=w_floor,
    )


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

    if isinstance(maps, jax.core.Tracer):
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
