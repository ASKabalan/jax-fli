from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jaxpm.distributed import fft3d, ifft3d, normal_field
from jaxpm.kernels import fftk
from jaxtyping import Array, PRNGKeyArray

from .fields import DensityField, DensityUnit, FieldStatus


def resample_white_field(
    white_field: Array,
    key: PRNGKeyArray,
    mesh_size: tuple[int, int, int],
    *,
    field_sharding: Any | None = None,
) -> Array:
    """Spectrally upsample a white field onto a larger mesh, preserving its modes exactly.

    Every mode of ``white_field`` with ``|n_i| < N_src_i / 2`` is copied into the target grid at the
    **same integer wavevector** ``n``; the remaining (higher ``|n|``) modes are drawn fresh from
    ``key``. The copied modes are therefore identical amplitude *and* phase, so the transfer
    function and coherence against the source are exactly 1 over the shared block — by construction,
    not by measurement. Filling the rest is not optional: zero-padding alone would leave the field
    with no small-scale power and a variance of ``prod(N_src) / prod(N_tgt)``.

    The source's integer wavevector ``n`` sits at a different **physical** ``k`` whenever the target
    box differs from the source's, so this reproduces a *realization*, not a physical field.

    Parameters
    ----------
    white_field : Array
        Real, unit-variance, zero-mean field. Not a ``DensityField``.
    key : jax.random.PRNGKey
        Seeds the modes above the source's Nyquist index.
    mesh_size : tuple[int, int, int]
        Target mesh. Must be >= the source mesh on every axis; equal returns the input unchanged.
    field_sharding : Any, optional
        JAX sharding descriptor for the target array.

    Returns
    -------
    Array
        Real white field of shape ``mesh_size``.
    """
    src = tuple(white_field.shape)
    tgt = tuple(mesh_size)
    if src == tgt:
        return white_field
    if any(t < s for s, t in zip(src, tgt)):
        raise ValueError(f"resample_white_field only upsamples: source {src} exceeds target {tgt} on some axis.")

    # NOT jitted, and the source transform is taken FIRST, while no sharded array is in play: the
    # source is small and replicated, and jaxdecomp's pfft rejects an unsharded operand once a
    # distributed mesh is active ("An unsharded dimension cannot correspond to a distributed mesh
    # axis"). Running eagerly keeps the two transforms in separate contexts. It is still the SAME
    # operator on both sides, which is what puts both k-arrays in jaxdecomp's transposed layout and
    # lets the block copy below avoid ever naming an axis.
    #
    # Unit variance per mode: an unnormalized FFT of a unit-variance white field gives
    # E|F[n]|^2 = prod(N), so dividing by sqrt(prod(N)) puts both grids on the same footing and a
    # copied mode keeps its variance. The shapes are static, so these are plain Python floats --
    # prod(2560^3) overflows float32's exact-integer range, which a jnp.prod would silently round.
    ks = fft3d(white_field) / math.sqrt(src[0] * src[1] * src[2])
    fresh = normal_field(seed=key, shape=tgt, sharding=field_sharding)
    kt = fft3d(fresh) / math.sqrt(tgt[0] * tgt[1] * tgt[2])

    # fft3d transposes (x, y, z) -> (y, z, x), the SAME way for both arrays, so axis a of `ks` and
    # axis a of `kt` are the same real-space axis and the index sets are read off the k-array shapes
    # directly -- no need to name which axis is which. In fftfreq order the shared modes are NOT a
    # centred block: they are n = 0..h-1 at the start of the axis and n = -1..-(h-1) at its end.
    # The source Nyquist index h is dropped so the copied set is symmetric under n -> -n, which is
    # what keeps the inverse transform exactly real.
    src_idx, tgt_idx = [], []
    for axis in range(3):
        h = ks.shape[axis] // 2
        low = jnp.arange(h)
        high = jnp.arange(1, h)
        src_idx.append(jnp.concatenate([low, ks.shape[axis] - high]))
        tgt_idx.append(jnp.concatenate([low, kt.shape[axis] - high]))

    # Cast the gathered block, not `ks`: the source k-array is REPLICATED on every rank (832^3 is
    # 4.7 GB at complex64, 9.4 GB at complex128), so it stays at the source's own precision.
    kt = kt.at[jnp.ix_(*tgt_idx)].set(ks[jnp.ix_(*src_idx)].astype(kt.dtype))
    return ifft3d(kt * math.sqrt(tgt[0] * tgt[1] * tgt[2]))


@jax.jit(
    static_argnames=[
        "mesh_size",
        "box_size",
        "cosmo",
        "pk_fn",
        "observer_position",
        "flatsky_npix",
        "nside",
        "field_size",
        "halo_size",
        "field_sharding",
    ],
)
def gaussian_initial_conditions(
    key: PRNGKeyArray,
    mesh_size: tuple[int, int, int],
    box_size: tuple[float, float, float],
    *,
    cosmo: jc.Cosmology | None = None,
    pk_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
    observer_position: tuple[float, float, float] = (0.5, 0.5, 0.5),
    flatsky_npix: tuple[int, int] | None = None,
    nside: int | None = None,
    field_size: tuple[int, int] | None = None,
    halo_size: int | tuple[int, int] = (0, 0),
    field_sharding: Any | None = None,
) -> DensityField:
    """
    Sample Gaussian initial conditions and package them as a Field PyTree.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator.
    mesh_size : tuple[int, int, int]
        Discretization of the simulation volume.
    box_size : tuple[float, float, float]
        Physical box side lengths (Mpc/h).
    pk_fn : callable
        Function mapping |k| to the linear matter power spectrum.
    observer_position : tuple[float, float, float], optional
        Fractional observer coordinates; defaults to box center.
    flatsky_npix : tuple[int, int], optional
        Requested flat-sky pixel resolution for downstream projections.
    nside : int, optional
        HEALPix resolution for spherical projections.
    halo_size : int | tuple[int, int], optional
        Halo exchange depth for distributed painting.
    sharding : Any, optional
        JAX sharding descriptor for distributed arrays.

    Returns
    -------
    DensityField
        Field PyTree populated with the linear density field.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax.random as jr
    >>> pk = lambda k: 1.0 / (1.0 + k**2)
    >>> field = gaussian_initial_conditions(
    ...     jr.PRNGKey(0),
    ...     mesh_size=(16, 16, 16),
    ...     box_size=(200.0, 200.0, 200.0),
    ...     pk_fn=pk,
    ... )
    >>> field.array.shape
    (16, 16, 16)
    """

    field = normal_field(seed=key, shape=mesh_size, sharding=field_sharding)
    return interpolate_initial_conditions(
        initial_field=field,
        mesh_size=mesh_size,
        box_size=box_size,
        cosmo=cosmo,
        pk_fn=pk_fn,
        observer_position=observer_position,
        flatsky_npix=flatsky_npix,
        nside=nside,
        field_size=field_size,
        halo_size=halo_size,
        field_sharding=field_sharding,
    )


@jax.jit(
    static_argnames=[
        "mesh_size",
        "box_size",
        "cosmo",
        "pk_fn",
        "observer_position",
        "flatsky_npix",
        "nside",
        "field_size",
        "halo_size",
        "field_sharding",
    ],
)
def interpolate_initial_conditions(
    initial_field: Array,
    mesh_size: tuple[int, int, int] = None,
    box_size: tuple[float, float, float] = None,
    *,
    cosmo: jc.Cosmology | None = None,
    pk_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
    observer_position: tuple[float, float, float] = (0.5, 0.5, 0.5),
    flatsky_npix: tuple[int, int] | None = None,
    nside: int | None = None,
    field_size: tuple[int, int] | None = None,
    halo_size: int | tuple[int, int] = (0, 0),
    field_sharding: Any | None = None,
) -> DensityField:
    """
    Sample Gaussian initial conditions and package them as a Field PyTree.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator.
    mesh_size : tuple[int, int, int]
        Discretization of the simulation volume.
    box_size : tuple[float, float, float]
        Physical box side lengths (Mpc/h).
    pk_fn : callable
        Function mapping |k| to the linear matter power spectrum.
    observer_position : tuple[float, float, float], optional
        Fractional observer coordinates; defaults to box center.
    flatsky_npix : tuple[int, int], optional
        Requested flat-sky pixel resolution for downstream projections.
    nside : int, optional
        HEALPix resolution for spherical projections.
    halo_size : int | tuple[int, int], optional
        Halo exchange depth for distributed painting.
    sharding : Any, optional
        JAX sharding descriptor for distributed arrays.

    Returns
    -------
    DensityField
        Field PyTree populated with the linear density field.
    """
    if pk_fn is None:
        if cosmo is None:
            raise ValueError("Either pk_fn or cosmo must be provided to compute the power spectrum.")
        else:
            k = jnp.logspace(-4, 1, 128)
            pk = jc.power.linear_matter_power(cosmo, k)
            pk_fn = lambda x: jnp.interp(x, k, pk)
    if isinstance(initial_field, DensityField):
        mesh_size = initial_field.mesh_size
        box_size = initial_field.box_size
        initial_field = initial_field.array
    else:
        if mesh_size is None or box_size is None:
            raise ValueError("mesh_size and box_size must be provided if initial_field is not a DensityField.")

    field = fft3d(initial_field)
    kvec = fftk(field)
    kmesh = sum((kk / box_size[i] * mesh_size[i]) ** 2 for i, kk in enumerate(kvec)) ** 0.5
    factor = (mesh_size[0] * mesh_size[1] * mesh_size[2]) / (box_size[0] * box_size[1] * box_size[2])
    pkmesh = pk_fn(kmesh) * factor

    field = field * jnp.sqrt(pkmesh)
    field = ifft3d(field)

    return DensityField(
        array=field,
        mesh_size=mesh_size,
        box_size=box_size,
        observer_position=observer_position,
        field_sharding=field_sharding,
        halo_size=halo_size,
        #
        nside=nside,
        field_size=field_size,
        flatsky_npix=flatsky_npix,
        #
        scale_factors=0.0,
        status=FieldStatus.INITIAL_FIELD,
        unit=DensityUnit.DENSITY,
    )
