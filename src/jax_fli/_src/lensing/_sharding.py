"""The lensing sharding convention and its single source of truth.

Mesh axes are read **positionally**: ``M`` = the first mesh axis (carries NPIX for spherical maps and
the first spatial dim for 3-D fields), ``N`` = the second mesh axis (carries the tomographic BINS for
convergence/shear and the second spatial dim for 3-D fields). With ``M = "x"``, ``N = "y"``:

    spherical density   (B,) NPIX          -> P([None,] "x")            NPIX/M, N unused
    convergence         (B,) BINS, NPIX    -> P([None,] "y", "x")       BINS/N, NPIX/M
    shear               (B,) BINS, 2, NPIX -> P([None,] "y", None, "x") BINS/N, NPIX/M, comp replicated

To shard BINS over the N axis JAX requires ``BINS % N_size == 0`` (N_size divides BINS). When it
cannot (``N_size == 1``, a single bin, or the *bad-divisor* case ``BINS % N_size != 0``), BINS is
replicated over N and NPIX stays sharded over M; the producer (``born`` via ``apply_sharding``) warns.

``mesh_MN`` and ``lensing_axes`` are the public mesh helpers. ``_convergence_spec`` / ``_shear_spec``
are the module-private spec builders consumed by the field ``apply_sharding`` methods
(``fields/lensing_maps.py``) and by the sharded Kaiser-Squires transform (``_kaiser_squires.py``):
the field methods are the public sharding API, not these helpers.
"""

from __future__ import annotations

from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

__all__ = ["mesh_MN", "lensing_axes"]


def _is_multi_device(field_sharding) -> bool:
    """True when ``field_sharding`` spans more than one device (otherwise sharding is a no-op)."""
    mesh = getattr(field_sharding, "mesh", None)
    return field_sharding is not None and mesh is not None and mesh.size > 1


def mesh_MN(field_sharding):
    """``(M, N, M_size, N_size)`` for a field sharding's mesh (positional convention).

    ``M`` is the first mesh axis, ``N`` the second; a single-axis mesh has ``N = None``, ``N_size = 1``.
    """
    mesh = field_sharding.mesh
    names = tuple(mesh.axis_names)
    m_axis = names[0]
    n_axis = names[1] if len(names) > 1 else None
    m_size = int(mesh.shape[m_axis])
    n_size = int(mesh.shape[n_axis]) if n_axis is not None else 1
    return m_axis, n_axis, m_size, n_size


def lensing_axes(field_sharding, nbins: int):
    """``(M, N, M_size, N_size, distribute_bins)`` for ``nbins`` tomographic bins.

    ``distribute_bins`` is True iff BINS can shard over the N axis (``N_size > 1`` and
    ``nbins % N_size == 0``); otherwise BINS is replicated over N (the bad-divisor mesh included).
    """
    m_axis, n_axis, m_size, n_size = mesh_MN(field_sharding)
    distribute_bins = n_size > 1 and nbins % n_size == 0
    return m_axis, n_axis, m_size, n_size, distribute_bins


def _convergence_spec(field_sharding, nbins: int, ndim: int) -> NamedSharding:
    """Convergence ``NamedSharding`` ``P([None,] N, M)`` (BINS/N, NPIX/M, batch replicated).

    ``ndim`` is the convergence array ndim (1 for ``(NPIX,)``, 2 for ``(BINS, NPIX)``, 3 for
    ``(B, BINS, NPIX)``). BINS is replicated when N does not divide it.
    """
    m_axis, n_axis, _, _, distribute = lensing_axes(field_sharding, nbins)
    bins_axis = n_axis if distribute else None
    spec = P(m_axis) if ndim == 1 else P(*([None] * (ndim - 2)), bins_axis, m_axis)
    return NamedSharding(field_sharding.mesh, spec)


def _shear_spec(field_sharding, nbins: int, conv_ndim: int) -> NamedSharding:
    """Shear ``NamedSharding`` ``P([None,] N, None, M)``; ``conv_ndim`` = the convergence ndim."""
    m_axis, n_axis, _, _, distribute = lensing_axes(field_sharding, nbins)
    bins_axis = n_axis if distribute else None
    spec = P(None, m_axis) if conv_ndim == 1 else P(*([None] * (conv_ndim - 2)), bins_axis, None, m_axis)
    return NamedSharding(field_sharding.mesh, spec)
