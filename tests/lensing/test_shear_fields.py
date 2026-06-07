"""Shear-field API contracts: the spin-2 guards on ``FlatShearField`` / ``SphericalShearField``.

These are *not* Kaiser-Squires round-trips (those live in ``test_kaiser_squires.py``). They pin the
field-level behaviour that the shear types add on top of their scalar bases:

- ``FlatShearField`` has **no flat-sky spin-2 power**, so the scalar ``FlatDensity`` helpers it would
  otherwise inherit (``angular_cl``, ``ud_sample``, ...) are disabled and must raise rather than return
  a silently-wrong scalar result.
- Indexing a **single** (non-batched) shear map must never slice into the spin-2 component axis.
"""

from __future__ import annotations

import pytest


def test_flat_shear_scalar_methods_raise(born_flat_kappa_single):
    """FlatShearField disables the inherited scalar FlatDensity helpers (no flat spin-2 power)."""
    sh = born_flat_kappa_single[0].get_shear()
    with pytest.raises(NotImplementedError):
        sh.angular_cl()
    with pytest.raises(NotImplementedError):
        sh.ud_sample(32)


def test_shear_getitem_keeps_component_axis(born_kappa_multi):
    """Indexing a batched shear selects a bin; indexing a single map never slices the spin-2 axis."""
    sh = born_kappa_multi.get_shear()  # (S, 2, npix)
    npix = sh.array.shape[-1]
    single = sh[0]  # indexes the batch -> (2, npix)
    assert single.array.shape == (2, npix)
    with pytest.warns(UserWarning):
        guarded = single[0]  # guarded: returns self, not a sliced (npix,)
    assert guarded.array.shape == (2, npix)
