from __future__ import annotations

import numpy as np

__all__ = ["get_desy3_mask", "build_observer_visibility_mask"]

# Center observer (normalized box coords) sees the whole sky -> no visibility mask.
_OBSERVER_CENTER = (0.5, 0.5, 0.5)


def get_desy3_mask(nside: int = 2048) -> np.ndarray:
    """Load the DES Y3 footprint mask at the requested HEALPix resolution.

    Loads the Nside=2048 reference mask, up/downgrades via healpy.ud_grade,
    and returns a uint8 boolean footprint (1 = inside DES Y3 footprint).
    """
    from importlib.resources import files

    import healpy as hp

    mask_path = files("jax_fli.data").joinpath("masks/des_y3_footprint_mask_nside2048.npy")
    with mask_path.open("rb") as f:
        mask_2048 = np.load(f)
    return hp.ud_grade(mask_2048 * 1.0, nside).astype(np.uint8)


def build_observer_visibility_mask(observer_position, nside, apodization_scale_deg: float = 1.0):
    """Apodized observer visibility footprint at ``nside``, or scalar ``1`` for a center observer.

    The footprint is the set of sky directions whose ray from ``observer_position`` (normalized
    box coords in ``[0, 1]^3``) crosses the box; it is apodized with a C2 window so it can be
    multiplied onto a convergence map before a Kaiser-Squires transform (suppressing
    footprint-edge ringing). The center observer ``(0.5, 0.5, 0.5)`` sees the whole sky, so this
    returns the scalar ``1`` — an identity multiplier that JIT eliminates, so callers can multiply
    unconditionally without a None check. Shared by the full-field forward model and
    ``fli-simulate`` so the two build the *same* mask.
    """
    if tuple(float(x) for x in observer_position) == _OBSERVER_CENTER:
        return 1
    if nside is None:
        raise ValueError("observer visibility mask requires a spherical nside")

    import jax.numpy as jnp
    from jaxpm.spherical import spherical_visibility_mask

    from .apodize import apodize

    binary = spherical_visibility_mask(nside, jnp.asarray(observer_position), box_size=1.0)
    return apodize(binary, apodization_scale_deg)
