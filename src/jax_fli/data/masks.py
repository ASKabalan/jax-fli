from __future__ import annotations

import numpy as np

__all__ = ["get_desy3_mask"]


def get_desy3_mask(nside: int = 2048) -> np.ndarray:
    """Load the DES Y3 footprint mask at the requested HEALPix resolution.

    Loads the Nside=2048 reference mask, up/downgrades via healpy.ud_grade,
    and returns a uint8 boolean footprint (1 = inside DES Y3 footprint).
    """
    import healpy as hp
    from importlib.resources import files

    mask_path = files("jax_fli.data").joinpath("masks/des_y3_footprint_mask_nside2048.npy")
    with mask_path.open("rb") as f:
        mask_2048 = np.load(f)
    return hp.ud_grade(mask_2048 * 1.0, nside).astype(np.uint8)
