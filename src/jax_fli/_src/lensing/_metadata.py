from __future__ import annotations

import jax_cosmo as jc

from jax_fli.data.metadata import _max_z_source

__all__ = ["_attach_source_metadata", "_max_z_source"]


def _attach_source_metadata(kappas, cosmo, source_kind, sources, min_z, max_z, n_integrate):
    """Replace shell-level metadata with per-source-bin metadata on a kappa field.

    For distribution sources: z_eff = ∫ z n(z) dz / ∫ n(z) dz (composite Simpson's rule).
    For scalar sources: z_eff = the source redshift itself.
    """
    import jax.numpy as jnp

    if source_kind == "distribution":
        N = n_integrate
        z_list = []
        for nz in sources:
            z_list.append(_max_z_source(nz, min_z, max_z, N))
        z_sources = jnp.stack(z_list)
    else:
        z_sources = jnp.atleast_1d(sources)

    scale_factors = 1.0 / (1.0 + z_sources)
    comoving_centers = jc.background.radial_comoving_distance(cosmo, scale_factors)
    density_width = jnp.zeros_like(z_sources)

    return kappas.replace(
        z_sources=z_sources,
        scale_factors=scale_factors,
        comoving_centers=comoving_centers,
        density_width=density_width,
    )
