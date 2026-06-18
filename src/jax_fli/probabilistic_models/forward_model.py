"""Deterministic forward models used by the probabilistic layer."""

from __future__ import annotations

import jax

from ..data import build_observer_visibility_mask
from ..fields.painting import PaintingOptions
from ..lensing import born
from ..pm import BullFrog, DoubleKickDrift, DriftInterp, DriftKickDrift, NoCorrection, NoInterp, lpt, nbody
from .config import Configurations

__all__ = ["make_full_field_model"]

# N-body solvers selectable via ``config.nbody_solver``. DoubleKickDrift uses uniform
# a-stepping (reversible); DriftKickDrift / BullFrog use growth-factor stepping (their
# own ``time_stepping`` defaults), so the factory never overrides ``time_stepping``.
_SOLVERS = {
    "DoubleKickDrift": DoubleKickDrift,
    "DriftKickDrift": DriftKickDrift,
    "BullFrog": BullFrog,
}

_LENSING_OUTPUTS = ("convergence", "shear", "reduced_shear")


def make_full_field_model(
    config: Configurations,
):
    """Build the deterministic forward model returning the lensing observable and lightcone.

    The observable is convergence, shear, or reduced shear depending on
    ``config.lensing_output``. For an off-center observer (spherical geometry) the apodized
    observer visibility mask is applied to the kappa map *before* the Kaiser-Squires transform
    (shear / reduced_shear only) to suppress footprint-edge ringing; convergence is returned
    unmasked (its footprint enters through the likelihood mask).
    """

    geometry = config.geometry
    if geometry not in {"flat", "spherical"}:
        raise ValueError("geometry must be either 'flat' or 'spherical'")
    if config.lensing == "raytrace":
        raise NotImplementedError("The forward model supports born lensing only; set config.lensing='born'.")
    if config.lensing_output not in _LENSING_OUTPUTS:
        raise ValueError(f"lensing_output must be one of {_LENSING_OUTPUTS}, got {config.lensing_output!r}")
    if config.nbody_solver not in _SOLVERS:
        raise ValueError(f"nbody_solver must be one of {tuple(_SOLVERS)}, got {config.nbody_solver!r}")

    painting = PaintingOptions(
        target=geometry,
        scheme=config.scheme,
        paint_nside=config.paint_nside,
        kernel_width_arcmin=config.kernel_width_arcmin,
        kernel_width_pixels=config.kernel_width_pixels,
        pixel_window_deconvolution=config.pixel_window_deconvolution,
    )
    interp_cls = DriftInterp if config.drift_on_lightcone else NoInterp
    interp_kernel = interp_cls(painting=painting)

    solver = _SOLVERS[config.nbody_solver](
        pgd_kernel=NoCorrection(),
        interp_kernel=interp_kernel,
        gradient_order=config.gradient_order,
        laplace_fd=config.laplace_fd,
        order=config.paint_order,
        deconvolution=config.deconvolution,
        t0=config.t0,
        t1=config.t1,
        n_steps=config.nb_steps,
    )

    # Observer-driven visibility mask (spherical only), apodized and precomputed once at build
    # time. Default 1 (identity) so the unconditional multiply below is a no-op that JIT
    # eliminates — covers flat geometry; the helper likewise returns 1 for a center observer
    # (whole sky) and an apodized array for an off-center one.
    visibility_mask = 1
    if geometry == "spherical":
        visibility_mask = build_observer_visibility_mask(
            config.observer_position, config.paint_nside or config.nside, config.apodization_scale_deg
        )

    def forward_model(cosmo, initial_conditions):
        # warmstart NZ
        if config.nz_shear is not None:
            if isinstance(config.nz_shear, (list | tuple)):
                for nz in config.nz_shear:
                    if callable(nz) and hasattr(nz, "_norm"):
                        nz._norm = None
            else:
                nz = config.nz_shear
                if callable(nz) and hasattr(nz, "_norm"):
                    nz._norm = None

        dx_field, p_field = lpt(
            cosmo,
            initial_conditions,
            ts=config.t0,
            order=config.lpt_order,
            paint_order=config.paint_order,
            gradient_order=config.gradient_order,
            laplace_fd=config.laplace_fd,
            dealiased=config.dealiased,
            exact_growth=config.exact_growth,
        )

        lightcone = nbody(
            cosmo,
            dx_field,
            p_field,
            solver=solver,
            nb_shells=config.number_of_shells,
            shell_spacing=config.shell_spacing,
            min_width=config.min_width,
            adjoint=config.adjoint,
            checkpoints=config.checkpoints,
        )

        kappa = born(
            cosmo,
            lightcone,
            nz_shear=config.nz_shear,
            min_z=config.min_redshift,
            max_z=config.max_redshift,
        )

        # The apodized observer visibility mask is a Kaiser-Squires concern: apodizing the
        # kappa map *before* map2alm suppresses ringing from the footprint edge, so it belongs
        # on the KS *input* for shear / reduced_shear. Convergence is returned untouched (no
        # map-space apodization — the survey footprint enters via the likelihood mask instead).
        # For a center observer visibility_mask is None, so no multiplication is emitted.
        if config.lensing_output == "convergence":
            observable = kappa
        else:
            ks_input = kappa
            ks_input *= visibility_mask
            observable = ks_input.get_shear(reduced_shear=config.lensing_output == "reduced_shear")

        return observable, lightcone

    return jax.jit(forward_model)
