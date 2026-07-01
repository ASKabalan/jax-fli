"""Deterministic forward models used by the probabilistic layer."""

from __future__ import annotations

import warnings

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

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

# Simulation pipeline depth: "pm" runs LPT then N-body; "lpt" paints the lightcone from LPT alone.
_SIM_MODES = ("pm", "lpt")


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
    if config.lensing_output not in _LENSING_OUTPUTS:
        raise ValueError(f"lensing_output must be one of {_LENSING_OUTPUTS}, got {config.lensing_output!r}")
    if config.nbody_solver not in _SOLVERS:
        raise ValueError(f"nbody_solver must be one of {tuple(_SOLVERS)}, got {config.nbody_solver!r}")
    if config.sim_mode not in _SIM_MODES:
        raise ValueError(f"sim_mode must be one of {_SIM_MODES}, got {config.sim_mode!r}")

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
        time_stepping=config.time_stepping,
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

    # ===========================================================================
    # Check for CUDA availability if map2alm_method is set to 'jax_cuda'
    # ===========================================================================
    map2alm_method = config.map2alm_method
    if jax.devices()[0].platform == "cpu" and map2alm_method == "jax_cuda":
        warnings.warn("map2alm_method 'jax_cuda' is not available on CPU. Using 'jax' instead.")
        map2alm_method = "jax"

    if map2alm_method == "jax_cuda":
        try:
            from s2fft_lib import _s2fft

            if not _s2fft.COMPILED_WITH_CUDA:
                warnings.warn(
                    "map2alm_method 'jax_cuda' is not available because s2fft was not compiled with CUDA. Using 'jax' instead."
                )
                map2alm_method = "jax"
        except ImportError:
            warnings.warn(
                "map2alm_method 'jax_cuda' is not available because s2fft is not installed. Using 'jax' instead."
            )
            map2alm_method = "jax"
    # ===========================================================================

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

        if config.sim_mode == "lpt":
            # LPT-only: paint the lightcone directly from a single multi-shell LPT call (no N-body).
            lightcone, _ = lpt(
                cosmo,
                initial_conditions,
                nb_shells=config.number_of_shells,
                order=config.lpt_order,
                painting=painting,
                shell_spacing=config.shell_spacing,
                min_width=config.min_width,
                paint_order=config.paint_order,
                gradient_order=config.gradient_order,
                laplace_fd=config.laplace_fd,
                dealiased=config.dealiased,
                exact_growth=config.exact_growth,
            )
        else:
            # pm: LPT snapshot at t0 -> N-body integration painting the lightcone.
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
            n_integrate=config.n_integrate,
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
            # Run Kaiser-Squires UNSHARDED: replicate the convergence so get_shear takes the plain
            # (differentiable) vmap path. The sharded shard_map path is correct in the forward but NOT
            # differentiable — s2fft's ``_ftm_to_flm`` transpose drops the shard_map manual-axis sharding
            # (a cotangent-sharding mismatch under value_and_grad). The shear observable comes out
            # REPLICATED; the gradient flows back through the replicate (XLA constrains the cotangent
            # back to the convergence layout), so the IC / density gradient upstream stays sharded.
            # This has to be fixed in s2fft
            if ks_input.field_sharding is not None and ks_input.field_sharding.mesh.size > 1:
                replicated = NamedSharding(ks_input.field_sharding.mesh, P(*([None] * ks_input.array.ndim)))
                ks_input = ks_input.replace(
                    array=jax.lax.with_sharding_constraint(ks_input.array, replicated), field_sharding=None
                )
            observable = ks_input.get_shear(
                reduced_shear=config.lensing_output == "reduced_shear", method=map2alm_method
            )

        return observable, lightcone

    return jax.jit(forward_model)
