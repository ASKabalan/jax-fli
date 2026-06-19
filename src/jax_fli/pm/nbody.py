from __future__ import annotations

import warnings
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax_cosmo as jc

from ..fields import FieldStatus, ParticleField, SphericalDensity
from ..fields.painting import PaintingOptions
from ._resolve_geometry import resolve_geometry
from .integrate import AdjointType, integrate
from .interp import NoInterp
from .solvers import AbstractNBodySolver, DoubleKickDrift

__all__ = ["nbody"]


def _warn_if_spherical_npix_unsharded(field):
    """Warn when a **spherical** density's npix (M = the first mesh axis) sits on a size-1 axis.

    The HEALPix map is partitioned over M only, so an ``(1, N)`` mesh leaves the spherical density
    unsharded across devices. Fires once per produced spherical lightcone; no-op for flat/volumetric
    targets and on a single device.
    """
    field_sharding = getattr(field, "field_sharding", None)
    mesh = getattr(field_sharding, "mesh", None)
    if mesh is None or mesh.size <= 1 or not isinstance(field, SphericalDensity):
        return
    if int(mesh.shape[mesh.axis_names[0]]) == 1:  # M = first mesh axis carries npix for spherical maps
        warnings.warn(
            "nbody: the mesh's first (M) axis has size 1, so the spherical density npix is not sharded "
            "across devices (the HEALPix map is partitioned over M only). Use a (devices, 1) or "
            "(devices // n_bins, n_bins) mesh to shard the spherical density over pixels.",
            stacklevel=2,
        )


def _validate_t0_cb(lpt_t0, t0):
    lpt_t0 = jnp.atleast_1d(lpt_t0).squeeze()
    if not jnp.isscalar(t0) or not jnp.isscalar(lpt_t0):
        raise ValueError("Starting scale factor t0 and LPT fields' scale factor must be scalars.")
    if lpt_t0 != t0:
        raise ValueError(f"Starting scale factor t0={t0} does not match LPT fields' scale factor {lpt_t0}.")


@partial(
    jax.jit,
    static_argnames=[
        "nb_shells",
        "adjoint",
        "checkpoints",
        "step_checkpoints",
        "shell_spacing",
        "min_width",
    ],
)
def nbody(
    cosmo,
    dx_field: ParticleField,
    p_field: ParticleField,
    *,
    solver: AbstractNBodySolver = DoubleKickDrift(interp_kernel=NoInterp(painting=PaintingOptions(target="particles"))),
    ts=None,
    nb_shells: int | None = None,
    density_widths=None,
    shell_spacing: str = "a",
    min_width: float = 50.0,
    adjoint: AdjointType | None = "checkpointed",
    checkpoints: int | None = None,
    step_checkpoints: int | None = None,
) -> jax.Array:
    """
    Evolve particles forward in time and save lightcone density planes.

    Takes LPT displacements and momenta, runs N-body integration using
    a symplectic solver, and returns density planes at shell centers.

    Parameters
    ----------
    cosmo : jax_cosmo.Cosmology
        Cosmology for background expansion.
    dx_field : ParticleField
        Displacement field from LPT.
    p_field : ParticleField
        Momentum field from LPT.
    solver : AbstractNBodySolver
        Solver instance with t0, t1 configured.
    ts : float, 1-D array, or 2-D ``(2, N)`` array, optional
        Scale factor specification. Mutually exclusive with *nb_shells*.
    nb_shells : int, optional
        Number of radial lightcone shells (alternative to *ts*).
    density_widths : float or array, optional
        Override shell widths.
    adjoint : AdjointType or None, default='checkpointed'
        'checkpointed' or 'reverse' for reverse-mode gradients (``grad``/``vjp``); ``None`` selects a
        plain ``jax.lax`` forward loop that supports forward-mode AD (``jvp``/``jacfwd``) but is not
        reverse-differentiable.
    checkpoints : int or None, optional
        Number of checkpoints for the *shell* scan (the outer loop over saved snapshots),
        used by the 'checkpointed' adjoint. ``None`` = equinox default.
    step_checkpoints : int or None, optional
        Number of checkpoints for the *integration-step* loop within each shell, used by the
        'checkpointed' adjoint. ``None`` = equinox default. This is distinct from *checkpoints*,
        which checkpoints the shell scan; *step_checkpoints* trades recompute for memory along the
        time-integration steps. Has no effect on the 'reverse' adjoint.

    Returns
    -------
    Field
        Lightcone as a stacked Field PyTree.
    """

    assert dx_field.status == FieldStatus.LPT1 or dx_field.status == FieldStatus.LPT2, (
        "dx_field must have status FieldStatus.LPT1 or FieldStatus.LPT2."
    )
    assert p_field.status == FieldStatus.LPT1 or p_field.status == FieldStatus.LPT2, (
        "p_field must have status FieldStatus.LPT1 or FieldStatus.LPT2."
    )

    if dx_field.mesh_size != p_field.mesh_size:
        raise ValueError("dx_field and p_field must have matching mesh_size")
    if dx_field.box_size != p_field.box_size:
        raise ValueError("dx_field and p_field must have matching box_size")

    # Derive t0 from solver or from LPT fields
    t0 = solver.t0
    if t0 is None:
        raise ValueError("t0 must be set on the solver before calling nbody().")
    t1 = solver.t1
    if t1 is None:
        raise ValueError("t1 must be set on the solver before calling nbody().")

    # Check that t0 matches the LPT fields' scale factor
    jax.debug.callback(_validate_t0_cb, dx_field.scale_factors, t0)

    # Always resolve geometry through resolve_geometry
    if ts is None and nb_shells is None:
        ts = jnp.array([t1])  # snapshot default

    # Sanity check, if ts is a scalar and painting target is flat or spherical, raise error if density_width is None because it is required for lightcone output
    if jnp.isscalar(ts) and solver.interp_kernel.painting.target in ("flat", "spherical") and density_widths is None:
        raise ValueError(
            f"Scalar ts={ts} with painting target {solver.interp_kernel.painting.target} requires density_widths to be set for lightcone output."
        )
    ts_resolved, r_centers, density_plane_width = resolve_geometry(
        cosmo,
        dx_field.max_comoving_radius,
        ts=ts,
        nb_shells=nb_shells,
        density_widths=density_widths,
        shell_spacing=shell_spacing,
        min_width=min_width,
        box_size_z=dx_field.box_size[2],
    )
    updated_interp = solver.interp_kernel.update_geometry(
        ts=ts_resolved,
        r_centers=r_centers,
        density_widths=density_plane_width,
        max_comoving_distance=dx_field.max_comoving_radius,
    )
    solver = eqx.tree_at(lambda s: s.interp_kernel, solver, updated_interp)

    if solver.n_steps is None:
        raise ValueError("n_steps must be set on the solver before calling nbody().")
    n_steps_final = solver.n_steps

    if nb_shells is not None and nb_shells > n_steps_final:
        raise ValueError(
            f"nb_shells={nb_shells} exceeds n_steps={n_steps_final}. "
            f"The number of shells cannot exceed the number of integration steps."
        )

    # Run integration
    lightcone = integrate(
        displacements=dx_field,
        velocities=p_field,
        cosmo=cosmo,
        ts=ts_resolved,
        solver=solver,
        t0=t0,
        t1=t1,
        n_steps=n_steps_final,
        adjoint=adjoint,
        checkpoints=checkpoints,
        step_checkpoints=step_checkpoints,
    )

    if ts_resolved.ndim == 1 and ts_resolved.size == 1:
        # Add singleton plane dimension for consistent output shape
        lightcone = lightcone.apply_fn(lambda x: x.squeeze(axis=0))
        r_centers = r_centers.squeeze(axis=0)
        density_plane_width = density_plane_width.squeeze(axis=0)

    # Reverse to get near-to-far ordering
    if lightcone.is_batched():
        lightcone = lightcone[::-1]
        r_centers = r_centers[::-1]
        density_plane_width = density_plane_width[::-1]

    # Set metadata
    scale_factors = lightcone.scale_factors
    z_sources = jc.utils.a2z(scale_factors)
    lightcone = lightcone.replace(
        z_sources=z_sources, comoving_centers=r_centers, density_width=density_plane_width, status=FieldStatus.LIGHTCONE
    )

    # Post-paint HEALPix window deconvolution (a_lm level), distinct from the 3D force window.
    painting = solver.interp_kernel.painting
    if painting.target == "spherical" and painting.pixel_window_deconvolution:
        lightcone = lightcone.deconvolve(
            method=painting.scheme,
            kernel_width_arcmin=painting.kernel_width_arcmin,
            kernel_width_pixels=painting.kernel_width_pixels,
            smoothing_interpretation=painting.smoothing_interpretation,
        )

    _warn_if_spherical_npix_unsharded(lightcone)
    return lightcone
