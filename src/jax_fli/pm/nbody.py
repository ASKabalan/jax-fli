from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax_cosmo as jc

from ..fields import FieldStatus, ParticleField
from ..fields.painting import PaintingOptions
from ._resolve_geometry import resolve_geometry
from .integrate import AdjointType, integrate
from .interp import NoInterp
from .solvers import AbstractNBodySolver, DoubleKickDrift

__all__ = ["nbody"]


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
    adjoint: AdjointType = "checkpointed",
    checkpoints: int | None = None,
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
    adjoint : AdjointType, default='checkpointed'
        Adjoint mode: 'checkpointed' or 'reverse'.
    checkpoints : int or None, optional
        Number of checkpoints for 'checkpointed' adjoint.

    Returns
    -------
    Field
        Lightcone as a stacked Field PyTree.
    """

    assert (
        dx_field.status == FieldStatus.LPT1 or dx_field.status == FieldStatus.LPT2
    ), "dx_field must have status FieldStatus.LPT1 or FieldStatus.LPT2."
    assert (
        p_field.status == FieldStatus.LPT1 or p_field.status == FieldStatus.LPT2
    ), "p_field must have status FieldStatus.LPT1 or FieldStatus.LPT2."

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

    ts_resolved, r_centers, density_plane_width = resolve_geometry(
        cosmo,
        dx_field.max_comoving_radius,
        ts=ts,
        nb_shells=nb_shells,
        density_widths=density_widths,
        shell_spacing=shell_spacing,
        min_width=min_width,
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

    return lightcone
