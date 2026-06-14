from functools import partial
from typing import Any, Literal

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
from jax._src.numpy.util import promote_dtypes_inexact

from ..fields import ParticleField
from .solvers import AbstractNBodySolver, NBodyState, _advance_time, _compute_dt_internal

AdjointType = Literal["reverse", "checkpointed"]


def _clip_to_end(tprev, tnext, t1):
    """
    Clips the computed next time step to ensure it does not exceed the final time t1, within a tolerance.

    The tolerance is set based on the data type of tnext (1e-10 for float64, otherwise 1e-6).

    Args:
        tprev: The previous time step value.
        tnext: The proposed next time step value.
        t1: The final time for integration.

    Returns:
        The adjusted next time step, set to t1 if tnext is within the tolerance of t1.
    """
    tol = 1e-10 if tnext.dtype == jnp.dtype("float64") else 1e-6
    clip = tnext > t1 - tol
    return jnp.where(clip, t1, tnext)


def _clip_to_start(tprev, tnext, t0):
    """
    Clips the computed previous time step to ensure it does not fall below the initial time t0, within a tolerance.

    The tolerance is determined by the data type of tprev (1e-10 for float64, otherwise 1e-6).

    Args:
        tprev: The computed previous time step value.
        tnext: The current time step value.
        t0: The initial time for integration.

    Returns:
        The adjusted previous time step, set to t0 if tprev is within the tolerance of t0.
    """
    tol = 1e-10 if tprev.dtype == jnp.dtype("float64") else 1e-6
    clip = tprev < t0 + tol
    return jnp.where(clip, t0, tprev)


def _boundary_t_prev(tc, t0_anchor, dt0):
    """Start time of the forward loop's final (clipped) step into snapshot ``tc``.

    The forward integrator (:func:`_fwd_loop`) re-anchors a uniform ``dt0`` grid at the
    previous snapshot ``t0_anchor`` and clips the last step into the snapshot ``tc``. Its
    final step therefore runs from ``a_last = t0_anchor + (m - 1) * dt0`` to ``tc``, where
    ``m = ceil((tc - t0_anchor) / dt0)``. The reversible backward pass must invert *that*
    step, so the boundary reverse/linearization point is ``a_last`` — not ``tc - dt0``. The
    two coincide only when the interval is an integer number of ``dt0`` steps (on-grid
    snapshots, no clip); for off-grid snapshots ``tc - dt0`` lands at the wrong time and the
    reconstructed trajectory (and step VJP) drift, giving wrong gradients.
    """
    tol = 1e-10 if jnp.asarray(tc).dtype == jnp.dtype("float64") else 1e-6
    n_full = jnp.floor((tc - t0_anchor) / dt0 - tol)  # full dt0 steps strictly below tc
    n_full = jnp.maximum(n_full, 0.0)
    t_prev = t0_anchor + n_full * dt0  # == tc - dt0 when on-grid
    return _clip_to_start(t_prev, tc, t0_anchor)


def integrate(
    displacements: ParticleField,
    velocities: ParticleField,
    cosmo: Any,
    ts: jnp.ndarray,
    solver: AbstractNBodySolver,
    t0: float,
    t1: float,
    n_steps: int,
    adjoint: AdjointType = "checkpointed",
    checkpoints: int | None = None,
) -> Any:
    """
    Main integration entry point.

    Integrates an N-body system from t0 through snapshot times ts using the provided solver.
    Supports two adjoint modes for gradient computation:
      - 'checkpointed': Memory-efficient autodiff using equinox checkpointed scan
      - 'reverse': Custom VJP with explicit backward pass using solver.reverse()

    Args:
        displacements: Displacement ParticleField from LPT.
        velocities: Velocity/momentum ParticleField from LPT.
        cosmo: Cosmology parameters.
        ts: Array of snapshot times at which to save output.
        solver: An AbstractNBodySolver instance.
        t0: Initial time for integration.
        t1: Final time for integration.
        n_steps: Number of integration steps from t0 to t1.
        adjoint: Adjoint mode, either 'checkpointed' or 'reverse'.

    Returns:
        Snapshots at each time in ts, as returned by solver.save_at().

    Raises:
        ValueError: If adjoint='reverse' is used with a non-reversible correction kernel.
        ValueError: If adjoint='reverse' is used with time_stepping != 'a'.
    """
    # Validate reversibility compatibility
    if adjoint == "reverse":
        if not solver.pgd_kernel.reversible:
            raise ValueError(
                f"Cannot use adjoint='reverse' with {type(solver.pgd_kernel).__name__}. "
                f"Use SharpeningKernel for reversible integration, or use adjoint='checkpointed'."
            )
        if solver.time_stepping != "a":
            raise ValueError(
                f"Cannot use adjoint='reverse' with time_stepping={solver.time_stepping!r}. "
                f"The custom VJP backward pass assumes uniform a-steps. Use adjoint='checkpointed'."
            )

    (ts,) = promote_dtypes_inexact(ts)

    # Compute step sizes
    dt0_a = (t1 - t0) / n_steps

    # Initialize solver OUTSIDE the loop
    disp0, vel0 = displacements, velocities
    t1_init = t0 + dt0_a
    disp, vel, state = solver.init(disp0, vel0, t0, t1_init, dt0_a, cosmo)

    # Bundle all differentiable args
    y0_cosmo_ts_solver = ((disp, vel, state), cosmo, ts, solver)

    if adjoint == "checkpointed":
        return _integrate_checkpointed(
            y0_cosmo_ts_solver, t0=t0, t1=t1, n_steps=n_steps, dt0_a=dt0_a, checkpoints=checkpoints
        )
    elif adjoint == "reverse":
        # For reverse adjoint, time_stepping must be "a" (validated above), so dt0_a is the step
        return _integrate_reverse_adjoint(y0_cosmo_ts_solver, t0=t0, t1=t1, dt0=dt0_a)
    elif adjoint == "lax":
        # For testing: just run the forward pass without custom VJP
        snapshots, _ = _fwd_loop(y0_cosmo_ts_solver, t0=t0, t1=t1, n_steps=n_steps, dt0_a=dt0_a, kind="bounded")
        return snapshots
    else:
        raise ValueError(f"Unknown adjoint type: {adjoint}")


def _integrate_checkpointed(
    y0_cosmo_ts_solver: tuple,
    *,
    t0: float,
    t1: float,
    n_steps: int,
    dt0_a: float,
    checkpoints: int | None = None,
) -> Any:
    """
    Simple forward pass with checkpointing for memory-efficient backprop.

    Uses equinox's checkpointed scan to trade computation for memory during
    automatic differentiation.

    Args:
        y0_cosmo_ts_solver: Bundled differentiable args (y0, cosmo, ts, solver).
        t0: Initial time.
        t1: Final time.
        n_steps: Number of integration steps.
        dt0_a: Step size in scale factor (for solver.step dt parameter).

    Returns:
        Snapshots at each time in ts.
    """
    snapshots, _ = _fwd_loop(
        y0_cosmo_ts_solver, t0=t0, t1=t1, n_steps=n_steps, dt0_a=dt0_a, kind="checkpointed", checkpoints=checkpoints
    )
    return snapshots


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def _integrate_reverse_adjoint(
    y0_cosmo_ts_solver: tuple,
    t0: float,
    t1: float,
    dt0: float,
) -> Any:
    """
    Integration with custom VJP using explicit backward pass.

    Only supports time_stepping="a" (validated in integrate()).
    dt0 here is dt0_a = (t1 - t0) / n_steps.

    Args:
        y0_cosmo_ts_solver: Bundled differentiable args (y0, cosmo, ts, solver).
        t0: Initial time.
        t1: Final time.
        dt0: Step size in scale factor.

    Returns:
        Snapshots at each time in ts.
    """
    n_steps = round((t1 - t0) / dt0)
    snapshots, _ = _fwd_loop(y0_cosmo_ts_solver, t0=t0, t1=t1, n_steps=n_steps, dt0_a=dt0, kind="lax")
    return snapshots


def _integrate_fwd(
    y0_cosmo_ts_solver: tuple,
    t0: float,
    t1: float,
    dt0: float,
) -> tuple[Any, tuple]:
    """
    Forward pass for the custom VJP.

    Executes forward integration and saves residuals for the backward pass.

    Args:
        y0_cosmo_ts_solver: Bundled differentiable args.
        t0: Initial time.
        t1: Final time.
        dt0: Step size in scale factor.

    Returns:
        Tuple of (snapshots, residuals) where residuals contain state needed for backward.
    """
    n_steps = round((t1 - t0) / dt0)
    snapshots, y_final = _fwd_loop(y0_cosmo_ts_solver, t0=t0, t1=t1, n_steps=n_steps, dt0_a=dt0, kind="lax")
    y0, cosmo, ts, solver = y0_cosmo_ts_solver
    return snapshots, (y_final, cosmo, ts, solver)


def _fwd_loop(
    y0_cosmo_ts_solver: tuple[tuple[ParticleField, ParticleField, NBodyState], Any, jnp.ndarray, AbstractNBodySolver],
    *,
    t0: float,
    t1: float,
    n_steps: int,
    dt0_a: float,
    kind: str = "lax",
    checkpoints: int | None = None,
) -> tuple[Any, tuple[ParticleField, ParticleField, NBodyState]]:
    """
    Forward integration loop for AbstractNBodySolver.

    The integration process is organized into two nested loops:
      - The inner loop steps forward using the solver's time_stepping variable until reaching the snapshot time.
      - The outer loop iterates over each snapshot time in ts, applying save_at to record the state.

    Args:
        y0_cosmo_ts_solver: A tuple containing:
            - y0: (disp, vel, state) tuple after solver.init() has been called
            - cosmo: Cosmology parameters (unused here, passed through for gradients)
            - ts: Array of snapshot times
            - solver: The AbstractNBodySolver instance
        t0: The starting time for integration.
        t1: The final time for integration.
        n_steps: Number of integration steps from t0 to t1.
        dt0_a: Step size in scale factor (passed to solver.step as dt parameter).
        kind: Loop type ('lax' or 'checkpointed').

    Returns:
        A tuple containing:
          - The collection of snapshots obtained via solver.save_at().
          - The final state (disp, vel, state) after integration.
    """
    (disp, vel, state), cosmo, ts, solver = y0_cosmo_ts_solver
    # n_steps is static, so max_steps is a compile-time constant
    max_steps = n_steps + 5

    # Compute step size in the solver's internal time variable
    dt_internal = _compute_dt_internal(solver.time_stepping, t0, t1, n_steps, cosmo)

    def inner_forward_step(carry):
        """Single integration step."""
        disp_, vel_, state_, t_curr, t_target = carry
        t_next = _advance_time(solver.time_stepping, t_curr, dt_internal, cosmo)
        t_next = _clip_to_end(t_curr, t_next, t_target)
        disp_next, vel_next, state_next = solver.step(disp_, vel_, t_curr, t_next, dt0_a, state_, cosmo)
        return (disp_next, vel_next, state_next, t_next, t_target)

    def inner_forward_cond(carry):
        """Continue while t_curr < t_target."""
        _, _, _, t_curr, t_target = carry
        return t_curr < t_target

    def outer_forward_step(outer_carry, t_target):
        """
        Integrate from the current time up to t_target, then save a snapshot.

        Args:
            outer_carry: (disp, vel, state, t_curr)
            t_target: The snapshot time to integrate to.

        Returns:
            Updated outer carry and the snapshot.
        """
        disp_, vel_, state_, t_curr = outer_carry
        inner_carry = (disp_, vel_, state_, t_curr, t_target)

        # Inner loop: step until reaching t_target
        disp_, vel_, state_, _, _ = eqxi.while_loop(
            inner_forward_cond, inner_forward_step, inner_carry, max_steps=max_steps, kind=kind
        )

        # Save snapshot at t_target
        snapshot = solver.save_at(disp_, vel_, t_target, dt0_a, state_, cosmo)

        outer_carry = (disp_, vel_, state_, t_target)
        return outer_carry, snapshot

    # Initialize carry
    init_carry = (disp, vel, state, t0)

    # Run outer scan over all snapshot times
    (disp_final, vel_final, state_final, _), snapshots = eqxi.scan(
        outer_forward_step, init_carry, ts, kind=kind, checkpoints=checkpoints if kind == "checkpointed" else None
    )

    y_final = (disp_final, vel_final, state_final)
    return snapshots, y_final


from typing import Any

import jax
import jax.numpy as jnp


def _integrate_bwd(
    t0: float,
    t1: float,
    dt0: float,
    residuals: tuple,
    cotangents: Any,
) -> tuple[tuple]:
    """
    Backward pass for the custom VJP, computing adjoint sensitivities.
    """
    y_final, cosmo, ts, solver = residuals
    ys_ct = cotangents

    # Partition differentiable and non-differentiable parts
    diff_cosmo, nondiff_cosmo = eqx.partition(cosmo, eqx.is_inexact_array_like)
    diff_solver, nondiff_solver = eqx.partition(solver, eqx.is_inexact_array_like)

    disp_final, vel_final, state_final = y_final
    diff_state_final, nondiff_state_final = eqx.partition(state_final, eqx.is_inexact_array_like)

    # Initialize zero adjoints
    adj_disp = jax.tree.map(jnp.zeros_like, disp_final)
    adj_vel = jax.tree.map(jnp.zeros_like, vel_final)
    adj_state = jax.tree.map(jnp.zeros_like, diff_state_final)
    adj_cosmo = jax.tree.map(jnp.zeros_like, diff_cosmo)
    adj_solver = jax.tree.map(jnp.zeros_like, diff_solver)
    adj_ts_scalar = jnp.zeros_like(ts[0])

    def inner_backward_step(carry):
        (
            disp,
            vel,
            state,
            diff_cosmo_,
            diff_solver_,
            adj_disp_,
            adj_vel_,
            adj_state_,
            adj_cosmo_,
            adj_solver_,
            t0_,
            tc,
        ) = carry

        t_prev = tc - dt0
        t_prev = _clip_to_start(t_prev, tc, t0_)

        # Reconstruct full objects for reverse
        solver_ = eqx.combine(diff_solver_, nondiff_solver)
        cosmo_ = eqx.combine(diff_cosmo_, nondiff_cosmo)

        # Reverse the forward step to get previous state
        disp_prev, vel_prev, state_prev = solver_.reverse(disp, vel, t_prev, tc, dt0, state, cosmo_)
        diff_state_prev, nondiff_state_prev = eqx.partition(state_prev, eqx.is_inexact_array_like)

        # VJP for the step function
        def _to_vjp_step(disp_in, vel_in, diff_cosmo_in, diff_solver_in, diff_state_in):
            solver_in = eqx.combine(diff_solver_in, nondiff_solver)
            cosmo_in = eqx.combine(diff_cosmo_in, nondiff_cosmo)
            # IMPORTANT: Use the nondiff part from the PREVIOUS state (reconstructed above)
            state_in = eqx.combine(diff_state_in, nondiff_state_prev)
            disp_out, vel_out, state_out = solver_in.step(disp_in, vel_in, t_prev, tc, dt0, state_in, cosmo_in)
            state_out, _ = eqx.partition(state_out, eqx.is_inexact_array_like)
            return (disp_out, vel_out, state_out)

        _, f_vjp_step = jax.vjp(_to_vjp_step, disp_prev, vel_prev, diff_cosmo_, diff_solver_, diff_state_prev)

        # Propagate adjoints backwards
        new_adj_disp, new_adj_vel, new_adj_cosmo_step, new_adj_solver_step, new_adj_state_step = f_vjp_step(
            (adj_disp_, adj_vel_, adj_state_)
        )

        # Accumulate parameter adjoints
        adj_cosmo_new = jax.tree.map(jnp.add, adj_cosmo_, new_adj_cosmo_step)
        adj_solver_new = jax.tree.map(jnp.add, adj_solver_, new_adj_solver_step)

        # FIX: Return new_adj_state_step, not adj_state_
        return (
            disp_prev,
            vel_prev,
            state_prev,
            diff_cosmo_,
            diff_solver_,
            new_adj_disp,
            new_adj_vel,
            new_adj_state_step,
            adj_cosmo_new,
            adj_solver_new,
            t0_,
            t_prev,
        )

    def inner_backward_cond(carry):
        *_, t0_, tc = carry
        return tc > t0_

    def outer_backward_step(outer_carry, vals):
        y_ct, t0_prev_snap = vals
        (
            disp,
            vel,
            state,
            diff_cosmo_,
            diff_solver_,
            adj_disp_,
            adj_vel_,
            adj_state_,
            adj_cosmo_,
            adj_solver_,
            adj_ts_scalar_,
            tc,
        ) = outer_carry

        solver_ = eqx.combine(diff_solver_, nondiff_solver)
        cosmo_ = eqx.combine(diff_cosmo_, nondiff_cosmo)
        diff_state_, nondiff_state_ = eqx.partition(state, eqx.is_inexact_array_like)

        # VJP of the snapshot function
        def _to_vjp_save(disp_in, vel_in, diff_state_in, diff_cosmo_in, diff_solver_in, tc_in):
            s_in = eqx.combine(diff_solver_in, nondiff_solver)
            c_in = eqx.combine(diff_cosmo_in, nondiff_cosmo)
            s_in_state = eqx.combine(diff_state_in, nondiff_state_)
            return s_in.save_at(disp_in, vel_in, tc_in, dt0, s_in_state, c_in)

        _, f_vjp_save = jax.vjp(_to_vjp_save, disp, vel, diff_state_, diff_cosmo_, diff_solver_, tc)
        save_adj_disp, save_adj_vel, save_adj_state, save_adj_cosmo, save_adj_solver, adj_t_save = f_vjp_save(y_ct)

        # ACCUMULATE: Add snapshot contributions to current adjoint carry
        adj_disp_sum = jax.tree.map(jnp.add, adj_disp_, save_adj_disp)
        adj_vel_sum = jax.tree.map(jnp.add, adj_vel_, save_adj_vel)
        adj_state_sum = jax.tree.map(jnp.add, adj_state_, save_adj_state)
        adj_cosmo_sum = jax.tree.map(jnp.add, adj_cosmo_, save_adj_cosmo)
        adj_solver_sum = jax.tree.map(jnp.add, adj_solver_, save_adj_solver)

        # Reverse to find the linearization point for the step VJP. The forward re-anchors its
        # dt0 grid at the previous snapshot and clips the last step INTO this snapshot, so its
        # final step runs a_last -> tc; invert exactly that step (see _boundary_t_prev), not
        # tc - dt0 (which only matches when the interval is an integer number of dt0 steps).
        t_prev = _boundary_t_prev(tc, t0_prev_snap, dt0)
        disp_prev, vel_prev, state_prev = solver_.reverse(disp, vel, t_prev, tc, dt0, state, cosmo_)

        diff_state_prev, nondiff_state_prev = eqx.partition(state_prev, eqx.is_inexact_array_like)

        # VJP of the forward step (includes time gradient)
        def _to_vjp_step_full(disp_in, vel_in, diff_state_in, diff_cosmo_in, diff_solver_in, tc_in):
            # Linearize the forward's final (clipped) step a_last -> tc. a_last is anchored at
            # the previous snapshot (independent of tc_in), so reuse the boundary t_prev.
            tp_local = t_prev
            s_in = eqx.combine(diff_solver_in, nondiff_solver)
            c_in = eqx.combine(diff_cosmo_in, nondiff_cosmo)
            s_in_state_prev = eqx.combine(diff_state_in, nondiff_state_prev)

            disp_out, vel_out, state_out = s_in.step(disp_in, vel_in, tp_local, tc_in, dt0, s_in_state_prev, c_in)
            diff_state_out = eqx.filter(state_out, eqx.is_inexact_array_like)
            return disp_out, vel_out, diff_state_out

        _, f_vjp_step = jax.vjp(_to_vjp_step_full, disp_prev, vel_prev, diff_state_prev, diff_cosmo_, diff_solver_, tc)

        # Pull the SUMMED adjoints through the step VJP
        (step_adj_disp, step_adj_vel, step_adj_state, step_adj_cosmo, step_adj_solver, step_adj_ts) = f_vjp_step(
            (adj_disp_sum, adj_vel_sum, adj_state_sum)
        )

        # Time gradient logic
        step_adj_ts = jnp.where(tc == t_prev, jnp.zeros_like(step_adj_ts), step_adj_ts)
        f_adj_ts = adj_t_save + step_adj_ts
        adj_ts_out = f_adj_ts - adj_ts_scalar_

        # Update global param adjoints
        adj_cosmo_final = jax.tree.map(jnp.add, adj_cosmo_sum, step_adj_cosmo)
        adj_solver_final = jax.tree.map(jnp.add, adj_solver_sum, step_adj_solver)

        # Run inner backward loop
        inner_carry = (
            disp_prev,
            vel_prev,
            state_prev,
            diff_cosmo_,
            diff_solver_,
            step_adj_disp,
            step_adj_vel,
            step_adj_state,  # Use step_adj_state as seed
            adj_cosmo_final,
            adj_solver_final,
            t0_prev_snap,
            t_prev,
        )

        (
            disp_out,
            vel_out,
            state_out,
            _,
            _,
            adj_disp_out,
            adj_vel_out,
            adj_state_out,
            adj_cosmo_out,
            adj_solver_out,
            _,
            t_out,
        ) = jax.lax.while_loop(inner_backward_cond, inner_backward_step, inner_carry)

        outer_carry = (
            disp_out,
            vel_out,
            state_out,
            diff_cosmo_,
            diff_solver_,
            adj_disp_out,
            adj_vel_out,
            adj_state_out,
            adj_cosmo_out,
            adj_solver_out,
            step_adj_ts,
            t_out,
        )

        return outer_carry, adj_ts_out

    # Set up reverse scan
    t1_ = ts[-1]
    t_steps = jnp.concatenate((jnp.asarray([t0]), ts[:-1]))

    init_carry = (
        disp_final,
        vel_final,
        state_final,
        diff_cosmo,
        diff_solver,
        adj_disp,
        adj_vel,
        adj_state,
        adj_cosmo,
        adj_solver,
        adj_ts_scalar,
        t1_,
    )

    vals = (ys_ct, t_steps)

    (
        (_, _, _, _, _, adj_disp_final, adj_vel_final, adj_state_final, adj_cosmo_final, adj_solver_final, _, _),
        adj_ts_contributions,
    ) = jax.lax.scan(outer_backward_step, init_carry, vals, reverse=True)

    adj_cosmo_full = eqx.combine(adj_cosmo_final, nondiff_cosmo)
    adj_solver_full = eqx.combine(adj_solver_final, nondiff_solver)
    adj_y0 = (adj_disp_final, adj_vel_final, adj_state_final)

    return ((adj_y0, adj_cosmo_full, adj_ts_contributions, adj_solver_full),)


_integrate_reverse_adjoint.defvjp(_integrate_fwd, _integrate_bwd)
