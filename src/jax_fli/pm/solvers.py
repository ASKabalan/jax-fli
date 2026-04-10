from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxpm.growth import (
    Dplus_to_a,
    E,
    Gf,
    dGfa,
    gp,
    growth_factor_second,
    growth_rate,
    growth_rate_second,
)
from jaxpm.growth import growth_factor as Gp
from jaxpm.pm import pm_forces

from ..fields import ParticleField, PositionUnit
from .correction import AbstractCorrection, NoCorrection
from .interp import AbstractInterp, InterpTilerState

__all__ = [
    "NBodyState",
    "AbstractNBodySolver",
    "DoubleKickDrift",
    "DriftKickDrift",
    "BullFrog",
    "_compute_dt_internal",
    "_advance_time",
]


def _forces(pos: ParticleField, gradient_order: int = 1, laplace_fd: bool = False) -> ParticleField:
    """Compute raw PM forces (gradient of gravitational potential).

    Parameters
    ----------
    gradient_order : int
        0 = exact ik kernel, 4 = 4th-order finite-difference, 1 = alias for 4
        (backward-compatible with jaxpm's order=1 which is 4th-order FD).
    """
    # Map to jaxpm convention: 0→0 (exact ik), 4→1 (4th-order FD), 1→1 (backward compat)
    jaxpm_gradient_order = 1 if gradient_order == 4 else gradient_order
    forces_array = pm_forces(
        pos.array,
        mesh_shape=pos.mesh_size,
        paint_absolute_pos=(pos.unit == PositionUnit.GRID_ABSOLUTE),
        halo_size=pos.halo_size,
        sharding=pos.field_sharding,
        gradient_order=jaxpm_gradient_order,
        laplace_fd=laplace_fd,
    )
    return pos.replace(array=forces_array)


def _compute_midpoint(time_var: str, t0, t1, cosmo):
    """Compute midpoint scale factor based on the chosen time variable.

    Parameters
    ----------
    time_var : str
        "a" (arithmetic), "D" (growth-factor), or "log_a" (geometric).
    """
    if time_var == "a":
        return (t0 + t1) / 2.0
    elif time_var == "D":
        D_mid = (Gp(cosmo, t0) + Gp(cosmo, t1)) / 2.0
        return Dplus_to_a(cosmo, D_mid)
    elif time_var == "log_a":
        return jnp.sqrt(t0 * t1)
    else:
        raise ValueError(f"Unknown time_var: {time_var!r}")


def _compute_dt_internal(time_stepping: str, t0, t1, n_steps: int, cosmo):
    """Compute step size in the solver's internal time variable."""
    if time_stepping == "a":
        return (t1 - t0) / n_steps
    elif time_stepping == "D":
        return (Gp(cosmo, t1) - Gp(cosmo, t0)) / n_steps
    elif time_stepping == "log_a":
        return (jnp.log(t1) - jnp.log(t0)) / n_steps
    else:
        raise ValueError(f"Unknown time_stepping: {time_stepping!r}")


def _advance_time(time_stepping: str, t_curr, dt_internal, cosmo):
    """Advance t_curr by one step in the solver's internal time variable, return new a."""
    if time_stepping == "a":
        return t_curr + dt_internal
    elif time_stepping == "D":
        return jnp.squeeze(Dplus_to_a(cosmo, Gp(cosmo, t_curr) + dt_internal))
    elif time_stepping == "log_a":
        return t_curr * jnp.exp(dt_internal)
    else:
        raise ValueError(f"Unknown time_stepping: {time_stepping!r}")


def _d2_norm_factor(cosmo):
    """Correction factor K to convert jax_cosmo's D2 to physical convention.

    jax_cosmo normalizes D2 independently (D2(1)=1), but BullFrog needs
    D2 consistent with D1 such that D2 ~ -3/7 * D1^2 in EdS.
    K = D2_physical(1) / D1(1)^2, computed from the EdS limit at early times.
    """
    a_small = jnp.asarray(1e-4)
    D1_small = Gp(cosmo, a_small)
    D2_small = growth_factor_second(cosmo, a_small)
    return -3.0 / 7.0 * D1_small**2 / D2_small


class NBodyState(eqx.Module):
    """
    Combined state for N-body integration (auxiliary states).

    Attributes
    ----------
    interp_state : InterpTilerState
        State for the interpolation kernel.
    t_initial : float
        Initial time of the simulation (used for clipping kicks).
    """

    interp_state: InterpTilerState
    t_initial: float


class AbstractNBodySolver(eqx.Module):
    """
    Abstract base class for N-body solvers.

    Note: cosmo and dt are passed as method parameters, not stored on the class.
    This allows the solver to be used with different cosmologies and time steps.
    """

    interp_kernel: AbstractInterp
    pgd_kernel: AbstractCorrection = NoCorrection()
    gradient_order: int = eqx.field(static=True, default=1)
    laplace_fd: bool = eqx.field(static=True, default=False)
    time_stepping: str = eqx.field(static=True, default="a")
    t0: float | None = eqx.field(static=True, default=0.001)
    t1: float = eqx.field(static=True, default=1.0)
    n_steps: int | None = eqx.field(static=True, default=None)

    @abstractmethod
    def init(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t0: float,
        t1: float,
        dt: float,
        cosmo: Any,
    ) -> tuple[ParticleField, ParticleField, NBodyState]:
        """Initialize solver state and perform first kick."""
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t0: float,
        t1: float,
        dt: float,
        state: NBodyState,
        cosmo: Any,
    ) -> tuple[ParticleField, ParticleField, NBodyState]:
        """Perform one integration step."""
        raise NotImplementedError

    @abstractmethod
    def save_at(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t1: float,
        dt: float,
        state: NBodyState,
        cosmo: Any,
    ) -> Any:
        """Extract output at the current step."""
        raise NotImplementedError

    @abstractmethod
    def reverse(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t0: float,
        t1: float,
        dt: float,
        state: NBodyState,
        cosmo: Any,
    ) -> tuple[ParticleField, ParticleField, NBodyState]:
        """Reverse step."""
        raise NotImplementedError


class DoubleKickDrift(AbstractNBodySolver):
    """
    Reversible symplectic KKD solver with PGD correction storage.

    Structure:
    - Step: Kick (prev_mid -> current) + Kick (current -> next_mid) -> PGD Boost -> Drift
    - Init: No-op physically, effectively sets up state such that the first 'prev' kick is zero.

    This ensures reversibility and correct time synchronization with the integrator loop.
    """

    def __check_init__(self):
        if self.time_stepping != "a":
            raise ValueError(
                f"DoubleKickDrift requires time_stepping='a', got {self.time_stepping!r}. "
                f"DoubleKickDrift uses dt for t_prev = t0 - dt, which assumes uniform a-steps."
            )

    def init(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t0: float,
        t1: float,
        dt: float,
        cosmo: Any,
    ) -> tuple[ParticleField, ParticleField, NBodyState]:
        """
        Initialization: No physical update.
        Sets t_initial in state to handle boundary conditions for kicks.
        """
        interp_state = self.interp_kernel.init()
        return displacement, velocities, NBodyState(interp_state=interp_state, t_initial=t0)

    def step(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t0: float,
        t1: float,
        dt: float,
        state: NBodyState,
        cosmo: Any,
    ) -> tuple[ParticleField, ParticleField, NBodyState]:
        """
        Step: Forces -> Double Kick -> PGD Boost -> Drift
        """
        # Times
        # t0 is current time (x_i), t1 is next time (x_{i+1})
        # We need to construct the previous interval midpoint for the kick.
        # We assume dt was constant or use t0-dt.
        # t_prev = t0 - dt. Midpoint is geometric mean (t_prev * t0)^0.5
        t_prev = t0 - dt
        t_prev_clamped = jnp.maximum(t_prev, state.t_initial)

        # Midpoints
        ac = t0  # Force evaluation time

        # Previous half-step (K1)
        ai_1 = (t_prev_clamped * t0) ** 0.5
        af_1 = t0

        # Next half-step (K2)
        ai_2 = t0
        af_2 = (t0 * t1) ** 0.5

        # 1. Forces at current position (x_i at t0)
        forces = _forces(displacement, self.gradient_order, self.laplace_fd) * (1.5 * cosmo.Omega_m)

        # 2. Compute Kick Factor
        # Note: If t0 == t_initial, then t_prev_clamped = t0, ai_1 = t0.
        # So K1 term Gf(t0)-Gf(t0) becomes 0.
        k1 = (Gf(cosmo, af_1) - Gf(cosmo, ai_1)) / dGfa(cosmo, ac)
        k2 = (Gf(cosmo, af_2) - Gf(cosmo, ai_2)) / dGfa(cosmo, ac)

        kick_factor = k1 + k2
        prefactor_kick = 1.0 / (ac**2 * E(cosmo, ac))

        dvel = forces * (prefactor_kick * kick_factor)

        # v_mid: velocity updated to the middle of the drift interval (af_2)
        v_mid = (velocities + dvel).replace(scale_factors=jnp.atleast_1d(af_2))

        # 3. PGD Boost (at x_i) + Drift (x_i -> x_{i+1})
        ai, ac_drift, af = t0, af_2, t1
        drift_factor = (Gp(cosmo, af) - Gp(cosmo, ai)) / gp(cosmo, ac_drift)
        prefactor_drift = 1.0 / (ac_drift**3 * E(cosmo, ac_drift))
        full_drift_factor = prefactor_drift * drift_factor

        # Apply PGD Boost at t0
        x_curr, v_boosted = self.pgd_kernel.apply(t0, displacement, v_mid, full_drift_factor=full_drift_factor)

        dpos = v_boosted * full_drift_factor

        x_new = (displacement + dpos).replace(scale_factors=jnp.atleast_1d(af))

        # 4. Advance Interp
        new_interp_state = self.interp_kernel.advance(state.interp_state, t1)

        new_state = NBodyState(interp_state=new_interp_state, t_initial=state.t_initial)

        return x_new, v_boosted, new_state

    def save_at(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t1: float,
        dt: float,
        state: NBodyState,
        cosmo: Any,
    ) -> Any:
        # In this scheme, t1 is the current time of the particles
        return self.interp_kernel.paint(state.interp_state, t1, (displacement, velocities), cosmo)

    def reverse(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t0: float,
        t1: float,
        dt: float,
        state: NBodyState,
        cosmo: Any,
    ) -> tuple[ParticleField, ParticleField, NBodyState]:
        """
        Reverse Step: Un-Drift -> Un-Boost -> Un-Kick
        Input: x_{i+1} (t1), v_{mid_boosted}
        Target: x_i (t0), v_{old}
        """
        # Reconstruct drift parameters
        # t0 is prev time (target), t1 is current time (source)
        # Note: in reverse, t0 and t1 are swapped relative to forward sense in integration loop?
        # integrate.py reverse loop passes: t_prev, tc.
        # t_prev becomes t0 (target), tc becomes t1 (source).

        # Drift calc
        t0t1 = (t0 * t1) ** 0.5
        ai, ac_drift, af = t0, t0t1, t1

        drift_factor = (Gp(cosmo, af) - Gp(cosmo, ai)) / gp(cosmo, ac_drift)
        prefactor_drift = 1.0 / (ac_drift**3 * E(cosmo, ac_drift))
        full_drift_factor = prefactor_drift * drift_factor

        # 1. Un-Drift (x_{i+1} -> x_i) using boosted velocity
        dpos = velocities * full_drift_factor
        x_old = (displacement - dpos).replace(scale_factors=jnp.atleast_1d(t0))

        # 2. Un-Boost (at x_i)
        # Recover unboosted velocity. Note: rewinding at t0 (target time)
        x_uncorrected, v_uncorrected = self.pgd_kernel.rewind(
            t0, x_old, velocities, full_drift_factor=full_drift_factor
        )

        # 3. Un-Kick
        # Need to reconstruct kick factors identical to forward pass
        t_prev = t0 - dt
        t_prev_clamped = jnp.maximum(t_prev, state.t_initial)

        # Previous half-step (K1)
        ai_1 = (t_prev_clamped * t0) ** 0.5
        af_1 = t0

        # Next half-step (K2)
        ai_2 = t0
        af_2 = t0t1  # (t0 * t1)**0.5

        ac_kick = t0

        k1 = (Gf(cosmo, af_1) - Gf(cosmo, ai_1)) / dGfa(cosmo, ac_kick)
        k2 = (Gf(cosmo, af_2) - Gf(cosmo, ai_2)) / dGfa(cosmo, ac_kick)

        kick_factor = k1 + k2
        prefactor_kick = 1.0 / (ac_kick**2 * E(cosmo, ac_kick))

        forces = _forces(x_uncorrected, self.gradient_order, self.laplace_fd) * (1.5 * cosmo.Omega_m)
        dvel = forces * (prefactor_kick * kick_factor)

        v_old = v_uncorrected - dvel  # Time unit isn't critical for v return in reverse

        # 4. Rewind Interp State
        assert self.interp_kernel.ts is not None
        max_ts = self.interp_kernel.ts[-1]
        t1_clipped = jnp.minimum(t1, max_ts)
        prev_interp_state = self.interp_kernel.rewind(state.interp_state, t1_clipped)
        prev_state = NBodyState(interp_state=prev_interp_state, t_initial=state.t_initial)

        return x_old, v_old, prev_state


class DriftKickDrift(AbstractNBodySolver):
    """
    FastPM DKD pi-integrator (Drift-Kick-Drift).

    Operates with D-time velocity pi = dx/dD instead of conformal momentum.
    Converts from conformal momentum in init(). Matches disco-dj's "fastpm" stepper.

    Parameters
    ----------
    time_stepping : str
        Time variable for step spacing and midpoint computation:
        "a", "D", or "log_a". Default is "D" (growth-factor midpoints).

    References
    ----------
    - arXiv:1603.00476 (FastPM)
    - arXiv:2301.09655 (Pi-integrator)
    """

    time_stepping: str = eqx.field(static=True, default="D")

    def _compute_coefficients(self, t0, t1, cosmo):
        """Compute FastPM DKD coefficients for a step from scale factor t0 to t1."""
        a_mid = _compute_midpoint(self.time_stepping, t0, t1, cosmo)

        # FastPM alpha = Gf(begin) / Gf(end)
        alpha = Gf(cosmo, t0) / Gf(cosmo, t1)

        # Beta from Zel'dovich consistency
        D_mid = Gp(cosmo, a_mid)
        beta = (1.0 - alpha) / D_mid

        # Drift in normalized D-time
        D_begin = Gp(cosmo, t0)
        D_end = Gp(cosmo, t1)
        ddrift1 = D_mid - D_begin
        ddrift2 = D_end - D_mid

        return alpha, beta, ddrift1, ddrift2

    def init(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t0: float,
        t1: float,
        dt: float,
        cosmo: Any,
    ) -> tuple[ParticleField, ParticleField, NBodyState]:
        """Convert conformal momentum to D-time velocity pi = p / Gf."""
        gf_t0 = Gf(cosmo, t0)
        pi = velocities * (1.0 / gf_t0)
        interp_state = self.interp_kernel.init()
        return displacement, pi, NBodyState(interp_state=interp_state, t_initial=t0)

    def step(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t0: float,
        t1: float,
        dt: float,
        state: NBodyState,
        cosmo: Any,
    ) -> tuple[ParticleField, ParticleField, NBodyState]:
        """DKD step: Drift1 -> Kick -> Drift2."""
        alpha, beta, ddrift1, ddrift2 = self._compute_coefficients(t0, t1, cosmo)

        # Drift 1
        x_mid = displacement + velocities * ddrift1

        # Kick (raw forces without 3/2 Omega_m — absorbed into pi-integrator formulation)
        forces = _forces(x_mid, self.gradient_order, self.laplace_fd)
        pi_new = velocities * alpha + forces * beta

        # Drift 2
        x_new = (x_mid + pi_new * ddrift2).replace(scale_factors=jnp.atleast_1d(t1))

        # Advance interp state
        new_interp_state = self.interp_kernel.advance(state.interp_state, t1)
        new_state = NBodyState(interp_state=new_interp_state, t_initial=state.t_initial)

        return x_new, pi_new, new_state

    def save_at(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t1: float,
        dt: float,
        state: NBodyState,
        cosmo: Any,
    ) -> Any:
        return self.interp_kernel.paint(state.interp_state, t1, (displacement, velocities), cosmo)

    def reverse(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t0: float,
        t1: float,
        dt: float,
        state: NBodyState,
        cosmo: Any,
    ) -> tuple[ParticleField, ParticleField, NBodyState]:
        """Reverse DKD step: Un-Drift2 -> Un-Kick -> Un-Drift1."""
        alpha, beta, ddrift1, ddrift2 = self._compute_coefficients(t0, t1, cosmo)

        # Un-Drift 2
        x_mid = displacement - velocities * ddrift2

        # Un-Kick (raw forces without 3/2 Omega_m — matches forward step)
        forces = _forces(x_mid, self.gradient_order, self.laplace_fd)
        pi_old = (velocities - forces * beta) * (1.0 / alpha)

        # Un-Drift 1
        x_old = (x_mid - pi_old * ddrift1).replace(scale_factors=jnp.atleast_1d(t0))

        # Rewind interp state
        assert self.interp_kernel.ts is not None
        max_ts = self.interp_kernel.ts[-1]
        t1_clipped = jnp.minimum(t1, max_ts)
        prev_interp_state = self.interp_kernel.rewind(state.interp_state, t1_clipped)
        prev_state = NBodyState(interp_state=prev_interp_state, t_initial=state.t_initial)

        return x_old, pi_old, prev_state


class BullFrog(AbstractNBodySolver):
    """
    BullFrog DKD integrator.

    Operates with D-time velocity pi = dx/dD instead of conformal momentum.
    Converts from conformal momentum in init(). Always uses NoCorrection.

    Parameters
    ----------
    time_stepping : str
        Time variable for step spacing and midpoint computation:
        "a", "D", or "log_a". Default is "D" (growth-factor midpoints).

    References
    ----------
    - arXiv:2409.19049 (BullFrog)
    - arXiv:2301.09655 (Pi-integrator)
    """

    time_stepping: str = eqx.field(static=True, default="D")

    def _compute_coefficients(self, t0, t1, cosmo):
        """Compute BullFrog DKD coefficients for a step from scale factor t0 to t1."""
        # Correction factor: jax_cosmo normalizes D2 independently (D2(1)=1).
        # BullFrog needs D2 ~ -3/7 * D1^2 (physical convention).
        K = _d2_norm_factor(cosmo)

        # Growth factors
        D1_begin = Gp(cosmo, t0)
        D1_end = Gp(cosmo, t1)
        D2_begin = growth_factor_second(cosmo, t0) * K
        D2_end = growth_factor_second(cosmo, t1) * K

        # Growth rates: f = d ln D / d ln a (normalization-independent)
        f1_begin = growth_rate(cosmo, t0)
        f1_end = growth_rate(cosmo, t1)
        f2_begin = growth_rate_second(cosmo, t0)
        f2_end = growth_rate_second(cosmo, t1)

        # R(a) = (dD2/da) / (dD1/da) = f2*D2 / (f1*D1), using corrected D2
        R_begin = f2_begin * D2_begin / (f1_begin * D1_begin)
        R_end = f2_end * D2_end / (f1_end * D1_end)

        # D-time quantities
        dD = D1_end - D1_begin
        xi = D1_begin / dD

        # BullFrog bracket
        bracket = (D2_begin + R_begin * dD / 2.0) / ((xi + 0.5) * dD) - (xi + 0.5) * dD

        # Alpha coefficient (independent of time_var)
        alpha = (R_end - bracket) / (R_begin - bracket)

        # Midpoint from time_stepping
        a_mid = _compute_midpoint(self.time_stepping, t0, t1, cosmo)
        D_mid = Gp(cosmo, a_mid)

        # Beta from Zel'dovich consistency
        beta = (1.0 - alpha) / D_mid

        # Drift lengths (normalized D1)
        ddrift1 = D_mid - D1_begin
        ddrift2 = D1_end - D_mid

        return alpha, beta, ddrift1, ddrift2

    def init(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t0: float,
        t1: float,
        dt: float,
        cosmo: Any,
    ) -> tuple[ParticleField, ParticleField, NBodyState]:
        """Convert conformal momentum to D-time velocity pi = p / Gf."""
        gf_t0 = Gf(cosmo, t0)
        pi = velocities * (1.0 / gf_t0)
        interp_state = self.interp_kernel.init()
        return displacement, pi, NBodyState(interp_state=interp_state, t_initial=t0)

    def step(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t0: float,
        t1: float,
        dt: float,
        state: NBodyState,
        cosmo: Any,
    ) -> tuple[ParticleField, ParticleField, NBodyState]:
        """DKD step: Drift1 -> Kick -> Drift2."""
        alpha, beta, ddrift1, ddrift2 = self._compute_coefficients(t0, t1, cosmo)

        # Drift 1
        x_mid = displacement + velocities * ddrift1

        # Kick (raw forces without 3/2 Omega_m — absorbed into BullFrog beta)
        forces = _forces(x_mid, self.gradient_order, self.laplace_fd)
        pi_new = velocities * alpha + forces * beta

        # Drift 2
        x_new = (x_mid + pi_new * ddrift2).replace(scale_factors=jnp.atleast_1d(t1))

        # Advance interp state
        new_interp_state = self.interp_kernel.advance(state.interp_state, t1)
        new_state = NBodyState(interp_state=new_interp_state, t_initial=state.t_initial)

        return x_new, pi_new, new_state

    def save_at(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t1: float,
        dt: float,
        state: NBodyState,
        cosmo: Any,
    ) -> Any:
        return self.interp_kernel.paint(state.interp_state, t1, (displacement, velocities), cosmo)

    def reverse(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t0: float,
        t1: float,
        dt: float,
        state: NBodyState,
        cosmo: Any,
    ) -> tuple[ParticleField, ParticleField, NBodyState]:
        """Reverse DKD step: Un-Drift2 -> Un-Kick -> Un-Drift1."""
        alpha, beta, ddrift1, ddrift2 = self._compute_coefficients(t0, t1, cosmo)

        # Un-Drift 2
        x_mid = displacement - velocities * ddrift2

        # Un-Kick (raw forces without 3/2 Omega_m — matches forward step)
        forces = _forces(x_mid, self.gradient_order, self.laplace_fd)
        pi_old = (velocities - forces * beta) * (1.0 / alpha)

        # Un-Drift 1
        x_old = (x_mid - pi_old * ddrift1).replace(scale_factors=jnp.atleast_1d(t0))

        # Rewind interp state
        assert self.interp_kernel.ts is not None
        max_ts = self.interp_kernel.ts[-1]
        t1_clipped = jnp.minimum(t1, max_ts)
        prev_interp_state = self.interp_kernel.rewind(state.interp_state, t1_clipped)
        prev_state = NBodyState(interp_state=prev_interp_state, t_initial=state.t_initial)

        return x_old, pi_old, prev_state
