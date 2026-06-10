from __future__ import annotations

import itertools
import warnings
from abc import abstractmethod
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import jax_healpy as jhp

from .._src.base._warn import warning_if
from ..fields import DensityUnit, FieldStatus, ParticleField, PositionUnit, SphericalDensity
from ..fields.painting import PaintingOptions

__all__ = [
    "InterpTilerState",
    "OnionTiler",
    "TelephotoInterp",
    "NoInterp",
    "DriftInterp",
    "get_all_47_symmetries",
    "AbstractInterp",
]


def get_all_47_symmetries() -> jnp.ndarray:
    """Generate all 48 cubic symmetry matrices."""
    permutations = list(itertools.permutations([0, 1, 2]))
    signs = list(itertools.product([1, -1], repeat=3))
    matrices = []
    for s in signs:
        for p in permutations:
            M = jnp.zeros((3, 3), dtype=jnp.int16)
            M = M.at[0, p[0]].set(s[0])
            M = M.at[1, p[1]].set(s[1])
            M = M.at[2, p[2]].set(s[2])
            matrices.append(M)
    return jnp.array(matrices)[1:]  # Exclude identity


class InterpTilerState(eqx.Module):
    """
    Dynamic state for interpolation kernels.
    """

    rotation_idx: int = -1
    shell_idx: int = -1
    # Rotations are generated at init and stored here
    rotations: jnp.ndarray | None = None


class AbstractInterp(eqx.Module):
    """
    Abstract base class for interpolation kernels (Executor).
    Holds configuration (Painting, Geometry).
    """

    painting: PaintingOptions
    ts: jnp.ndarray | None = None
    r_centers: jnp.ndarray | None = None
    density_widths: jnp.ndarray | None = None
    max_comoving_distance: float | None = None

    def update_geometry(
        self,
        ts: jnp.ndarray,
        r_centers: jnp.ndarray,
        density_widths: jnp.ndarray,
        max_comoving_distance: float,
    ) -> AbstractInterp:
        """Update geometry configuration."""
        return eqx.tree_at(
            lambda s: (s.ts, s.r_centers, s.density_widths, s.max_comoving_distance),
            self,
            (ts, r_centers, density_widths, max_comoving_distance),
            is_leaf=lambda x: x is None,
        )

    @abstractmethod
    def init(self) -> InterpTilerState:
        """Initialize dynamic state."""
        raise NotImplementedError

    @abstractmethod
    def advance(self, state: InterpTilerState, t: float) -> InterpTilerState:
        """Advance the interpolation state to the next shell (only at snapshot times)."""
        raise NotImplementedError

    @abstractmethod
    def rewind(self, state: InterpTilerState, t: float) -> InterpTilerState:
        """Rewind the interpolation state to the previous shell (only at snapshot times)."""
        raise NotImplementedError

    @abstractmethod
    def paint(
        self,
        state: InterpTilerState,
        t: float,
        y: tuple[ParticleField, ParticleField],
        cosmo,
        drift_factor: Callable | None = None,
    ) -> Any:
        """Produce a map.

        ``drift_factor`` is the solver-provided drift multiplier
        (``AbstractNBodySolver.drift_factor``); it carries the solver's
        momentum convention. Kernels that do not drift ignore it.
        """
        raise NotImplementedError

    def _drift_positions(
        self,
        positions: jnp.ndarray,
        velocities: jnp.ndarray,
        a_from: float,
        a_to: jnp.ndarray,
        drift_factor: Callable,
        cosmo,
    ) -> jnp.ndarray:
        """Drift particles from ``a_from`` to their per-particle ``a_to``.

        ``positions`` and ``velocities`` must be in the *same* spatial unit. The
        solver-provided ``drift_factor(a_from, a_to, cosmo)`` returns the per-particle
        multiplier in the solver's momentum convention (conformal ``p`` for KKD,
        ``pi = dx/dD`` for the FastPM/BullFrog steppers), so the same call is correct
        for every solver.
        """
        factor = drift_factor(a_from, a_to, cosmo)
        return positions + factor * velocities


class NoInterp(AbstractInterp):
    """Simple painting without tiling."""

    def init(self) -> InterpTilerState:
        # Check geometry
        if self.r_centers is not None:
            assert self.density_widths is not None
            furthest_shell = self.r_centers[0] + self.density_widths[0] / 2
            furthest_shell = warning_if(
                furthest_shell,
                furthest_shell > self.max_comoving_distance,
                """
                NoInterp does not support tiling.
                Your furthest shell is at {:.2f} Mpc/h,
                but your box only extends to {:.2f} Mpc/h.
                Your simulations will run but will have artifacts for the far shells.
            """,
                furthest_shell,
                self.max_comoving_distance,
            )

        return InterpTilerState(
            rotation_idx=-1,
            shell_idx=-1,
            rotations=None,
        )

    def advance(self, state: InterpTilerState, t: float) -> InterpTilerState:
        is_snapshot = jnp.isin(t, self.ts)
        new_shell_idx = jnp.where(is_snapshot, state.shell_idx + 1, state.shell_idx)
        return eqx.tree_at(lambda s: s.shell_idx, state, new_shell_idx)

    def rewind(self, state: InterpTilerState, t: float) -> InterpTilerState:
        is_snapshot = jnp.isin(t, self.ts)
        new_shell_idx = jnp.where(is_snapshot, state.shell_idx - 1, state.shell_idx)
        return eqx.tree_at(lambda s: s.shell_idx, state, new_shell_idx)

    def paint(
        self,
        state: InterpTilerState,
        t: float,
        y: tuple[ParticleField, ParticleField],
        cosmo,
        drift_factor: Callable | None = None,
    ) -> Any:
        dx, p = y

        assert self.r_centers is not None
        assert self.density_widths is not None
        r_center = self.r_centers[state.shell_idx]
        width = self.density_widths[state.shell_idx]

        if self.painting is None or self.painting.target == "particles":
            result = dx
        elif self.painting.target == "spherical":
            result = dx.paint_spherical(
                center=r_center,
                density_plane_width=width,
                scheme=self.painting.scheme,
                weights=self.painting.weights,
                kernel_width_arcmin=self.painting.kernel_width_arcmin,
                kernel_width_pixels=self.painting.kernel_width_pixels,
                smoothing_interpretation=self.painting.smoothing_interpretation,
                paint_nside=self.painting.paint_nside,
                ud_grade_power=self.painting.ud_grade_power,
                ud_grade_order_in=self.painting.ud_grade_order_in,
                ud_grade_order_out=self.painting.ud_grade_order_out,
                ud_grade_pess=self.painting.ud_grade_pess,
                batch_size=self.painting.batch_size,
            )
        elif self.painting.target == "flat":
            result = dx.paint_2d(
                center=r_center,
                density_plane_width=width,
                weights=self.painting.weights,
                batch_size=self.painting.batch_size,
            )
        elif self.painting.target == "density":
            result = dx.paint(
                weights=self.painting.weights,
                batch_size=self.painting.batch_size,
                order=self.painting.order,
                deconvolution=self.painting.deconvolution,
            )
        else:
            result = dx

        result = result.replace(scale_factors=t, comoving_centers=r_center, density_width=width)
        return result


class DriftInterp(AbstractInterp):
    """
    Painting without tiling, but drifting particles to the lightcone surface.

    Identical to :class:`NoInterp` except that every particle is drifted from the
    snapshot scale factor to the scale factor at which it actually crosses the
    lightcone (``a_of_chi`` of its comoving distance from the observer), correcting
    the discrete time stepping. ``DriftInterp`` always drifts — there is no on/off
    switch; use :class:`NoInterp` for a plain snapshot.

    Useful when the box is large enough to cover the lightcone without replication.
    """

    def init(self) -> InterpTilerState:
        # Check geometry
        if self.r_centers is not None:
            assert self.density_widths is not None
            furthest_shell = self.r_centers[0] + self.density_widths[0] / 2
            furthest_shell = warning_if(
                furthest_shell,
                furthest_shell > self.max_comoving_distance,
                """
                DriftInterp does not support tiling.
                Your furthest shell is at {:.2f} Mpc/h,
                but your box only extends to {:.2f} Mpc/h.
                Your simulations will run but will have artifacts for the far shells.
            """,
                furthest_shell,
                self.max_comoving_distance,
            )

        return InterpTilerState(
            rotation_idx=-1,
            shell_idx=-1,
            rotations=None,
        )

    def advance(self, state: InterpTilerState, t: float) -> InterpTilerState:
        is_snapshot = jnp.isin(t, self.ts)
        new_shell_idx = jnp.where(is_snapshot, state.shell_idx + 1, state.shell_idx)
        return eqx.tree_at(lambda s: s.shell_idx, state, new_shell_idx)

    def rewind(self, state: InterpTilerState, t: float) -> InterpTilerState:
        is_snapshot = jnp.isin(t, self.ts)
        new_shell_idx = jnp.where(is_snapshot, state.shell_idx - 1, state.shell_idx)
        return eqx.tree_at(lambda s: s.shell_idx, state, new_shell_idx)

    def paint(
        self,
        state: InterpTilerState,
        t: float,
        y: tuple[ParticleField, ParticleField],
        cosmo,
        drift_factor: Callable | None = None,
    ) -> Any:
        dx, p = y
        assert drift_factor is not None, "DriftInterp requires a solver-provided drift_factor"

        assert self.r_centers is not None
        assert self.density_widths is not None
        r_center = self.r_centers[state.shell_idx]
        width = self.density_widths[state.shell_idx]

        # Per-particle lightcone-crossing scale factor from comoving distance to observer.
        observer_mpc = jnp.array(dx.observer_position_mpc)
        positions_mpc = dx.to(PositionUnit.MPC_H).array
        dist_mpc = jnp.linalg.norm(positions_mpc - observer_mpc, axis=-1)
        a_cross = jc.background.a_of_chi(cosmo, dist_mpc)[..., None]

        # Drift in the native GRID_RELATIVE unit (the unit of both dx and p), so this
        # reduces to NoInterp plus exactly the drift term and painting is unchanged.
        drifted = self._drift_positions(dx.array, p.array, t, a_cross, drift_factor, cosmo)
        dx_drifted = dx.replace(array=drifted)

        if self.painting.target == "particles":
            result = dx_drifted
        elif self.painting.target == "spherical":
            result = dx_drifted.paint_spherical(
                center=r_center,
                density_plane_width=width,
                scheme=self.painting.scheme,
                weights=self.painting.weights,
                kernel_width_arcmin=self.painting.kernel_width_arcmin,
                kernel_width_pixels=self.painting.kernel_width_pixels,
                smoothing_interpretation=self.painting.smoothing_interpretation,
                paint_nside=self.painting.paint_nside,
                ud_grade_power=self.painting.ud_grade_power,
                ud_grade_order_in=self.painting.ud_grade_order_in,
                ud_grade_order_out=self.painting.ud_grade_order_out,
                ud_grade_pess=self.painting.ud_grade_pess,
                batch_size=self.painting.batch_size,
            )
        elif self.painting.target == "flat":
            result = dx_drifted.paint_2d(
                center=r_center,
                density_plane_width=width,
                weights=self.painting.weights,
                batch_size=self.painting.batch_size,
            )
        elif self.painting.target == "density":
            result = dx_drifted.paint(
                weights=self.painting.weights,
                batch_size=self.painting.batch_size,
                order=self.painting.order,
                deconvolution=self.painting.deconvolution,
            )
        else:
            result = dx_drifted

        result = result.replace(scale_factors=t, comoving_centers=r_center, density_width=width)
        return result


class OnionTiler(AbstractInterp):
    """27-tile spherical painting with rotation decorrelation.

    When ``drift_on_lightcone`` is set, particles are additionally drifted to their
    lightcone-crossing scale factor (per tile, using each tile's comoving distance).
    """

    drift_on_lightcone: bool = eqx.field(default=False, static=True)

    def init(self) -> InterpTilerState:
        if self.painting.target != "spherical":
            raise ValueError("OnionTiler only supports spherical painting.")

        # Generate rotations
        all_symmetries = get_all_47_symmetries()
        rng = jax.random.PRNGKey(42)
        rot_indices = jax.random.randint(rng, shape=(27,), minval=0, maxval=47)
        selected_rotations = all_symmetries[rot_indices]

        return InterpTilerState(
            rotation_idx=-1,
            shell_idx=-1,
            rotations=selected_rotations,
        )

    def advance(self, state: InterpTilerState, t: float) -> InterpTilerState:
        is_snapshot = jnp.isin(t, self.ts)
        new_shell_idx = jnp.where(is_snapshot, state.shell_idx + 1, state.shell_idx)
        return eqx.tree_at(lambda s: s.shell_idx, state, new_shell_idx)

    def rewind(self, state: InterpTilerState, t: float) -> InterpTilerState:
        is_snapshot = jnp.isin(t, self.ts)
        new_shell_idx = jnp.where(is_snapshot, state.shell_idx - 1, state.shell_idx)
        return eqx.tree_at(lambda s: s.shell_idx, state, new_shell_idx)

    def paint(
        self,
        state: InterpTilerState,
        t: float,
        y: tuple[ParticleField, ParticleField],
        cosmo,
        drift_factor: Callable | None = None,
    ) -> Any:
        dx, p = y
        assert self.r_centers is not None
        assert self.density_widths is not None
        r_center = self.r_centers[state.shell_idx]
        width = self.density_widths[state.shell_idx]

        R_min = r_center - width / 2
        R_max = r_center + width / 2

        inside_box = (r_center + width / 2) <= self.max_comoving_distance

        def inside_branch(operand):
            current_state, c = operand
            return self._paint_single(t, p, dx, c, r_center, width, current_state, drift_factor)

        def outside_branch(operand):
            current_state, c = operand
            return self._paint_tiled_27(t, p, dx, c, r_center, R_min, R_max, current_state, drift_factor)

        sph_map = jax.lax.cond(inside_box, inside_branch, outside_branch, (state, cosmo))
        sph_map = sph_map.replace(scale_factors=t, comoving_centers=r_center, density_width=width)
        return sph_map

    def _paint_single(
        self,
        t: float,
        p: ParticleField,
        dx: ParticleField,
        cosmo,
        R_center: float,
        width: float,
        state: InterpTilerState,
        drift_factor: Callable | None = None,
    ) -> SphericalDensity:
        # Inside the box: no tiling, drift in native GRID_RELATIVE units (like DriftInterp).
        if self.drift_on_lightcone:
            observer_mpc = jnp.array(dx.observer_position_mpc)
            positions_mpc = dx.to(PositionUnit.MPC_H).array
            dist_mpc = jnp.linalg.norm(positions_mpc - observer_mpc, axis=-1)
            a_cross = jc.background.a_of_chi(cosmo, dist_mpc)[..., None]
            drifted = self._drift_positions(dx.array, p.array, t, a_cross, drift_factor, cosmo)
            dx = dx.replace(array=drifted)

        sph_map = dx.paint_spherical(
            center=R_center,
            density_plane_width=width,
            scheme=self.painting.scheme,
            weights=self.painting.weights,
            kernel_width_arcmin=self.painting.kernel_width_arcmin,
            kernel_width_pixels=self.painting.kernel_width_pixels,
            smoothing_interpretation=self.painting.smoothing_interpretation,
            paint_nside=self.painting.paint_nside,
            ud_grade_power=self.painting.ud_grade_power,
            ud_grade_order_in=self.painting.ud_grade_order_in,
            ud_grade_order_out=self.painting.ud_grade_order_out,
            ud_grade_pess=self.painting.ud_grade_pess,
            batch_size=self.painting.batch_size,
        )
        return sph_map.replace(
            status=FieldStatus.LIGHTCONE,
            unit=DensityUnit.DENSITY,
            scale_factors=t,
            comoving_centers=R_center,
            density_width=width,
        )

    def _paint_tiled_27(
        self,
        t: float,
        p: ParticleField,
        dx: ParticleField,
        cosmo,
        r_center: float,
        R_min: float,
        R_max: float,
        state: InterpTilerState,
        drift_factor: Callable | None = None,
    ) -> SphericalDensity:
        nside = dx.nside
        observer_mpc = jnp.array(dx.observer_position_mpc)
        observer_relative = jnp.array(dx.observer_position)
        box_size = jnp.array(dx.box_size)
        mesh_size = jnp.array(dx.mesh_size)
        width = R_max - R_min

        positions = dx.to(PositionUnit.MPC_H).array
        positions_centered = positions - observer_mpc

        # The momentum p is a GRID_RELATIVE displacement; scale by the cell size to get
        # an Mpc/h displacement (do NOT use p.to(MPC_H), which would add the uniform grid).
        velocities_mpc = p.array * (box_size / mesh_size)

        shifts = jnp.array(list(itertools.product([0, 1, -1], repeat=3)), dtype=jnp.int16)
        shifts = jnp.where(observer_relative == 0, shifts + 1, shifts)
        shifts = jnp.where(observer_relative == 1, shifts - 1, shifts)
        shifts = shifts * box_size

        rot_matrices = state.rotations

        npix = jhp.nside2npix(nside)
        init_map = SphericalDensity.FromDensityMetadata(
            array=jnp.zeros(npix),
            field=dx,
            unit=DensityUnit.DENSITY,
        )

        def scan_body(accum_map, tile_data):
            shift_idx, rot_mat = tile_data
            shift_vec = shift_idx.astype(jnp.float32)

            p_rotated = jnp.einsum("ij,...j->...i", rot_mat, positions_centered)
            p_tiled = p_rotated + shift_vec

            p_final_pos = p_tiled + observer_mpc

            if self.drift_on_lightcone:
                # Rotate the Mpc/h velocity with the same rotation, drift to each tile's
                # own lightcone-crossing epoch (distance differs per tile/shift).
                v_rotated = jnp.einsum("ij,...j->...i", rot_mat, velocities_mpc)
                dist_mpc = jnp.linalg.norm(p_final_pos - observer_mpc, axis=-1)
                a_cross = jc.background.a_of_chi(cosmo, dist_mpc)[..., None]
                p_final_pos = self._drift_positions(p_final_pos, v_rotated, t, a_cross, drift_factor, cosmo)

            p_final = dx.replace(array=p_final_pos, unit=PositionUnit.MPC_H)

            tile_map = p_final.paint_spherical(
                center=r_center,
                density_plane_width=width,
                scheme=self.painting.scheme,
                weights=self.painting.weights,
                kernel_width_arcmin=self.painting.kernel_width_arcmin,
                kernel_width_pixels=self.painting.kernel_width_pixels,
                smoothing_interpretation=self.painting.smoothing_interpretation,
                paint_nside=nside,
                ud_grade_power=self.painting.ud_grade_power,
                ud_grade_order_in=self.painting.ud_grade_order_in,
                ud_grade_order_out=self.painting.ud_grade_order_out,
                ud_grade_pess=self.painting.ud_grade_pess,
                batch_size=self.painting.batch_size,
            )
            return accum_map + tile_map, None

        final_map, _ = jax.lax.scan(scan_body, init_map, (shifts, rot_matrices))
        final_map = final_map.replace(
            status=FieldStatus.LIGHTCONE,
            unit=DensityUnit.DENSITY,
            scale_factors=t,
            comoving_centers=r_center,
            density_width=width,
        )
        return final_map


class TelephotoInterp(AbstractInterp):
    """Single rotation+shift painting for narrow FOV.

    When ``drift_on_lightcone`` is set, particles are drifted to their
    lightcone-crossing scale factor after the rotation+z-shift.
    """

    drift_on_lightcone: bool = eqx.field(default=False, static=True)

    def init(self) -> InterpTilerState:
        if self.painting.target != "spherical" and self.painting.target != "flat":
            raise ValueError("TelephotoInterp only supports projections (flat or spherical).")

        if self.painting.target == "spherical":
            warnings.warn(
                """
            TelephotoInterp is designed for narrow FOV flat-sky projections.
            Using it for spherical painting will cause significant artifacts.
            use for illustriative purposes only.
            """
            )

        # Check logic moved to update_geometry/advance or assumed checked

        return InterpTilerState(
            rotation_idx=-1,
            shell_idx=-1,
            rotations=get_all_47_symmetries(),
        )

    def advance(self, state: InterpTilerState, t: float) -> InterpTilerState:
        assert self.r_centers is not None
        is_snapshot = jnp.isin(t, self.ts)
        next_shell_idx = jnp.where(is_snapshot, state.shell_idx + 1, state.shell_idx)

        # Geometry check uses next_shell_idx (harmless when not snapshot — computed but discarded)
        r_center = self.r_centers[next_shell_idx]
        inside_box = (r_center) <= self.max_comoving_distance

        would_advance_rot = jnp.where(inside_box, state.rotation_idx, state.rotation_idx + 1)
        next_rot_idx = jnp.where(is_snapshot, would_advance_rot, state.rotation_idx)

        return eqx.tree_at(lambda s: (s.shell_idx, s.rotation_idx), state, (next_shell_idx, next_rot_idx))

    def rewind(self, state: InterpTilerState, t: float) -> InterpTilerState:
        assert self.r_centers is not None
        assert self.density_widths is not None
        is_snapshot = jnp.isin(t, self.ts)

        # Check if CURRENT shell is outside box (meaning we incremented rot_idx when entering)
        r_center = self.r_centers[state.shell_idx]
        width = self.density_widths[state.shell_idx]
        current_outside = (r_center + width / 2) > self.max_comoving_distance

        would_rewind_rot = jnp.where(current_outside, state.rotation_idx - 1, state.rotation_idx)
        prev_rot_idx = jnp.where(is_snapshot, would_rewind_rot, state.rotation_idx)
        prev_shell_idx = jnp.where(is_snapshot, state.shell_idx - 1, state.shell_idx)

        return eqx.tree_at(lambda s: (s.shell_idx, s.rotation_idx), state, (prev_shell_idx, prev_rot_idx))

    def paint(
        self,
        state: InterpTilerState,
        t: float,
        y: tuple[ParticleField, ParticleField],
        cosmo,
        drift_factor: Callable | None = None,
    ) -> Any:
        dx, p = y

        assert self.r_centers is not None
        assert self.density_widths is not None

        r_center = self.r_centers[state.shell_idx]
        width = self.density_widths[state.shell_idx]
        inside_box = (r_center) <= self.max_comoving_distance

        def inside_branch(operand):
            current_state, c = operand
            return self._paint_single(t, p, dx, c, r_center, width, current_state, drift_factor)

        def outside_branch(operand):
            current_state, c = operand
            return self._paint_telephoto(t, p, dx, c, r_center, width, current_state, drift_factor)

        # Clear jax_cosmo workspace before cond to prevent tracer leaks:
        # traced values stored there escape the cond scope.
        sph_map = jax.lax.cond(inside_box, inside_branch, outside_branch, (state, cosmo))

        sph_map = sph_map.replace(scale_factors=t, comoving_centers=r_center, density_width=width)
        return sph_map

    def _paint_single(
        self,
        t: float,
        p: ParticleField,
        dx: ParticleField,
        cosmo,
        r_center: float,
        width: float,
        state: InterpTilerState,
        drift_factor: Callable | None = None,
    ) -> SphericalDensity:
        # Inside the box: no shift, drift in native GRID_RELATIVE units (like DriftInterp).
        if self.drift_on_lightcone:
            observer_mpc = jnp.array(dx.observer_position_mpc)
            positions_mpc = dx.to(PositionUnit.MPC_H).array
            dist_mpc = jnp.linalg.norm(positions_mpc - observer_mpc, axis=-1)
            a_cross = jc.background.a_of_chi(cosmo, dist_mpc)[..., None]
            drifted = self._drift_positions(dx.array, p.array, t, a_cross, drift_factor, cosmo)
            dx = dx.replace(array=drifted)

        if self.painting.target == "spherical":
            return dx.paint_spherical(
                center=r_center,
                density_plane_width=width,
                scheme=self.painting.scheme,
                weights=self.painting.weights,
                kernel_width_arcmin=self.painting.kernel_width_arcmin,
                kernel_width_pixels=self.painting.kernel_width_pixels,
                smoothing_interpretation=self.painting.smoothing_interpretation,
                paint_nside=self.painting.paint_nside,
                ud_grade_power=self.painting.ud_grade_power,
                ud_grade_order_in=self.painting.ud_grade_order_in,
                ud_grade_order_out=self.painting.ud_grade_order_out,
                ud_grade_pess=self.painting.ud_grade_pess,
                batch_size=self.painting.batch_size,
            )
        else:  # flat
            return dx.paint_2d(
                center=r_center,
                density_plane_width=width,
                weights=self.painting.weights,
                batch_size=self.painting.batch_size,
            )

    def _paint_telephoto(
        self,
        t: float,
        p: ParticleField,
        dx: ParticleField,
        cosmo,
        r_center: float,
        width: float,
        state: InterpTilerState,
        drift_factor: Callable | None = None,
    ) -> SphericalDensity:
        observer_mpc = jnp.array(dx.observer_position_mpc)
        box_size = jnp.array(dx.box_size)
        mesh_size = jnp.array(dx.mesh_size)

        assert state.rotations is not None
        rot_idx = state.rotation_idx % 47
        M = state.rotations[rot_idx]

        shift_distance = jnp.floor(r_center / self.max_comoving_distance) * self.max_comoving_distance

        positions = dx.to(PositionUnit.MPC_H).array
        # Scale the GRID_RELATIVE momentum to an Mpc/h displacement (cell size).
        velocities_mpc = p.array * (box_size / mesh_size)

        p_centered = positions - observer_mpc
        p_rotated = jnp.einsum("ij,...j->...i", M, p_centered)
        z_axis = jnp.array([0.0, 0.0, 1.0])
        p_shifted = p_rotated + z_axis * shift_distance
        v_rotated = jnp.einsum("ij,...j->...i", M, velocities_mpc)

        p_final_pos = p_shifted

        if self.drift_on_lightcone:
            # p_shifted is already observer-relative, so its distance is ||p_shifted||.
            dist_mpc = jnp.linalg.norm(p_shifted, axis=-1)
            a_cross = jc.background.a_of_chi(cosmo, dist_mpc)[..., None]
            p_final_pos = self._drift_positions(p_shifted, v_rotated, t, a_cross, drift_factor, cosmo)

        dx_transformed = dx.replace(array=p_final_pos + observer_mpc, unit=PositionUnit.MPC_H)

        if self.painting.target == "spherical":
            return dx_transformed.paint_spherical(
                center=r_center,
                density_plane_width=width,
                scheme=self.painting.scheme,
                weights=self.painting.weights,
                kernel_width_arcmin=self.painting.kernel_width_arcmin,
                kernel_width_pixels=self.painting.kernel_width_pixels,
                smoothing_interpretation=self.painting.smoothing_interpretation,
                paint_nside=self.painting.paint_nside,
                ud_grade_power=self.painting.ud_grade_power,
                ud_grade_order_in=self.painting.ud_grade_order_in,
                ud_grade_order_out=self.painting.ud_grade_order_out,
                ud_grade_pess=self.painting.ud_grade_pess,
                batch_size=self.painting.batch_size,
            )
        else:  # flat
            return dx_transformed.paint_2d(
                center=r_center,
                density_plane_width=width,
                weights=self.painting.weights,
                batch_size=self.painting.batch_size,
            )
