from typing import Any

import jax.numpy as jnp
from jaxpm.distributed import uniform_particles
from jaxtyping import Array

from .._src.base._enums import ConvergenceUnit, DensityUnit, PhysicalUnit, PositionUnit

__all__ = [
    # Density units exports
    "DENSITY",
    "OVERDENSITY",
    "COUNTS",
    "MSUN_H_PER_MPC3",
    # Position units exports
    "MPC_H",
    "GRID_ABSOLUTE",
    "GRID_RELATIVE",
]


def convert_units(
    array: Array,
    origin: PhysicalUnit,
    destination: PhysicalUnit,
    mesh_size: tuple[int, int, int],
    box_size: tuple[float, float, float],
    h: float | None = None,  # Hubble parameter, needed for MPC conversions
    omega_m: float | None = None,  # Matter density parameter, needed for MSUN conversions
    mean_density: float | None = None,  # Mean density for overdensity conversions
    volume_element: float | None = None,  # Volume per voxel/pixel for density conversions
    field_sharding: Any | None = None,
    normalization: str = "global",  # "global" or "per_plane"
    mask: Array | None = None,  # Optional mask for density conversions
) -> Array:
    """
    Convert array between units of the same physical quantity.

    Parameters
    ----------
    array : Array
        The data to convert
    origin : PhysicalUnit
        Current unit of the array
    destination : PhysicalUnit
        Target unit
    mesh_size : tuple of int
        Grid dimensions (nx, ny, nz)
    box_size : tuple of float
        Box size in Mpc/h
    h : float, optional
        Hubble parameter (h = H0 / 100 km/s/Mpc), required for MPC conversions
    normalization : str, optional
        Overdensity normalization mode. "global" (default) divides by the
        mean of the entire array. "per_plane" divides each leading slice
        (shell) by its own spatial mean — useful for lightcone maps where
        each shell is a separate density plane.

    Returns
    -------
    Array
        Array in the new units
    """
    # Same unit, no conversion needed
    if origin == destination:
        return array

    # Validate same unit family
    if type(origin) is not type(destination):
        raise TypeError(
            f"Cannot convert between different unit families: {type(origin).__name__} -> {type(destination).__name__}"
        )

    # Dispatch to specific converter
    if isinstance(origin, PositionUnit):
        return _convert_position(array, origin, destination, mesh_size, box_size, field_sharding)
    elif isinstance(origin, DensityUnit):
        return _convert_density(
            array, origin, destination, volume_element, omega_m, h, mean_density, normalization, mask
        )
    elif isinstance(origin, ConvergenceUnit):
        return _convert_convergence(array, origin, destination)
    else:
        raise NotImplementedError(f"Conversion not implemented for {type(origin).__name__}")


def _convert_position(
    array: Array,
    origin: PositionUnit,
    destination: PositionUnit,
    mesh_size: tuple[int, int, int],
    box_size: tuple[float, float, float],
    field_sharding: Any | None = None,
) -> Array:
    """
    Convert position units via GRID_ABSOLUTE as the canonical hub.

    Parameters
    ----------
    array : Array
        Position data to convert
    origin : PositionUnit
        Current unit
    destination : PositionUnit
        Target unit
    mesh_size : tuple of int
        Grid dimensions (nx, ny, nz)
    box_size : tuple of float
        Box size in Mpc/h
    field_sharding : optional
        JAX sharding for distributed computation

    Returns
    -------
    Array
        Positions in new units
    """
    mesh_size_arr = jnp.array(mesh_size)
    box_size_arr = jnp.array(box_size)

    # Step 1: Convert to GRID_ABSOLUTE (hub)
    if origin == PositionUnit.GRID_ABSOLUTE:
        grid_coords = array
    elif origin == PositionUnit.GRID_RELATIVE:
        grid_coords = array + uniform_particles(mesh_size, sharding=field_sharding)
    elif origin == PositionUnit.MPC_H:
        grid_coords = (array / box_size_arr) * mesh_size_arr
    else:
        raise ValueError(f"Unknown origin unit: {origin}")

    # Step 2: Convert from GRID_ABSOLUTE to destination
    if destination == PositionUnit.GRID_ABSOLUTE:
        return grid_coords
    elif destination == PositionUnit.GRID_RELATIVE:
        return grid_coords - uniform_particles(mesh_size, sharding=field_sharding)
    elif destination == PositionUnit.MPC_H:
        return (grid_coords / mesh_size_arr) * box_size_arr
    else:
        raise ValueError(f"Unknown destination unit: {destination}")


def _convert_density(
    array: Array,
    origin: DensityUnit,
    destination: DensityUnit,
    volume_element: float | None = None,
    omega_m: float | None = None,
    h: float | None = None,
    mean_density: float | None = None,
    normalization: str = "global",
    mask: Array | None = None,
) -> Array:
    """
    Convert between density units.

    Uses DENSITY as the canonical hub.

    Parameters
    ----------
    array : Array
        Input density field
    origin : DensityUnit
        Current unit of the array
    destination : DensityUnit
        Target unit
    volume_element : float
        Volume per voxel/pixel in (Mpc/h)³.

        3D voxel:
            volume_element = (Lx * Ly * Lz) / (Nx * Ny * Nz)

        Flat sky pixel:
            volume_element = (θ_pix × r)² × Δr
            where θ_pix is pixel angular size in radians,
            r is comoving distance in Mpc/h

        HEALPix pixel (thin shell):
            volume_element = (4π / 12 / nside²) × r² × Δr

        HEALPix pixel (thick shell):
            volume_element = (4π / 12 / nside²) × (R_max³ - R_min³) / 3
    mask : Array, optional
        Optional mask to apply to the density field before computing the mean density.

    omega_m : float, optional
        Matter density parameter. Required for MSUN_H_PER_MPC3 conversions.
    h : float, optional
        Hubble parameter (H0 / 100 km/s/Mpc). Required for MSUN_H_PER_MPC3 conversions.
    mean_density : float, optional
        Mean density ρ̄ in particles per (Mpc/h)³. Required when converting
        FROM OVERDENSITY to other units.
    normalization : str, optional
        "global" (default): overdensity is relative to the global mean of
        the whole array. "per_plane": each plane along axis 0 is normalised
        by its own spatial mean (mean over all remaining axes). Useful for
        lightcone shells — SphericalDensity (n_shells, npix) or FlatDensity
        (n_shells, ny, nx). Ignored when mean_density is provided explicitly.

    Returns
    -------
    Array
        Converted density field

    Notes
    -----
    Conversion formulas:

        COUNTS → DENSITY:       ρ = N / V
        DENSITY → OVERDENSITY:  δ = ρ / ρ̄ - 1
        DENSITY → MSUN:         ρ_phys = ρ × m_particle / V × h

    where:
        m_particle = Ωm × ρ_crit × V_physical
        ρ_crit = 2.775 × 10¹¹ h² M☉/Mpc³
        V_physical = V / h³
    """
    if volume_element is None:
        raise ValueError("volume_element is required for density unit conversions")

    volume_element = jnp.expand_dims(volume_element, axis=tuple(range(volume_element.ndim, array.ndim)))
    assert volume_element is not None  # reassignment widens type; re-narrow for checker
    # Compute mass per particle if needed for MSUN conversions
    mass_per_particle = None
    if origin == DensityUnit.MSUN_H_PER_MPC3 or destination == DensityUnit.MSUN_H_PER_MPC3:
        if omega_m is None or h is None:
            raise ValueError("omega_m and h required for MSUN_H_PER_MPC3 conversion")
        rho_crit = 2.775e11 * h**2  # M☉/Mpc³
        rho_matter = omega_m * rho_crit  # M☉/Mpc³
        volume_physical = volume_element / h**3  # Mpc³
        mass_per_particle = rho_matter * volume_physical  # M☉ (for mean 1 particle/voxel)

    # ===== Step 1: Convert to DENSITY (hub) =====

    if origin == DensityUnit.DENSITY:
        density = array

    elif origin == DensityUnit.COUNTS:
        density = array / volume_element

    elif origin == DensityUnit.OVERDENSITY:
        if mean_density is None:
            raise ValueError("mean_density required to convert from OVERDENSITY")
        density = (array + 1.0) * mean_density

    elif origin == DensityUnit.MSUN_H_PER_MPC3:
        mass_msun_h = array * volume_element  # M☉/h per voxel
        mass_msun = mass_msun_h / h  # M☉ per voxel
        counts = mass_msun / mass_per_particle  # particles per voxel
        density = counts / volume_element  # particles per (Mpc/h)³

    else:
        raise ValueError(f"Unknown origin unit: {origin}")

    # ===== Step 2: Convert from DENSITY to destination =====

    if destination == DensityUnit.DENSITY:
        return density

    elif destination == DensityUnit.COUNTS:
        return density * volume_element

    elif destination == DensityUnit.OVERDENSITY:
        if mean_density is None:
            # Mean density ρ̄ for δ = ρ/ρ̄ - 1. With a partial-sky visibility ``mask``
            # (off-center observer), average over VISIBLE pixels only so the unseen,
            # zero-filled region does not dilute ρ̄ (which would inflate δ by ~1/f_sky).
            # mask=None (full sky / center observer) keeps the plain mean — unchanged.
            # Cast the mask to float first: a uint8 sum over ~50M pixels overflows.
            w = None if mask is None else jnp.asarray(mask, dtype=density.dtype)
            if normalization == "per_plane" and density.ndim > 1:
                # Mean over all spatial axes (axis 1 onward), keepdims for broadcast.
                # e.g. SphericalDensity (n_shells, npix)  → mean shape (n_shells, 1)
                #      FlatDensity      (n_shells, ny, nx) → mean shape (n_shells, 1, 1)
                reduce_axes = tuple(range(1, density.ndim))
                if w is None:
                    mean_density = jnp.mean(density, axis=reduce_axes, keepdims=True)
                else:
                    wb = jnp.broadcast_to(w, density.shape)
                    mean_density = jnp.sum(density * wb, axis=reduce_axes, keepdims=True) / jnp.sum(
                        wb, axis=reduce_axes, keepdims=True
                    )
            else:
                if w is None:
                    mean_density = jnp.mean(density)
                else:
                    wb = jnp.broadcast_to(w, density.shape)
                    mean_density = jnp.sum(density * wb) / jnp.sum(wb)
        eps = jnp.finfo(density.dtype).eps
        safe_mean = jnp.where(mean_density == 0, eps, mean_density)
        return density / safe_mean - 1.0

    elif destination == DensityUnit.MSUN_H_PER_MPC3:
        counts = density * volume_element  # particles per voxel
        mass_msun = counts * mass_per_particle  # M☉ per voxel
        mass_msun_h = mass_msun * h  # M☉/h per voxel
        return mass_msun_h / volume_element  # M☉/h / (Mpc/h)³

    else:
        raise ValueError(f"Unknown destination unit: {destination}")


def _convert_convergence(
    array: Array,
    origin: ConvergenceUnit,
    destination: ConvergenceUnit,
) -> Array:
    """
    Convert convergence units.

    Both DIMENSIONLESS and EFFECTIVE_DENSITY are κ = Σ/Σ_crit,
    just different semantic interpretations.
    """
    # Both are numerically identical
    return array


# Density units exports
DENSITY = DensityUnit.DENSITY
OVERDENSITY = DensityUnit.OVERDENSITY
COUNTS = DensityUnit.COUNTS
MSUN_H_PER_MPC3 = DensityUnit.MSUN_H_PER_MPC3
# Position units exports
MPC_H = PositionUnit.MPC_H
GRID_ABSOLUTE = PositionUnit.GRID_ABSOLUTE
GRID_RELATIVE = PositionUnit.GRID_RELATIVE
