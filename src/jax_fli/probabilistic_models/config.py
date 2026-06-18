from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array


@dataclass
class Configurations:
    """Configuration shared by full-field and power-spectrum models.

    This dataclass intentionally stores only probabilistic / cosmological
    configuration. Geometry and mesh/box metadata belong to DensityField
    instances or separate function arguments, not here.
    """

    # Mandatory parameters
    mesh_size: tuple[int, int, int]
    box_size: tuple[float, float, float]
    # Cosmological parameters
    fiducial_cosmology: Any
    nz_shear: list[Any]
    # Inference parameters
    priors: dict[str, Any]
    sigma_e: float
    log_lightcone: bool = False
    # Simulation settings
    halo_size: tuple[float, float] = (0, 0)
    # Observer location in normalized box coords [0, 1]^3. The center (0.5, 0.5, 0.5)
    # sees the whole sky (no visibility mask); any other position triggers an
    # observer-driven visibility mask in the forward model (spherical geometry only).
    observer_position: tuple[float, float, float] = (0.5, 0.5, 0.5)
    field_sharding: Any = None
    nside: int = None
    flatsky_npix: tuple[int, int] = None
    field_size: tuple[float, float] = None
    # Lensing parameters
    min_redshift: float = 0.01
    max_redshift: float = 1.5
    lensing: str = "born"  # born only; "raytrace" is not supported by the forward model
    # Observable returned by the forward model: "convergence" | "shear" | "reduced_shear"
    lensing_output: str = "convergence"
    # Masking / likelihood
    # Survey footprint mask (e.g. DES Y3), a (npix,) HEALPix array at the model nside, or None.
    # Used only in the likelihood: pixels with mask == 0 get an inflated sigma_unobserved.
    mask: Any = None
    sigma_unobserved: float = 1e6  # likelihood sigma on pixels outside the survey mask
    apodization_scale_deg: float = 1.0  # C2 apodization scale for the observer visibility mask
    # Simulation parameters
    lpt_order: int = 2
    t0: float = 0.01
    nb_steps: int = 100
    t1: float = 1.0
    adjoint: str = "checkpointed"
    checkpoints: int = 10
    number_of_shells: int = 8
    shell_spacing: str = "comoving"
    min_width: float = 50.0  # Mpc/h comoving, minimum shell width
    geometry: str = "spherical"
    scheme: str = "bilinear"
    paint_nside: int | None = None
    kernel_width_arcmin: float | None = None
    kernel_width_pixels: float | None = None
    # Post-paint HEALPix window deconvolution (a_lm level) for spherical maps; distinct from the
    # 3D force `deconvolution` below. Requires scheme in {ngp, rbf_neighbor}.
    pixel_window_deconvolution: bool = False
    drift_on_lightcone: bool = False
    # Force / painting knobs shared by LPT and the N-body solver
    paint_order: str = "cic"  # mass assignment: NGP / CIC / TSC / PCS
    gradient_order: int = 1  # force gradient order (0 = exact ik, 4 = 4th-order FD)
    laplace_fd: bool = False  # finite-difference Laplacian for the force
    deconvolution: bool = False  # deconvolve the mass-assignment window (solver)
    # LPT-only knobs
    dealiased: bool = False  # dealias the painted density in Fourier space
    exact_growth: bool = False  # use exact (ODE) growth factors
    # N-body solver: "DoubleKickDrift" | "DriftKickDrift" | "BullFrog"
    nbody_solver: str = "BullFrog"
    # Power spectrum settings (for power-spectrum model, not used in full-field model)
    ells: Array = field(default_factory=lambda: jnp.arange(2, 2048))
    f_sky: float = 1.0  # sky fraction for Gaussian covariance mode-count
