from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array


@dataclass(kw_only=True)
class Configurations:
    """Configuration shared by full-field and power-spectrum models.

    This dataclass intentionally stores only probabilistic / cosmological
    configuration. Geometry and mesh/box metadata belong to DensityField
    instances or separate function arguments, not here.
    """

    # Simulation setting
    mesh_size: tuple[int, int, int]
    box_size: tuple[float, float, float]
    halo_size: tuple[float, float] = (0, 0)
    nb_steps: int = 100
    field_sharding: Any = None

    # N-body / force / painting knobs
    # N-body solver: "DoubleKickDrift" | "DriftKickDrift" | "BullFrog"
    nbody_solver: str = "BullFrog"
    t0: float = 0.01
    t1: float = 1.0
    lpt_order: int = 2
    number_of_shells: int = 8
    drift_on_lightcone: bool = False
    paint_order: str = "cic"  # mass assignment: NGP / CIC / TSC / PCS
    gradient_order: int = 1  # force gradient order (0 = exact ik, 4 = 4th-order FD)
    laplace_fd: bool = False  # finite-difference Laplacian for the force
    deconvolution: bool = False  # deconvolve the mass-assignment window (solver)
    dealiased: bool = False  # dealias the painted density in Fourier space
    exact_growth: bool = False  # use exact (ODE) growth factors
    shell_spacing: str = "a"
    min_width: float = 50.0  # Mpc/h comoving, minimum shell width

    # Lensing
    # Observable returned by the forward model: "convergence" | "shear" | "reduced_shear"
    lensing_output: str = "convergence"
    # Spherical map->alm method for the shear (Kaiser-Squires) path: "jax" | "jax_cuda".
    map2alm_method: str = "jax"
    min_redshift: float = 0.01
    max_redshift: float = 1.5
    n_integrate: int = 32  # number of points for the lensing integral along the line of sight
    apodization_scale_deg: float = 1.0  # C2 apodization scale for the observer visibility mask

    # Geometry / painting (spherical only for now)
    nside: int = None
    flatsky_npix: tuple[int, int] = None
    field_size: tuple[float, float] = None
    geometry: str = "spherical"
    scheme: str = "bilinear"
    # Observer location in normalized box coords [0, 1]^3. The center (0.5, 0.5, 0.5)
    # sees the whole sky (no visibility mask); any other position triggers an
    # observer-driven visibility mask in the forward model (spherical geometry only).
    observer_position: tuple[float, float, float] = (0.5, 0.5, 0.5)
    paint_nside: int | None = None
    kernel_width_arcmin: float | None = None
    kernel_width_pixels: float | None = None
    # Post-paint HEALPix window deconvolution (a_lm level) for spherical maps; distinct from the
    # 3D force `deconvolution` below. Requires scheme in {ngp, rbf_neighbor}.
    pixel_window_deconvolution: bool = False

    # Cosmology and nz
    fiducial_cosmology: Any
    nz_shear: list[Any]

    # Masking / likelihood + observer visibility mask
    # Survey footprint mask (e.g. DES Y3), a (npix,) HEALPix array at the model nside, or None.
    # Used only in the likelihood: pixels with mask == 0 get an inflated sigma_unobserved.
    mask: Any = None
    sigma_unobserved: float = 1e6  # likelihood sigma on pixels outside the survey mask
    log_lightcone: bool = False

    # Priors and inference settings
    priors: dict[str, Any]
    sigma_e: float
    adjoint: str = "checkpointed"
    checkpoints: int = 10

    # Power spectrum settings (for power-spectrum model, not used in full-field model)
    ells: Array = field(default_factory=lambda: jnp.arange(2, 2048))
    f_sky: float = 1.0  # sky fraction for Gaussian covariance mode-count

    # Sampler settings (BlackJAX NUTS/MCLMC)
    sampler: str = "NUTS"  # "NUTS" | "MCLMC"
    nuts_max_num_doublings: int = 10  # leapfrog trajectory doubling depth (BlackJAX max_tree_depth analogue)
    nuts_target_accept: float = 0.8  # window-adaptation target acceptance rate
    mclmc_desired_energy_var: float = 1e-3
    mclmc_num_tune: int | None = None  # tuning steps; defaults to num_warmup
    mclmc_init_step_size_scale: float = 1e-4  # initial step = sqrt(total_dim) * scale (avoids high-dim collapse)
    mclmc_diagonal_preconditioning: bool = False
