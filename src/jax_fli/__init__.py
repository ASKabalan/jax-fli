"""
jax_fli: Forward-modeling and sampling on top of JAXPM + JAX-Decomp.
"""

from . import fields, infer, initial, io, lensing, pm, utils
from . import probabilistic_models as ppl

# From fields
from .fields import (
    AbstractField,
    DensityField,
    DensityUnit,
    FieldMetadata,
    FieldStatus,
    FlatDensity,
    FlatKappaField,
    FlatShearField,
    PaintingOptions,
    ParticleField,
    PhysicalUnit,
    PositionUnit,
    SpectralUnit,
    SphericalDensity,
    SphericalKappaField,
    SphericalShearField,
    StarletCoefficients,
    convert_units,
    units,
)

# From initial
from .initial import gaussian_initial_conditions, interpolate_initial_conditions

# From lensing
from .lensing import born, raytrace

# From pm
from .pm import (
    AbstractNBodySolver,
    BullFrog,
    DoubleKickDrift,
    DriftInterp,
    DriftKickDrift,
    InterpTilerState,
    NoCorrection,
    NoInterp,
    OnionTiler,
    PGDKernel,
    SharpeningKernel,
    TelephotoInterp,
    compute_n_steps_for_shells,
    lpt,
    nbody,
    resolve_geometry,
    simulation_stepping,
)

# From summary_statistics (formerly jax_fli.power)
from .summary_statistics import (
    PDF,
    BinnedStatistic,
    PeakCounts,
    PowerSpectrum,
    angular_cl_flat,
    angular_cl_spherical,
    coherence,
    comoving_tophat,
    compute_theory_cl,
    compute_theory_cl_for_density,
    pdf_spherical,
    peak_counts_spherical,
    power,
    starlet_coefficients_spherical,
    tophat_z,
    transfer,
)

# From utils
from .utils import (
    centers,
    compute_box_size_from_redshift,
    compute_max_redshift_from_box_size,
    compute_particle_scale_factors,
    distances,
    edges,
)

__version__ = "0.1.0"

__all__ = [
    # Submodules
    "fields",
    "initial",
    "io",
    "lensing",
    "pm",
    "ppl",
    "infer",
    "utils",
    # From initial
    "gaussian_initial_conditions",
    "interpolate_initial_conditions",
    # From fields
    "AbstractField",
    "FieldMetadata",
    "DensityField",
    "DensityUnit",
    "FieldStatus",
    "FlatDensity",
    "FlatKappaField",
    "FlatShearField",
    "PaintingOptions",
    "ParticleField",
    "PhysicalUnit",
    "PositionUnit",
    "SpectralUnit",
    "SphericalDensity",
    "SphericalKappaField",
    "SphericalShearField",
    "StarletCoefficients",
    "convert_units",
    "units",
    # From pm
    "lpt",
    "nbody",
    "BullFrog",
    "DoubleKickDrift",
    "DriftKickDrift",
    "AbstractNBodySolver",
    "PGDKernel",
    "SharpeningKernel",
    "NoCorrection",
    "OnionTiler",
    "TelephotoInterp",
    "NoInterp",
    "DriftInterp",
    "InterpTilerState",
    "resolve_geometry",
    "compute_n_steps_for_shells",
    "simulation_stepping",
    # From summary_statistics
    "PowerSpectrum",
    "power",
    "transfer",
    "coherence",
    "angular_cl_flat",
    "angular_cl_spherical",
    "comoving_tophat",
    "compute_theory_cl",
    "compute_theory_cl_for_density",
    "tophat_z",
    "BinnedStatistic",
    "PDF",
    "pdf_spherical",
    "PeakCounts",
    "peak_counts_spherical",
    "starlet_coefficients_spherical",
    # From lensing
    "born",
    "raytrace",
    # From utils
    "compute_box_size_from_redshift",
    "compute_max_redshift_from_box_size",
    "compute_particle_scale_factors",
    "edges",
    "distances",
    "centers",
]
