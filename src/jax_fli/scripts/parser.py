"""Shared argument-group builders used by entry-point scripts and the launcher.

All functions are pure argparse — no jax_fli imports.
"""
from __future__ import annotations


def add_distributed_args(p):
    """Process-grid dimensions (pdim) and node count — single-device defaults."""
    g = p.add_argument_group("distributed")
    g.add_argument(
        "--pdim",
        type=int,
        nargs=2,
        default=[1, 1],
        metavar=("PX", "PY"),
        help="Process mesh dimensions (default: 1 1 = single device)",
    )
    g.add_argument("--nodes", type=int, default=1, help="Number of nodes (default: 1)")


def add_slurm_args(p):
    """SLURM / cluster arguments owned by fli-launcher.

    ``--pdim`` and ``--nodes`` are duplicated here so the launcher can validate
    ``gpus_per_node * nodes == prod(pdim)`` and then forward them to the
    entry-script command after the ``--`` separator.
    """
    g = p.add_argument_group("SLURM / cluster")
    g.add_argument(
        "--mode", choices=["local", "sbatch", "dryrun"], default="dryrun", help="Execution mode (default: dryrun)"
    )
    g.add_argument("--account", default="XXX", help="SLURM account")
    g.add_argument("--constraint", default="h100", help="Node constraint (set to 'cpu' for CPU-only jobs)")
    g.add_argument("--nodes", type=int, default=1, help="Number of nodes (default: 1)")
    g.add_argument("--gpus-per-node", type=int, default=4, dest="gpus_per_node", help="GPUs per node (default: 4)")
    g.add_argument("--cpus-per-node", type=int, default=16, dest="cpus_per_node", help="CPUs per node (default: 16)")
    g.add_argument(
        "--tasks-per-node",
        type=int,
        default=None,
        dest="tasks_per_node",
        help="MPI tasks per node (defaults to --gpus-per-node)",
    )
    g.add_argument("--qos", default="qos_gpu_h100-t3", help="SLURM QoS")
    g.add_argument("--time-limit", default="00:30:00", dest="time_limit", help="SLURM time limit (HH:MM:SS)")
    g.add_argument(
        "--slurm-script",
        default=None,
        dest="slurm_script",
        help="Path to the SLURM wrapper script (required when --mode=sbatch)",
    )
    g.add_argument("--output-logs", default="SLURM_LOGS", dest="output_logs", help="Directory for SLURM log files")
    g.add_argument(
        "--pdim",
        type=int,
        nargs=2,
        default=[1, 1],
        metavar=("PX", "PY"),
        help="JAX process mesh dimensions; forwarded to the command (default: 1 1)",
    )


def add_integration_settings_args(p):
    """Integration / lightcone / lensing parameters.

    Covers everything shown in the Integration Settings form: physics,
    shell timing, and lensing source-distribution parameters.
    """
    g = p.add_argument_group("integration")
    p.add_argument(
        "--sim-mode",
        choices=["lpt", "pm", "lensing"],
        required=True,
        dest="sim_mode",
        help="Simulation pipeline: lpt, pm (N-body), or lensing (N-body + Born)",
    )
    g.add_argument("--nb-shells", type=int, default=8, dest="nb_shells", help="Number of lightcone shells (default: 8)")
    g.add_argument("--lpt-order", type=int, default=2, choices=[1, 2], dest="lpt_order", help="LPT order (default: 2)")
    g.add_argument("--t0", type=float, default=0.001, help="LPT starting scale factor (default: 0.001)")
    g.add_argument("--t1", type=float, default=1.0, help="Final scale factor (default: 1.0)")
    g.add_argument(
        "--nb-steps", type=int, default=30, dest="nb_steps", help="Number of integration steps (default: 30)"
    )
    g.add_argument(
        "--interp", choices=["none", "onion", "telephoto"], default="none", help="Interpolation kernel (default: none)"
    )
    g.add_argument("--solver", choices=["kdk", "dkd", "bf"], default="kdk", help="N-body integrator (default: kdk)")
    g.add_argument(
        "--time-stepping",
        choices=["a", "D", "log_a"],
        default="a",
        dest="time_stepping",
        help="Integrator time-stepping variable (default: a)",
    )
    g.add_argument("--dealiased", action="store_true", help="Enable dealiased mode (default: False)")
    g.add_argument(
        "--exact-growth",
        action="store_true",
        dest="exact_growth",
        help="Use exact growth factor computation (default: False)",
    )
    g.add_argument(
        "--gradient-order",
        type=int,
        default=1,
        choices=[0, 1],
        dest="gradient_order",
        help="Force gradient order (0=exact ik, 1=finite-difference) (default: 1)",
    )
    g.add_argument(
        "--laplace-fd", action="store_true", dest="laplace_fd", help="Use finite-difference Laplacian (default: False)"
    )
    g.add_argument(
        "--shell-spacing",
        choices=["comoving", "equal_vol", "a", "growth"],
        default="comoving",
        dest="shell_spacing",
        help="Shell spacing mode (default: comoving)",
    )
    g.add_argument(
        "--density-widths", type=float, nargs="+", default=None, metavar="W", help="Override shell widths in Mpc/h"
    )
    # Lightcone timing
    ts_group = p.add_mutually_exclusive_group()
    ts_group.add_argument(
        "--ts", type=float, nargs="+", default=None, metavar="A", help="Scale factors for snapshot/shell output"
    )
    ts_group.add_argument(
        "--ts-near",
        type=float,
        nargs="+",
        default=None,
        metavar="A_NEAR",
        help="Near scale factor edges (use with --ts-far)",
    )
    g.add_argument(
        "--ts-far",
        type=float,
        nargs="+",
        default=None,
        metavar="A_FAR",
        help="Far scale factor edges (use with --ts-near)",
    )
    g.add_argument(
        "--drift-on-lightcone",
        action="store_true",
        dest="drift_on_lightcone",
        help="Apply drift correction when painting lightcone shells",
    )
    g.add_argument(
        "--min-width", type=float, default=50.0, dest="min_width", help="Minimum shell width in Mpc/h (default: 50.0)"
    )
    add_lensing_args(p)


def add_cosmo_args(p):
    """Cosmological parameters (Omega_c, sigma8, Omega_b, h, n_s, etc.).

    Note: ``--seed`` is part of ``add_simulation_settings_args``, not here.
    """
    g = p.add_argument_group("cosmology")
    g.add_argument("--Omega-b", type=float, default=0.0486, dest="Omega_b", help="Baryon density (default: 0.0486)")
    g.add_argument("--h", type=float, default=0.6774, help="Dimensionless Hubble parameter (default: 0.6774)")
    g.add_argument("--n-s", type=float, default=0.9667, dest="n_s", help="Spectral index (default: 0.9667)")
    g.add_argument("--Omega-k", type=float, default=0.0, dest="Omega_k", help="Curvature density (default: 0.0)")
    g.add_argument("--w0", type=float, default=-1.0, help="Dark energy EOS w0 (default: -1.0)")
    g.add_argument("--wa", type=float, default=0.0, help="Dark energy EOS wa (default: 0.0)")
    g.add_argument("--Omega-nu", type=float, default=0.0, dest="Omega_nu", help="Neutrino density (default: 0.0)")
    g.add_argument(
        "--Omega-c", type=float, default=0.2589, dest="Omega_c", help="Cold dark matter density (default: 0.2589)"
    )
    g.add_argument("--sigma8", type=float, default=0.8159, help="sigma8 (default: 0.8159)")


def add_lensing_args(p):
    """Lensing / source-distribution parameters."""
    g = p.add_argument_group("lensing")
    g.add_argument(
        "--nz-shear",
        nargs="+",
        default=["s3"],
        metavar="Z",
        help="Source redshifts or 's3' for Stage-3 (default: s3)",
    )
    g.add_argument("--min-z", type=float, default=0.01, help="Minimum redshift for n(z) integration (default: 0.01)")
    g.add_argument("--max-z", type=float, default=1.5, help="Maximum redshift for n(z) integration (default: 1.5)")
    g.add_argument("--n-integrate", type=int, default=32, help="Number of integration points for n(z) (default: 32)")


def add_simulation_settings_args(p):
    """Simulation settings: mesh, box, halo, observer, seed, painting scheme, x64."""
    g = p.add_argument_group("simulation settings")
    g.add_argument(
        "--mesh-size",
        type=int,
        nargs=3,
        default=[64, 64, 64],
        metavar=("NX", "NY", "NZ"),
        help="Mesh resolution (default: 64 64 64)",
    )
    g.add_argument(
        "--box-size",
        type=float,
        nargs=3,
        default=[200.0, 200.0, 200.0],
        metavar=("LX", "LY", "LZ"),
        help="Box side lengths in Mpc/h (default: 200 200 200)",
    )
    g.add_argument(
        "--halo-multiplier",
        type=float,
        default=0.5,
        dest="halo_multiplier",
        metavar="M",
        help="Halo size as local_mesh × multiplier (default: 0.5)",
    )
    g.add_argument(
        "--observer-position",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 0.5],
        metavar=("OX", "OY", "OZ"),
        help="Observer position in box coordinates (default: 0.5 0.5 0.5)",
    )
    g.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")

    g = p.add_argument_group("output target")
    ex = g.add_mutually_exclusive_group()
    ex.add_argument("--nside", type=int, default=None, help="HEALPix NSIDE for spherical painting")
    ex.add_argument("--density", action="store_true", default=False, help="3D density field output")
    ex.add_argument(
        "--flatsky-npix",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        dest="flatsky_npix",
        help="Flat-sky pixel resolution (H×W)",
    )
    g.add_argument(
        "--field-size",
        nargs=2,
        type=int,
        default=[10, 10],
        metavar=("H", "W"),
        dest="field_size",
        help="Angular field size in degrees H×W (use with --flatsky-npix)",
    )
    g.add_argument(
        "--scheme",
        choices=["ngp", "bilinear", "rbf_neighbor"],
        default="bilinear",
        help="Spherical painting interpolation scheme (default: bilinear)",
    )
    g.add_argument(
        "--paint-nside",
        type=int,
        default=None,
        dest="paint_nside",
        help="Override nside for spherical painting (default: same as --nside)",
    )
    g.add_argument(
        "--kernel-width-arcmin",
        type=float,
        default=None,
        dest="kernel_width_arcmin",
        help="RBF smoothing kernel width in arcmin (default: None)",
    )
    g.add_argument(
        "--enable-x64",
        action="store_true",
        dest="enable_x64",
        help="Enable JAX 64-bit floating-point precision (default: False)",
    )


# ---------------------------------------------------------------------------
# Spectra computation argument groups (used by fli-spectra)
# ---------------------------------------------------------------------------


def add_prior_args(p):
    """Prior bounds for cosmological and IC sampling.

    Used by fli-samples, fli-infer, and fli-2pcf.
    """
    g = p.add_argument_group("priors")
    g.add_argument(
        "--sample",
        nargs="+",
        choices=["cosmo", "ic"],
        default=["cosmo", "ic"],
        metavar="WHAT",
        help="Space-separated subset of {cosmo, ic} to sample (default: cosmo ic).",
    )
    g.add_argument(
        "--prior-omega-c",
        type=float,
        nargs=2,
        default=[0.1, 0.5],
        dest="prior_omega_c",
        metavar=("MIN", "MAX"),
        help="Uniform prior bounds for Omega_c (default: 0.1 0.5)",
    )
    g.add_argument(
        "--prior-sigma8",
        type=float,
        nargs=2,
        default=[0.6, 1.0],
        dest="prior_sigma8",
        metavar=("MIN", "MAX"),
        help="Uniform prior bounds for sigma8 (default: 0.6 1.0)",
    )
    g.add_argument(
        "--prior-h",
        type=float,
        nargs=2,
        default=[0.5, 0.9],
        dest="prior_h",
        metavar=("MIN", "MAX"),
        help="Uniform prior bounds for h (default: 0.5 0.9)",
    )
    g.add_argument(
        "--prior-ic-gaussian",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        dest="prior_ic_gaussian",
        metavar=("MIN", "MAX"),
        help="Gaussian prior bounds for initial conditions (default: 0.0 1.0)",
    )


def add_infer_args(p):
    """Sampling configuration shared between fli-infer and launcher/infer.

    Does NOT include path args (--observable, --path) — those differ between
    the entry script (full paths, required) and the launcher (constructed from
    --observable-dir / --output-dir / --chain-index).
    """
    g = p.add_argument_group("inference")
    g.add_argument(
        "--initial-condition",
        type=str,
        default=None,
        metavar="PATH",
        dest="initial_condition",
        help="Parquet Catalog with IC DensityField for initialization or fixing IC.",
    )
    g.add_argument(
        "--init-cosmo",
        action="store_true",
        dest="init_cosmo",
        help="Warm-start cosmological parameters from the observable's stored cosmology.",
    )
    g.add_argument("--sigma-e", type=float, default=0.26, dest="sigma_e", help="Shape noise dispersion (default: 0.26)")
    g.add_argument(
        "--num-warmup", type=int, default=500, dest="num_warmup", help="MCMC warmup iterations (default: 500)"
    )
    g.add_argument(
        "--num-samples", type=int, default=1000, dest="num_samples", help="Samples per batch (default: 1000)"
    )
    g.add_argument(
        "--batch-count", type=int, default=5, dest="batch_count", help="Number of sequential batches (default: 5)"
    )
    g.add_argument(
        "--adjoint",
        choices=["checkpointed", "recursive"],
        default="checkpointed",
        help="Gradient strategy for NUTS (default: checkpointed)",
    )
    g.add_argument("--checkpoints", type=int, default=10, help="Number of gradient checkpoints (default: 10)")
    g.add_argument("--sampler", choices=["NUTS", "HMC", "MCLMC"], default="NUTS", help="MCMC sampler (default: NUTS)")
    g.add_argument(
        "--backend", choices=["numpyro", "blackjax"], default="numpyro", help="Sampling backend (default: numpyro)"
    )
    g.add_argument("--no-progress-bar", action="store_true", dest="no_progress_bar", help="Suppress tqdm progress bars")


def add_spectra_scan_args(p):
    """Scan and filter arguments for fli-spectra."""
    g = p.add_argument_group("scan")
    g.add_argument(
        "folder",
        help="Folder to scan for parquet files",
    )
    g.add_argument(
        "-r",
        "--regex",
        default=r".*\.parquet$",
        dest="regex",
        metavar="PATTERN",
        help="Regex pattern to match parquet filenames (default: all .parquet files)",
    )
    g.add_argument(
        "-R",
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories (default: False)",
    )
    g.add_argument(
        "--force-regen",
        action="store_true",
        help="Force regeneration of spectra even if output files already exist (default: False)",
    )
    g.add_argument(
        "--normalization",
        choices=["global", "per_plane"],
        default="global",
        dest="normalization",
        help="Overdensity normalization: 'global' divides by array mean, "
        "'per_plane' normalises each shell independently (default: global)",
    )


def add_spectra_flat_args(p):
    """Flat-sky spectra arguments for fli-spectra.

    Note: field_size and pixel_size are read from the stored field metadata.
    """
    g = p.add_argument_group("flat-sky spectra")
    g.add_argument(
        "--ell-edges",
        type=float,
        nargs="+",
        default=None,
        dest="ell_edges",
        metavar="E",
        help="Ell bin edges for flat-sky angular Cl (default: auto)",
    )


def add_spectra_spherical_args(p):
    """Spherical (HEALPix) spectra arguments for fli-spectra."""
    g = p.add_argument_group("spherical spectra")
    g.add_argument(
        "--lmax",
        type=int,
        default=None,
        help="Maximum multipole lmax for spherical Cl (default: 3*nside-1)",
    )
    g.add_argument(
        "--method",
        choices=["healpy", "jax"],
        default="healpy",
        help="SHT method for spherical Cl computation (default: healpy)",
    )


def add_spectra_density_args(p):
    """3D density P(k) arguments for fli-spectra."""
    g = p.add_argument_group("3D P(k)")
    g.add_argument(
        "--kedges",
        type=float,
        nargs="+",
        default=None,
        metavar="K",
        help="k bin edges for P(k) (default: auto)",
    )
    p.add_argument(
        "--kmax",
        type=float,
        default=None,
        help="Maximum k for P(k) (default: Nyquist frequency based on mesh size)",
    )
    g.add_argument(
        "--multipoles",
        type=int,
        nargs="+",
        default=[0],
        metavar="L",
        help="Multipole moments to compute (default: 0 = monopole only)",
    )
    g.add_argument(
        "--los",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 1.0],
        metavar=("LX", "LY", "LZ"),
        help="Line-of-sight direction for multipole decomposition (default: 0 0 1)",
    )


def add_spectra_common_args(p):
    """Common arguments shared across all fli-spectra field types."""
    g = p.add_argument_group("common")
    g.add_argument(
        "--batch-size",
        type=int,
        default=None,
        dest="batch_size",
        help="Batch size for jax.lax.map (default: None = no batching)",
    )
    g.add_argument(
        "--enable-x64",
        action="store_true",
        dest="enable_x64",
        help="Enable JAX 64-bit floating-point precision (default: False)",
    )
