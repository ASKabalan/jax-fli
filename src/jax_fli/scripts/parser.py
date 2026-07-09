"""Shared argument-group builders used by entry-point scripts.

All functions are pure argparse — no jax_fli imports. Each builder owns **one** concern so
entry scripts compose only the groups they actually consume:

* runtime               — ``add_common_args`` (``--enable-x64``)
* distributed           — ``add_distributed_args`` (``--pdim`` / ``--nodes`` / ``--gpus-per-node``)
* cosmology             — ``add_cosmo_args``
* simulation geometry   — ``add_simulation_settings_args`` (box/mesh/halo/observer/seed)
                          + ``add_output_target_args`` (nside / density / flat-sky + painting)
* integration           — ``add_integration_settings_args`` (physics, shell timing)
* lensing               — ``add_lensing_args``
* source                — ``add_source_args`` (local glob or HuggingFace repo; ``prefix=`` for a 2nd
                          source, ``multi=`` for one-pattern-per-chain; used by born/dorian/infer/extract)
* lensing post-proc     — ``add_lensing_postproc_args`` (output/nside/normalization for fli-born-rt / fli-dorian-rt)
* priors / inference    — ``add_prior_args`` / ``add_infer_args`` (sampler-only)
* forward-model         — ``add_forward_model_args`` (shape noise / mask / lightcone) +
                          ``add_scale_cut_args`` (optional pixel-likelihood ell cut) + ``add_gradient_args``
                          (adjoint / checkpoints) — the last two used by fli-infer

``--sim-mode`` lives in ``add_integration_settings_args``: the full-field entry points (fli-infer /
fli-samples) get the ``lpt``/``pm`` choice (default ``pm``); fli-simulate parametrizes it to be
required and to add the ``lensing`` choice.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Runtime / device
# ---------------------------------------------------------------------------


def add_common_args(p):
    """JAX runtime knobs shared by every compute script (currently ``--enable-x64``)."""
    g = p.add_argument_group("runtime")
    g.add_argument(
        "--enable-x64",
        action="store_true",
        dest="enable_x64",
        help="Enable JAX 64-bit floating-point precision (default: False)",
    )


def add_distributed_args(p):
    """Process-grid dimensions (pdim), node count, and GPUs-per-node — single-device defaults."""
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
    g.add_argument(
        "--gpus-per-node",
        type=int,
        default=None,
        dest="gpus_per_node",
        help="GPUs per node (intra-node NVLink slice width) for the hybrid device mesh on "
        "non-uniform interconnects. Default: None (falls back to $SLURM_GPUS_ON_NODE).",
    )


# ---------------------------------------------------------------------------
# Cosmology
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Simulation geometry — box + output target
# ---------------------------------------------------------------------------


def add_simulation_settings_args(p):
    """Box geometry and RNG: mesh, box, halo, observer position, apodization, seed.

    The *output target* (nside / density / flat-sky) and painting scheme live in the separate
    ``add_output_target_args`` builder so the box geometry and the projection choice stay
    independent concerns.
    """
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


def add_output_target_args(p):
    """Output projection target and painting scheme.

    Selects the field the pipeline emits — a 3D density, a flat-sky map, a HEALPix map, or
    (when none is given) particles — plus the spherical painting interpolation knobs.
    """
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
        "--kernel-width-pixels",
        type=float,
        default=None,
        dest="kernel_width_pixels",
        help="RBF smoothing kernel width in HEALPix pixels, e.g. 0.8 (default: None)",
    )
    g.add_argument(
        "--pixel-window-deconvolution",
        action="store_true",
        dest="pixel_window_deconvolution",
        help="Deconvolve the HEALPix pixel window (a_lm level) from spherical maps after painting "
        "(requires --scheme ngp or rbf_neighbor; default: False).",
    )


# ---------------------------------------------------------------------------
# Integration / lightcone
# ---------------------------------------------------------------------------


def add_integration_settings_args(p, solver_default="kdk", sim_mode_default="pm", sim_mode_choices=("lpt", "pm")):
    """Integration / lightcone physics: solver, time-stepping, shell timing, force kernels.

    ``solver_default`` lets each command pick the N-body integrator default: ``fli-simulate``
    keeps ``kdk`` (DoubleKickDrift), while the full-field model entry points (``fli-infer`` /
    ``fli-samples``) pass ``bf`` (BullFrog, the Configurations default).

    ``--sim-mode`` selects the pipeline depth. The full-field entry points (``fli-infer`` /
    ``fli-samples``) keep the defaults (choices ``lpt``/``pm``, default ``pm``); ``fli-simulate``
    passes ``sim_mode_default=None`` (making it required) and ``sim_mode_choices`` adding ``lensing``.

    Lensing parameters are **not** added here — scripts call ``add_lensing_args`` explicitly when
    they need source-distribution / Born options.
    """
    g = p.add_argument_group("integration")
    g.add_argument(
        "--sim-mode",
        choices=list(sim_mode_choices),
        default=sim_mode_default,
        required=sim_mode_default is None,  # fli-simulate passes None => required
        dest="sim_mode",
        help="Pipeline depth: 'lpt' = LPT-only lightcone (no N-body); 'pm' = LPT + N-body lightcone; "
        "'lensing' (fli-simulate only) = pm + Born -> kappa.",
    )
    g.add_argument(
        "--nb-shells", type=int, default=None, dest="nb_shells", help="Number of lightcone shells (default: 8)"
    )
    g.add_argument("--lpt-order", type=int, default=2, choices=[1, 2], dest="lpt_order", help="LPT order (default: 2)")
    g.add_argument("--t0", type=float, default=0.001, help="LPT starting scale factor (default: 0.001)")
    g.add_argument("--t1", type=float, default=1.0, help="Final scale factor (default: 1.0)")
    g.add_argument(
        "--nb-steps", type=int, default=30, dest="nb_steps", help="Number of integration steps (default: 30)"
    )
    g.add_argument(
        "--interp", choices=["none", "onion", "telephoto"], default="none", help="Interpolation kernel (default: none)"
    )
    g.add_argument(
        "--solver",
        choices=["kdk", "dkd", "bf"],
        default=solver_default,
        help=f"N-body integrator (default: {solver_default})",
    )
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
        "--paint-order",
        type=str,
        default="cic",
        choices=["ngp", "cic", "tsc", "pcs"],
        dest="paint_order",
        help="Mass-assignment order for force painting/readout (default: cic)",
    )
    g.add_argument(
        "--deconvolution",
        action="store_true",
        dest="deconvolution",
        help="Deconvolve the mass-assignment window in the force computation (default: False)",
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
    ts_group = g.add_mutually_exclusive_group()
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


# ---------------------------------------------------------------------------
# Lensing
# ---------------------------------------------------------------------------


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
    g.add_argument(
        "--quadrature",
        choices=["midpoint", "gauss_legendre"],
        default="midpoint",
        help="Born per-shell weight quadrature: 'midpoint' evaluates the lensing kernel at shell centers "
        "(historic); 'gauss_legendre' integrates it exactly over each shell (default: midpoint)",
    )


def add_source_args(p, *, prefix="", multi=False):
    """Generic catalog source: a local parquet glob OR a HuggingFace dataset repo.

    Shared by the post-processing scripts (``fli-born-rt`` / ``fli-dorian-rt``) and the inference and
    extraction entry points. The source is EITHER ``--input`` (a local file/glob) OR ``--repo`` +
    ``--data-files`` (a glob of parquet files inside a HuggingFace dataset repo); ``--input`` and
    ``--repo`` are mutually exclusive (enforced by argparse) and consumed by ``scripts._common``
    (``_load_lightcone`` / ``_resolve_source`` / ``_resolve_chain_sources``).

    ``prefix`` renames the flags so one parser can carry two independent sources: the default ``""``
    yields ``--input`` / ``--repo`` / ``--data-files`` (dests ``input`` / ``repo`` / ``data_files``);
    ``prefix="ic"`` yields ``--ic-input`` / ``--ic-repo`` / ``--ic-data-files`` (dests ``ic_input`` /
    ``ic_repo`` / ``ic_data_files``), used by ``fli-infer`` for its optional initial-condition source
    alongside the unprefixed observable source.

    ``multi`` makes ``--input`` and ``--data-files`` accept several patterns (``nargs="+"``); each
    pattern is a *separate* source — used by ``fli-extract`` where one pattern maps to one MCMC chain.
    The single-source consumers (born / dorian / infer) keep ``multi=False``.

    The Streamlit UI mirrors this builder in
    ``jax-fli-result/app/components/source_form.py::render_source_form`` (same ``prefix`` / ``multi``).
    """
    dash = f"{prefix}-" if prefix else ""
    under = f"{prefix}_" if prefix else ""
    nargs = "+" if multi else None
    repeat = " Repeatable — each pattern is one chain." if multi else ""

    g = p.add_argument_group(f"{prefix} source".strip())
    src = g.add_mutually_exclusive_group()
    src.add_argument(
        f"--{dash}input",
        dest=f"{under}input",
        default=None,
        nargs=nargs,
        metavar="FILE_OR_GLOB",
        help=f"Local parquet file(s): a path or glob (e.g. 'results/*.parquet').{repeat} "
        f"Mutually exclusive with --{dash}repo/--{dash}data-files.",
    )
    src.add_argument(
        f"--{dash}repo",
        dest=f"{under}repo",
        default=None,
        metavar="REPO_ID",
        help=f"HuggingFace dataset repo id (e.g. ASKabalan/jax-fli-experiments). Use with --{dash}data-files.",
    )
    g.add_argument(
        f"--{dash}data-files",
        dest=f"{under}data_files",
        default=None,
        nargs=nargs,
        metavar="GLOB",
        help=f"Glob of parquet files within --{dash}repo (e.g. '01-resolution/density/*.parquet').{repeat}",
    )


def add_lensing_postproc_args(p):
    """Output + density→κ knobs for the post-processing lensing scripts (fli-born-rt / fli-dorian-rt).

    ``--nside`` ud_grade-downsamples the stacked density lightcone before lensing; ``--normalization``
    selects the density→δ overdensity normalization passed to ``jfli.born`` / ``jfli.raytrace``. The
    density source itself is the separate ``add_source_args`` group.
    """
    g = p.add_argument_group("lensing post-processing")
    g.add_argument(
        "--output",
        "-o",
        default=".",
        metavar="DIR",
        help="Output directory (default: .)",
    )
    g.add_argument(
        "--nside",
        type=int,
        default=None,
        help="Downsample the density lightcone to this HEALPix nside before lensing (default: native).",
    )
    g.add_argument(
        "--normalization",
        choices=["global", "per_plane"],
        default="global",
        help="Overdensity normalization for the density→δ conversion, used by BOTH fli-born-rt and "
        "fli-dorian-rt (passed as normalization= to jfli.born / jfli.raytrace): 'global' divides by one "
        "mean across all shells, 'per_plane' normalizes each shell independently (default: global).",
    )


# ---------------------------------------------------------------------------
# Priors / inference
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


def add_infer_args(p, *, with_initial_condition=True):
    """Sampler-only configuration for fli-infer (NUTS / MCLMC tuning).

    Forward-model gradient (``--adjoint`` / ``--checkpoints``) lives in ``add_gradient_args`` and the
    shape noise (``--sigma-e``) in ``add_forward_model_args`` — both are shared with fli-muse, so this
    builder no longer owns them. Does NOT include path args (--observable, --path) — those differ between
    the entry script (full paths, required) and the launcher (constructed from
    --observable-dir / --output-dir / --chain-index).

    ``with_initial_condition`` controls the single-path ``--initial-condition`` flag: it stays on for
    ``fli-samples`` (the default), but ``fli-infer`` passes ``False`` and instead takes its optional IC
    through a prefixed ``add_source_args(p, prefix="ic")`` source (local glob or HF repo), mirroring
    its observable source.
    """
    g = p.add_argument_group("inference")
    if with_initial_condition:
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
    g.add_argument(
        "--num-warmup", type=int, default=500, dest="num_warmup", help="MCMC warmup iterations (default: 500)"
    )
    g.add_argument(
        "--num-samples", type=int, default=1000, dest="num_samples", help="Samples per batch (default: 1000)"
    )
    g.add_argument(
        "--batch-count", type=int, default=5, dest="batch_count", help="Number of sequential batches (default: 5)"
    )
    g.add_argument("--sampler", choices=["NUTS", "MCLMC"], default="NUTS", help="MCMC sampler (default: NUTS)")
    # NUTS tuning
    g.add_argument(
        "--max-num-doublings",
        type=int,
        default=10,
        dest="max_num_doublings",
        help="NUTS leapfrog trajectory doubling depth (default: 10)",
    )
    g.add_argument(
        "--target-accept",
        type=float,
        default=0.8,
        dest="target_accept",
        help="NUTS window-adaptation target acceptance rate (default: 0.8)",
    )
    # MCLMC tuning
    g.add_argument(
        "--mclmc-desired-energy-var",
        type=float,
        default=1e-3,
        dest="mclmc_desired_energy_var",
        help="MCLMC desired energy variance for L/step_size tuning (default: 1e-3)",
    )
    g.add_argument(
        "--mclmc-init-step-size",
        type=float,
        default=1e-4,
        dest="mclmc_init_step_size",
        help="MCLMC initial step size = sqrt(total_dim) * scale (default: 1e-4)",
    )
    g.add_argument("--no-progress-bar", action="store_true", dest="no_progress_bar", help="Suppress tqdm progress bars")


def add_forward_model_args(p):
    """Full-field likelihood knobs (used by fli-infer, fli-samples, and fli-muse).

    These map onto Configurations fields used by the survey-mask-aware likelihood: the shape-noise
    dispersion ``--sigma-e`` (the single shared home for it), a footprint mask, the inflated sigma on
    pixels outside it, and whether to record the lightcone.
    ``--lensing-output`` (convergence vs shear) lives here because shear is a forward-model
    concern only — the simulation / ray-tracing scripts (fli-simulate / fli-born-rt /
    fli-dorian-rt) emit density or convergence, never shear. ``--apodization-scale-deg`` is in
    the simulation-settings group (shared with fli-simulate).
    """
    g = p.add_argument_group("forward model")
    g.add_argument("--sigma-e", type=float, default=0.26, dest="sigma_e", help="Shape-noise dispersion (default: 0.26)")
    g.add_argument(
        "--lensing-output",
        choices=["convergence", "shear", "reduced_shear"],
        default="convergence",
        dest="lensing_output",
        help="Observable the forward model produces: convergence (kappa) or spin-2 shear / "
        "reduced_shear via Kaiser-Squires (default: convergence).",
    )
    g.add_argument(
        "--mask",
        type=str,
        default=None,
        metavar="MASK",
        help="Survey footprint for the likelihood: 'des_y3' or a path to a HEALPix map "
        "(.npy/.npz/.fits). Pixels outside it get --sigma-unobserved (default: no mask).",
    )
    g.add_argument(
        "--sigma-unobserved",
        type=float,
        default=1e6,
        dest="sigma_unobserved",
        help="Likelihood sigma applied on pixels outside --mask (default: 1e6)",
    )
    g.add_argument(
        "--log-lightcone",
        action="store_true",
        dest="log_lightcone",
        help="Record the lightcone as a deterministic site in the trace (default: False)",
    )
    g.add_argument(
        "--apodization-scale-deg",
        type=float,
        default=1.0,
        dest="apodization_scale_deg",
        help="C2 apodization scale (deg) for the off-center observer visibility mask (default: 1.0)",
    )
    g.add_argument(
        "--map2alm-method",
        choices=["jax", "jax_cuda"],
        default="jax",
        dest="map2alm_method",
        help="Method for map to alm conversion (default: jax)",
    )


# ---------------------------------------------------------------------------
# Pixel-likelihood scale cut + forward-model gradient (shared by the gradient-based
# full-field entry fli-infer; NOT fli-samples / post-processing)
# ---------------------------------------------------------------------------


def add_scale_cut_args(p):
    """Optional map-level scale cut for the pixel likelihood (used by fli-infer).

    If ``--ell-max`` is set, each observable map is band-limited to that multipole (map2alm ->
    cosine taper -> alm2map) before the per-pixel Gaussian; the taper reaches zero at ``--ell-max``
    with a cosine roll-off of ``--ell-taper-width`` in ell. Spherical geometry only. If unset, no cut.
    """
    g = p.add_argument_group("scale cut")
    g.add_argument(
        "--ell-max",
        type=int,
        default=None,
        dest="ell_max",
        help="Scale cut: band-limit each observable map to this multipole before the pixel Gaussian (default: off)",
    )
    g.add_argument(
        "--ell-taper-width",
        type=int,
        default=8,
        dest="ell_taper_width",
        help="Cosine roll-off width of the scale-cut taper, in ell (default: 8)",
    )


def add_gradient_args(p):
    """Forward-model backprop strategy (used by fli-infer's NUTS and fli-muse's score gradient).

    ``checkpointed`` trades recomputation for memory when differentiating the N-body integration;
    ``--checkpoints`` sets the number of gradient checkpoints.
    """
    g = p.add_argument_group("gradient")
    g.add_argument(
        "--adjoint",
        choices=["checkpointed", "recursive"],
        default="checkpointed",
        help="Gradient strategy for the N-body backprop (default: checkpointed)",
    )
    g.add_argument("--checkpoints", type=int, default=10, help="Number of gradient checkpoints (default: 10)")
