"""Shared argument-group builders used by both entry-point scripts and the launcher.

All functions are pure argparse — no jax_fli imports.
"""
from __future__ import annotations

DEFAULT_NAME_TEMPLATE = "%constraint%_cosmo_M%mesh_size%_B%box_size%_STEPS%nb_steps%_c%omega_c%_S8%sigma8%_s%seed%"


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


def add_mesh_args(p, nargs=3):
    """Mesh resolution and box size.

    Parameters
    ----------
    nargs : int or str
        Use ``3`` for fixed 3-element meshes (entry-point scripts).
        Use ``"+"`` for flat multi-mesh lists (launcher).
    """
    if nargs == 3:
        p.add_argument(
            "--mesh-size",
            type=int,
            nargs=3,
            default=[64, 64, 64],
            metavar=("NX", "NY", "NZ"),
            help="Mesh resolution (default: 64 64 64)",
        )
        p.add_argument(
            "--box-size",
            type=float,
            nargs=3,
            default=[200.0, 200.0, 200.0],
            metavar=("LX", "LY", "LZ"),
            help="Box side lengths in Mpc/h (default: 200 200 200)",
        )
    else:
        p.add_argument(
            "--mesh-size",
            type=int,
            nargs="+",
            default=[64, 64, 64, 32, 32, 32],
            metavar="N",
            help="Flat list of mesh sizes grouped into triples (e.g. 64 64 64 32 32 32)",
        )
        p.add_argument(
            "--box-size",
            type=float,
            nargs="+",
            default=[200.0, 200.0, 200.0],
            metavar="L",
            help="Flat list of box sizes grouped into triples (e.g. 200 200 200)",
        )


def add_common_sim_args(p):
    """Simulation parameters shared across all subcommands.

    Excludes: ``--enable-x64`` (added inline per script), ``--nb-shells``
    (each script adds it inline with the appropriate type).
    """
    g = p.add_argument_group("simulation")
    g.add_argument("--lpt-order", type=int, default=2, choices=[1, 2], help="LPT order (default: 2)")
    g.add_argument("--t0", type=float, default=0.1, help="LPT starting scale factor (default: 0.1)")
    g.add_argument("--t1", type=float, default=1.0, help="Final scale factor (default: 1.0)")
    g.add_argument(
        "--nb-steps", type=int, default=30, dest="nb_steps", help="Number of integration steps (default: 30)"
    )
    g.add_argument(
        "--interp", choices=["none", "onion", "telephoto"], default="none", help="Interpolation kernel (default: none)"
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
    g.add_argument("--solver", choices=["kdk", "dkd", "bf"], default="kdk", help="N-body integrator (default: kdk)")
    g.add_argument(
        "--time-stepping",
        choices=["a", "D", "log_a"],
        default="a",
        dest="time_stepping",
        help="Integrator time-stepping variable (default: a)",
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


def add_cosmo_args(p, sweep=False):
    """Cosmological parameters.

    Parameters
    ----------
    sweep : bool
        When ``False`` (single-run scripts), ``--Omega-c``, ``--sigma8``,
        and ``--seed`` accept single scalar values.  When ``True``
        (grid / launcher), they accept one or more values or
        ``start:stop:step`` range strings.
    """
    g = p.add_argument_group("cosmology")
    g.add_argument("--Omega-b", type=float, default=0.0486, dest="Omega_b", help="Baryon density (default: 0.0486)")
    g.add_argument("--h", type=float, default=0.6774, help="Dimensionless Hubble parameter (default: 0.6774)")
    g.add_argument("--n-s", type=float, default=0.9667, dest="n_s", help="Spectral index (default: 0.9667)")
    g.add_argument("--Omega-k", type=float, default=0.0, dest="Omega_k", help="Curvature density (default: 0.0)")
    g.add_argument("--w0", type=float, default=-1.0, help="Dark energy EOS w0 (default: -1.0)")
    g.add_argument("--wa", type=float, default=0.0, help="Dark energy EOS wa (default: 0.0)")
    g.add_argument("--Omega-nu", type=float, default=0.0, dest="Omega_nu", help="Neutrino density (default: 0.0)")
    if sweep:
        g.add_argument(
            "--Omega-c",
            nargs="+",
            type=str,
            default=["0.2589"],
            dest="Omega_c",
            help="Omega_c values or range strings (e.g. 0.2:0.4:0.05)",
        )
        g.add_argument("--sigma8", nargs="+", type=str, default=["0.8159"], help="sigma8 values or range strings")
        g.add_argument("--seed", nargs="+", type=str, default=["0"], help="Seed values or range strings")
    else:
        g.add_argument(
            "--Omega-c", type=float, default=0.2589, dest="Omega_c", help="Cold dark matter density (default: 0.2589)"
        )
        g.add_argument("--sigma8", type=float, default=0.8159, help="sigma8 (default: 0.8159)")
        g.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")


def add_lensing_args(p):
    """Lensing / source-distribution parameters."""
    g = p.add_argument_group("lensing")
    g.add_argument(
        "--nz-shear",
        nargs="+",
        default=["s3"],
        metavar="Z",
        help="Source redshifts or 's3' for Stage-3 preset (default: s3)",
    )
    g.add_argument("--min-z", type=float, default=0.01, help="Minimum redshift for n(z) integration (default: 0.01)")
    g.add_argument("--max-z", type=float, default=1.5, help="Maximum redshift for n(z) integration (default: 1.5)")
    g.add_argument("--n-integrate", type=int, default=32, help="Number of integration points for n(z) (default: 32)")


def add_lightcone_args(p):
    """Lightcone / shell geometry parameters.

    Does **not** include ``--nb-shells`` — each script adds it inline with
    the appropriate type (``int`` for single-run, ``str nargs='+'`` for grid).
    """
    g = p.add_argument_group("lightcone")
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
        help="Apply drift correction when painting lightcone shells",
        default=False,
    )
    g.add_argument(
        "--min-width", type=float, default=50.0, dest="min_width", help="Minimum shell width in Mpc/h (default: 50.0)"
    )
    g.add_argument("--equal-vol", action="store_true", help="Use equal-volume shell partitioning")


def add_output_target_args(p):
    """Output-target flags (mutually exclusive painting targets + companion field-size)."""
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
