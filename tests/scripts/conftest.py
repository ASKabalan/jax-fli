"""Shared fixtures, constants, and helpers for fli-simulate CLI validation tests."""

from __future__ import annotations

import os
import subprocess

import jax_cosmo as jc
import pytest

# ---------------------------------------------------------------------------
# Shared simulation constants
# ---------------------------------------------------------------------------

_RES = 32
_BOX = 256.0
_SEED = 42
_HALO_MULT = 0.5
_HALO_SIZE = (int(_RES * _HALO_MULT), int(_RES * _HALO_MULT))  # (16, 16)

_COSMO_PARAMS = dict(
    Omega_c=0.2589,
    sigma8=0.8159,
    Omega_b=0.0486,
    h=0.6774,
    n_s=0.9667,
    Omega_k=0.0,
    w0=-1.0,
    wa=0.0,
    Omega_nu=0.0,
)

# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

_COSMO_CLI = [
    "--Omega-c",
    str(_COSMO_PARAMS["Omega_c"]),
    "--sigma8",
    str(_COSMO_PARAMS["sigma8"]),
    "--Omega-b",
    str(_COSMO_PARAMS["Omega_b"]),
    "--h",
    str(_COSMO_PARAMS["h"]),
    "--n-s",
    str(_COSMO_PARAMS["n_s"]),
    "--Omega-k",
    str(_COSMO_PARAMS["Omega_k"]),
    "--w0",
    str(_COSMO_PARAMS["w0"]),
    "--wa",
    str(_COSMO_PARAMS["wa"]),
    "--Omega-nu",
    str(_COSMO_PARAMS["Omega_nu"]),
]

_MESH_CLI = [
    "--mesh-size",
    str(_RES),
    str(_RES),
    str(_RES),
    "--box-size",
    str(_BOX),
    str(_BOX),
    str(_BOX),
]

_BASE_CLI = (
    [
        "--enable-x64",
        "--seed",
        str(_SEED),
        "--halo-multiplier",
        str(_HALO_MULT),
        "--observer-position",
        "0.5",
        "0.5",
        "0.5",
    ]
    + _MESH_CLI
    + _COSMO_CLI
)


def run_sim(cmd: list[str]) -> None:
    """Run a fli-simulate command on CPU, failing loudly on error."""
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    subprocess.run(cmd, check=True, env=env)


def lpt_order_cli(lpt_order: int, dealiased: bool, exact_growth: bool) -> list[str]:
    """Build the --lpt-order / --dealiased / --exact-growth CLI fragment."""
    args = ["--lpt-order", str(lpt_order)]
    if dealiased:
        args.append("--dealiased")
    if exact_growth:
        args.append("--exact-growth")
    return args


def scheme_cli(scheme: str, kernel_width: float | None) -> list[str]:
    """Build the --scheme / --kernel-width-arcmin CLI fragment."""
    args = ["--scheme", scheme]
    if kernel_width is not None:
        args += ["--kernel-width-arcmin", str(kernel_width)]
    return args


# ---------------------------------------------------------------------------
# Pytest parametrize reusables
# ---------------------------------------------------------------------------

#: (scheme, kernel_width_arcmin) covering all three painting schemes
SCHEME_PARAMS = [
    ("ngp", None),
    ("bilinear", None),
    ("rbf_neighbor", None),
    ("rbf_neighbor", 5.0),
    ("rbf_neighbor", 10.0),
]
SCHEME_IDS = ["ngp", "bilinear", "rbf-nokw", "rbf-5arcmin", "rbf-10arcmin"]

#: (lpt_order, dealiased, exact_growth)
LPT_ORDER_PARAMS = [
    (1, False, False),
    (2, False, False),
    (2, True, True),
]
LPT_ORDER_IDS = ["order1", "order2", "order2-dealiased-exact"]

# ---------------------------------------------------------------------------
# Session-scoped cosmology fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cosmo():
    return jc.Cosmology(**_COSMO_PARAMS)
