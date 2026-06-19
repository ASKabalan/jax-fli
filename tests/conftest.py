"""Shared fixtures for jax_fli tests."""

from __future__ import annotations

import os

import pytest


def pytest_configure(config):
    """Set JAX environment before any import that triggers JAX initialization.

    This runs before collection and before any test file is imported, so environment
    variables set here take effect when JAX is first initialized.
    """
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

    # When running distributed tests from the project root (pytest -m distributed),
    # pytest collects catalog/ and cosmology/ before distributed/, so those test
    # files' module-level `import jax` would initialize JAX with 1 device before
    # tests/distributed/conftest.py can set XLA_FLAGS. Set it here instead.
    markexpr = str(getattr(config.option, "markexpr", "") or "")
    is_distributed = "distributed" in markexpr and "not distributed" not in markexpr
    if is_distributed:
        if "SLURM_JOB_ID" in os.environ:
            import jax

            jax.distributed.initialize()
        else:
            os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")


def pytest_collection_modifyitems(items):
    """Auto-mark non-distributed tests as 'single_device'."""
    for item in items:
        if not item.get_closest_marker("distributed"):
            item.add_marker(pytest.mark.single_device)


def pytest_collection_finish(session):
    """Block mixed distributed/single_device sessions (checked after marker filtering)."""
    has_distributed = False
    has_single_device = False

    for item in session.items:
        if item.get_closest_marker("distributed"):
            has_distributed = True
        else:
            has_single_device = True

        if has_distributed and has_single_device:
            pytest.exit(
                "Cannot mix 'distributed' and 'single_device' tests in one pytest session.\n"
                "  Run: pytest -m distributed\n"
                "   or: pytest -m 'not distributed'",
                returncode=1,
            )


@pytest.fixture(scope="session", autouse=True)
def enable_x64():
    """Enable float64 for all tests (required for lossless parquet round-trips)."""
    import jax

    jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="session")
def cosmology():
    """Planck18 cosmology instance."""
    import jax_cosmo as jc

    return jc.Planck18()


@pytest.fixture(scope="session")
def lpt_spherical(cosmology):
    """A 5-shell spherical LPT density map (batched ``SphericalDensity``, nside 16).

    The shells (comoving ~30-156 Mpc/h) fit inside the 512 Mpc/h box, so the painted map carries
    real structure (non-degenerate) and full per-shell lightcone metadata — the data source for the
    summary-statistic API and catalog tests. Built once per session.
    """
    import jax
    import jax.numpy as jnp

    import jax_fli as jfli

    ic = jfli.gaussian_initial_conditions(
        jax.random.PRNGKey(0), (16, 16, 16), (512.0, 512.0, 512.0), cosmo=cosmology, nside=16
    )
    painting = jfli.PaintingOptions(target="spherical", paint_nside=16)
    sph, _ = jfli.lpt(
        cosmology,
        ic,
        ts=jnp.linspace(0.95, 0.99, 5),
        density_widths=jnp.full(5, 20.0),
        painting=painting,
    )
    return sph
