"""Lightcone-mode tests for NBody."""

from __future__ import annotations

import jax.numpy as jnp
import jax_cosmo as jc
import jax_fli as jfli
import numpy as np
import pytest

PAINT_NSIDE = 16

T0 = 0.1
T1 = 1.0


def _make_solver(target="flat", shell_spacing="comoving", min_width=1.0, n_steps=10):
    return jfli.DoubleKickDrift(
        interp_kernel=jfli.NoInterp(painting=jfli.PaintingOptions(target=target, paint_nside=PAINT_NSIDE)),
        t0=T0,
        t1=T1,
        n_steps=n_steps,
        shell_spacing=shell_spacing,
        min_width=min_width,
    )


@pytest.fixture(scope="session")
def lpt_fields(cosmology, small_initial_field):
    """LPT particle snapshot at T0."""
    dx, p = jfli.lpt(cosmology, small_initial_field, ts=T0, order=1)
    return dx, p


@pytest.mark.parametrize("target", ["particles", "density"])
@pytest.mark.parametrize("a_target", [0.5, 0.8])
def test_scalar_ts(cosmology, lpt_fields, target, a_target):
    """Scalar ts → snapshot at single epoch."""
    solver = _make_solver(target=target)
    dx, p = lpt_fields
    result = jfli.nbody(
        cosmology,
        dx,
        p,
        solver=solver,
        ts=a_target,
    )

    assert not result.is_batched()
    assert result.status == jfli.FieldStatus.LIGHTCONE

    if target == "particles":
        assert isinstance(result, jfli.ParticleField)
    else:
        assert isinstance(result, jfli.DensityField)


@pytest.mark.parametrize("target", ["flat", "spherical", "particles", "density"])
@pytest.mark.parametrize("nb_shells", [1, 5, 10])
@pytest.mark.parametrize(
    "shell_spacing",
    [
        "a",
        "growth",
        "comoving",
        "equal_vol",
    ],
)
def test_nb_shells(cosmology, lpt_fields, target, nb_shells, shell_spacing):
    solver = _make_solver(target=target, shell_spacing=shell_spacing, min_width=1.0, n_steps=15)
    dx, p = lpt_fields
    result = jfli.nbody(
        cosmology,
        dx,
        p,
        solver=solver,
        nb_shells=nb_shells,
    )

    assert result.status == jfli.FieldStatus.LIGHTCONE

    if nb_shells >= 2:
        assert result.is_batched()
        a = jnp.atleast_1d(result.scale_factors)
        assert a.shape[0] == nb_shells
        assert result.comoving_centers is not None
        assert result.comoving_centers.shape[0] == nb_shells
        assert result.density_width is not None
        assert result.density_width.shape[0] == nb_shells
        assert np.all(np.asarray(result.density_width) > 0)
        z = jc.utils.a2z(a)
        np.testing.assert_allclose(np.asarray(result.z_sources), np.asarray(z), rtol=1e-5)
    else:
        assert not result.is_batched()

    if target == "flat":
        assert isinstance(result, jfli.FlatDensity)
    elif target == "spherical":
        assert isinstance(result, jfli.SphericalDensity)
    elif target == "density":
        assert isinstance(result, jfli.DensityField)
    else:
        assert isinstance(result, jfli.ParticleField)


@pytest.mark.parametrize("target", ["flat", "spherical", "particles", "density"])
def test_1d_ts(cosmology, lpt_fields, target):
    """1-D ts array → multi-shell lightcone."""
    if target == "spherical":
        pytest.importorskip("jax_healpy")

    solver = _make_solver(target=target)
    ts = jnp.linspace(0.5, 0.9, 4)
    widths = jnp.full(4, 20.0)

    dx, p = lpt_fields
    result = jfli.nbody(
        cosmology,
        dx,
        p,
        solver=solver,
        ts=ts,
        density_widths=widths,
    )

    assert result.status == jfli.FieldStatus.LIGHTCONE
    assert result.is_batched()
    a = jnp.atleast_1d(result.scale_factors)
    assert a.shape[0] == 4

    if target == "flat":
        assert isinstance(result, jfli.FlatDensity)
    elif target == "spherical":
        assert isinstance(result, jfli.SphericalDensity)
    elif target == "density":
        assert isinstance(result, jfli.DensityField)
    else:
        assert isinstance(result, jfli.ParticleField)


@pytest.mark.parametrize("target", ["flat", "spherical", "particles", "density"])
def test_2d_ts(cosmology, lpt_fields, target):
    """2-D ts array (near/far) → multi-shell lightcone."""
    if target == "spherical":
        pytest.importorskip("jax_healpy")

    solver = _make_solver(target=target)
    a_near = jnp.array([0.9, 0.8, 0.7])
    a_far = jnp.array([0.85, 0.75, 0.65])
    ts = jnp.stack([a_near, a_far])

    dx, p = lpt_fields
    result = jfli.nbody(
        cosmology,
        dx,
        p,
        solver=solver,
        ts=ts,
    )

    assert result.status == jfli.FieldStatus.LIGHTCONE
    assert result.is_batched()
    a = jnp.atleast_1d(result.scale_factors)
    assert a.shape[0] == 3

    if target == "flat":
        assert isinstance(result, jfli.FlatDensity)
    elif target == "spherical":
        assert isinstance(result, jfli.SphericalDensity)
    elif target == "density":
        assert isinstance(result, jfli.DensityField)
    else:
        assert isinstance(result, jfli.ParticleField)


@pytest.mark.parametrize("target", ["particles", "density"])
def test_snapshot_default(cosmology, lpt_fields, target):
    """No ts/nb_shells → snapshot at t1."""
    solver = jfli.DoubleKickDrift(
        interp_kernel=jfli.NoInterp(painting=jfli.PaintingOptions(target=target)),
        t0=T0,
        t1=T1,
        n_steps=10,
    )

    dx, p = lpt_fields
    result = jfli.nbody(
        cosmology,
        dx,
        p,
        solver=solver,
    )

    assert result.status == jfli.FieldStatus.LIGHTCONE
    assert not result.is_batched()

    if target == "particles":
        assert isinstance(result, jfli.ParticleField)
    else:
        assert isinstance(result, jfli.DensityField)


def test_nb_shells_exceeds_n_steps(cosmology, lpt_fields):
    """nb_shells > n_steps should raise ValueError."""
    solver = _make_solver(target="particles", n_steps=5)
    dx, p = lpt_fields

    with pytest.raises(ValueError, match="nb_shells.*n_steps"):
        jfli.nbody(
            cosmology,
            dx,
            p,
            solver=solver,
            nb_shells=20,
        )
