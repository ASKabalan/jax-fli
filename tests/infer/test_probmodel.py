"""Gates for the field-level model and the pixel-space likelihood (with optional scale cut).

The initial conditions are sampled WHITE from ``DistributedNormal`` (no transform, so the sampler
traverses white space) and colored inline for the forward model; the cosmology comes from
``PreconditionnedUniform`` priors. The likelihood is a per-pixel Gaussian registering conditionable
``observable_i`` sites; when ``config.ell_max`` is set each observable map is band-limited by
``scale_cut`` (map2alm -> cosine taper -> alm2map) before the Gaussian.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import jax_healpy as jhp
import numpy as np
import numpyro
import pytest
from numpyro.handlers import condition, seed, trace

import jax_fli as jfli

MESH = (16, 16, 16)
NSIDE = 16
MAX_Z = 0.25


def _config(**overrides):
    cosmo = jc.Planck18()
    box = tuple(float(x) for x in jfli.utils.compute_box_size_from_redshift(cosmo, MAX_Z, (0.5, 0.5, 0.5)))
    defaults = dict(
        mesh_size=MESH,
        box_size=box,
        halo_size=(0, 0),
        nb_steps=2,
        sim_mode="lpt",
        nbody_solver="BullFrog",
        t0=0.1,
        t1=1.0,
        lpt_order=1,
        number_of_shells=2,
        paint_order="cic",
        gradient_order=4,
        laplace_fd=True,
        shell_spacing="a",
        time_stepping="D",
        min_width=10.0,
        lensing_output="convergence",
        map2alm_method="jax",
        min_redshift=0.001,
        max_redshift=MAX_Z,
        n_integrate=8,
        nside=NSIDE,
        geometry="spherical",
        scheme="rbf_neighbor",
        observer_position=(0.5, 0.5, 0.5),
        paint_nside=NSIDE,
        kernel_width_pixels=0.8,
        fiducial_cosmology=jc.Planck18,
        nz_shear=[MAX_Z],
        log_lightcone=False,
        priors={},
        sigma_e=0.3,
        adjoint="checkpointed",
        checkpoints=2,
    )
    defaults.update(overrides)
    return jfli.ppl.Configurations(**defaults)


@pytest.fixture(scope="module")
def pixel_mock():
    """Mock data traced from the default (centered + pixel) model at fixed cosmology."""
    config = _config()
    model = jfli.ppl.full_field_probmodel(config)
    tr = trace(seed(model, 0)).get_trace()
    data = {"observable_0": tr["observable_0"]["value"]}
    return config, data, tr


def test_default_config_samples_white_ic(pixel_mock):
    """The IC site is the WHITE field: `initial_conditions` is a plain `DistributedNormal` sample
    (no transform; colored inline by the forward model), with no whitened `initial_conditions_base`
    site."""
    _, _, tr = pixel_mock
    assert tr["initial_conditions"]["type"] == "sample"
    assert "initial_conditions_base" not in tr
    assert tr["observable_0"]["type"] == "sample"
    # the sampled IC is the WHITE field, drawn from DistributedNormal (no transform)
    from jax_fli.fields import DensityField

    assert isinstance(tr["initial_conditions"]["fn"], jfli.infer.dist.DistributedNormal)
    assert isinstance(tr["initial_conditions"]["value"], DensityField)


def test_cosmology_sites(pixel_mock):
    """PreconditionnedUniform priors are reparam'd: NUTS moves the white `<name>_base`, and `<name>`
    is the in-bounds physical deterministic."""
    priors = {
        "Omega_c": jfli.infer.dist.PreconditionnedUniform(0.1, 0.5),
        "sigma8": jfli.infer.dist.PreconditionnedUniform(0.6, 1.0),
    }
    model = jfli.ppl.full_field_probmodel(_config(priors=priors))
    tr = trace(seed(model, 3)).get_trace()

    for name, prior in priors.items():
        assert tr[f"{name}_base"]["type"] == "sample"  # white latent NUTS moves
        assert tr[name]["type"] == "deterministic"  # physical value
        val = float(tr[name]["value"])
        assert float(prior.low) < val < float(prior.high)


def _cell_config(geometry, lensing_output, **ov):
    """Config for one (geometry, observable) cell of the pixel-likelihood matrix."""
    if geometry == "spherical":
        geom = dict(
            nside=NSIDE, geometry="spherical", scheme="rbf_neighbor", paint_nside=NSIDE, kernel_width_pixels=0.8
        )
    else:
        geom = dict(nside=NSIDE, geometry="flat", flatsky_npix=(16, 16), field_size=(3.0, 3.0), scheme="bilinear")
    return _config(lensing_output=lensing_output, **geom, **ov)


@pytest.mark.parametrize(
    "geometry,lensing_output",
    [
        ("spherical", "shear"),
        ("spherical", "density"),
        ("flat", "convergence"),
        ("flat", "shear"),
        ("flat", "density"),
    ],
)
def test_matrix_cells_finite(geometry, lensing_output):
    """Every (geometry, observable) pixel-likelihood cell builds a finite model + IC gradient."""
    model = jfli.ppl.full_field_probmodel(_cell_config(geometry, lensing_output))
    tr = trace(seed(model, 0)).get_trace()
    obs = tr["observable_0"]["value"]
    assert bool(jnp.isfinite(jnp.asarray(obs)).all())

    # condition on the traced data and check the pixel-likelihood potential + IC gradient are finite
    conditioned = condition(model, data={"observable_0": obs})
    _, potential_fn, _, _ = numpyro.infer.util.initialize_model(jax.random.PRNGKey(2), conditioned, dynamic_args=False)
    s = jax.random.normal(jax.random.PRNGKey(7), MESH)
    pe, g = jax.value_and_grad(potential_fn)({"initial_conditions": s})
    assert bool(jnp.isfinite(jnp.asarray(pe)))
    assert bool(jnp.isfinite(g["initial_conditions"]).all())


def test_number_counts_projection_properties():
    """The density observable is a PROJECTED_DENSITY overdensity map (zero-mean, finite)."""
    from jax_fli.fields import FieldStatus

    cfg = _cell_config("spherical", "density")
    tr = trace(seed(jfli.ppl.full_field_probmodel(cfg), 0)).get_trace()
    obs = tr["observable_0"]["value"]
    assert obs.shape == (12 * NSIDE**2,)
    assert bool(jnp.isfinite(obs).all())
    assert abs(float(obs.mean())) < 1e-3  # per-plane overdensity -> zero mean
    assert tr["observable_meta_data"]["value"].status == FieldStatus.PROJECTED_DENSITY


@pytest.mark.parametrize("lensing_output", ["density", "shear"])
def test_scale_cut_bandlimits_observable(lensing_output):
    """With config.ell_max set, the pixel-likelihood loc (band-limited model map) has ~zero power
    above ell_max — spherical spin-0 (density) and spin-2 (shear)."""
    ell_max, width = 12, 4
    cfg = _cell_config("spherical", lensing_output, ell_max=ell_max, ell_taper_width=width)
    tr = trace(seed(jfli.ppl.full_field_probmodel(cfg), 0)).get_trace()
    loc = tr["observable_0"]["fn"].loc  # the band-limited model map (a field), noise-free
    arr = np.asarray(loc.array if hasattr(loc, "array") else loc)
    m = arr if arr.ndim == 1 else arr[0]  # spin-2 shear is (2, npix); use the first component
    cl = np.asarray(jhp.anafast(m, lmax=2 * NSIDE - 1))
    assert np.median(cl[ell_max + 1 :]) < 1e-2 * np.median(cl[2 : ell_max - width])


@pytest.mark.parametrize("lensing_output", ["density", "shear"])
def test_scale_cut_gradient_finite(lensing_output):
    """value_and_grad through the ell_max scale-cut pixel likelihood (map2alm -> taper -> alm2map) is
    finite in the IC coordinates -- the actual inference path (spin-0 density and spin-2 shear)."""
    cfg = _cell_config("spherical", lensing_output, ell_max=12, ell_taper_width=4)
    model = jfli.ppl.full_field_probmodel(cfg)
    tr = trace(seed(model, 0)).get_trace()
    conditioned = condition(model, data={"observable_0": tr["observable_0"]["value"]})
    _, potential_fn, _, _ = numpyro.infer.util.initialize_model(jax.random.PRNGKey(2), conditioned, dynamic_args=False)
    s = jax.random.normal(jax.random.PRNGKey(7), MESH)
    pe, g = jax.value_and_grad(potential_fn)({"initial_conditions": s})
    assert bool(jnp.isfinite(jnp.asarray(pe)))
    assert bool(jnp.isfinite(g["initial_conditions"]).all())


def test_scale_cut_flat_raises():
    """Scale cut is spherical-only: flat geometry with ell_max set raises at model build."""
    with pytest.raises(NotImplementedError, match="spherical-only"):
        jfli.ppl.full_field_probmodel(_cell_config("flat", "convergence", ell_max=12, ell_taper_width=4))
