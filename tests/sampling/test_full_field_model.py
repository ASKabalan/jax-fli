"""End-to-end full-field forward/probabilistic model: output switch, survey-mask
likelihood, observer visibility mask, and solver variants.

Runs a coarse (16^3 mesh, nside=16) but physically valid spherical model (box large
enough to reach the z=1.5 sources) under a NumPyro seed+trace, and inspects the
observation sites (loc/scale of the per-bin DistributedNormal).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numpyro")
datasets = pytest.importorskip("datasets")

import jax_cosmo as jc
import jax_fli as jfli
from numpyro.handlers import seed, trace

NSIDE = 16
NPIX = 12 * NSIDE**2
N_BINS = 2


@pytest.fixture(scope="module")
def base_kwargs():
    """Base Configurations kwargs: coarse mesh, big box (reaches z=1.5), 2 source bins."""
    return dict(
        mesh_size=(16, 16, 16),
        box_size=(6000.0, 6000.0, 6000.0),
        nside=NSIDE,
        fiducial_cosmology=jc.Planck18,
        nz_shear=[0.5, 1.0],
        priors={
            "Omega_c": jfli.infer.PreconditionnedUniform(0.1, 0.5),
            "sigma8": jfli.infer.PreconditionnedUniform(0.6, 1.0),
        },
        sigma_e=0.26,
        geometry="spherical",
        lpt_order=1,
        t0=0.1,
        nb_steps=6,
        number_of_shells=4,
        max_redshift=1.5,
    )


@pytest.mark.parametrize(
    "lensing_output, expected_shape",
    [
        ("convergence", (NPIX,)),
        ("shear", (2, NPIX)),
        ("reduced_shear", (2, NPIX)),
    ],
)
def test_observable_shape(base_kwargs, lensing_output, expected_shape):
    """The per-bin observation site has the convergence/shear shape for the chosen output."""
    config = jfli.ppl.Configurations(**base_kwargs, lensing_output=lensing_output)
    tr = trace(seed(jfli.ppl.full_field_probmodel(config), 0)).get_trace()
    for i in range(N_BINS):
        assert tr[f"kappa_{i}"]["value"].shape == expected_shape
        assert np.all(np.isfinite(tr[f"kappa_{i}"]["value"]))


def test_survey_mask_inflates_sigma(base_kwargs):
    """With a DES-like mask, the likelihood sigma is inflated on unobserved pixels."""
    mask = np.ones(NPIX)
    mask[: NPIX // 2] = 0.0  # half the sky unobserved
    config = jfli.ppl.Configurations(**base_kwargs, lensing_output="convergence", mask=mask, sigma_unobserved=1e3)
    tr = trace(seed(jfli.ppl.full_field_probmodel(config), 0)).get_trace()

    scale = np.asarray(tr["kappa_0"]["fn"].scale)
    assert scale.shape == (NPIX,)
    # unobserved pixels carry sigma_unobserved; observed pixels a single smaller value
    assert np.allclose(scale[mask == 0], 1e3)
    obs_vals = np.unique(scale[mask > 0])
    assert obs_vals.size == 1 and obs_vals[0] < 1e3


def test_offcenter_observer_visibility_mask(base_kwargs):
    """The observer visibility mask is a Kaiser-Squires concern: it is built for an off-center
    observer and applied to the kappa map *before* get_shear (shear / reduced_shear). Convergence
    is returned unmasked, so an off-center convergence observable has no forced zeros.

    (The mask cannot be isolated by a center-vs-off-center comparison: observer_position also flows
    into the lightcone geometry, so both convergence and shear already differ between observers.)
    """
    # The mask itself: a center observer -> identity 1; off-center -> an apodized partial footprint.
    assert jfli.data.build_observer_visibility_mask((0.5, 0.5, 0.5), NSIDE, 1.0) == 1
    vm = np.asarray(jfli.data.build_observer_visibility_mask((0.0, 0.5, 1.0), NSIDE, 1.0))
    assert vm.shape == (NPIX,)
    assert (vm == 0.0).any() and (vm > 0.0).any()  # non-empty, non-full footprint

    def loc(lensing_output):
        config = jfli.ppl.Configurations(
            **base_kwargs, lensing_output=lensing_output, observer_position=(0.0, 0.5, 1.0)
        )
        tr = trace(seed(jfli.ppl.full_field_probmodel(config), 0)).get_trace()
        return np.asarray(tr["kappa_0"]["fn"].loc)

    # Convergence is returned unmasked -> no forced zeros (the old code masked it here).
    conv = loc("convergence")
    assert np.all(np.isfinite(conv)) and not (conv == 0.0).any()

    # Shear applies the apodized mask to kappa before KS -> runs end-to-end with finite output.
    shear = loc("shear")
    assert shear.shape == (2, NPIX) and np.all(np.isfinite(shear))


@pytest.mark.parametrize("solver", ["DoubleKickDrift", "DriftKickDrift", "BullFrog"])
def test_solver_variants_run(base_kwargs, solver):
    """All three selectable solvers build and run end-to-end with finite output."""
    config = jfli.ppl.Configurations(**base_kwargs, nbody_solver=solver)
    tr = trace(seed(jfli.ppl.full_field_probmodel(config), 0)).get_trace()
    assert np.all(np.isfinite(tr["kappa_0"]["value"]))
