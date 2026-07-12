"""Gates for the fli-infer conditioning fixes (Task 1).

The model samples the WHITE initial conditions and colors them inline with the sampled cosmology, and
the cosmology sample sites are the reparam bases ``<name>_base``. Two bugs are fixed here:

1. Fixing/warm-starting the IC must condition the WHITE field, obtained by DE-coloring the catalog's
   colored physical delta with the inverse power-spectrum transform (else the model colors it twice).
2. Fixing cosmology must PIN the fiducial to the IC cosmology (conditioning the deterministic
   ``Omega_c`` name is a no-op under ``TransformReparam``); warm-starting must seed the ``<name>_base``
   sites via the inverse ``PreconditionnedUniform`` bijector ``ndtri((phys - lo) / (hi - lo))``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
import pytest
from jax.scipy.special import ndtri
from jaxpm.distributed import fft3d, ifft3d
from jaxpm.kernels import fftk
from numpyro.handlers import seed, trace

import jax_fli as jfli
from jax_fli.initial import interpolate_initial_conditions

MESH = (16, 16, 16)
BOX = (512.0, 512.0, 512.0)


def test_ic_decolor_roundtrip():
    """De-coloring (inverse transform) inverts the model's inline coloring: the white field fixed by
    fli-infer, re-colored by the model, reproduces the catalog's physical delta."""
    cosmo = jc.Planck18()
    white = jax.random.normal(jax.random.PRNGKey(0), MESH)
    # model coloring: interpolate_initial_conditions(white, cosmo)
    colored = interpolate_initial_conditions(white, MESH, BOX, cosmo=cosmo).array
    # fli-infer de-coloring: exact inverse (same 128-point P(k), no k=0 special case)
    k = jnp.logspace(-4, 1, 128)
    pk = jc.power.linear_matter_power(cosmo, k)
    fc = fft3d(colored)
    kmesh = sum((kk / BOX[i] * MESH[i]) ** 2 for i, kk in enumerate(fftk(fc))) ** 0.5
    pkmesh = jnp.interp(kmesh, k, pk) * (MESH[0] * MESH[1] * MESH[2]) / (BOX[0] * BOX[1] * BOX[2])
    recovered_white = ifft3d(fc / jnp.sqrt(pkmesh)).real
    # and re-coloring the recovered white must reproduce the colored field (what the model does)
    recolored = interpolate_initial_conditions(recovered_white, MESH, BOX, cosmo=cosmo).array
    assert np.allclose(np.asarray(recovered_white), np.asarray(white), atol=1e-5, rtol=1e-4)
    assert np.allclose(np.asarray(recolored).real, np.asarray(colored).real, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("phys,lohi", [(0.25, (0.1, 0.5)), (0.81, (0.6, 1.0)), (0.67, (0.5, 0.9))])
def test_base_warmstart_bijector_roundtrip(phys, lohi):
    """The base warm-start ``ndtri((phys - lo)/(hi - lo))`` pushed through the PreconditionnedUniform
    transform returns the physical value -- i.e. it seeds the correct latent-space point."""
    lo, hi = lohi
    base = float(ndtri((phys - lo) / (hi - lo)))
    prior = jfli.infer.PreconditionnedUniform(lo, hi)
    # apply the prior's transform chain to the base -> physical
    recovered = jnp.asarray(base)
    for t in prior.transforms:
        recovered = t(recovered)
    assert np.isclose(float(recovered), phys, atol=1e-6)


def test_fixed_cosmo_pins_fiducial():
    """With no cosmo priors, the model builds ``fiducial_cosmology()``; pinning it to the IC cosmology
    makes the traced ``cosmo`` equal the IC cosmology (not Planck18)."""
    ic_cosmo = jc.Planck18(Omega_c=0.2, sigma8=0.9)  # deliberately != default
    priors = {}
    config = jfli.ppl.Configurations(
        mesh_size=MESH,
        box_size=BOX,
        nside=16,
        geometry="spherical",
        sim_mode="lpt",
        lensing_output="density",
        fiducial_cosmology=lambda **_: ic_cosmo,  # the fli-infer fixed-cosmo pin
        nz_shear=[0.25],
        priors=priors,
        sigma_e=0.3,
    )
    tr = trace(seed(jfli.ppl.mock_probmodel(config), 0)).get_trace()
    cosmo = tr["cosmo"]["value"]
    assert np.isclose(float(cosmo.Omega_c), 0.2) and np.isclose(float(cosmo.sigma8), 0.9)
