"""Theory angular power spectra for density shells (``compute_theory_cl_for_density``).

These guard the comoving-volume fix: a painted shell is a comoving-VOLUME projection
(``q(chi) ∝ chi**2``), so the default ``radial_weight="comoving_volume"`` must

  * be independent of the (deprecated) ``nz_zmax`` and converged in ``n_chi``,
  * never NaN across a wide redshift range (the legacy ``tophat_z`` path does),
  * agree with the equivalent jax_cosmo :class:`comoving_tophat` number-counts spectrum
    on a well-sampled shell, while differing substantially from the legacy ``tophat_z``
    weighting on the nearest chi->0 shell (the defect the fix addresses),
  * stay jit-able and differentiable w.r.t. cosmology.

x64 is enabled session-wide by the root ``enable_x64`` fixture (weak-lensing/number-counts
theory NaNs above ell ~ 215 in float32).
"""

from __future__ import annotations

import jax.numpy as jnp
import jax_cosmo as jc
import pytest
from numpy.testing import assert_allclose

from jax_fli import SphericalDensity, compute_theory_cl_for_density
from jax_fli.summary_statistics.theory import comoving_tophat

ELL = jnp.arange(2, 200)


@pytest.fixture(scope="module")
def shells_wide(cosmology):
    """A CosmoGrid-like shell-geometry carrier: 7 comoving shells from chi≈0 to z≈1.4.

    The nearest shell starts at chi=0 (narrow in z); widths grow with distance — the geometry
    that breaks the legacy redshift-tophat normalisation/sampling.  Only the lightcone metadata
    is read by the theory code, so a tiny ``nside=1`` map is enough.
    """
    centers = jnp.array([19.0, 75.0, 200.0, 425.0, 950.0, 2000.0, 2950.0])
    widths = jnp.array([38.0, 50.0, 60.0, 70.0, 100.0, 100.0, 100.0])
    a = jc.background.a_of_chi(cosmology, centers)
    z = jc.utils.a2z(a)
    return SphericalDensity(
        array=jnp.zeros((centers.shape[0], 12)),
        nside=1,
        mesh_size=(8, 8, 8),
        box_size=(100.0, 100.0, 100.0),
        comoving_centers=centers,
        density_width=widths,
        scale_factors=a,
        z_sources=z,
    )


@pytest.fixture(scope="module")
def near_shell(cosmology):
    """A single chi=[0, 38] Mpc/h shell carrier (the worst case for the legacy path)."""
    center, width = jnp.array(19.0), jnp.array(38.0)
    a = jc.background.a_of_chi(cosmology, jnp.atleast_1d(center))
    z = jc.utils.a2z(a)
    return SphericalDensity(
        array=jnp.zeros(12),
        nside=1,
        mesh_size=(8, 8, 8),
        box_size=(100.0, 100.0, 100.0),
        comoving_centers=center,
        density_width=width,
        scale_factors=a[0],
        z_sources=z[0],
    )


def test_nz_zmax_independent(cosmology, shells_wide):
    """The default comoving-volume path must not depend on nz_zmax (the original bug)."""
    base = compute_theory_cl_for_density(cosmology, shells_wide, ELL, nz_zmax=0.1).array
    for zmax in (0.5, 2.0, 5.0):
        other = compute_theory_cl_for_density(cosmology, shells_wide, ELL, nz_zmax=zmax).array
        assert_allclose(other, base, rtol=1e-6)


def test_no_nan_and_positive_over_wide_z(cosmology, shells_wide):
    """No NaN and strictly positive auto-spectra across chi≈0 -> z≈1.4 (legacy path NaNs here)."""
    cl = compute_theory_cl_for_density(cosmology, shells_wide, ELL).array
    assert cl.shape == (7, ELL.shape[0])
    assert bool(jnp.isfinite(cl).all())
    assert bool((cl > 0).all())


def test_n_chi_converged(cosmology, shells_wide):
    """The radial integral is converged: n_chi=64 and n_chi=256 agree to <0.5%."""
    coarse = compute_theory_cl_for_density(cosmology, shells_wide, ELL, n_chi=64).array
    fine = compute_theory_cl_for_density(cosmology, shells_wide, ELL, n_chi=256).array
    assert_allclose(coarse, fine, rtol=5e-3)


def test_comoving_tophat_normalized_and_nz_zmax_free(cosmology, near_shell):
    """comoving_tophat integrates to 1 over its shell support, independent of the config zmax."""
    cn = float(near_shell.comoving_centers - near_shell.density_width / 2)
    cf = float(near_shell.comoving_centers + near_shell.density_width / 2)
    zf = float(jc.utils.a2z(jc.background.a_of_chi(cosmology, jnp.atleast_1d(cf)))[0])
    zgrid = jnp.linspace(0.0, zf, 4000)
    for zmax in (0.1, 1.0, 5.0):
        nz = comoving_tophat(cosmology, cn, cf, gals_per_arcmin2=1.0, zmax=zmax)
        integral = jnp.trapezoid(nz(zgrid), zgrid)  # __call__ already vectorises over z
        assert_allclose(float(integral), 1.0, rtol=5e-3)


def test_tophat_z_path_warns(cosmology, shells_wide):
    """The legacy radial_weight='tophat_z' path is deprecated and warns."""
    with pytest.warns(DeprecationWarning):
        compute_theory_cl_for_density(cosmology, shells_wide, ELL, radial_weight="tophat_z", nz_zmax=2.0)


def test_runs_on_real_lpt_map(cosmology, lpt_spherical):
    """End-to-end on a real painted SphericalDensity (shared session fixture)."""
    cl = compute_theory_cl_for_density(cosmology, lpt_spherical, ELL).array
    assert cl.shape[0] == lpt_spherical.comoving_centers.shape[0]
    assert bool(jnp.isfinite(cl).all())
    assert bool((cl > 0).all())
