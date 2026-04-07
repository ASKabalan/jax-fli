"""Compare jax_cosmo/jaxpm growth functions against native fastpm-python, pyccl, and disco-dj."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
import pytest
from numpy.testing import assert_allclose

jax.config.update("jax_enable_x64", True)

# Import native fastpm BEFORE any patching
from discodj.cosmology.cosmology import Cosmology as DiscoDJCosmology
from fastpm.background import MatterDominated as NativeMatterDominated

# jaxpm growth functions
from jaxpm.growth import (
    Dplus_to_a,
    Dplusdada,
    E,
    Gf,
    Gf2,
    dGf2a,
    dGfa,
    gp,
    growth_factor,
    growth_factor_second,
    growth_rate,
    growth_rate_second,
)

COSMO = jc.Planck18()
SCALE_FACTORS = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0])

FPM_PT = NativeMatterDominated(COSMO.Omega_m, a=list(SCALE_FACTORS), a_normalize=1.0)
_GROWTH_ATOL = 1e-9
_GROWTH_RTOL = 1e-5
_GROWTH_DERIV_RTOL = 5e-5


def _scalar(x):
    """Convert a jax array to a Python float."""
    return float(jnp.squeeze(x))


# ---------------------------------------------------------------------------
# Fastpm-python comparison
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a", SCALE_FACTORS)
class TestGrowthFunctions:
    def test_D1(self, a):
        jc_val = _scalar(growth_factor(COSMO, jnp.atleast_1d(a)))
        fpm_val = float(FPM_PT.D1(a))
        assert_allclose(jc_val, fpm_val, rtol=_GROWTH_RTOL, atol=_GROWTH_ATOL)

    def test_D2(self, a):
        jc_val = _scalar(growth_factor_second(COSMO, jnp.atleast_1d(a)))
        fpm_val = float(FPM_PT.D2(a))
        assert_allclose(jc_val, fpm_val, rtol=_GROWTH_DERIV_RTOL, atol=_GROWTH_ATOL)

    def test_f1(self, a):
        jc_val = _scalar(growth_rate(COSMO, jnp.atleast_1d(a)))
        fpm_val = float(FPM_PT.f1(a))
        assert_allclose(jc_val, fpm_val, rtol=_GROWTH_RTOL, atol=_GROWTH_ATOL)

    def test_f2(self, a):
        jc_val = _scalar(growth_rate_second(COSMO, jnp.atleast_1d(a)))
        fpm_val = float(FPM_PT.f2(a))
        assert_allclose(jc_val, fpm_val, rtol=_GROWTH_RTOL, atol=_GROWTH_ATOL)

    def test_E(self, a):
        jc_val = _scalar(E(COSMO, jnp.atleast_1d(a)))
        fpm_val = float(FPM_PT.E(a))
        assert_allclose(jc_val, fpm_val, rtol=_GROWTH_RTOL, atol=_GROWTH_ATOL)

    def test_Gf(self, a):
        jc_val = _scalar(Gf(COSMO, jnp.atleast_1d(a)))
        fpm_val = float(FPM_PT.Gf(a))
        assert_allclose(jc_val, fpm_val, rtol=_GROWTH_RTOL, atol=_GROWTH_ATOL)

    def test_gf(self, a):
        jc_val = _scalar(dGfa(COSMO, jnp.atleast_1d(a)))
        fpm_val = float(FPM_PT.gf(a))
        assert_allclose(jc_val, fpm_val, rtol=_GROWTH_DERIV_RTOL, atol=_GROWTH_ATOL)

    def test_gp(self, a):
        jc_val = _scalar(gp(COSMO, jnp.atleast_1d(a)))
        fpm_val = float(FPM_PT.gp(a))
        assert_allclose(jc_val, fpm_val, rtol=_GROWTH_RTOL, atol=_GROWTH_ATOL)

    def test_Gf2(self, a):
        jc_val = _scalar(Gf2(COSMO, jnp.atleast_1d(a)))
        fpm_val = float(FPM_PT.Gf2(a))
        assert_allclose(jc_val, fpm_val, rtol=_GROWTH_DERIV_RTOL, atol=_GROWTH_ATOL)

    def test_dGf2a(self, a):
        jc_val = _scalar(dGf2a(COSMO, jnp.atleast_1d(a)))
        fpm_val = float(FPM_PT.gf2(a))
        assert_allclose(jc_val, fpm_val, rtol=_GROWTH_DERIV_RTOL, atol=_GROWTH_ATOL)


# ---------------------------------------------------------------------------
# Disco-DJ comparison
# ---------------------------------------------------------------------------


class TestDiscoDJComparison:
    @pytest.fixture(autouse=True, scope="class")
    def ddj_cosmo(self):
        # Use Omega_c = Omega_m - Omega_b to absorb neutrino mass into CDM,
        # ensuring disco-dj's Omega_m matches jax_cosmo's.
        ddj = DiscoDJCosmology(
            Omega_c=float(COSMO.Omega_m - COSMO.Omega_b),
            Omega_b=float(COSMO.Omega_b),
            h=float(COSMO.h),
            sigma8=float(COSMO.sigma8),
            n_s=float(COSMO.n_s),
            dtype_num=64,
        )
        TestDiscoDJComparison._ddj_cosmo = ddj.compute_timetables({"steps": 32768, "a_min": 1e-7})
        return TestDiscoDJComparison._ddj_cosmo

    @pytest.mark.parametrize("a", SCALE_FACTORS)
    def test_D1(self, a, ddj_cosmo):
        jc_val = _scalar(growth_factor(COSMO, jnp.atleast_1d(a)))
        ddj_val = float(ddj_cosmo.Dplus(a))
        assert_allclose(jc_val, ddj_val, rtol=_GROWTH_RTOL, atol=_GROWTH_ATOL)

    @pytest.mark.parametrize("a", SCALE_FACTORS)
    def test_growth_rate(self, a, ddj_cosmo):
        # disco-dj interpolates dD/da then computes f = a*(dD/da)/D, while
        # jax_cosmo interpolates f directly — different interpolation strategies
        # yield ~9e-6 relative error at late times.
        jc_val = _scalar(growth_rate(COSMO, jnp.atleast_1d(a)))
        ddj_val = float(ddj_cosmo.growth_rate(a))
        assert_allclose(jc_val, ddj_val, rtol=_GROWTH_RTOL, atol=_GROWTH_ATOL)

    @pytest.mark.parametrize("a", SCALE_FACTORS)
    def test_E(self, a, ddj_cosmo):
        # E(a) is analytic — essentially exact match
        jc_val = _scalar(E(COSMO, jnp.atleast_1d(a)))
        ddj_val = float(ddj_cosmo.E(a))
        assert_allclose(jc_val, ddj_val, rtol=_GROWTH_DERIV_RTOL, atol=_GROWTH_ATOL)

    @pytest.mark.parametrize("a", SCALE_FACTORS)
    def test_Dplusda(self, a, ddj_cosmo):
        jc_val = _scalar(gp(COSMO, jnp.atleast_1d(a)))
        ddj_val = float(ddj_cosmo.Dplusda(a))
        assert_allclose(jc_val, ddj_val, rtol=_GROWTH_DERIV_RTOL, atol=_GROWTH_ATOL)

    @pytest.mark.parametrize("a", SCALE_FACTORS[SCALE_FACTORS < 1.0])
    def test_Dplusdada(self, a, ddj_cosmo):
        # disco-dj's Dplusdada has a bug (passes a instead of ln_a to ODE function),
        # so we compare against a numerical finite-difference of ddj.Dplusda instead.
        # Excluded a=1.0: numerical FD is unreliable at the table boundary.
        jc_val = _scalar(Dplusdada(COSMO, jnp.atleast_1d(a)))
        eps = max(a * 1e-5, 1e-8)
        ddj_val = (float(ddj_cosmo.Dplusda(a + eps)) - float(ddj_cosmo.Dplusda(a - eps))) / (2 * eps)
        assert_allclose(jc_val, ddj_val, rtol=5e-4, atol=_GROWTH_ATOL)

    @pytest.mark.parametrize("a", SCALE_FACTORS)
    def test_Fplus(self, a, ddj_cosmo):
        jc_val = _scalar(Gf(COSMO, jnp.atleast_1d(a)))
        ddj_val = float(ddj_cosmo.Fplus(a))
        assert_allclose(jc_val, ddj_val, rtol=_GROWTH_DERIV_RTOL, atol=_GROWTH_ATOL)

    @pytest.mark.parametrize("a", SCALE_FACTORS[SCALE_FACTORS < 1.0])
    def test_Fplusda(self, a, ddj_cosmo):
        # disco-dj's Fplusda depends on its buggy Dplusdada, so we compare
        # against a numerical finite-difference of ddj.Fplus instead.
        # Excluded a=1.0: numerical FD is unreliable at the table boundary.
        jc_val = _scalar(dGfa(COSMO, jnp.atleast_1d(a)))
        eps = max(a * 1e-5, 1e-8)
        ddj_val = (float(ddj_cosmo.Fplus(a + eps)) - float(ddj_cosmo.Fplus(a - eps))) / (2 * eps)
        assert_allclose(jc_val, ddj_val, rtol=5e-4, atol=_GROWTH_ATOL)

    @pytest.mark.parametrize("a", SCALE_FACTORS)
    def test_Dplus_to_a(self, a, ddj_cosmo):
        D = _scalar(growth_factor(COSMO, jnp.atleast_1d(a)))
        jc_val = _scalar(Dplus_to_a(COSMO, jnp.atleast_1d(D)))
        ddj_val = float(ddj_cosmo.Dplus_to_a(D))
        assert_allclose(jc_val, ddj_val, rtol=_GROWTH_RTOL, atol=_GROWTH_ATOL)
