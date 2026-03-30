"""Shared fixtures for lensing tests — Born vs glass comparison."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import jax_cosmo.background as bg
import jax_cosmo.constants as const
import jax_fli as jfli
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
MESH_SIZE = (64, 64, 64)
BOX_SIZE = (6000.0, 6000.0, 6000.0)
NSIDE = 64
NB_SHELLS = 10
LPT_TS = 0.001
LPT_ORDER = 2
T0 = 0.001
T1 = 1.0
N_STEPS = 20
Z_SOURCES_SINGLE = 0.5
Z_SOURCES_MULTI = [0.5, 1.0]


# ---------------------------------------------------------------------------
# jax_cosmo adapter for glass Cosmology protocol
# ---------------------------------------------------------------------------
class JaxCosmoForGlass:
    """Wraps a jax_cosmo.Cosmology so glass can use it directly.

    All distances are in Mpc/h (jax_cosmo convention).  Ratios cancel
    the h-factor, so glass's lensing formula gives consistent results.
    """

    def __init__(self, jc_cosmo):
        self._cosmo = jc_cosmo

    @property
    def Omega_m0(self):
        return float(self._cosmo.Omega_m)

    @property
    def hubble_distance(self):
        return float(const.rh)

    def _chi(self, z):
        """Comoving distance for scalar or array redshift."""
        z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
        a = 1.0 / (1.0 + z_arr)
        return np.asarray(jc.background.radial_comoving_distance(self._cosmo, jnp.array(a)))

    def transverse_comoving_distance(self, z, z2=None):
        chi = self._chi(z)
        if z2 is not None:
            chi2 = self._chi(z2)
            result = chi2 - chi
            return result.item() if np.ndim(result) == 0 else result
        return chi.item() if np.ndim(chi) == 0 else chi

    def H_over_H0(self, z):
        z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
        a = 1.0 / (1.0 + z_arr)
        esqr = np.asarray(bg.Esqr(self._cosmo, jnp.array(a)))
        result = np.sqrt(esqr)
        return result.item() if np.ndim(result) == 0 else result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def lensing_initial_field(cosmology):
    """Gaussian initial conditions for lensing tests."""
    return jfli.gaussian_initial_conditions(
        jax.random.PRNGKey(42),
        MESH_SIZE,
        BOX_SIZE,
        cosmo=cosmology,
        nside=NSIDE,
    )


@pytest.fixture(scope="session")
def lensing_lightcone(cosmology, lensing_initial_field):
    """Spherical lightcone from LPT + N-body."""
    dx, p = jfli.lpt(cosmology, lensing_initial_field, ts=LPT_TS, order=LPT_ORDER)
    solver = jfli.DoubleKickDrift(
        interp_kernel=jfli.NoInterp(
            painting=jfli.PaintingOptions(target="spherical", paint_nside=NSIDE),
        ),
        t0=T0,
        t1=T1,
    )
    lightcone = jfli.nbody(
        cosmology,
        dx,
        p,
        solver=solver,
        nb_shells=NB_SHELLS,
        n_steps=N_STEPS,
    )
    return lightcone


@pytest.fixture(scope="session")
def glass_cosmo(cosmology):
    """jax_cosmo-backed cosmology wrapped for glass."""
    return JaxCosmoForGlass(cosmology)


@pytest.fixture(scope="session")
def born_kappa_single(cosmology, lensing_lightcone):
    """Born convergence map at z_source=0.5."""
    return jfli.born(cosmology, lensing_lightcone, nz_shear=Z_SOURCES_SINGLE)


@pytest.fixture(scope="session")
def born_kappa_multi(cosmology, lensing_lightcone):
    """Born convergence maps at z_source=[0.5, 1.0]."""
    return jfli.born(cosmology, lensing_lightcone, nz_shear=Z_SOURCES_MULTI)


@pytest.fixture(scope="session")
def raytrace_born_kappa_single(cosmology, lensing_lightcone):
    """Raytrace(born=True) convergence at z=0.5 (requires dorian)."""
    pytest.importorskip("dorian")
    _, born_field = jfli.raytrace(cosmology, lensing_lightcone, nz_shear=Z_SOURCES_SINGLE, born=True, raytrace=False)
    return born_field


@pytest.fixture(scope="session")
def raytrace_born_kappa_multi(cosmology, lensing_lightcone):
    """Raytrace(born=True) convergence at z=[0.5, 1.0] (requires dorian)."""
    pytest.importorskip("dorian")
    _, born_field = jfli.raytrace(cosmology, lensing_lightcone, nz_shear=Z_SOURCES_MULTI, born=True, raytrace=False)
    return born_field


@pytest.fixture(scope="session")
def glass_kappa_maps(cosmology, lensing_lightcone, glass_cosmo):
    """Glass MultiPlaneConvergence maps at z_source=[0.5, 1.0].

    Converts jax-fli lightcone shells to overdensity, builds glass
    tophat windows from shell edges, and iterates the glass convergence
    accumulator.  Returns a dict mapping source redshift to numpy kappa array.
    """
    import glass

    planes = np.asarray(lensing_lightcone.array)
    comoving_centers = np.asarray(lensing_lightcone.comoving_centers)
    density_width = np.asarray(lensing_lightcone.density_width)

    # Overdensity: delta = plane / mean(plane) - 1
    deltas = []
    for i in range(planes.shape[0]):
        plane = planes[i]
        mean_val = np.mean(plane)
        eps = np.finfo(mean_val.dtype).eps
        safe_mean = mean_val if mean_val != 0 else eps
        deltas.append(plane / safe_mean - 1)

    # Redshift edges from comoving shell boundaries
    r_near = comoving_centers - density_width / 2
    r_far = comoving_centers + density_width / 2

    a_near = jc.background.a_of_chi(cosmology, jnp.asarray(r_near))
    a_far = jc.background.a_of_chi(cosmology, jnp.asarray(r_far))
    z_near = np.asarray(jc.utils.a2z(a_near))
    z_far = np.asarray(jc.utils.a2z(a_far))

    # Sort near-to-far (increasing z) for glass
    sort_idx = np.argsort(z_near)
    z_near = z_near[sort_idx]
    z_far = z_far[sort_idx]
    deltas = [deltas[i] for i in sort_idx]

    z_edges = np.concatenate([[z_near[0]], z_far])
    windows = glass.tophat_windows(z_edges)

    # Per-target iteration: flush pending shell with a dummy plane at
    # the exact source redshift so every shell with zeff < z_target
    # contributes and the source plane is placed correctly.
    npix = deltas[0].shape[0]
    result = {}
    for z_target in Z_SOURCES_MULTI:
        convergence = glass.MultiPlaneConvergence(glass_cosmo)
        for delta_i, win in zip(deltas, windows):
            if win.zeff >= z_target:
                break
            convergence.add_window(delta_i, win)
        # Flush the pending shell and set source at exact z_target
        convergence.add_plane(np.zeros(npix), z_target, wlens=0.0)
        result[z_target] = np.copy(convergence.kappa)

    return result
