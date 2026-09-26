"""Spherical-map operators of the mass-mapping pipeline: ``scale_cut(l_min=...)``, the per-shell
``SphericalDensity.resolution_cut`` and ``DensityField.sky_projection``."""

from __future__ import annotations

import healpy as hp
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
import pytest

import jax_fli as jfli
from jax_fli import SphericalDensity

NSIDE = 16
MESH = (16, 16, 16)
BOX = (512.0, 512.0, 512.0)


def _gaussian_maps(n, seed=0):
    rng = np.random.default_rng(seed)
    lmax = 3 * NSIDE - 1
    cl = 1.0 / (np.arange(lmax + 1) + 5.0) ** 2
    maps = []
    for _ in range(n):
        np.random.seed(int(rng.integers(1 << 30)))
        maps.append(hp.synfast(cl, nside=NSIDE, lmax=lmax, new=True) + 0.3)  # nonzero monopole
    return np.array(maps)


def _cl(m):
    return hp.anafast(np.asarray(m), lmax=2 * NSIDE - 1)


def test_scale_cut_l_min():
    field = SphericalDensity(array=jnp.asarray(_gaussian_maps(1)[0]), nside=NSIDE, mesh_size=MESH, box_size=BOX)
    full = field.scale_cut(20, 4).array
    cut = field.scale_cut(20, 4, l_min=2).array
    c_cut, c_full = _cl(cut), _cl(full)
    # anafast's HEALPix quadrature is not exact, so 'zero' means ~1e-6 of the retained power
    assert c_cut[0] < 1e-6 * c_full[2:16].max() and c_cut[1] < 1e-6 * c_full[2:16].max()
    assert c_full[0] > 1e-2 * c_full[2:16].max()  # the input really had a monopole
    np.testing.assert_allclose(c_cut[2:16], c_full[2:16], rtol=1e-4)


def test_resolution_cut():
    maps = _gaussian_maps(3, seed=1)
    lc = SphericalDensity(
        array=jnp.asarray(maps),
        nside=NSIDE,
        mesh_size=MESH,
        box_size=BOX,
        comoving_centers=jnp.array([60.0, 200.0, 600.0]),
        density_width=jnp.array([120.0, 100.0, 100.0]),
    )
    ell_max = 40  # k_Nyq = pi * 16 / 512 ~ 0.098 h/Mpc: ell_res ~ 9 and ~ 20 for the two inner shells, ~ 59 outside
    out = np.asarray(lc.resolution_cut(ell_max).array)
    np.testing.assert_array_equal(out[2], maps[2])  # resolved shell untouched
    for s in (0, 1):
        c = _cl(out[s])
        r_lo, r_hi = [0.0, 150.0][s], [120.0, 250.0][s]
        ell_res = np.pi * 16 / 512 * 0.75 * (r_hi**4 - r_lo**4) / (r_hi**3 - r_lo**3)
        l_cut = int(max(np.floor(ell_res), 4))
        assert c[l_cut + 1 :].max() < 1e-6 * c[:l_cut].max()
    with pytest.raises(ValueError, match="shell radii"):
        lc.resolution_cut(ell_max, r_centers=np.array([1.0, 2.0]), density_width=np.array([1.0, 1.0]))


@pytest.fixture(scope="module")
def ic_field():
    cosmo = jc.Planck18()
    return jfli.gaussian_initial_conditions(jax.random.PRNGKey(0), MESH, BOX, cosmo=cosmo, nside=8)


def test_sky_projection_constant_and_linear(ic_field):
    cosmo = jc.Planck18()
    nz = jc.redshift.smail_nz(1.0, 2.0, 0.3, zmax=2.0)
    const = ic_field.replace(array=jnp.full(MESH, 2.0))
    proj = np.asarray(const.sky_projection(cosmo, [nz]).array)
    assert proj.shape == (1, hp.nside2npix(8))
    # independent quadrature of 2 * int_0^r_max q(chi) dchi
    r_max = float(ic_field.max_comoving_radius)
    chi = np.linspace(1e-3, r_max, 4000)
    z = np.linspace(1e-3, 2.0, 4000)
    chi_z = np.asarray(jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(jnp.asarray(z))))
    n = np.array(nz(jnp.asarray(z)))
    n /= np.trapezoid(n, z)
    a = np.asarray(jc.background.a_of_chi(cosmo, jnp.asarray(chi)))
    q = (
        1.5
        * cosmo.Omega_m
        / 2997.92458**2
        * chi
        / a
        * np.trapezoid(n * np.clip(1 - chi[:, None] / chi_z, 0, None), z, axis=1)
    )
    np.testing.assert_allclose(proj, 2.0 * np.trapezoid(q, chi), rtol=2e-2)
    inner = np.asarray(const.sky_projection(cosmo, [nz], r_min=64.0).array)
    np.testing.assert_allclose(inner, 2.0 * np.trapezoid(np.where(chi >= 64.0, q, 0.0), chi), rtol=2e-2)
    # linear in the field
    f, g = ic_field.array, jnp.roll(ic_field.array, 3, axis=0)
    lhs = ic_field.replace(array=2 * f + g).sky_projection(cosmo, [nz]).array
    rhs = (
        2 * ic_field.replace(array=f).sky_projection(cosmo, [nz]).array
        + ic_field.replace(array=g).sky_projection(cosmo, [nz]).array
    )
    np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), rtol=1e-8, atol=1e-12)


def test_sky_projection_float32_field(ic_field):
    """A float32 field (e.g. a frame stored in float32) projects like its float64 original."""
    cosmo = jc.Planck18()
    nz = jc.redshift.smail_nz(1.0, 2.0, 0.3, zmax=2.0)
    p64 = np.asarray(ic_field.sky_projection(cosmo, [nz]).array)
    p32 = np.asarray(ic_field.replace(array=ic_field.array.astype(jnp.float32)).sky_projection(cosmo, [nz]).array)
    np.testing.assert_allclose(p32, p64, rtol=1e-5, atol=1e-6 * np.abs(p64).max())
