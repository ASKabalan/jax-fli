"""The per-shell resolution cut on a sharded lightcone matches the single-device forward model.

The painted spherical lightcone comes out sharded over pixels (``P(None, "x")``). The resolution cut's SHTs must run
on a replicated copy with a replicated output: left sharded, XLA all-gathers inside ``alm2map`` right before its FFT
and the CPU FFT thunk rejects the transposed layout (``RET_CHECK ... fft_thunk.cc:168``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
import pytest
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P

import jax_fli as jfli
from jax_fli.data.nz import get_des_y3_nz_shear
from jax_fli.initial import interpolate_initial_conditions

MESH = (64, 64, 64)  # the smallest set-up seen to fail without the replicated output


def _forward_kappa(field_sharding, halo):
    cosmo = jc.Planck18()
    box = tuple(float(x) for x in jfli.utils.compute_box_size_from_redshift(cosmo, 1.0, (0.5, 0.5, 0.5)))
    cfg = jfli.ppl.Configurations(
        mesh_size=MESH,
        box_size=box,
        halo_size=halo,
        field_sharding=field_sharding,
        sim_mode="lpt",
        lpt_order=2,
        paint_order="cic",
        gradient_order=4,
        laplace_fd=True,
        number_of_shells=22,
        shell_spacing="equal_vol",
        min_width=50.0,
        max_width=150.0,
        r_min=300.0,
        resolution_cut=True,
        geometry="spherical",
        scheme="bilinear",
        nside=32,
        paint_nside=16,
        kernel_width_pixels=0.8,
        observer_position=(0.5, 0.5, 0.5),
        lensing_output="convergence",
        normalization="global",
        map2alm_method="jax",
        min_redshift=0.001,
        max_redshift=1.0,
        n_integrate=8,
        fiducial_cosmology=jc.Planck18,
        nz_shear=[get_des_y3_nz_shear(zmax=1.0)[i] for i in (1, 2)],
        priors={},
        sigma_e=0.26,
        ell_max=28,
        ell_taper_width=4,
        ell_min=2,
    )
    forward = jfli.ppl.make_full_field_model(cfg)
    white = jax.random.normal(jax.random.PRNGKey(0), MESH)
    if field_sharding is not None:
        white = jax.device_put(white, field_sharding)

    @jax.jit
    def run(w):
        ic = interpolate_initial_conditions(
            w,
            MESH,
            box,
            cosmo=cosmo,
            observer_position=(0.5, 0.5, 0.5),
            halo_size=halo,
            nside=32,
            field_sharding=field_sharding,
        )
        kappa, lightcone = forward(cosmo, ic)
        return kappa.array, lightcone.array

    return jax.device_get(run(white))


@pytest.mark.distributed
@pytest.mark.slow
def test_resolution_cut_sharded_matches_single_device():
    n = jax.device_count()
    mesh = jax.make_mesh((n, 1), ("x", "y"), axis_types=(AxisType.Auto, AxisType.Auto))
    kappa_1, lc_1 = _forward_kappa(None, (0, 0))
    kappa_n, lc_n = _forward_kappa(NamedSharding(mesh, P("x", "y")), (4, 0))
    assert np.all(np.isfinite(kappa_n))
    np.testing.assert_allclose(lc_n, lc_1, rtol=0, atol=1e-10 * np.abs(lc_1).max())
    np.testing.assert_allclose(kappa_n, kappa_1, rtol=0, atol=1e-10 * np.abs(kappa_1).max())
    assert float(jnp.std(kappa_1)) > 0
