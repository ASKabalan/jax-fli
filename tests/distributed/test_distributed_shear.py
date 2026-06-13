"""Distributed lensing sharding: the convergence/shear follow the lensing convention and the mesh
cases warn / error consistently.

Convention (mesh axes **M = "x"** carries NPIX, **N = "y"** carries the tomographic BINS):

    spherical density  (B,) NPIX          -> P([None,] "x")             NPIX/M
    convergence        (B,) BINS, NPIX    -> P([None,] "y", "x")        BINS/N, NPIX/M
    shear              (B,) BINS, 2, NPIX -> P([None,] "y", None, "x")  BINS/N, NPIX/M

``field_sharding`` always stays the **canonical 3-D mesh layout** (``P("x","y")``) — the transposed
convergence/shear layouts above live on the *array* (``field.array.sharding``), not on
``field_sharding``. The tests assert the two separately.

The four cases (``NBINS`` source bins; ``M_size``/``N_size`` = mesh axis sizes):

    D  N divides NBINS, N>1   convergence/shear distribute over bins   no warning; P("y", ..., "x")
    A  N == 1                 bins not distributed                     born warns
    B  M == 1 (spherical)     spherical density npix not sharded       nbody warns
    C  N>1, NBINS % N != 0    mis-shaped lensing mesh                  ValueError

The warnings live on the *producers* (born for the lensing N axis, nbody for the density M axis);
``get_shear`` is silent and only raises the bad-divisor ``ValueError``.

Run locally (8 CPU devices):
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
        pytest tests/distributed/test_distributed_shear.py -v
"""

from __future__ import annotations

import gc

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import jax_fli as jfli
import numpy as np
import pytest
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P
from jax_fli.fields import SphericalKappaField

NSIDE = 16
NPIX = 12 * NSIDE**2
META = dict(mesh_size=(32, 32, 32), box_size=(256.0, 256.0, 256.0), nside=NSIDE)

# Warning / error message fragments (kept in sync with the implementation; used as regex by pytest).
WARN_BORN_N1 = r"second \(N\) axis has size 1"
WARN_NBODY_M1 = r"first \(M\) axis has size 1"
ERR_BAD_DIVISOR = r"Cannot distribute .* bins over the mesh's bins axis"


# --------------------------------------------------------------------------- #
# Fixtures (no closures / return-helpers — global pytest rule)
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _free_memory():
    """Drop XLA executables and collect garbage after each test.

    The 8-CPU mesh holds per-device array copies and every jit + s2fft SHT adds a compiled
    executable; without this the full file accumulates enough host memory to abort mid-run.
    """
    yield
    jax.clear_caches()
    gc.collect()


@pytest.fixture
def kappa_stack(request):
    """Zero-mean random convergence stack ``(nbin, npix)`` (``nbin`` via indirect param)."""
    nbin = request.param
    a = np.random.default_rng(1234 + nbin).standard_normal((nbin, NPIX))
    return jnp.asarray(a - a.mean(axis=-1, keepdims=True))


@pytest.fixture(scope="module")
def cosmo():
    return jc.Planck18()


@pytest.fixture
def lightcone(request, cosmo):
    """A spherical lightcone painted on the ``pdims`` mesh (indirect). Used by the born tests; the
    pdims here always have ``M_size > 1`` so building it does not itself emit the nbody M=1 warning."""
    pdims = request.param
    mesh = jax.make_mesh(pdims, ("x", "y"), axis_types=(AxisType.Auto, AxisType.Auto))
    ics = jfli.gaussian_initial_conditions(
        jax.random.PRNGKey(42),
        (32, 32, 32),
        (256.0, 256.0, 256.0),
        cosmo=cosmo,
        nside=NSIDE,
        field_sharding=NamedSharding(mesh, P("x", "y")),
        halo_size=(8, 8),
    )
    painting = jfli.PaintingOptions(target="spherical", paint_nside=NSIDE)
    solver = jfli.DoubleKickDrift(interp_kernel=jfli.NoInterp(painting=painting), t0=0.1, t1=1.0, n_steps=6)
    dx, p = jfli.lpt(cosmo, ics, ts=0.1, order=1)
    return jfli.nbody(cosmo, dx, p, solver=solver, nb_shells=4, shell_spacing="growth", min_width=1.0)


# --------------------------------------------------------------------------- #
# Assertion helper (returns nothing — not a data builder)
# --------------------------------------------------------------------------- #
def _assert_sharding(array, mesh, expected_spec):
    expected = NamedSharding(mesh, expected_spec)
    assert array.sharding.is_equivalent_to(expected, ndim=array.ndim), f"expected {expected}, got {array.sharding}"


# --------------------------------------------------------------------------- #
# get_shear mechanics on a hand-sharded convergence (fast — no pipeline)
# --------------------------------------------------------------------------- #
@pytest.mark.distributed
@pytest.mark.parametrize(
    "kappa_stack, pdims",
    [(4, (2, 4)), (4, (4, 2)), (8, (2, 4)), (8, (1, 8))],  # case D: N divides NBINS (1 or 2+ bins/device)
    indirect=["kappa_stack"],
)
def test_D_get_shear_distributes_bins(kappa_stack, pdims, recwarn):
    """Case D: convergence ``P("y","x")`` -> shear ``P("y", None, "x")``, bit-exact, no warning."""
    mesh = jax.make_mesh(pdims, ("x", "y"), axis_types=(AxisType.Auto, AxisType.Auto))
    sharding = NamedSharding(mesh, P("y", "x"))
    field = SphericalKappaField(array=jax.device_put(kappa_stack, sharding), field_sharding=sharding, **META)
    reference = SphericalKappaField(array=kappa_stack, **META).get_shear()

    shear = field.get_shear()
    shear_jit = jax.jit(lambda f: f.get_shear())(field)

    assert len(recwarn) == 0, [str(w.message) for w in recwarn]  # get_shear is silent
    _assert_sharding(shear.array, mesh, P("y", None, "x"))
    _assert_sharding(shear_jit.array, mesh, P("y", None, "x"))
    np.testing.assert_array_equal(process_allgather(shear.array, tiled=True), np.asarray(reference.array))
    np.testing.assert_array_equal(process_allgather(shear_jit.array, tiled=True), np.asarray(reference.array))


@pytest.mark.distributed
@pytest.mark.parametrize("kappa_stack", [4], indirect=True)
def test_A_get_shear_replicates_when_N_is_1(kappa_stack, recwarn):
    """Case A (N=1): convergence ``P(None,"x")`` -> shear ``P(None, None, "x")``, bit-exact, get_shear silent."""
    mesh = jax.make_mesh((8, 1), ("x", "y"), axis_types=(AxisType.Auto, AxisType.Auto))
    sharding = NamedSharding(mesh, P(None, "x"))
    field = SphericalKappaField(array=jax.device_put(kappa_stack, sharding), field_sharding=sharding, **META)
    reference = SphericalKappaField(array=kappa_stack, **META).get_shear()

    shear = field.get_shear()
    assert len(recwarn) == 0, [str(w.message) for w in recwarn]  # the N=1 warning belongs to born, not get_shear
    _assert_sharding(shear.array, mesh, P(None, None, "x"))
    np.testing.assert_array_equal(process_allgather(shear.array, tiled=True), np.asarray(reference.array))


@pytest.mark.distributed
@pytest.mark.parametrize("kappa_stack", [3], indirect=True)
def test_C_get_shear_raises_on_bad_divisor(kappa_stack):
    """Case C: 3 bins on a size-4 N axis -> clean ``ValueError`` (not the XLA IndivisibleError)."""
    mesh = jax.make_mesh((2, 4), ("x", "y"), axis_types=(AxisType.Auto, AxisType.Auto))
    sharding = NamedSharding(mesh, P(None, "x"))  # constructible (bins replicated)
    field = SphericalKappaField(array=jax.device_put(kappa_stack, sharding), field_sharding=sharding, **META)
    with pytest.raises(ValueError, match=ERR_BAD_DIVISOR):
        field.get_shear()


@pytest.mark.distributed
@pytest.mark.parametrize("kappa_stack", [4], indirect=True)
def test_lone_map_and_reduced_shear(kappa_stack):
    """A lone ``(npix,)`` map replicates without crashing; ``reduced_shear`` distributes like ``gamma``."""
    mesh = jax.make_mesh((2, 4), ("x", "y"), axis_types=(AxisType.Auto, AxisType.Auto))

    # lone (npix,) slice -> no bins to distribute, no crash
    stack = SphericalKappaField(
        array=jax.device_put(kappa_stack, NamedSharding(mesh, P(None, "x"))),
        field_sharding=NamedSharding(mesh, P(None, "x")),
        **META,
    )
    one = stack[0]
    ref_one = SphericalKappaField(array=kappa_stack[0], **META).get_shear()
    np.testing.assert_array_equal(process_allgather(one.get_shear().array, tiled=True), np.asarray(ref_one.array))

    # reduced shear on the distributed (case D) path
    field = SphericalKappaField(
        array=jax.device_put(kappa_stack, NamedSharding(mesh, P("y", "x"))),
        field_sharding=NamedSharding(mesh, P("y", "x")),
        **META,
    )
    ref = SphericalKappaField(array=kappa_stack, **META).get_shear(reduced_shear=True)
    reduced = field.get_shear(reduced_shear=True)
    _assert_sharding(reduced.array, mesh, P("y", None, "x"))
    np.testing.assert_array_equal(process_allgather(reduced.array, tiled=True), np.asarray(ref.array))


# --------------------------------------------------------------------------- #
# born shards the convergence (full pipeline)
# --------------------------------------------------------------------------- #
@pytest.mark.distributed
@pytest.mark.parametrize("lightcone", [(2, 4), (4, 2)], indirect=True)
def test_D_born_shards_convergence(lightcone, cosmo, recwarn):
    """Case D: born returns convergence ``P("y","x")`` (no warning) and ``born().get_shear()`` -> ``P("y",None,"x")``."""
    mesh = lightcone.field_sharding.mesh
    nz = jfli.io.get_stage3_nz_shear()[:4]  # 4 bins; N divides 4 for (2,4) and (4,2)

    kappa = jfli.born(cosmo, lightcone, nz_shear=nz)
    assert not [w for w in recwarn if "axis has size" in str(w.message)]  # case D is warning-free
    assert kappa.field_sharding.spec == P("x", "y")  # field_sharding stays canonical (not transposed)
    _assert_sharding(kappa.array, mesh, P("y", "x"))  # the lensing layout lives on the array

    shear = kappa.get_shear()
    _assert_sharding(shear.array, mesh, P("y", None, "x"))
    # bit-exact vs single-device get_shear of the gathered convergence
    gathered = process_allgather(kappa.array, tiled=True)
    ref = SphericalKappaField(array=jnp.asarray(gathered), **META).get_shear()
    np.testing.assert_array_equal(process_allgather(shear.array, tiled=True), np.asarray(ref.array))


@pytest.mark.distributed
@pytest.mark.parametrize("lightcone", [(8, 1)], indirect=True)
def test_A_born_warns_when_N_is_1(lightcone, cosmo):
    """Case A (mesh (8,1), N=1): born warns; the convergence *array* falls back to ``P(None,"x")``
    (bins not distributed) while ``field_sharding`` stays the canonical ``P("x","y")``."""
    mesh = lightcone.field_sharding.mesh
    nz = jfli.io.get_stage3_nz_shear()[:4]
    with pytest.warns(UserWarning, match=WARN_BORN_N1):
        kappa = jfli.born(cosmo, lightcone, nz_shear=nz)
    assert kappa.field_sharding.spec == P("x", "y")  # field_sharding stays canonical (not transposed)
    _assert_sharding(kappa.array, mesh, P(None, "x"))  # N=1: bins replicated, npix on M


@pytest.mark.distributed
@pytest.mark.parametrize("lightcone", [(2, 4)], indirect=True)
def test_C_born_raises_on_bad_divisor(lightcone, cosmo):
    """Case C (mesh (2,4), 3 bins): born raises the bad-divisor ``ValueError``."""
    nz = jfli.io.get_stage3_nz_shear()[:3]  # 3 bins; 3 % 4 != 0
    with pytest.raises(ValueError, match=ERR_BAD_DIVISOR):
        jfli.born(cosmo, lightcone, nz_shear=nz)


# --------------------------------------------------------------------------- #
# nbody warns when the spherical density npix axis (M) is size 1
# --------------------------------------------------------------------------- #
@pytest.mark.distributed
def test_B_nbody_warns_when_M_is_1(cosmo):
    """Case B (mesh (1,8), M=1): nbody warns that the spherical density npix is not sharded."""
    mesh = jax.make_mesh((1, 8), ("x", "y"), axis_types=(AxisType.Auto, AxisType.Auto))
    ics = jfli.gaussian_initial_conditions(
        jax.random.PRNGKey(0),
        (32, 32, 32),
        (256.0, 256.0, 256.0),
        cosmo=cosmo,
        nside=NSIDE,
        field_sharding=NamedSharding(mesh, P("x", "y")),
        halo_size=(8, 8),
    )
    painting = jfli.PaintingOptions(target="spherical", paint_nside=NSIDE)
    solver = jfli.DoubleKickDrift(interp_kernel=jfli.NoInterp(painting=painting), t0=0.1, t1=1.0, n_steps=6)
    dx, p = jfli.lpt(cosmo, ics, ts=0.1, order=1)
    with pytest.warns(UserWarning, match=WARN_NBODY_M1):
        jfli.nbody(cosmo, dx, p, solver=solver, nb_shells=4, shell_spacing="growth", min_width=1.0)
