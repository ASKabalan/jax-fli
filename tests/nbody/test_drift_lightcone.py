"""drift_on_lightcone tests for the lightcone interpolators.

Covers the 4 interpolation kernels x 2 solver momentum conventions (= 8 cases):

* conformal momentum  -> ``DoubleKickDrift`` (KKD)
* pi = dx/dD momentum -> ``DriftKickDrift`` (DKD), shared by ``BullFrog``

The drift continuation factor is delegated to the solver
(``AbstractNBodySolver.drift_factor``) so the *same* interp code is correct for
either convention. The parametrized test checks every kernel runs and that the
drifted map stays consistent with (a small correction of) its non-drift baseline.
A second, ``distributed``-marked test checks single- vs multi-device equivalence.
"""

from __future__ import annotations

import jax
import jax_cosmo as jc
import jax_fli as jfli
import numpy as np
import pytest

MESH_SIZE = (32, 32, 32)
BOX_SIZE = (256.0, 256.0, 256.0)
FLATSKY_NPIX = (32, 32)
FIELD_SIZE = (10.0, 10.0)
PAINT_NSIDE = 16
T0 = 0.1
T1 = 1.0
N_STEPS = 10
NB_SHELLS = 4
SEED = 42

# (label, solver builder name) — the two momentum conventions.
CONVENTIONS = {"KKD": "conformal", "DKD": "pi"}


def _make_ic(cosmo):
    return jfli.gaussian_initial_conditions(
        jax.random.PRNGKey(SEED),
        MESH_SIZE,
        BOX_SIZE,
        cosmo=cosmo,
        flatsky_npix=FLATSKY_NPIX,
        field_size=FIELD_SIZE,
        nside=PAINT_NSIDE,
    )


def _make_solver(name, interp):
    kw = dict(interp_kernel=interp, t0=T0, t1=T1, n_steps=N_STEPS)
    if name == "KKD":
        return jfli.DoubleKickDrift(**kw)
    if name == "DKD":
        return jfli.DriftKickDrift(time_stepping="a", **kw)
    if name == "BullFrog":
        return jfli.BullFrog(time_stepping="a", **kw)
    raise ValueError(name)


def _spherical():
    return jfli.PaintingOptions(target="spherical", scheme="rbf_neighbor", paint_nside=PAINT_NSIDE)


def _flat():
    return jfli.PaintingOptions(target="flat")


def _build_interp(kind, *, drift):
    """Build interp kernel `kind` with its natural painting target."""
    if kind == "NoInterp":
        return jfli.NoInterp(painting=_spherical())  # never drifts
    if kind == "DriftInterp":
        return jfli.DriftInterp(painting=_spherical())  # always drifts
    if kind == "OnionTiler":
        return jfli.OnionTiler(painting=_spherical(), drift_on_lightcone=drift)
    if kind == "TelephotoInterp":
        return jfli.TelephotoInterp(painting=_flat(), drift_on_lightcone=drift)
    raise ValueError(kind)


def _run(cosmo, ic, kind, conv, *, drift):
    interp = _build_interp(kind, drift=drift)
    solver = _make_solver(conv, interp)
    dx, p = jfli.lpt(cosmo, ic, ts=T0, order=1)
    return jfli.nbody(cosmo, dx, p, nb_shells=NB_SHELLS, solver=solver, shell_spacing="comoving", min_width=1.0)


def _corr(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def _rel_diff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).mean() / (np.abs(b).mean() + 1e-30))


# ---------------------------------------------------------------------------
# Direct check that the delegated drift factor matches each solver's own step
# ---------------------------------------------------------------------------
def test_drift_factor_matches_solver_convention(cosmology):
    """The solver drift_factor must equal the in-step drift it would apply."""
    from jaxpm.growth import E, Gf, gp
    from jaxpm.growth import growth_factor as Gp

    a0, a1 = 0.4, 0.7
    interp = jfli.NoInterp(painting=_spherical())

    # Conformal (KKD): (Gp(a1)-Gp(a0)) / Gf(ac) == 1/(ac^3 E) * dGp / gp(ac)
    kkd = _make_solver("KKD", interp)
    ac = (a0 * a1) ** 0.5
    expected_kkd = 1.0 / (ac**3 * E(cosmology, ac)) * (Gp(cosmology, a1) - Gp(cosmology, a0)) / gp(cosmology, ac)
    np.testing.assert_allclose(float(kkd.drift_factor(a0, a1, cosmology)), float(expected_kkd), rtol=1e-6)
    # ... and the same as dGp / Gf(ac)
    np.testing.assert_allclose(
        float(kkd.drift_factor(a0, a1, cosmology)),
        float((Gp(cosmology, a1) - Gp(cosmology, a0)) / Gf(cosmology, ac)),
        rtol=1e-6,
    )

    # pi (DKD / BullFrog): Gp(a1) - Gp(a0)
    expected_pi = float(Gp(cosmology, a1) - Gp(cosmology, a0))
    for name in ("DKD", "BullFrog"):
        sv = _make_solver(name, interp)
        np.testing.assert_allclose(float(sv.drift_factor(a0, a1, cosmology)), expected_pi, rtol=1e-6)


# ---------------------------------------------------------------------------
# 8-case matrix: 4 interp kernels x 2 momentum conventions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("conv", list(CONVENTIONS))
@pytest.mark.parametrize("kind", ["NoInterp", "DriftInterp", "OnionTiler", "TelephotoInterp"])
def test_drift_lightcone_8cases(cosmology, kind, conv):
    ic = _make_ic(cosmology)

    out = _run(cosmology, ic, kind, conv, drift=True)
    arr = np.asarray(out.array)

    assert np.all(np.isfinite(arr)), f"{kind}/{conv}: non-finite values"
    assert out.status == jfli.FieldStatus.LIGHTCONE
    assert arr.shape[0] == NB_SHELLS
    assert float(arr.std()) > 0.0, f"{kind}/{conv}: degenerate (zero-variance) map"

    if kind == "NoInterp":
        return  # no drift to compare against

    # Drift is a small physical correction of the non-drift baseline, not a
    # decorrelating artifact: high correlation + small relative difference.
    base_kind = "NoInterp" if kind == "DriftInterp" else kind
    base = np.asarray(_run(cosmology, ic, base_kind, conv, drift=False).array)

    corr = _corr(arr, base)
    rel = _rel_diff(arr, base)
    assert corr > 0.95, f"{kind}/{conv}: drift map decorrelated from baseline (corr={corr:.4f})"
    assert rel < 0.2, f"{kind}/{conv}: drift changed the map too much (rel diff={rel:.4f})"
    assert rel > 1e-6, f"{kind}/{conv}: drift had no effect (rel diff={rel:.2e})"


def _tiling_ts(cosmo):
    """Scale factors whose comoving distances sit just beyond the box radius.

    With box 256 and a centred observer, ``max_comoving_radius`` is 128 Mpc/h, so
    these shells fall outside the box (within the +/-1-box tiling reach) and force
    the ``_paint_tiled_27`` / ``_paint_telephoto`` branches that the inside-box
    cases never touch. ``nb_shells`` geometry always stays inside the box, so an
    explicit ``ts`` is required here.
    """
    import jax.numpy as jnp

    chi = jnp.array([150.0, 180.0, 210.0])  # > 128 Mpc/h
    return jc.background.a_of_chi(cosmo, chi), jnp.full(3, 25.0)


@pytest.mark.parametrize("conv", list(CONVENTIONS))
@pytest.mark.parametrize("kind", ["OnionTiler", "TelephotoInterp"])
def test_drift_tiling_branch(cosmology, kind, conv):
    """Exercise the outside-box tiling branches (incl. the Telephoto distance fix)."""
    ic = _make_ic(cosmology)
    ts, widths = _tiling_ts(cosmology)
    target = "spherical" if kind == "OnionTiler" else "flat"

    def run(interp):
        solver = _make_solver(conv, interp)
        dx, p = jfli.lpt(cosmology, ic, ts=T0, order=1)
        out = jfli.nbody(
            cosmology,
            dx,
            p,
            solver=solver,
            ts=ts,
            density_widths=widths,
            shell_spacing="comoving",
            min_width=1.0,
        )
        return np.asarray(out.array)

    no = run(jfli.NoInterp(painting=jfli.PaintingOptions(target=target, paint_nside=PAINT_NSIDE)))
    tiled = run(_build_interp(kind, drift=False))
    tiled_drift = run(_build_interp(kind, drift=True))

    assert np.all(np.isfinite(tiled)), f"{kind}/{conv}: non-finite tiled map"
    assert np.all(np.isfinite(tiled_drift)), f"{kind}/{conv}: non-finite drifted tiled map"
    # The tiling branch fills far shells that the single box (NoInterp) cannot:
    # if it were inactive, tiled would equal NoInterp.
    assert _rel_diff(tiled, no) > 0.1, f"{kind}/{conv}: tiling branch not exercised (== NoInterp)"
    # Drift remains a bounded correction of the tiled map, not a decorrelating artifact.
    assert _corr(tiled_drift, tiled) > 0.9, f"{kind}/{conv}: drift decorrelated the tiled map"
    assert _rel_diff(tiled_drift, tiled) > 1e-6, f"{kind}/{conv}: drift had no effect on tiles"


def test_drift_lightcone_differentiable(cosmology):
    """grad through the drifted lightcone w.r.t. cosmology (the inference path)."""
    import jax.numpy as jnp

    ic = _make_ic(cosmology)
    dx, p = jfli.lpt(cosmology, ic, ts=T0, order=1)
    solver = _make_solver("KKD", jfli.DriftInterp(painting=_spherical()))
    # Shells inside the box (radius 128 Mpc/h) so they actually catch particles;
    # DriftInterp does not tile, so far shells would be empty -> zero gradient.
    ts = jc.background.a_of_chi(cosmology, jnp.array([40.0, 70.0, 100.0]))
    widths = jnp.full(3, 20.0)

    def scalar(omega_c):
        cosmo = jc.Cosmology(
            Omega_c=omega_c,
            Omega_b=0.0492,
            h=0.6726,
            n_s=0.9645,
            sigma8=0.81,
            Omega_k=0.0,
            w0=-1.0,
            wa=0.0,
        )
        lc = jfli.nbody(
            cosmo,
            dx,
            p,
            solver=solver,
            ts=ts,
            density_widths=widths,
            shell_spacing="comoving",
            min_width=1.0,
        )
        return jnp.sum(lc.array**2)

    g = float(jax.grad(scalar)(0.2589))
    assert np.isfinite(g), "gradient through drift is non-finite"
    assert abs(g) > 0.0, "gradient through drift is identically zero"


# ---------------------------------------------------------------------------
# Single- vs multi-device equivalence for the drifted lightcone
# ---------------------------------------------------------------------------
@pytest.mark.distributed
def test_drift_single_vs_distributed():
    from jax import lax
    from jax.experimental.multihost_utils import process_allgather
    from jax.sharding import AxisType, NamedSharding
    from jax.sharding import PartitionSpec as P

    from tests.helpers import compare_fields

    cosmo = jc.Planck18()
    n_dev = jax.device_count()
    ic = _make_ic(cosmo)

    # Shard across the first spatial axis with a halo for ghost exchange.
    mesh = jax.make_mesh((n_dev, 1), ("x", "y"), axis_types=(AxisType.Auto, AxisType.Auto))
    sharding = NamedSharding(mesh, P("x", "y"))
    halo = (ic.mesh_size[0] // 4, ic.mesh_size[1] // 4)
    ic_sharded = ic.replace(
        array=lax.with_sharding_constraint(ic.array, sharding),
        field_sharding=sharding,
        halo_size=halo,
    )

    # Use shells *outside* the box so the OnionTiler tiling branch (rotate/shift the
    # sharded particles, sum across devices) is exercised — the path the user flagged
    # as "patterns worse on multi-device", not the inside-box single-paint branch.
    ts, widths = _tiling_ts(cosmo)

    def run_tiled(field):
        interp = jfli.OnionTiler(painting=_spherical(), drift_on_lightcone=True)
        solver = _make_solver("KKD", interp)
        dx, p = jfli.lpt(cosmo, field, ts=T0, order=1)
        return jfli.nbody(
            cosmo,
            dx,
            p,
            solver=solver,
            ts=ts,
            density_widths=widths,
            shell_spacing="comoving",
            min_width=1.0,
        )

    single = run_tiled(ic)
    multi = run_tiled(ic_sharded)

    single_arr = np.asarray(single.array)
    assert float(single_arr.std()) > 0.0, "tiled lightcone is degenerate (empty)"
    multi_arr = process_allgather(multi.array, tiled=True)
    compare_fields(single_arr, multi_arr, "OnionTiler tiled+drift single-vs-distributed")
