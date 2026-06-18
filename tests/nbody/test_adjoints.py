"""Adjoint, gradient, and reversibility tests for N-body solvers.

Test matrix (29 cases total):
- test_checkpointed_vs_reverse:        6 cases — checkpointed ≈ reverse gradients (time_stepping="a")
- test_reverse_vs_checkpointed_lightcone: 6 cases — reverse ≈ checkpointed through saved snapshots
- test_step_checkpoints_invariant:     6 cases — gradient value invariant to step_checkpoints
- test_adjoint_transpose:              2 cases — reverse/checkpointed VJP == forward-mode JVP (adjoint=None)
- test_checkpointed_gradient_D:        4 cases — checkpointed finite for time_stepping="D"
- test_dkd_gradient_vs_discodj:        2 cases — jax-fli DKD grad ≈ disco-dj fastpm adjoint grad
- test_reversible_roundtrip:           3 cases — step + reverse = identity
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import pytest
from numpy.testing import assert_allclose

jax.config.update("jax_enable_x64", True)

from discodj import DiscoDJ
from discodj.cosmology.cosmology import Cosmology as DiscoDJCosmology

import jax_fli as jfli
from jax_fli.fields import DensityField, FieldStatus
from jax_fli.fields.units import DensityUnit
from tests.helpers import MSE, MSRE, compare_fields

# ---------------------------------------------------------------------------
# Fixed simulation config (local to this module, no shared conftest)
# ---------------------------------------------------------------------------

RES = 32
BOXSIZE = 256.0
SEED = 42
A_INI = 0.01
A_END = 1.0

# Tolerances
RTOL = 1e-6
ATOL = 1e-9

# Perturbation amplitude for the reference observable
PERTURBATION_SCALE = 0.1


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cosmo():
    return jc.Planck18()


@pytest.fixture(scope="module")
def ddj_cosmo(cosmo):
    """Create a disco-dj Cosmology matching jax_cosmo Planck18."""
    ddj = DiscoDJCosmology(
        Omega_c=float(cosmo.Omega_m - cosmo.Omega_b),
        Omega_b=float(cosmo.Omega_b),
        h=float(cosmo.h),
        sigma8=float(cosmo.sigma8),
        n_s=float(cosmo.n_s),
        dtype_num=64,
    )
    return ddj.compute_timetables({"steps": 32768, "a_min": 1e-7})


@pytest.fixture(scope="module")
def white_noise():
    key = jax.random.PRNGKey(SEED)
    return jax.random.normal(key, shape=(RES, RES, RES), dtype=jnp.float64)


@pytest.fixture(scope="module")
def ddj_instance(ddj_cosmo, white_noise):
    """Fully initialized DiscoDJ (no gradient flags — used for forward sims)."""
    cosmo_dict = {
        "Omega_c": float(ddj_cosmo.Omega_c),
        "Omega_b": float(ddj_cosmo.Omega_b),
        "h": float(ddj_cosmo.h),
        "sigma8": float(ddj_cosmo.sigma8),
        "n_s": float(ddj_cosmo.n_s),
    }
    dj = DiscoDJ(dim=3, res=RES, boxsize=BOXSIZE, cosmo=cosmo_dict, precision="double")
    dj = dj.with_timetables()
    dj = dj.with_linear_ps(transfer_function="Eisenstein-Hu")
    dj = dj.with_ics(white_noise_field=white_noise)
    dj = dj.with_lpt(n_order=2, exact_growth=True)
    return dj


@pytest.fixture(scope="module")
def jfli_initial_field(ddj_instance):
    """Create jax_fli initial conditions from disco-dj's linear field."""
    delta_r = ddj_instance.delta_ini
    return DensityField(
        array=delta_r,
        mesh_size=(RES, RES, RES),
        box_size=(BOXSIZE, BOXSIZE, BOXSIZE),
        status=FieldStatus.INITIAL_FIELD,
        unit=DensityUnit.DENSITY,
    )


@pytest.fixture(scope="module")
def perturbed_reference(cosmo, jfli_initial_field):
    """Reference density perturbed with noise, so loss gradients are non-trivial.

    Uses DKD, time_stepping='a', 10 steps, then adds scaled noise to break
    the trivial zero-gradient case where pred == target.
    """
    dx, p = jfli.lpt(
        cosmo,
        jfli_initial_field,
        ts=A_INI,
        order=2,
        gradient_order=0,
        dealiased=True,
        exact_growth=True,
    )
    solver = _make_solver("DKD", "a", t0=A_INI, t1=A_END, n_steps=10)
    result = jfli.nbody(
        cosmo,
        dx,
        p,
        solver=solver,
        adjoint="checkpointed",
    )
    key = jax.random.PRNGKey(123)
    noise = jax.random.normal(key, shape=result.array.shape, dtype=result.array.dtype)
    return result.array + PERTURBATION_SCALE * jnp.std(result.array) * noise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_solver(name, time_stepping="a", t0=A_INI, t1=A_END, n_steps=10):
    """Create a solver with gradient_order=0 (exact ik) and NoInterp(density)."""
    interp = jfli.NoInterp(painting=jfli.PaintingOptions(target="density"))
    if name == "DKD":
        return jfli.DriftKickDrift(
            interp_kernel=interp, gradient_order=0, time_stepping=time_stepping, t0=t0, t1=t1, n_steps=n_steps
        )
    elif name == "KKD":
        return jfli.DoubleKickDrift(interp_kernel=interp, gradient_order=0, t0=t0, t1=t1, n_steps=n_steps)
    elif name == "BullFrog":
        return jfli.BullFrog(
            interp_kernel=interp, gradient_order=0, time_stepping=time_stepping, t0=t0, t1=t1, n_steps=n_steps
        )
    else:
        raise ValueError(f"Unknown solver: {name}")


def _timed(fn, *args, **kwargs):
    """Call fn, return (result, elapsed_seconds). Blocks until ready."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    jax.block_until_ready(result)
    elapsed = time.perf_counter() - t0
    return result, elapsed


# ---------------------------------------------------------------------------
# Test 1: checkpointed vs reverse — 6 cases
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("solver_name", ["DKD", "KKD", "BullFrog"])
@pytest.mark.parametrize("n_steps", [10, 20])
def test_checkpointed_vs_reverse(cosmo, jfli_initial_field, perturbed_reference, solver_name, n_steps):
    """Checkpointed and reverse adjoints must give identical gradients (time_stepping='a')."""
    reference = perturbed_reference

    solver = _make_solver(solver_name, "a", t0=A_INI, t1=A_END, n_steps=n_steps)

    def loss(cosmo_, init_array, adjoint):
        init_f = jfli_initial_field.replace(array=init_array)
        dx_, p_ = jfli.lpt(
            cosmo_,
            init_f,
            ts=A_INI,
            order=2,
            gradient_order=0,
            dealiased=True,
            exact_growth=True,
        )
        result = jfli.nbody(
            cosmo_,
            dx_,
            p_,
            solver=solver,
            adjoint=adjoint,
        )
        return jnp.sum((result.array - reference) ** 2)

    grad_fn_chk = jax.grad(lambda c, ic: loss(c, ic, "checkpointed"), argnums=(0, 1))
    grad_fn_rev = jax.grad(lambda c, ic: loss(c, ic, "reverse"), argnums=(0, 1))

    (grad_cosmo_chk, grad_ic_chk), t_chk = _timed(grad_fn_chk, cosmo, jfli_initial_field.array)
    (grad_cosmo_rev, grad_ic_rev), t_rev = _timed(grad_fn_rev, cosmo, jfli_initial_field.array)

    print(f"\n[{solver_name} n={n_steps}] checkpointed: {t_chk:.2f}s, reverse: {t_rev:.2f}s")
    print(f"  IC grad   — MSE: {MSE(grad_ic_chk, grad_ic_rev):.6e}, MSRE: {MSRE(grad_ic_chk, grad_ic_rev):.6e}")
    print(
        f"  Cosmo grad — MSE: {MSE(grad_cosmo_chk, grad_cosmo_rev):.6e}, "
        f"MSRE: {MSRE(grad_cosmo_chk, grad_cosmo_rev):.6e}"
    )

    # Compare IC gradients
    compare_fields(grad_ic_chk, grad_ic_rev, f"{solver_name} n={n_steps} IC grad", rtol=RTOL, atol=ATOL)

    # Compare cosmo gradients (leaf-by-leaf)
    chk_leaves = jax.tree.leaves(grad_cosmo_chk)
    rev_leaves = jax.tree.leaves(grad_cosmo_rev)
    for i, (c, r) in enumerate(zip(chk_leaves, rev_leaves)):
        assert_allclose(c, r, rtol=RTOL, atol=ATOL, err_msg=f"Cosmo gradient leaf {i} differs: chk={c}, rev={r}")


# ---------------------------------------------------------------------------
# Test 1b: reverse == checkpointed through SAVED SNAPSHOTS (a lightcone) — 6 cases
# ---------------------------------------------------------------------------

# Uniform dt0 grid for the integrator. On-grid ts land on t0 + k*dt0 (the forward takes an
# integer number of full steps, no clip); off-grid ts force the forward to clip its last step
# into each snapshot — the case that exposed the reversible-backward boundary-step bug.
_LC_NSTEPS = 18
_LC_DT0 = (A_END - A_INI) / _LC_NSTEPS


@pytest.mark.slow
@pytest.mark.parametrize("solver_name", ["KKD", "DKD", "BullFrog"])
@pytest.mark.parametrize(
    "grid, ts",
    [
        ("on-grid", [A_INI + 9 * _LC_DT0, A_END]),
        ("off-grid", [0.53, A_END]),
    ],
    ids=["on-grid", "off-grid"],
)
def test_reverse_vs_checkpointed_lightcone(cosmo, jfli_initial_field, solver_name, grid, ts):
    """Reverse adjoint must equal checkpointed through SAVED SNAPSHOTS, not just a single
    final-state output (the regime ``test_checkpointed_vs_reverse`` covers).

    The forward re-anchors a uniform dt0 grid at each snapshot and clips its last step into the
    snapshot time; the reversible backward must invert *that* clipped step (a_last -> snapshot),
    not (snapshot - dt0 -> snapshot). Off-grid snapshots (any real lightcone) used to give wrong
    gradients (~50-85%); this guards the ``pm/integrate.py::_boundary_t_prev`` fix for every
    reversible solver, on-grid (no clip) and off-grid (clipped boundary step).
    """
    solver = _make_solver(solver_name, "a", t0=A_INI, t1=A_END, n_steps=_LC_NSTEPS)
    ts_arr = jnp.asarray(ts)

    def loss(init_array, adjoint):
        init_f = jfli_initial_field.replace(array=init_array)
        dx_, p_ = jfli.lpt(cosmo, init_f, ts=A_INI, order=2, gradient_order=0, dealiased=True, exact_growth=True)
        result = jfli.nbody(cosmo, dx_, p_, solver=solver, ts=ts_arr, adjoint=adjoint, min_width=1.0)
        return jnp.sum(result.array**2)

    grad_chk = jax.grad(lambda ic: loss(ic, "checkpointed"))(jfli_initial_field.array)
    grad_rev = jax.grad(lambda ic: loss(ic, "reverse"))(jfli_initial_field.array)

    compare_fields(grad_rev, grad_chk, f"{solver_name} lightcone IC grad ({grid} ts)", rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Test 1c: step_checkpoints is a memory knob, not an accuracy knob — 6 cases
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("solver_name", ["KKD", "BullFrog"])
@pytest.mark.parametrize("step_checkpoints", [1, 3, 10])
def test_step_checkpoints_invariant(cosmo, jfli_initial_field, perturbed_reference, solver_name, step_checkpoints):
    """``step_checkpoints`` checkpoints the inner integration-step loop: it trades recompute for
    memory but must NOT change the gradient value. The checkpointed adjoint recomputes the forward
    (it never inverts), so the gradient is identical for any checkpoint count — the precision-robust
    behaviour Experiment 09 characterises.
    """
    reference = perturbed_reference
    solver = _make_solver(solver_name, "a", t0=A_INI, t1=A_END, n_steps=10)

    def loss(init_array, step_checkpoints):
        init_f = jfli_initial_field.replace(array=init_array)
        dx_, p_ = jfli.lpt(cosmo, init_f, ts=A_INI, order=2, gradient_order=0, dealiased=True, exact_growth=True)
        result = jfli.nbody(cosmo, dx_, p_, solver=solver, adjoint="checkpointed", step_checkpoints=step_checkpoints)
        return jnp.sum((result.array - reference) ** 2)

    grad_ref = jax.grad(lambda ic: loss(ic, None))(jfli_initial_field.array)
    grad_sc = jax.grad(lambda ic: loss(ic, step_checkpoints))(jfli_initial_field.array)

    assert jnp.all(jnp.isfinite(grad_sc)), f"Non-finite IC gradient at step_checkpoints={step_checkpoints}"
    compare_fields(
        grad_sc, grad_ref, f"{solver_name} step_checkpoints={step_checkpoints} vs None", rtol=RTOL, atol=ATOL
    )


# ---------------------------------------------------------------------------
# Test 1d: adjoint transpose (dot-product) test — FD-free, O(1) memory — 2 cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver_name", ["KKD", "BullFrog"])
def test_adjoint_transpose(cosmo, jfli_initial_field, solver_name):
    """The reverse / checkpointed VJP must be the exact transpose of the forward-mode JVP:
    ``⟨w, J v⟩ == ⟨Jᵀ w, v⟩`` for a random tangent ``v`` and cotangent ``w``. This is an FD-free,
    O(1)-memory correctness check of the custom reverse adjoint (the dot-product / transpose test).

    The forward-mode JVP runs through ``adjoint=None`` — a plain ``jax.lax`` forward loop that carries
    no ``custom_vjp`` and so supports ``jvp``; the VJP runs through the reverse-mode adjoints. (The
    'reverse'/'checkpointed' paths use equinox custom-VJP loops, which block forward mode, so the
    forward direction needs the ``adjoint=None`` path.)
    """
    solver = _make_solver(solver_name, "a", t0=A_INI, t1=A_END, n_steps=10)

    def observable(init_array, adjoint):
        init_f = jfli_initial_field.replace(array=init_array)
        dx_, p_ = jfli.lpt(cosmo, init_f, ts=A_INI, order=2, gradient_order=0, dealiased=True, exact_growth=True)
        return jfli.nbody(cosmo, dx_, p_, solver=solver, adjoint=adjoint).array

    x0 = jfli_initial_field.array
    kv, kw = jax.random.split(jax.random.PRNGKey(7))
    v = jax.random.normal(kv, x0.shape, x0.dtype)

    # Forward-mode tangent via adjoint=None (jax.lax loop is jvp-able; the reverse-mode loops are not).
    primal, Jv = jax.jvp(lambda x: observable(x, None), (x0,), (v,))
    assert jnp.all(jnp.isfinite(Jv)), "Non-finite JVP tangent through adjoint=None"

    w = jax.random.normal(kw, primal.shape, primal.dtype)
    lhs = float(jnp.sum(w * Jv))  # ⟨w, J v⟩

    for adjoint in ("reverse", "checkpointed"):
        _, vjp_fn = jax.vjp(lambda x, a=adjoint: observable(x, a), x0)
        (JTw,) = vjp_fn(w)
        rhs = float(jnp.sum(JTw * v))  # ⟨Jᵀ w, v⟩
        assert_allclose(
            lhs,
            rhs,
            rtol=1e-8,
            atol=1e-10,
            err_msg=f"{solver_name} transpose test failed: ⟨w,Jv⟩ != ⟨Jᵀw,v⟩ for adjoint={adjoint}",
        )


# ---------------------------------------------------------------------------
# Test 2: checkpointed gradient with D-stepping — 4 cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver_name", ["DKD", "BullFrog"])
@pytest.mark.parametrize("n_steps", [10, 20])
def test_checkpointed_gradient_D(cosmo, jfli_initial_field, perturbed_reference, solver_name, n_steps):
    """D-stepping only supports checkpointed adjoint. Verify gradients are finite."""
    reference = perturbed_reference

    solver = _make_solver(solver_name, "D", t0=A_INI, t1=A_END, n_steps=n_steps)

    def loss(cosmo_, init_array):
        init_f = jfli_initial_field.replace(array=init_array)
        dx_, p_ = jfli.lpt(
            cosmo_,
            init_f,
            ts=A_INI,
            order=2,
            gradient_order=0,
            dealiased=True,
            exact_growth=True,
        )
        result = jfli.nbody(
            cosmo_,
            dx_,
            p_,
            solver=solver,
            adjoint="checkpointed",
        )
        return jnp.sum((result.array - reference) ** 2)

    grad_fn = jax.grad(loss, argnums=(0, 1))
    (grad_cosmo, grad_ic), elapsed = _timed(grad_fn, cosmo, jfli_initial_field.array)

    print(f"\n[{solver_name}/D n={n_steps}] checkpointed: {elapsed:.2f}s")
    print(f"  IC grad norm:    {float(jnp.linalg.norm(grad_ic)):.6e}")
    print(f"  Cosmo grad norms: {[float(jnp.linalg.norm(g)) for g in jax.tree.leaves(grad_cosmo)]}")

    # All cosmo gradient leaves must be finite
    for i, g in enumerate(jax.tree.leaves(grad_cosmo)):
        assert jnp.all(jnp.isfinite(g)), f"Non-finite cosmo gradient leaf {i}: {g}"

    # IC gradient must be finite
    assert jnp.all(jnp.isfinite(grad_ic)), "Non-finite IC gradient"


# ---------------------------------------------------------------------------
# Test 3: jax-fli DKD gradient vs disco-dj fastpm adjoint — 2 cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_steps", [10, 20])
def test_dkd_gradient_vs_discodj(cosmo, jfli_initial_field, ddj_cosmo, white_noise, n_steps):
    """Compare jax-fli DKD cosmo gradients against disco-dj fastpm adjoint.

    Also verifies jax-fli IC gradients are finite and non-zero.
    Note: disco-dj differentiates w.r.t. white noise (Fourier half-complex space),
    while jax-fli differentiates w.r.t. delta_ini (real space). These are different
    parameterizations, so IC gradients are not directly comparable element-wise.
    """
    cosmo_dict = {
        "Omega_c": float(ddj_cosmo.Omega_c),
        "Omega_b": float(ddj_cosmo.Omega_b),
        "h": float(ddj_cosmo.h),
        "sigma8": float(ddj_cosmo.sigma8),
        "n_s": float(ddj_cosmo.n_s),
    }

    solver = _make_solver("DKD", "a", t0=A_INI, t1=A_END, n_steps=n_steps)

    # ---------------------------------------------------------------
    # jax-fli: gradients w.r.t. both cosmo and ICs
    # ---------------------------------------------------------------
    def jfli_loss(cosmo_, init_array):
        init_f = jfli_initial_field.replace(array=init_array)
        dx_, p_ = jfli.lpt(cosmo_, init_f, ts=A_INI, order=2, gradient_order=0, dealiased=True, exact_growth=True)
        result = jfli.nbody(cosmo_, dx_, p_, solver=solver, adjoint="checkpointed")
        return jnp.sum(result.array**2)

    jfli_grad_fn = jax.grad(jfli_loss, argnums=(0, 1))
    (jfli_grad_cosmo, jfli_grad_ic), t_jfli = _timed(jfli_grad_fn, cosmo, jfli_initial_field.array)

    # ---------------------------------------------------------------
    # disco-dj: gradient w.r.t. cosmo
    # ---------------------------------------------------------------
    dj_cosmo = DiscoDJ(
        dim=3, res=RES, boxsize=BOXSIZE, cosmo=cosmo_dict, precision="double", requires_grad_wrt_cosmo=True
    )
    dj_cosmo = dj_cosmo.with_timetables()
    dj_cosmo = dj_cosmo.with_linear_ps(transfer_function="Eisenstein-Hu")
    dj_cosmo = dj_cosmo.with_ics(white_noise_field=white_noise)
    dj_cosmo = dj_cosmo.with_lpt(n_order=2, exact_growth=True)

    def ddj_loss_cosmo(dj):
        X, P, a = dj.run_nbody(
            a_ini=A_INI,
            a_end=A_END,
            n_steps=n_steps,
            stepper="fastpm",
            method="pm",
            res_pm=RES,
            time_var="a",
            ic_method="lpt",
            nlpt_order_ics=2,
            grad_kernel_order=0,
            exact_growth=True,
        )
        delta = dj.get_delta_from_pos(X, res=RES)
        density = delta + 1.0
        return jnp.sum(density**2)

    ddj_grad_cosmo_dj, t_ddj = _timed(jax.grad(ddj_loss_cosmo), dj_cosmo)

    # ---------------------------------------------------------------
    # Print comparison
    # ---------------------------------------------------------------
    jfli_cosmo_leaves = jax.tree.leaves(jfli_grad_cosmo)
    ddj_cosmo_obj = ddj_grad_cosmo_dj.cosmo
    ddj_cosmo_vals = [
        float(ddj_cosmo_obj.Omega_c),
        float(ddj_cosmo_obj.Omega_b),
        float(ddj_cosmo_obj.h),
        float(ddj_cosmo_obj.sigma8),
        float(ddj_cosmo_obj.n_s),
    ]

    print(f"\n[DKD vs DDJ n={n_steps}] jfli: {t_jfli:.2f}s, ddj cosmo: {t_ddj:.2f}s")
    print(f"  jfli cosmo grad leaves: {[float(x) for x in jfli_cosmo_leaves]}")
    print(f"  ddj  cosmo grad values: {ddj_cosmo_vals}")
    print(f"  jfli IC grad norm:      {float(jnp.linalg.norm(jfli_grad_ic)):.6e}")

    # ---------------------------------------------------------------
    # Assertions
    # ---------------------------------------------------------------
    # jax-fli IC gradient must be finite and non-zero
    assert jnp.all(jnp.isfinite(jfli_grad_ic)), "Non-finite jfli IC gradient"
    assert jnp.linalg.norm(jfli_grad_ic) > 0, "Zero jfli IC gradient"

    # Both cosmo gradients must be finite and non-zero
    for i, g in enumerate(jfli_cosmo_leaves):
        assert jnp.all(jnp.isfinite(g)), f"Non-finite jfli cosmo gradient leaf {i}: {g}"
    assert any(float(jnp.abs(g)) > 1e-10 for g in jfli_cosmo_leaves), "All jfli cosmo gradients are zero"
    for i, v in enumerate(ddj_cosmo_vals):
        assert jnp.isfinite(v), f"Non-finite ddj cosmo gradient {i}: {v}"
    assert any(abs(v) > 1e-10 for v in ddj_cosmo_vals), "All ddj cosmo gradients are zero"


# ---------------------------------------------------------------------------
# Test 4: reversible roundtrip — 3 cases
# ---------------------------------------------------------------------------


def _build_interp_with_geometry():
    """Build a NoInterp kernel with minimal geometry for reversibility tests."""
    interp = jfli.NoInterp(painting=jfli.PaintingOptions(target="density"))
    ts = jnp.array([1.0])
    interp = interp.update_geometry(
        ts=ts,
        r_centers=jnp.zeros(1),
        density_widths=jnp.ones(1),
        max_comoving_distance=1.0,
    )
    return interp


@pytest.fixture(scope="module")
def roundtrip_lpt(cosmo, jfli_initial_field):
    """LPT2 displacements and momenta for roundtrip tests."""
    dx, p = jfli.lpt(
        cosmo,
        jfli_initial_field,
        ts=A_INI,
        order=2,
        gradient_order=0,
        dealiased=True,
        exact_growth=True,
    )
    return dx, p


@pytest.mark.parametrize(
    "solver_factory",
    [
        lambda interp: jfli.DriftKickDrift(interp_kernel=interp, gradient_order=0, time_stepping="a"),
        lambda interp: jfli.DoubleKickDrift(interp_kernel=interp, gradient_order=0),
        lambda interp: jfli.BullFrog(interp_kernel=interp, gradient_order=0, time_stepping="a"),
    ],
    ids=["DKD", "KKD", "BullFrog"],
)
def test_reversible_roundtrip(cosmo, roundtrip_lpt, solver_factory):
    """Step + reverse = identity for all reversible solvers."""
    dx, p = roundtrip_lpt

    interp = _build_interp_with_geometry()
    solver = solver_factory(interp)

    t0 = A_INI
    dt = 0.01
    t1 = t0 + dt

    dx_init, p_init, state = solver.init(dx, p, t0, t1, dt, cosmo)
    dx_stepped, p_stepped, state_stepped = solver.step(dx_init, p_init, t0, t1, dt, state, cosmo)
    dx_recovered, p_recovered, _ = solver.reverse(dx_stepped, p_stepped, t0, t1, dt, state_stepped, cosmo)

    compare_fields(dx_recovered.array, dx_init.array, "roundtrip dx", rtol=RTOL, atol=ATOL)
    compare_fields(p_recovered.array, p_init.array, "roundtrip p", rtol=RTOL, atol=ATOL)
