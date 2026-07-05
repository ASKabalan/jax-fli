"""MUSE (Marginal Unbiased Score Expansion) inference.

MUSE (Millea & Seljak 2021, arXiv:2112.09354) gives a Gaussian posterior over a few *parameters of
interest* ``theta`` while marginalizing a high-dimensional *latent* ``z`` (here the white initial
conditions), far more cheaply than full HMC. For data ``x`` it takes the score at the latent MAP,

    s(theta, x) = grad_theta logLike(theta, z_MAP(theta, x), x),      z_MAP = argmax_z logLike,

which is biased, then debiases it with the mean score over simulations drawn at ``theta``. The MUSE
estimate solves ``s(theta, x_obs) - <s(theta, x_sim)> + grad log prior(theta) = 0`` by damped Newton,
re-simulating each step. Everything runs in *white* space (standard-normal ``theta`` and ``z``), so the
prior is ``N(0, 1)`` (``grad = -theta``, ``hess = -I``).

The algebra (Newton step, ``J``, ``H``, covariance) is lifted from the reference implementation
``marius311/muse_inference`` (``muse_inference.py``). The module is deliberately SLURM-agnostic: it
exposes the primitives :func:`muse_map` (real-data score), :func:`muse_simulate` (one simulation's
score -- the unit a SLURM array task or ``lax.map`` calls) and :func:`muse_infer` (the reducer:
Newton step + covariance), plus an in-process driver :func:`muse_inference` mirroring
:func:`~jax_fli.infer.batched_sampling` for laptop-scale problems.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Key, PyTree

__all__ = [
    "MuseProblem",
    "MuseResult",
    "muse_map",
    "muse_simulate",
    "muse_infer",
    "muse_covariance",
    "muse_inference",
    "muse_problem_from_model",
]


# ---------------------------------------------------------------------------
# Problem specification
# ---------------------------------------------------------------------------


def _standard_grad_logprior(theta: Array) -> Array:
    return -theta


def _standard_hess_logprior(theta: Array) -> Array:
    return -jnp.eye(theta.shape[0])


@dataclass(frozen=True)
class MuseProblem:
    """The three callables a MUSE problem must supply, plus the theta prior.

    ``theta`` is always a flat ``(n_theta,)`` array (white / unconstrained); ``z`` and ``x`` are
    arbitrary PyTrees. ``loglike`` is ``log p(x, z | theta)`` EXCLUDING the theta prior (the latent
    prior ``log p(z)`` stays inside -- it is theta-independent, so it does not bias the theta score).
    The prior defaults to the standard normal used by the white-space reparametrization.

    Parameters
    ----------
    sample_x_z : (rng, theta) -> (x, z)
        Forward-simulate data and latent from ``p(x, z | theta)``.
    loglike : (theta, z, x) -> scalar
        ``log p(x, z | theta)`` (no theta prior).
    init_z : (theta, x) -> z0
        Latent MAP warm-start (e.g. ``zeros_like``).
    n_theta : int
        Dimension of ``theta``.
    grad_logprior_theta, hess_logprior_theta : (theta) -> vector / matrix
        Gradient and Hessian of ``log prior(theta)``; default standard-normal.
    """

    sample_x_z: Callable[[Key, Array], tuple[PyTree, PyTree]]
    loglike: Callable[[Array, PyTree, PyTree], Array]
    init_z: Callable[[Array, PyTree], PyTree]
    n_theta: int
    grad_logprior_theta: Callable[[Array], Array] = _standard_grad_logprior
    hess_logprior_theta: Callable[[Array], Array] = _standard_hess_logprior


@dataclass(frozen=True)
class MuseResult:
    """Result of a MUSE run.

    ``theta`` / ``Sigma`` are in the white (unconstrained) space MUSE operates in; map them to
    physical parameters by drawing ``N(theta, Sigma)`` and pushing samples through the parameter
    bijector (do NOT linearize ``Sigma`` through the Jacobian). ``J`` is the score covariance and
    ``H`` the response Jacobian (``Sigma^-1 = H^T J^-1 H + prior``). ``history`` is the theta path.
    """

    theta: Array
    Sigma: Array
    J: Array
    H: Array
    history: Array
    n_steps: int
    map_gnorm: float


# ---------------------------------------------------------------------------
# Inner latent MAP + score  (the primitives every entry point shares)
# ---------------------------------------------------------------------------


def _z_map(problem: MuseProblem, theta: Array, x: PyTree, z_guess: PyTree, maxiter: int, gtol: float):
    """Latent MAP ``argmax_z loglike(theta, z, x)`` via L-BFGS, warm-started at ``z_guess``.

    Early-exits when ``||grad_z (-loglike)|| < gtol`` (a ``while_loop``, so a warm-started solve costs
    only the few iterations it needs). Returns ``(z_map, gnorm)`` where ``gnorm`` is the final
    gradient norm -- the caller MUST check ``gnorm < gtol``: a loosely-converged MAP silently biases
    the score (the whole method rests on ``grad_z = 0`` at the MAP, via the envelope theorem). L-BFGS
    operates on the ``z`` PyTree directly, so a sharded field keeps its sharding.
    """
    otu = optax.tree_utils
    negll = lambda z: -problem.loglike(theta, z, x)
    opt = optax.lbfgs()
    value_and_grad = optax.value_and_grad_from_state(negll)

    def cond(carry):
        _, state = carry
        it = otu.tree_get(state, "count")
        g = otu.tree_get(state, "grad")
        return (it == 0) | ((it < maxiter) & (otu.tree_norm(g) >= gtol))

    def body(carry):
        z, state = carry
        value, grad = value_and_grad(z, state=state)
        updates, state = opt.update(grad, state, z, value=value, grad=grad, value_fn=negll)
        return optax.apply_updates(z, updates), state

    z_map, _ = jax.lax.while_loop(cond, body, (z_guess, opt.init(z_guess)))
    gnorm = otu.tree_norm(jax.grad(negll)(z_map))
    return z_map, gnorm


def _score(problem: MuseProblem, theta: Array, x: PyTree, z_guess: PyTree, maxiter: int, gtol: float):
    """MAP-score ``s = grad_theta loglike(theta, z_MAP, x)`` (``z_MAP`` pinned via ``stop_gradient``,
    so no gradient flows through the optimizer -- exact by the envelope theorem). Returns
    ``(s, z_map, gnorm)``."""
    z_map, gnorm = _z_map(problem, theta, x, z_guess, maxiter, gtol)
    z_map = jax.lax.stop_gradient(z_map)
    s = jax.grad(lambda th: problem.loglike(th, z_map, x))(theta)
    return s, z_map, gnorm


@partial(jax.jit, static_argnames=("problem", "maxiter", "gtol"))
def muse_map(problem: MuseProblem, x_obs: PyTree, theta: Array, *, z_guess=None, maxiter=256, gtol=1e-4):
    """Score of the REAL data at ``theta`` (one MUSE entry point).

    Returns ``(s_obs, z_map, gnorm)``. ``z_guess`` defaults to ``problem.init_z(theta, x_obs)``.
    Check ``gnorm < gtol`` (the inner-MAP convergence gate).
    """
    z0 = problem.init_z(theta, x_obs) if z_guess is None else z_guess
    return _score(problem, theta, x_obs, z0, maxiter, gtol)


@partial(jax.jit, static_argnames=("problem", "maxiter", "gtol"))
def muse_simulate(problem: MuseProblem, theta: Array, rng: Key, *, z_guess=None, maxiter=256, gtol=1e-4):
    """Score of ONE simulation drawn at ``theta`` (the SLURM-array / ``lax.map`` unit).

    Draws ``(x, z) ~ p(.|theta)``, finds the latent MAP and returns ``(s_sim, z_map, gnorm)``.
    ``z_guess`` (default ``problem.init_z``) warm-starts the MAP -- pass the previous Newton
    iteration's MAP for the field-scale speedup.
    """
    x, _z_true = problem.sample_x_z(rng, theta)
    z0 = problem.init_z(theta, x) if z_guess is None else z_guess
    return _score(problem, theta, x, z0, maxiter, gtol)


# ---------------------------------------------------------------------------
# Reducers: Newton step, J, H, covariance  (formulas from muse_inference.py)
# ---------------------------------------------------------------------------


def _newton_update(problem, s_obs, s_sims, theta, alpha):
    """One damped-Newton MUSE step. ``s_sims`` is ``(M, n_theta)``.

    ``s_MUSE = s_obs - mean(s_sims)``; ``s_post = s_MUSE + grad_logprior``; the step Hessian is the
    diagonal score-covariance estimate ``Hinv_like = diag(-1/var(s_sims))`` (Fisher ~ score
    covariance), combined with the prior curvature (muse_inference.py:313-336).
    """
    s_muse = s_obs - s_sims.mean(0)
    s_post = s_muse + problem.grad_logprior_theta(theta)
    hinv_like = jnp.diag(-1.0 / s_sims.var(0))
    hinv_post = jnp.linalg.pinv(jnp.linalg.pinv(hinv_like) + problem.hess_logprior_theta(theta))
    step = alpha * (hinv_post @ s_post)
    return theta - step, step


def _response_H(problem, theta, keys, eps, maxiter, gtol, z_guess=None):
    """Response Jacobian ``H`` by central finite differences, averaged over sims (muse:428-509).

    For each sim key, perturb ONLY the GENERATING theta (``sample_x_z(key, theta +/- eps)``) with
    COMMON random numbers (same key at + and -), while the score is EVALUATED at the fiducial
    ``theta`` (warm start held fixed). ``H_ij = d<s_i>/d theta_gen_j``. ``eps`` is a per-coordinate
    step ``(n_theta,)``. ``z_guess`` (a converged MAP) warm-starts the inner MAPs.
    """
    n = problem.n_theta
    warm = problem.init_z(theta, None) if z_guess is None else z_guess

    def s_at_gen(theta_gen, key):
        x, _ = problem.sample_x_z(key, theta_gen)
        s, _, _ = _score(problem, theta, x, warm, maxiter, gtol)
        return s

    def jac_one(key):
        cols = []
        for j in range(n):
            e = jnp.zeros(n).at[j].set(eps[j])
            cols.append((s_at_gen(theta + e, key) - s_at_gen(theta - e, key)) / (2.0 * eps[j]))
        return jnp.stack(cols, axis=1)  # (n_theta, n_theta), column j = d s / d theta_j

    return jax.vmap(jac_one)(keys).mean(0)


def muse_infer(problem, s_obs, s_sims, theta, *, alpha=0.7, J_sims=None, H=None):
    """Reduce collected scores.

    Always returns the Newton-updated ``theta`` (the debiased-mean step). If both ``J_sims``
    (``(M, n_theta)`` scores at the current ``theta``) and ``H`` (the response Jacobian) are given,
    also returns the covariance ``Sigma = (H^T J^-1 H + H_prior)^-1`` with
    ``H_prior = -hess_logprior`` (muse_inference.py:39-46).

    Returns a dict with ``theta_next``/``step`` and, when covariance inputs are present, ``Sigma``
    and ``J``.
    """
    theta_next, step = _newton_update(problem, s_obs, s_sims, theta, alpha)
    out = {"theta_next": theta_next, "step": step}
    if J_sims is not None and H is not None:
        J = jnp.atleast_2d(jnp.cov(J_sims.T))
        h_prior = -problem.hess_logprior_theta(theta)
        sigma_inv = H.T @ jnp.linalg.pinv(J) @ H + h_prior
        out["Sigma"] = jnp.linalg.pinv(sigma_inv)
        out["J"] = J
    return out


# ---------------------------------------------------------------------------
# In-process driver (mirrors batched_sampling)
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("problem", "maxiter", "gtol"))
def _sim_and_obs(problem, theta, sim_keys, z_sims, z_obs_guess, x_obs, maxiter, gtol):
    """One iteration's expensive work (jitted, ``theta`` traced so it compiles ONCE across the loop):
    a sim score for each key in ``sim_keys`` at ``theta`` via ``lax.map`` (warm-started at ``z_sims``)
    + the real-data score. ``sim_keys`` are held FIXED across Newton iterations (common random
    numbers): the draws stay coupled as ``theta`` moves, so the warm-start is effective and the Newton
    path is low-variance. Returns ``(s_obs, s_sims, z_sims_new, z_obs_new, max_gnorm)``."""

    def body(kz):
        key, zg = kz
        x, _ = problem.sample_x_z(key, theta)
        return _score(problem, theta, x, zg, maxiter, gtol)

    s_sims, z_sims_new, gnorms = jax.lax.map(body, (sim_keys, z_sims))
    s_obs, z_obs_new, gnorm_obs = _score(problem, theta, x_obs, z_obs_guess, maxiter, gtol)
    return s_obs, s_sims, z_sims_new, z_obs_new, jnp.maximum(gnorm_obs, gnorms.max())


@partial(jax.jit, static_argnames=("problem", "n_sims", "maxiter", "gtol"))
def _cov_sims(problem, theta, key, n_sims, maxiter, gtol, z_guess=None):
    """``n_sims`` fresh sim scores at ``theta`` (for ``J``), warm-started from ``z_guess`` (a shared
    converged MAP) if given, else cold from ``init_z``."""
    keys = jax.random.split(key, n_sims)
    if z_guess is None:
        z0 = jax.vmap(lambda k: problem.init_z(theta, problem.sample_x_z(k, theta)[0]))(keys)
    else:
        z0 = jax.tree.map(lambda a: jnp.broadcast_to(a, (n_sims, *a.shape)), z_guess)

    def body(kz):
        k, zg = kz
        x, _ = problem.sample_x_z(k, theta)
        return _score(problem, theta, x, zg, maxiter, gtol)[0]

    return jax.lax.map(body, (keys, z0))


def muse_covariance(
    problem: MuseProblem,
    theta: Array,
    rng_key: Key,
    *,
    n_sims_cov: int = 100,
    n_sims_H: int = 10,
    map_maxiter: int = 256,
    map_gtol: float = 1e-4,
    z_guess=None,
):
    """MUSE covariance at ``theta`` (usually ``theta_MUSE``): ``Sigma = (H^T J^-1 H + H_prior)^-1``.

    ``J`` is the score covariance over ``n_sims_cov`` fresh sims; ``H`` the response Jacobian from
    ``n_sims_H`` central-difference sims (``step = 0.1/std(score)``); ``H_prior = -hess_logprior``.
    Sims run via ``lax.map`` on the current device mesh. ``z_guess`` (a converged MAP) warm-starts the
    ``J`` sims when given. Returns ``(Sigma, J, H)``. Standalone entry point for the final covariance of
    a decomposed (e.g. SLURM-driven) run.
    """
    s_J = _cov_sims(problem, theta, jax.random.fold_in(rng_key, 1), n_sims_cov, map_maxiter, map_gtol, z_guess)
    eps = 0.1 / jnp.std(s_J, axis=0)
    keys_H = jax.random.split(jax.random.fold_in(rng_key, 2), n_sims_H)
    H = _response_H(problem, theta, keys_H, eps, map_maxiter, map_gtol, z_guess)
    J = jnp.atleast_2d(jnp.cov(s_J.T))
    h_prior = -problem.hess_logprior_theta(theta)
    Sigma = jnp.linalg.pinv(H.T @ jnp.linalg.pinv(J) @ H + h_prior)
    return Sigma, J, H


def muse_inference(
    problem: MuseProblem,
    x_obs: PyTree,
    path: str | None,
    rng_key: Key,
    theta0: Array,
    *,
    n_sims: int = 100,
    maxsteps: int = 30,
    alpha: float = 0.7,
    rtol: float = 1e-3,
    map_maxiter: int = 256,
    map_gtol: float = 1e-4,
    n_sims_cov: int | None = None,
    n_sims_H: int | None = None,
    progress: bool = True,
) -> MuseResult:
    """Run the full MUSE Newton loop in-process, simulations via ``lax.map`` (memory-safe, not vmap).

    Each iteration re-simulates ``n_sims`` data sets at the current ``theta`` and warm-starts every
    sim's inner MAP from its previous-iteration MAP (the field-scale speedup). At convergence,
    estimates the covariance from ``n_sims_cov`` fresh sims (``J``) and ``n_sims_H`` central-difference
    sims (``H``). Checkpoints the ``theta`` history to ``path`` for resumption.

    Parameters mirror :func:`~jax_fli.infer.batched_sampling` where sensible. ``x_obs`` is the observed
    data, ``theta0`` the flat white starting point; warm-starting ``theta0`` near the truth cuts
    ``maxsteps`` sharply.
    """
    theta = jnp.asarray(theta0, float)
    n_cov = n_sims if n_sims_cov is None else n_sims_cov
    n_H = max(1, n_sims // 10) if n_sims_H is None else n_sims_H

    # resume theta from a prior run
    start = 0
    history = [np.asarray(theta)]
    state_path = None if path is None else os.path.join(path, "muse_state.npz")
    if path is not None:
        os.makedirs(path, exist_ok=True)
        if os.path.exists(state_path):
            saved = np.load(state_path)
            theta = jnp.asarray(saved["theta"])
            history = list(saved["history"])
            start = int(saved["step"])
            if progress:
                print(f"Resuming MUSE from step {start}, theta={np.asarray(theta)}")

    # Compute the real-data latent MAP once (fully converged), then warm-start EVERY simulation's inner
    # MAP from it: sims at the same theta have similar MAPs, so this turns most cold field MAPs
    # (thousands of L-BFGS iterations) into cheap warm ones -- the dominant field-scale speedup.
    if progress:
        print("Computing the real-data MAP (shared warm start for all sims)...")
    _s_obs0, z_obs_guess, _g0 = muse_map(problem, x_obs, theta, maxiter=map_maxiter, gtol=map_gtol)
    sim_keys = jax.random.split(jax.random.fold_in(rng_key, 1), n_sims)  # fixed across iterations
    z_sims = jax.tree.map(lambda a: jnp.broadcast_to(a, (n_sims, *a.shape)), z_obs_guess)
    last_gnorm = 0.0
    s_obs = None

    for i in range(start, maxsteps):
        s_obs, s_sims, z_sims, z_obs_guess, gnorm = _sim_and_obs(
            problem, theta, sim_keys, z_sims, z_obs_guess, x_obs, map_maxiter, map_gtol
        )
        last_gnorm = float(gnorm)
        if last_gnorm > 10.0 * map_gtol:
            print(f"WARNING: inner MAP not converged at step {i}: max||grad_z||={last_gnorm:.2e} (gtol={map_gtol})")

        theta, step = _newton_update(problem, s_obs, s_sims, theta, alpha)
        history.append(np.asarray(theta))
        step_norm = float(jnp.linalg.norm(step))
        if progress:
            print(f"MUSE step {i + 1}: theta={np.asarray(theta)} |step|={step_norm:.3e} max||grad_z||={last_gnorm:.1e}")
        if state_path is not None:
            np.savez(state_path, theta=np.asarray(theta), history=np.stack(history), step=i + 1)
        if step_norm < rtol:
            break

    n_steps = len(history) - 1

    # ---- covariance at theta_MUSE: J from fresh sims, H from central differences ----
    if progress:
        print(f"Estimating MUSE covariance: J from {n_cov} sims, H from {n_H} central-diff sims...")
    Sigma, J, H = muse_covariance(
        problem,
        theta,
        jax.random.fold_in(rng_key, maxsteps + 1),
        n_sims_cov=n_cov,
        n_sims_H=n_H,
        map_maxiter=map_maxiter,
        map_gtol=map_gtol,
        z_guess=z_obs_guess,  # the converged real-data MAP at theta_MUSE warm-starts the J sims
    )
    result = MuseResult(
        theta=theta,
        Sigma=Sigma,
        J=J,
        H=H,
        history=np.stack(history),
        n_steps=n_steps,
        map_gnorm=last_gnorm,
    )
    if path is not None:
        np.savez(
            os.path.join(path, "muse_result.npz"),
            theta=np.asarray(result.theta),
            Sigma=np.asarray(result.Sigma),
            J=np.asarray(result.J),
            H=np.asarray(result.H),
            history=result.history,
        )
    if progress:
        std = np.sqrt(np.diag(np.asarray(result.Sigma)))
        print(f"MUSE done in {n_steps} steps. theta={np.asarray(theta)} sigma={std}")
    return result


# ---------------------------------------------------------------------------
# NumPyro adapter: build a MuseProblem from a full_field_probmodel
# ---------------------------------------------------------------------------


def muse_problem_from_model(config, theta_names=None):
    """Build a :class:`MuseProblem` from a jax-fli ``full_field_probmodel`` configuration.

    The parameters of interest are the WHITE cosmology bases ``{name}_base`` (standard normal under
    the ``TransformReparam``), the latent is the WHITE ``initial_conditions`` field, and the theta
    prior is the default standard normal. ``theta`` is the flat vector of base values in the order of
    ``theta_names`` (default ``tuple(config.priors)``).

    - ``loglike(theta, z, x)`` is ``log p(x, z | theta)`` via :func:`numpyro.infer.util.log_density`
      of the (data-bound) model at the substituted bases + IC. For ``likelihood_space == 'pixel'`` the
      data ``x`` conditions the ``observable_i`` sites; for ``'harmonic'`` the model is rebuilt with
      ``observed_maps=x`` (the ell-tapered ``numpyro.factor``). The data ``x`` is always a free
      argument (MUSE varies it over simulations), so it is passed to ``muse_inference``/``muse_map``,
      NOT baked into the problem.
    - ``sample_x_z(rng, theta)`` ALWAYS generates via the pixel-noise predictive (a seeded run of the
      pixel-likelihood model at fixed bases), because both likelihood spaces assume the same flat white
      pixel noise. Returns ``(observable_maps (n_bins, ...), white_IC_array)``.

    Import is lazy (numpyro + the probabilistic_models package, which imports back into ``infer``).
    """
    import dataclasses

    import numpyro  # noqa: F401
    from numpyro.handlers import condition, seed, substitute, trace
    from numpyro.infer.util import log_density

    from ..probabilistic_models import full_field_probmodel

    theta_names = tuple(config.priors) if theta_names is None else tuple(theta_names)
    base_names = [f"{name}_base" for name in theta_names]
    n_theta = len(theta_names)
    n_bins = len(config.nz_shear)
    mesh = tuple(config.mesh_size)

    # Generation is ALWAYS the pixel-noise predictive (same white pixel noise both spaces assume).
    gen_model = full_field_probmodel(dataclasses.replace(config, likelihood_space="pixel"))

    def _params(theta, z):
        d = {bn: theta[i] for i, bn in enumerate(base_names)}
        d["initial_conditions"] = z
        return d

    if config.likelihood_space == "harmonic":

        def loglike(theta, z, x):
            model = full_field_probmodel(config, observed_maps=x)
            return log_density(model, (), {}, _params(theta, z))[0]

    else:
        score_model = full_field_probmodel(config)

        def loglike(theta, z, x):
            model = condition(score_model, data={f"observable_{i}": x[i] for i in range(n_bins)})
            return log_density(model, (), {}, _params(theta, z))[0]

    def sample_x_z(rng, theta):
        fixed = {bn: theta[i] for i, bn in enumerate(base_names)}
        tr = trace(seed(substitute(gen_model, data=fixed), rng)).get_trace()
        x = jnp.stack([tr[f"observable_{i}"]["value"] for i in range(n_bins)])
        z = tr["initial_conditions"]["value"]
        return x, z.array if hasattr(z, "array") else z

    # The inner-MAP warm start must NOT be exactly zero: a zero white field gives a uniform density
    # (particles exactly on the mesh), where the mass-assignment gradient is non-differentiable and the
    # score comes back NaN. A small fixed random field moves particles off-grid so grad_z is finite.
    z_init = 1e-2 * jax.random.normal(jax.random.PRNGKey(0), mesh)

    return MuseProblem(
        sample_x_z=sample_x_z,
        loglike=loglike,
        init_z=lambda theta, x: z_init,
        n_theta=n_theta,
    )
