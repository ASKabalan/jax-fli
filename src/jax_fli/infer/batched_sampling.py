"""Batched MCMC sampling utilities.

NumPyro is used purely as the probabilistic-programming front-end (it builds the
log-density from the model), while **BlackJAX** is the only sampling backend.
Two samplers are supported: ``"NUTS"`` and ``"MCLMC"``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import partial, wraps
from typing import ParamSpec, TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Key, PyTree

from ..io import load_sharded, save_sharded
from .sample_converter import default_save

_Param = ParamSpec("_Param")
_Return = TypeVar("_Return")

__all__ = ["batched_sampling", "requires_samplers"]


def requires_samplers(func: Callable[_Param, _Return]) -> Callable[_Param, _Return]:
    """Decorator that raises ImportError when 'blackjax' is not installed."""
    try:
        import blackjax  # noqa: F401
        import numpyro  # noqa: F401
        import orbax.checkpoint  # noqa: F401

        return func
    except ImportError:
        pass

    @wraps(func)
    def _deferred(*args: _Param.args, **kwargs: _Param.kwargs) -> _Return:
        raise ImportError("Missing optional dependency 'blackjax'. Install with: pip install jax-fli[sampling]")

    return _deferred


@requires_samplers
def batched_sampling(
    model,
    path: str,
    rng_key: Key,
    num_warmup: int = 500,
    num_samples: int = 1000,
    batch_count: int = 5,
    sampler: str = "NUTS",
    thinning: int = 1,
    init_params: PyTree | None = None,
    progress_bar: bool = True,
    save_callback: Callable[[dict, str, int, dict | None], None] = default_save,
    post_process: Callable[[dict], dict] | None = None,
    # ── NUTS tuning ──
    max_num_doublings: int = 10,
    target_accept: float = 0.8,
    # ── MCLMC tuning (init step scale / num_tune / preconditioning shared with MAMS) ──
    mclmc_desired_energy_var: float = 1e-3,
    mclmc_num_tune: int | None = None,
    mclmc_init_step_size_scale: float = 1e-4,
    mclmc_diagonal_preconditioning: bool = False,
    # ── MAMS (adjusted MCLMC) tuning ──
    mams_target_accept: float = 0.9,
    mams_l_proposal_factor: float = float("inf"),
    *model_args,
    **model_kwargs,
):
    """Run BlackJAX MCMC in batches, checkpointing after each batch to disk.

    The ``model`` is a (possibly conditioned) NumPyro model; its log-density is
    built once via :func:`numpyro.infer.util.initialize_model` and sampled with
    BlackJAX. Results are saved under ``path`` as ``samples_*`` and ``sampling_state``.

    Parameters
    ----------
    model : callable
        A NumPyro model. For inference it must already be conditioned on the data
        (e.g. ``numpyro.handlers.condition(model, {...})``).
    sampler : {"NUTS", "MCLMC", "MAMS"}
        Sampling algorithm. ``"MAMS"`` is the Metropolis-adjusted microcanonical sampler
        (BlackJAX ``adjusted_mclmc_dynamic``: quasi-random trajectory lengths, unbiased);
        it reuses ``mclmc_num_tune`` / ``mclmc_init_step_size_scale`` /
        ``mclmc_diagonal_preconditioning`` and adds ``mams_target_accept`` and
        ``mams_l_proposal_factor``.
    thinning : int
        Store one sample every ``thinning`` kernel steps. Mostly useful for MCLMC,
        where one kernel step is a single integration step (vs a full trajectory
        for NUTS), so consecutive unthinned draws are highly correlated.
    post_process : callable, optional
        A ``(sample_dict) -> sample_dict`` transform chained AFTER numpyro's ``postprocess_fn``,
        applied once per sample inside the sampling scan. Use it for per-sample field-sized work
        (e.g. recoloring a white ``initial_conditions`` to the physical field with its sampled
        cosmology) so it never runs batched across the whole draw.
    max_num_doublings, target_accept : int, float
        NUTS knobs. ``max_num_doublings`` is the leapfrog trajectory doubling depth
        (the BlackJAX equivalent of ``max_tree_depth``); ``target_accept`` is the
        window-adaptation target acceptance rate.
    mclmc_desired_energy_var, mclmc_num_tune, mclmc_init_step_size_scale, mclmc_diagonal_preconditioning
        MCLMC knobs. ``mclmc_init_step_size_scale`` sets the initial tuning step size
        to ``sqrt(total_dim) * scale`` — overriding the BlackJAX default of
        ``sqrt(total_dim) * 0.25``, which NaNs and collapses to zero at high dimension.
        ``mclmc_num_tune`` defaults to ``num_warmup``.

    Examples
    --------
    >>> import jax.random as jr
    >>> import numpyro
    >>> import numpyro.distributions as dist
    >>> def toy_model():  # zero-arg, self-conditioned model for brevity
    ...     x = numpyro.sample("x", dist.Normal(0, 1))
    ...     numpyro.sample("obs", dist.Normal(x, 1.0), obs=0.5)
    >>> batched_sampling(
    ...     toy_model,
    ...     path="output/toy",
    ...     rng_key=jr.PRNGKey(0),
    ...     num_warmup=10,
    ...     num_samples=20,
    ...     batch_count=2,
    ...     sampler="NUTS",
    ...     progress_bar=False,
    ... )
    """
    import blackjax
    import numpyro
    import orbax.checkpoint as ocp
    from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
    from blackjax.base import SamplingAlgorithm

    assert sampler in {"NUTS", "MCLMC", "MAMS"}, "Only 'NUTS', 'MCLMC' and 'MAMS' samplers are supported"
    assert thinning >= 1, "thinning must be a positive integer"

    os.makedirs(path, exist_ok=True)
    state_path = f"{path}/sampling_state"
    samples_prefix = f"{path}/samples"
    nb_samples = 0
    init_params = jax.tree.map(jnp.asarray, init_params)

    rng_key, init_key, warmup_key, run_key = jax.random.split(rng_key, 4)

    # ── Build the log-density from the NumPyro model (PPL front-end) ──
    kwargs = {}
    if init_params is not None:
        kwargs["init_strategy"] = partial(numpyro.infer.init_to_value, values=init_params)

    init_params_obj, potential_fn, postprocess_fn, _ = numpyro.infer.util.initialize_model(
        init_key, model, model_args=model_args, model_kwargs=model_kwargs, dynamic_args=True, **kwargs
    )
    logdensity_fn = lambda position: -potential_fn(*model_args, **model_kwargs)(position)
    initial_position = init_params_obj.z
    total_dim = sum(p.size for p in jax.tree.leaves(initial_position))

    state_exists = os.path.exists(state_path)

    if state_exists:
        # ── Resume from checkpoint: build cheap dummy state for Orbax shape inference ──
        print("Found existing sampling state, skipping warmup compilation...")
        if sampler == "NUTS":
            last_state = blackjax.nuts.init(initial_position, logdensity_fn)
            parameters = {
                "step_size": jnp.array(0.0),
                "inverse_mass_matrix": jnp.zeros(total_dim),
            }
        elif sampler == "MAMS":
            last_state = blackjax.mcmc.adjusted_mclmc_dynamic.init(
                position=initial_position, logdensity_fn=logdensity_fn, random_generator_arg=init_key
            )
            parameters = {
                "L": jnp.array(0.0),
                "step_size": jnp.array(0.0),
                "inverse_mass_matrix": jnp.ones(total_dim),
            }
        else:  # MCLMC
            last_state = blackjax.mcmc.mclmc.init(
                position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
            )
            parameters = {
                "L": jnp.array(0.0),
                "step_size": jnp.array(0.0),
                "inverse_mass_matrix": jnp.ones(total_dim),
            }

        abstract_state = jax.tree.map(
            ocp.tree.to_shape_dtype_struct,
            {"nb_samples": jnp.array(0), "last_state": last_state, "parameters": parameters},
        )
        try:
            saved_state = load_sharded(state_path, abstract_pytree=abstract_state)
        except Exception as e:
            raise RuntimeError(
                f"Saved sampling state at '{state_path}' was created by a different model "
                "configuration (e.g., different priors, kappa bins, or mesh size). "
                "Delete the directory or use a different output path."
            ) from e
        nb_samples, last_state, parameters = (
            saved_state["nb_samples"],
            saved_state["last_state"],
            saved_state["parameters"],
        )
    else:
        # ── Fresh run: full warmup / adaptation ──
        if sampler == "NUTS":
            print(f"Tuning NUTS parameters (step size and inverse mass matrix) with {num_warmup} warmup steps...")
            print(f"  ==> max_num_doublings={max_num_doublings}, target_accept={target_accept}")
            adapt = blackjax.window_adaptation(
                blackjax.nuts, logdensity_fn, progress_bar=progress_bar, target_acceptance_rate=target_accept
            )
            (last_state, tuned), _ = adapt.run(warmup_key, initial_position, num_warmup)
            parameters = {
                "step_size": tuned["step_size"],
                "inverse_mass_matrix": tuned["inverse_mass_matrix"],
            }
        elif sampler == "MAMS":
            from blackjax.mcmc.adjusted_mclmc_dynamic import rescale

            initial_state = blackjax.mcmc.adjusted_mclmc_dynamic.init(
                position=initial_position, logdensity_fn=logdensity_fn, random_generator_arg=init_key
            )

            def mams_kernel(rng_key, state, avg_num_integration_steps, step_size, inverse_mass_matrix):
                return blackjax.mcmc.adjusted_mclmc_dynamic.build_kernel(
                    integration_steps_fn=lambda k: jnp.ceil(jax.random.uniform(k) * rescale(avg_num_integration_steps)),
                    inverse_mass_matrix=inverse_mass_matrix,
                )(
                    rng_key=rng_key,
                    state=state,
                    step_size=step_size,
                    logdensity_fn=logdensity_fn,
                    L_proposal_factor=mams_l_proposal_factor,
                )

            num_tune = mclmc_num_tune if mclmc_num_tune is not None else num_warmup
            print("Tuning MAMS parameters (L and step_size)...")
            print(f"  ==> mams_target_accept={mams_target_accept}, num_tune={num_tune}")
            init_step = jnp.sqrt(total_dim) * mclmc_init_step_size_scale
            # Cap the INITIAL trajectory at ~50 integration steps: L = sqrt(dim) with a small
            # step would make the first tuning trajectories cost L/step ~ 1000s of gradients
            # at field dimensions (the tuner re-estimates L from the data in its later phases).
            init_mams_params = MCLMCAdaptationState(
                L=jnp.minimum(jnp.sqrt(total_dim), 50.0 * init_step),
                step_size=init_step,
                inverse_mass_matrix=jnp.ones(total_dim),
            )
            tuned_state, tuned_params, _ = blackjax.adjusted_mclmc_find_L_and_step_size(
                mclmc_kernel=mams_kernel,
                num_steps=num_tune,
                state=initial_state,
                rng_key=warmup_key,
                target=mams_target_accept,
                diagonal_preconditioning=mclmc_diagonal_preconditioning,
                params=init_mams_params,
            )
            last_state = tuned_state
            parameters = {
                "L": tuned_params.L,
                "step_size": tuned_params.step_size,
                "inverse_mass_matrix": tuned_params.inverse_mass_matrix,
            }
        else:  # MCLMC
            initial_state = blackjax.mcmc.mclmc.init(
                position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
            )
            kernel_builder = lambda imm: blackjax.mcmc.mclmc.build_kernel(
                logdensity_fn=logdensity_fn,
                integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
                inverse_mass_matrix=imm,
            )
            # Override the BlackJAX default initial step size (sqrt(dim) * 0.25), which NaNs on
            # the first leapfrog iteration at high dim and decays to 0 before tuning converges.
            print("Tuning MCLMC parameters (L and step_size)...")
            print(
                f"  ==> mclmc_desired_energy_var={mclmc_desired_energy_var}, mclmc_init_step_size_scale={mclmc_init_step_size_scale}"
            )
            print(f"  ==> step_size = jnp.sqrt(dim) * scale = {jnp.sqrt(total_dim) * mclmc_init_step_size_scale}")
            init_mclmc_params = MCLMCAdaptationState(
                L=jnp.sqrt(total_dim),
                step_size=jnp.sqrt(total_dim) * mclmc_init_step_size_scale,
                inverse_mass_matrix=jnp.ones(total_dim),
            )
            tuned_state, tuned_params, _ = blackjax.mclmc_find_L_and_step_size(
                mclmc_kernel=kernel_builder,
                num_steps=mclmc_num_tune if mclmc_num_tune is not None else num_warmup,
                state=initial_state,
                rng_key=warmup_key,
                diagonal_preconditioning=mclmc_diagonal_preconditioning,
                params=init_mclmc_params,
                desired_energy_var=mclmc_desired_energy_var,
                frac_tune1=0.1,
                frac_tune2=0.1,
                frac_tune3=0.1,
                # progress_bar=progress_bar,
            )
            last_state = tuned_state
            parameters = {
                "L": tuned_params.L,
                "step_size": tuned_params.step_size,
                "inverse_mass_matrix": tuned_params.inverse_mass_matrix,
            }

        inference_state = {"nb_samples": jnp.array(0), "last_state": last_state, "parameters": parameters}
        save_sharded(inference_state, state_path, overwrite=True, dump_structure=False)

    if sampler == "NUTS":
        sampler_fn = blackjax.nuts(logdensity_fn, max_num_doublings=max_num_doublings, **parameters)
    elif sampler == "MAMS":
        from blackjax.mcmc.adjusted_mclmc_dynamic import rescale

        avg_steps = parameters["L"] / parameters["step_size"]
        sampler_fn = blackjax.adjusted_mclmc_dynamic(
            logdensity_fn,
            step_size=parameters["step_size"],
            inverse_mass_matrix=parameters["inverse_mass_matrix"],
            L_proposal_factor=mams_l_proposal_factor,
            integration_steps_fn=lambda k: jnp.ceil(jax.random.uniform(k) * rescale(avg_steps)),
        )
    else:  # MCLMC
        sampler_fn = blackjax.mclmc(logdensity_fn, **parameters)

    if thinning > 1:
        base_sampler_fn = sampler_fn

        def thinned_step(rng_key, state):
            keys = jax.random.split(rng_key, thinning)
            return jax.lax.scan(lambda s, k: base_sampler_fn.step(k, s), state, keys)

        sampler_fn = SamplingAlgorithm(base_sampler_fn.init, thinned_step)

    start_batch = nb_samples // num_samples
    if start_batch >= batch_count:
        print(f"All {batch_count} batches already completed. Exiting.")
        return

    def transform(state, info):
        position = (
            postprocess_fn(*model_args, **model_kwargs)(state.position)
            if postprocess_fn is not None
            else state.position
        )
        # Chain a per-sample post-process after numpyro's postprocess (runs once per scan step, so
        # field-sized work like recoloring the white IC happens one sample at a time -- never batched).
        if post_process is not None:
            position = post_process(position)
        return position, info

    for i in range(start_batch, batch_count):
        print(f"Sampling batch {i + 1}/{batch_count} using {sampler} (blackjax)...")
        print(f"At sample batch {i + 1}, total samples so far: {nb_samples}")

        run_key, batch_key = jax.random.split(run_key)

        last_state, (samples, infos) = blackjax.util.run_inference_algorithm(
            rng_key=batch_key,
            initial_state=last_state,
            inference_algorithm=sampler_fn,
            num_steps=num_samples,
            transform=transform,
            progress_bar=progress_bar,
        )
        if sampler == "MCLMC":
            metrics = {
                "mean_num_steps": None,
                "total_divergences": None,
                # MCLMCInfo has no acceptance rate; report the fraction of NaN-free steps.
                "mean_accept_prob": float(jnp.mean(infos.nonans.astype(jnp.float32))),
            }
        else:  # NUTS and MAMS share the HMCInfo-style fields
            metrics = {
                "mean_num_steps": float(jnp.mean(infos.num_integration_steps)),
                "total_divergences": int(jnp.sum(infos.is_divergent)),
                "mean_accept_prob": float(jnp.mean(infos.acceptance_rate)),
            }

        nb_samples += num_samples

        print(f"Saving batch {i + 1} samples and state...")
        save_callback(samples, samples_prefix, i, metrics)
        inference_state = {"nb_samples": jnp.array(nb_samples), "last_state": last_state, "parameters": parameters}
        save_sharded(inference_state, state_path, overwrite=True, dump_structure=False)
        del samples
