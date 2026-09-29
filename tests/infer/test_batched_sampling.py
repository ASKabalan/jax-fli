"""Smoke + resume gates for batched_sampling across all three samplers."""

from __future__ import annotations

import importlib
import os

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from jax_fli.infer import batched_sampling

OBS = jnp.array([0.5, -0.3, 0.2])


def toy_model():
    x = numpyro.sample("x", dist.Normal(0.0, 1.0).expand([3]))
    numpyro.sample("obs", dist.Normal(x, 1.0), obs=OBS)


@pytest.fixture(params=["NUTS", "MCLMC", "MAMS"])
def sampler_name(request):
    return request.param


def test_run_and_resume(tmp_path, sampler_name):
    path = str(tmp_path / sampler_name)
    kwargs = dict(
        rng_key=jr.PRNGKey(0),
        num_warmup=100,
        num_samples=20,
        sampler=sampler_name,
        thinning=1 if sampler_name == "NUTS" else 3,
        progress_bar=False,
        mclmc_init_step_size=0.1,
    )

    batched_sampling(toy_model, path=path, batch_count=2, **kwargs)

    for batch in range(2):
        f = os.path.join(path, "samples", f"samples_batch_{batch}.npz")
        assert os.path.exists(f)
        x = np.load(f)["x"]
        assert x.shape == (20, 3)
        assert np.isfinite(x).all()
    assert os.path.exists(os.path.join(path, "metrics.md"))

    # Resume: same path, one more batch — must skip warmup and run only batch 3.
    batched_sampling(toy_model, path=path, batch_count=3, **kwargs)
    f = os.path.join(path, "samples", "samples_batch_2.npz")
    assert os.path.exists(f)
    assert np.isfinite(np.load(f)["x"]).all()


def test_resume_matches_uninterrupted(tmp_path, sampler_name):
    """A resumed run draws the same batches as an uninterrupted one, not the keys of the first batches again."""
    kwargs = dict(
        rng_key=jr.PRNGKey(0),
        num_warmup=100,
        num_samples=20,
        sampler=sampler_name,
        thinning=1 if sampler_name == "NUTS" else 3,
        progress_bar=False,
        mclmc_init_step_size=0.1,
    )
    whole, resumed = str(tmp_path / "whole"), str(tmp_path / "resumed")
    batched_sampling(toy_model, path=whole, batch_count=3, **kwargs)
    batched_sampling(toy_model, path=resumed, batch_count=2, **kwargs)
    batched_sampling(toy_model, path=resumed, batch_count=3, **kwargs)

    def load(path, batch):
        return np.load(os.path.join(path, "samples", f"samples_batch_{batch}.npz"))["x"]

    np.testing.assert_allclose(load(resumed, 2), load(whole, 2))
    assert not np.allclose(np.diff(load(resumed, 2), axis=0), np.diff(load(resumed, 0), axis=0))


def test_posterior_mean_roughly_correct(tmp_path):
    """MAMS on the conjugate toy: posterior mean = OBS/2 within loose Monte-Carlo error."""
    path = str(tmp_path / "mams_mean")
    batched_sampling(
        toy_model,
        path=path,
        rng_key=jr.PRNGKey(1),
        num_warmup=500,
        num_samples=300,
        batch_count=1,
        sampler="MAMS",
        thinning=2,
        progress_bar=False,
        mclmc_init_step_size=0.1,
        mclmc_diagonal_preconditioning=True,
    )
    x = np.load(os.path.join(path, "samples", "samples_batch_0.npz"))["x"]
    # posterior is N(OBS/2, 1/2); with ~300 correlated draws allow ~5 sigma_mean slack
    assert np.allclose(x.mean(axis=0), np.asarray(OBS) / 2, atol=0.35)


@pytest.mark.parametrize("sampler", ["NUTS", "MCLMC"])
def test_chunked_warmup_matches_one_chunk_and_resumes(tmp_path, sampler, monkeypatch):
    """The warmup gives the same tuning and draws whatever its chunk size, and when it is interrupted and resumed."""
    module = importlib.import_module("jax_fli.infer.batched_sampling")

    kwargs = dict(
        rng_key=jr.PRNGKey(0),
        num_warmup=50,
        num_samples=10,
        batch_count=1,
        sampler=sampler,
        thinning=1 if sampler == "NUTS" else 3,
        progress_bar=False,
        mclmc_init_step_size=0.1,
    )
    one_chunk, chunked, resumed = (str(tmp_path / name) for name in ("one_chunk", "chunked", "resumed"))
    batched_sampling(toy_model, path=one_chunk, warmup_chunk=50, **kwargs)
    batched_sampling(toy_model, path=chunked, warmup_chunk=7, **kwargs)

    save_sharded = module.save_sharded
    calls = []

    def interrupted_save(*args, **kw):
        # the third warmup checkpoint never comes: the run stops after two chunks
        calls.append(args[1])
        if len(calls) == 3:
            raise KeyboardInterrupt
        return save_sharded(*args, **kw)

    monkeypatch.setattr(module, "save_sharded", interrupted_save)
    with pytest.raises(KeyboardInterrupt):
        batched_sampling(toy_model, path=resumed, warmup_chunk=7, **kwargs)
    monkeypatch.setattr(module, "save_sharded", save_sharded)
    assert os.path.exists(os.path.join(resumed, "warmup_state"))
    batched_sampling(toy_model, path=resumed, warmup_chunk=7, **kwargs)
    assert not os.path.exists(os.path.join(resumed, "warmup_state"))

    def load(path):
        return np.load(os.path.join(path, "warmup_params.npz")), np.load(
            os.path.join(path, "samples", "samples_batch_0.npz")
        )["x"]

    params, draws = load(one_chunk)
    for path in (chunked, resumed):
        other_params, other_draws = load(path)
        for name in params.files:
            np.testing.assert_array_equal(other_params[name], params[name])
        np.testing.assert_array_equal(other_draws, draws)
    with open(os.path.join(chunked, "warmup.log")) as log:
        assert log.read().splitlines()[-1] == "warmup 50/50"
