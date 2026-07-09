"""Smoke + resume gates for batched_sampling across all three samplers."""

from __future__ import annotations

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
