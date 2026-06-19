"""PowerSpectrum n_components discriminator: scalar vs multi-component layouts."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jax_fli.power import PowerSpectrum

ELL = jnp.arange(10.0)


def test_scalar_single_not_batched():
    ps = PowerSpectrum(wavenumber=ELL, array=jnp.ones(10))
    assert ps.n_components == 1
    assert ps.is_batched() is False


def test_scalar_batched():
    ps = PowerSpectrum(wavenumber=ELL, array=jnp.ones((4, 10)))
    assert ps.is_batched() is True
    assert ps[0].array.shape == (10,)  # int key drops batch axis -> single


def test_shear_single_components_axis():
    ps = PowerSpectrum(wavenumber=ELL, array=jnp.ones((3, 10)), n_components=3)
    assert ps.is_batched() is False  # (3, n_ell) is a single 3-component spectrum, not a batch
    sliced = ps[0]  # not batched -> slices the ell axis, keeps the 3 components
    assert sliced.array.shape == (3, 1)
    assert sliced.wavenumber is not None
    assert sliced.wavenumber.shape == (1,)


def test_shear_batched():
    ps = PowerSpectrum(wavenumber=ELL, array=jnp.ones((4, 3, 10)), n_components=3)
    assert ps.is_batched() is True
    assert ps[0].array.shape == (3, 10)  # one realisation -> single 3-component spectrum
    assert ps[0].n_components == 3


def test_stack_preserves_components():
    sh = PowerSpectrum(wavenumber=ELL, array=jnp.ones((3, 10)), n_components=3)
    st = PowerSpectrum.stack([sh, sh])
    assert st.array.shape == (2, 3, 10)
    assert st.n_components == 3
    assert st.is_batched() is True


def test_bad_components_axis_rejected():
    with pytest.raises(ValueError):
        PowerSpectrum(wavenumber=ELL, array=jnp.ones((3, 10)), n_components=2)


def test_last_axis_must_match_wavenumber():
    with pytest.raises(ValueError):
        PowerSpectrum(wavenumber=ELL, array=jnp.ones((3, 7)), n_components=3)


def test_negative_index_selects_last_bin():
    """ps[-1] on a non-batched spectrum selects the last ell bin (not slice(-1, 0) -> empty)."""
    ps = PowerSpectrum(wavenumber=ELL, array=jnp.arange(30.0).reshape(3, 10), n_components=3)
    last = ps[-1]
    assert last.array.shape == (3, 1)
    assert last.wavenumber is not None
    assert float(last.wavenumber[0]) == float(ELL[-1])
    assert jnp.allclose(last.array[:, 0], ps.array[:, -1])


def test_negative_index_scalar_last_bin():
    ps = PowerSpectrum(wavenumber=ELL, array=jnp.arange(10.0))
    last = ps[-1]
    assert last.array.shape == (1,)
    assert jnp.allclose(last.array[0], ps.array[-1])


def test_plot_lines_count():
    import matplotlib

    matplotlib.use("Agg")
    ps = PowerSpectrum(wavenumber=ELL, array=jnp.abs(jnp.ones((3, 10))), n_components=3, name="cl")
    _, _, artists = ps.plot()
    assert len(artists) == 3
