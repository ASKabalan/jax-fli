"""Shared result object for binned, one-dimensional map statistics (PDF, peak counts).

``BinnedStatistic`` is the binned-statistic analogue of :class:`PowerSpectrum`: an immutable
``AbstractField`` PyTree carrying the bin centers (``bins``), the statistic values (``array``),
and full field metadata, with ``.plot()/.show()/.stack()/__getitem__`` and batching. ``kind``
(static) drives the default axis labels. :class:`~jax_fli.summary_statistics.pdf.PDF` and
:class:`~jax_fli.summary_statistics.peak_counts.PeakCounts` are thin subclasses.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from .._src.base._core import AbstractField
from .._src.fields._plotting import generate_titles

__all__ = ["BinnedStatistic", "resolve_bin_edges"]

# Default axis labels per ``kind``.
_AXIS_LABELS: dict[str, tuple[str, str]] = {
    "pdf": (r"$\kappa$", r"PDF"),
    "peak_counts": (r"peak height", r"$N_{\rm peaks}$"),
    "binned": (r"bin", r"value"),
}


def resolve_bin_edges(
    values,
    bins: int | jnp.ndarray,
    range: tuple[float, float] | None = None,
    *,
    default_range: tuple[float, float] | None = None,
) -> jnp.ndarray:
    """Turn ``bins`` (an int count or an explicit edges array) into a 1-D edges array.

    When ``bins`` is an int, the span is ``range`` if given, else ``default_range`` if given,
    else the concrete ``[min, max]`` of ``values`` (eager-only — pass ``range``/edges under jit).
    """
    if not isinstance(bins, int):
        return jnp.asarray(bins)
    if range is not None:
        lo, hi = range
    elif default_range is not None:
        lo, hi = default_range
    else:
        lo, hi = float(jnp.min(values)), float(jnp.max(values))
    return jnp.linspace(lo, hi, bins + 1)


class BinnedStatistic(AbstractField):
    """Container for a 1-D binned statistic over a fixed bin grid.

    ``array`` holds the statistic values, last axis length ``len(bins)``; a leading axis (2-D)
    is a batch of statistics. ``bins`` are the bin centers (the x-axis).
    """

    # Sentinel defaults so ``BinnedStatistic(bins=..., array=...)`` works without geometry.
    mesh_size: tuple[int, int, int] = eqx.field(static=True, default=(0, 0, 0))
    box_size: tuple[float, float, float] = eqx.field(static=True, default=(0.0, 0.0, 0.0))

    # Bin centers (x-axis). Validated non-None and 1-D in __check_init__.
    bins: jax.Array | None = None
    # Discriminator driving default axis labels ("pdf", "peak_counts", ...).
    kind: str = eqx.field(static=True, default="binned")

    @property
    def bin_centers(self) -> jax.Array:
        """Alias for ``bins`` (the bin centers)."""
        return self.bins  # type: ignore[return-value]

    def __check_init__(self):
        super().__check_init__()
        if self.bins is None:
            raise ValueError("bins (bin centers) is required for BinnedStatistic.")
        if self.bins.ndim != 1:
            raise ValueError("bins must be 1D")
        n_bins = self.bins.shape[0]
        if self.array.shape[-1] != n_bins:
            raise ValueError(f"array last axis {self.array.shape[-1]} does not match bins {n_bins}.")
        if self.array.ndim not in (1, 2):
            raise ValueError(f"BinnedStatistic array must be 1D (n_bins) or 2D (B, n_bins), got {self.array.shape}.")

    # ---- AbstractField abstract methods -----------------------------------
    def is_batched(self) -> bool:
        return self.array.ndim == 2

    def is_multi_batched(self) -> bool:
        return False

    @classmethod
    def full_like(cls, field: BinnedStatistic, fill_value: float = 0.0) -> BinnedStatistic:
        return cls(
            array=jnp.full_like(field.array, fill_value),
            bins=field.bins,
            kind=field.kind,
            mesh_size=field.mesh_size,
            box_size=field.box_size,
            comoving_centers=field.comoving_centers,
            density_width=field.density_width,
            z_sources=field.z_sources,
            scale_factors=field.scale_factors,
            nside=field.nside,
            flatsky_npix=field.flatsky_npix,
            field_size=field.field_size,
            status=field.status,
            unit=field.unit,
            name=field.name,
        )

    # ---- Indexing ---------------------------------------------------------
    def __getitem__(self, key) -> BinnedStatistic:
        """Batched: index the batch axis (and per-batch metadata), keeping ``bins``. Non-batched:
        index the bin axis."""
        if self.is_batched():
            B = self.array.shape[0]

            def _idx(x):
                if x is not None and getattr(x, "ndim", 0) >= 1 and x.shape[0] == B:
                    return x[key]
                return x

            return self.replace(
                array=self.array[key],
                scale_factors=_idx(self.scale_factors),
                comoving_centers=_idx(self.comoving_centers),
                density_width=_idx(self.density_width),
                z_sources=_idx(self.z_sources),
            )
        if isinstance(key, int):
            k = key if key >= 0 else self.array.shape[-1] + key
            ksel = slice(k, k + 1)
        else:
            ksel = key
        return self.replace(
            bins=jnp.atleast_1d(self.bins[ksel]),  # type: ignore[reportOptionalSubscript]
            array=self.array[..., ksel],
        )

    # ---- Plotting ---------------------------------------------------------
    def plot(
        self,
        *,
        ax: Axes | None = None,
        logx: bool = False,
        logy: bool = False,
        label: Sequence[str] | str | None = None,
        color: str | None = None,
        figsize: tuple[float, float] | None = None,
        grid: bool = True,
        legend: bool = False,
        drawstyle: str = "steps-mid",
        **kwargs: Any,
    ) -> tuple[Figure, Axes, list[Any]]:
        """Overlay every (batched) statistic in this object on a single axis."""
        if isinstance(self.bins, jax.core.Tracer):
            raise ValueError("Cannot plot traced arrays. Use .plot() outside of a jit context.")

        x = self.bins
        y_2d = self.array.reshape(-1, self.array.shape[-1])
        n = y_2d.shape[0]

        if label is None:
            label = generate_titles(self.name or self.kind, self.scale_factors, n)
        if isinstance(label, str):
            label = [label] * n
        if len(label) != n:
            raise ValueError(f"label must have length {n}, got {len(label)}.")

        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=figsize or (8, 6))
        else:
            fig = ax.get_figure()
        assert ax is not None

        artists: list[Any] = []
        for i in range(n):
            (line,) = ax.plot(x, y_2d[i], label=label[i], color=color, drawstyle=drawstyle, **kwargs)
            artists.append(line)

        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        if grid:
            ax.grid(True, which="both", ls=":", alpha=0.5)
        xlabel, ylabel = _AXIS_LABELS.get(self.kind, _AXIS_LABELS["binned"])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if legend:
            ax.legend()
        return fig, ax, artists

    def show(self, **kwargs: Any):
        fig, ax, artists = self.plot(**kwargs)
        import matplotlib.pyplot as plt

        plt.show()
        return fig, ax, artists

    # ---- Stacking ---------------------------------------------------------
    @classmethod
    def stack(cls, stats: Sequence[BinnedStatistic]) -> BinnedStatistic:
        """Stack multiple statistics along a new leading batch axis (bin grids must match)."""
        ref = stats[0]
        for s in stats[1:]:
            if s.shape != ref.shape:
                raise ValueError("All statistics must share the same shape to be stacked.")
            if s.kind != ref.kind:
                raise ValueError("All statistics must share the same kind to be stacked.")
        stacked = jnp.stack([s.array for s in stats], axis=0)
        # Stack the per-shell lightcone metadata alongside the array so the batched result keeps
        # one metadata row per stacked statistic (skip fields that are None on the reference).
        meta_updates = {
            f: jnp.stack([getattr(s, f) for s in stats], axis=0)
            for f in ("z_sources", "comoving_centers", "scale_factors", "density_width")
            if getattr(ref, f) is not None
        }
        return ref.replace(array=stacked, **meta_updates)
