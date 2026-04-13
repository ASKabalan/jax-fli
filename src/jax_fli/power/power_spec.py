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

__all__ = ["PowerSpectrum"]


class PowerSpectrum(AbstractField):
    """
    Container for power spectrum data (P(k), C_ell, transfer, coherence, ...).

    Inherits from AbstractField so it carries full field metadata (mesh_size,
    box_size, comoving_centers, density_width, etc.) needed for theory
    predictions via compute_theory_cl_for_density.

    The power spectrum values are stored in `array` (inherited from AbstractPytree).
    A `spectra` property is provided for backwards compatibility.

    When created by angular_cl() or power(), status is set to FieldStatus.SPECTRA
    and unit to SpectralUnit.ANGULAR_CL or SpectralUnit.POWER_SPECTRA.
    """

    # Override to provide sentinel defaults so backward-compat construction
    # PowerSpectrum(wavenumber=k, array=pk, name="pk") still works.
    mesh_size: tuple[int, int, int] = eqx.field(static=True, default=(0, 0, 0))
    box_size: tuple[float, float, float] = eqx.field(static=True, default=(0.0, 0.0, 0.0))

    # Wavenumber grid (ell or k). Default None; validated as non-None in __check_init__.
    wavenumber: jax.Array | None = None

    # name and scale_factors are inherited from AbstractField (static and dynamic respectively).

    @property
    def spectra(self) -> jax.Array:
        """Alias for array (backwards compatibility)."""
        return self.array

    def __check_init__(self):
        """Validation hook called after Equinox auto-initialization."""
        # Call parent — sentinel defaults (0,0,0), (0.,0.,0.), (0.5,0.5,0.5), (0,0)
        # all pass every check in AbstractField.__check_init__.
        super().__check_init__()

        if self.wavenumber is None:
            raise ValueError("wavenumber is required for PowerSpectrum.")
        if self.wavenumber.ndim != 1:
            raise ValueError("wavenumber must be 1D")

        n_k = self.wavenumber.shape[0]

        if self.array.ndim == 1:
            if self.array.shape[0] != n_k:
                raise ValueError(f"Spectra length {self.array.shape[0]} does not match wavenumber {n_k}.")
        elif self.array.ndim == 2:
            if self.array.shape[1] != n_k:
                raise ValueError(
                    f"Spectra shape {self.array.shape} incompatible with wavenumber {n_k}. " "Use shape (n_spec, n_k)."
                )
        else:
            raise ValueError("Spectra must be 1D or 2D.")

    # ---- AbstractField abstract methods -----------------------------------

    def is_batched(self) -> bool:
        """Return True if the spectrum has a leading batch dimension (2D array)."""
        return self.array.ndim == 2

    def is_multi_batched(self) -> bool:
        """PowerSpectrum does not support multi-batching."""
        return False

    @classmethod
    def full_like(cls, field: PowerSpectrum, fill_value: float = 0.0) -> PowerSpectrum:
        """Create a new PowerSpectrum with the same metadata filled with fill_value."""
        return cls(
            array=jnp.full_like(field.array, fill_value),
            wavenumber=field.wavenumber,
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

    # ---- Representation ---------------------------------------------------
    def __repr__(self) -> str:
        return (
            "PowerSpectrum("
            f"wavenumber=Array{tuple(self.wavenumber.shape)}, "  # type: ignore[reportOptionalMemberAccess]
            f"array=Array{tuple(self.array.shape)}, "
            f"name={self.name!r}, "
            f"status={self.status.name}, "
            f"unit={self.unit.name}, "
            f"scale_factors={self.scale_factors})"
        )

    def __getitem__(self, key) -> PowerSpectrum:
        """
        Slice spectra while keeping the wavenumber grid aligned.

        Examples
        --------
        ps[:5]          -> first 5 k and spectra entries
        ps[:2, :5]      -> first 2 spectra and first 5 k (for batched spectra)
        """
        # Normalize key into (spec_sel, k_sel)
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError("__getitem__ expects key like spectra_sel, k_sel")
            k_sel, spec_sel = key
        else:
            k_sel, spec_sel = key, slice(None)

        k_new = jnp.atleast_1d(self.wavenumber[spec_sel])  # type: ignore[reportOptionalSubscript]
        if self.array.ndim == 1:
            array_out = self.array[k_sel]
        else:
            array_out = self.array[k_sel, spec_sel]

        if self.scale_factors is not None and self.array.squeeze().ndim == 1:
            sf_new = jnp.atleast_1d(self.scale_factors[k_sel])
        else:
            sf_new = self.scale_factors

        return PowerSpectrum(
            wavenumber=k_new,
            array=array_out,
            name=self.name,
            scale_factors=sf_new,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            comoving_centers=self.comoving_centers,
            density_width=self.density_width,
            z_sources=self.z_sources,
            nside=self.nside,
            flatsky_npix=self.flatsky_npix,
            field_size=self.field_size,
            status=self.status,
            unit=self.unit,
        )

    # ---- Plotting ---------------------------------------------------------
    def plot(
        self,
        *,
        ax: Axes | None = None,
        logx: bool = True,
        logy: bool = True,
        label: Sequence[str] | None = None,
        color: str | None = None,
        figsize: tuple[float, float] | None = None,
        grid: bool = True,
        legend: bool = False,
        **kwargs: Any,
    ) -> tuple[Figure, Axes, list[Any]]:
        """
        Overlay all spectra in this object on a single axis.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Target axes; if None, a new figure/axes is created.
        logx, logy : bool
            Whether to use log scaling on x/y.
        label : sequence of str, optional
            One label per spectrum; must match the batch dimension length.
        color : str, optional
            Fixed color to use; otherwise matplotlib cycle is used.
        figsize : tuple, optional
            Used only when creating a new axes.
        grid : bool
            Enable grid lines.
        """
        if not jax.core.is_concrete(self.wavenumber):
            raise ValueError("Cannot plot traced arrays. Use PowerSpectrum.plot() outside of a jit context.")

        k_1d = self.wavenumber
        pk_2d = self.array[None, :] if self.array.ndim == 1 else self.array
        n_spec = pk_2d.shape[0]

        # Always generate labels if none are provided, so the metadata exists
        if label is None:
            base_name = self.name or "spectrum"
            label = generate_titles(base_name, self.scale_factors, n_spec)

        if isinstance(label, str):
            label = [label] * n_spec

        if not isinstance(label, (list | tuple)):
            raise TypeError("label must be a list/tuple of strings or None.")
        if len(label) != n_spec:
            raise ValueError(f"label must have length {n_spec}, got {len(label)}.")

        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=figsize or (8, 6))
        else:
            fig = ax.get_figure()
        assert ax is not None

        artists: list[Any] = []
        for i in range(n_spec):
            lab = label[i]
            (line,) = ax.plot(k_1d, pk_2d[i], label=lab, color=color, **kwargs)
            artists.append(line)

        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        if grid:
            ax.grid(True, which="both", ls=":", alpha=0.5)
        if (self.name or "").lower() == "cl":
            ax.set_xlabel(r"$\ell$")
            ax.set_ylabel(r"$C_\ell$")
        else:
            ax.set_xlabel(r"$k$")
            ax.set_ylabel(r"$P(k)$")
        if legend:
            ax.legend()
        return fig, ax, artists

    def show(
        self,
        *,
        ax: Axes | None = None,
        logx: bool = True,
        logy: bool = True,
        label: Sequence[str] | None = None,
        color: str | None = None,
        figsize: tuple[float, float] | None = None,
        grid: bool = True,
        legend: bool = False,
        **kwargs: Any,
    ):
        fig, ax, artists = self.plot(
            ax=ax,
            logx=logx,
            logy=logy,
            label=label,
            color=color,
            figsize=figsize,
            grid=grid,
            legend=legend,
            **kwargs,
        )
        import matplotlib.pyplot as plt

        plt.show()
        return fig, ax, artists

    # ---- Stacking helper ------------------------------------------------
    @classmethod
    def stack(cls, power_spectra: Sequence[PowerSpectrum]) -> PowerSpectrum:
        """Stack multiple PowerSpectrum objects along a new leading axis.

        All wavenumber grids must match (allclose). Arrays are concatenated
        along batch axis (introducing a leading dimension if needed).
        Metadata is copied from the first element.
        """
        # Make sure that all wavenumber grids match and they have the same name
        ref_k = power_spectra[0].wavenumber
        name = power_spectra[0].name
        for spec in power_spectra[1:]:
            if spec.shape != power_spectra[0].shape:
                raise ValueError("All PowerSpectrum instances must share the same shape to be stacked.")
            if spec.name != name:
                raise ValueError("All PowerSpectrum instances must share the same name to be stacked.")

        stacked_array = jnp.stack([spec.array for spec in power_spectra], axis=0)
        ref = power_spectra[0]
        return cls(
            wavenumber=ref_k,
            array=stacked_array,
            name=name,
            mesh_size=ref.mesh_size,
            box_size=ref.box_size,
            comoving_centers=ref.comoving_centers,
            density_width=ref.density_width,
            z_sources=ref.z_sources,
            scale_factors=ref.scale_factors,
            nside=ref.nside,
            flatsky_npix=ref.flatsky_npix,
            field_size=ref.field_size,
            status=ref.status,
            unit=ref.unit,
        )
