from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from warnings import warn

from .._src.base._core import AbstractField
from .._src.base._enums import DensityUnit, FieldStatus, SpectralUnit
from .._src.base._tri_map import tri_map
from .._src.fields._plotting import generate_titles, plot_3d_density, prepare_axes
from ..summary_statistics import PowerSpectrum, coherence, transfer
from ..summary_statistics import power as power_fn
from .lightcone import FlatDensity
from .units import convert_units


class DensityField(AbstractField):
    """
    PyTree container for volumetric simulation arrays and their static metadata.
    """

    def __check_init__(self):
        """Validation hook called after Equinox auto-initialization."""
        super().__check_init__()
        # Validate array shape
        if not (
            (self.array.ndim == 3 and self.array.shape == self.mesh_size)
            or (self.array.ndim == 4 and self.array.shape[1:] == self.mesh_size)
            or (self.array.ndim == 5 and self.array.shape[2:] == self.mesh_size)
        ):
            raise ValueError(
                "DensityField array must have shape (mesh_size), (S, mesh_size), or (N, S, mesh_size), "
                f"got array shape {self.array.shape} and mesh_size {self.mesh_size}"
            )
        # Validate unit type
        if not isinstance(self.unit, DensityUnit):
            raise TypeError(
                f"DensityField.unit must be a DensityUnit, got {self.unit!r}. "
                "Please set the correct unit when constructing the field."
            )

    def apply_sharding(self) -> DensityField:
        """Shard the 3-D density: ``P([None,] "x", "y", None)`` (X→M, Y→N, Z replicated).

        Uses the canonical layout in ``field_sharding`` directly (see ``AbstractField.apply_sharding``);
        the volumetric box is the reference layout the lensing maps transpose from.
        """
        return super().apply_sharding()

    def __getitem__(self, key) -> DensityField:
        """
        Index into batched DensityField.

        For a 4D array (S, X, Y, Z), slices the snapshot dimension.
        For a 5D array (N, S, X, Y, Z), slices the simulation-batch dimension.
        """
        if self.array.ndim not in (4, 5):
            warn(
                "Indexing only supported for batched DensityField with 4D or 5D array, "
                f"got array with {self.array.ndim} dimensions"
                "Returning self without indexing."
            )
            return self
        # Use the AbstractField __getitem__ implementation (tree.map).
        return super().__getitem__(key)

    # --------------------------------------------------------------- properties

    def to(
        self,
        unit: DensityUnit,
        *,
        omega_m: float | None = None,
        h: float | None = None,
        mean_density: float | None = None,
        normalization: str = "global",
    ) -> DensityField:
        """
        Convert the field to a different unit.

        Parameters
        ----------
        unit : DensityUnit
            Target unit for conversion.
        omega_m : float, optional
            Matter density parameter. Required for MSUN_H_PER_MPC3 conversions.
        h : float, optional
            Hubble parameter (H0 / 100 km/s/Mpc). Required for MSUN_H_PER_MPC3 conversions.
        mean_density : float, optional
            Mean density ρ̄ in particles per (Mpc/h)³. Required when converting
            FROM OVERDENSITY to other units.
        normalization : str, optional
            Overdensity normalization: "global" (default) uses the mean over
            the whole array. Ignored unless converting to OVERDENSITY and
            mean_density is not provided explicitly.

        Returns
        -------
        DensityField
            New field with converted array and updated unit attribute.

        Examples
        --------
        >>> field = DensityField(array=arr, mesh_size=(128,128,128), box_size=(500,500,500))
        >>> overdensity_field = field.to(DensityUnit.OVERDENSITY)
        >>> physical_field = field.to(DensityUnit.MSUN_H_PER_MPC3, omega_m=0.3, h=0.7)
        """
        if self.unit == unit:
            return self

        if not isinstance(unit, DensityUnit):
            raise TypeError(f"DensityField.to() expected a DensityUnit for 'unit', got {unit!r}")

        # Compute volume element for 3D voxel
        box_size_arr = np.array(self.box_size)
        mesh_size_arr = np.array(self.mesh_size)
        volume_element = np.prod(box_size_arr / mesh_size_arr)

        new_array = convert_units(
            array=self.array,
            origin=self.unit,
            destination=unit,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            h=h,
            omega_m=omega_m,
            mean_density=mean_density,
            volume_element=volume_element,
            field_sharding=self.field_sharding,
            normalization=normalization,
        )

        return self.replace(array=new_array, unit=unit)

    @partial(jax.jit, static_argnames=["nz_slices"])
    def project(self, nz_slices: int = 10) -> FlatDensity:
        """
        Create a 2D projection by summing slices along the z-axis.

        Parameters
        ----------
        nz_slices : int, default=10
            Number of z-slices to sum from the end of the array.

        Returns
        -------
        FlatDensity
            2D flat-sky density map.
        """
        data = jnp.asarray(self.array)

        if data.ndim not in (3, 4):
            raise ValueError(f"project() expects 3D array or batch of 3D arrays, got shape {data.shape}")

        # Vectorized: sum over last nz_slices along z-axis (last dimension)
        projection = jnp.sum(data[..., -nz_slices:], axis=-1)

        # Update flatsky_npix to match projected shape
        # For 3D: shape is (X, Y), for 4D: shape is (N, X, Y) - take last 2 dims
        flatsky_npix = projection.shape[-2:] if data.ndim == 4 else projection.shape
        projected_field = self.replace(flatsky_npix=flatsky_npix)
        from .lightcone import FlatDensity

        return FlatDensity.FromDensityMetadata(
            array=projection,
            field=projected_field,
            status=FieldStatus.PROJECTED_DENSITY,
        )

    def plot(
        self,
        *,
        ax: Axes | Sequence[Axes] | None = None,
        cmap: str = "magma",
        figsize: tuple[float, float] | None = None,
        ncols: int = 3,
        titles: str | Sequence[str] | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        colorbar: bool = True,
        project_slices: int = 10,
        crop: tuple[slice, slice, slice] = (slice(None), slice(None), slice(None)),
        labels: tuple[str, str, str] = ("X", "Y", "Z"),
        ticks: tuple[Sequence[float], Sequence[float], Sequence[float]] = ([], [], []),
        elev: float = 40,
        azim: float = -30,
        zoom: float = 0.8,
        edges: bool = True,
        levels: int = 64,
    ):
        """Plot 3D density field as orthogonal slice visualization."""

        if not jax.core.is_concrete(self.array):
            raise ValueError("Cannot plot traced arrays. Use outside of jit context.")

        data = np.asarray(self.array)

        # Handle batch dimension
        if data.ndim == 3:
            data = data[None, ...]
        elif data.ndim != 4:
            raise ValueError(f"Expected 3D or 4D array, got {data.ndim}D")

        n_plots = data.shape[0]
        fig, axes = prepare_axes(ax, n_plots, ncols, projection="3d", figsize=figsize)

        if isinstance(titles, str):
            titles = [titles]

        if titles is None:
            titles = generate_titles("3D Density", self.scale_factors, n_plots)

        for idx, ax_i in enumerate(axes[:n_plots]):
            plot_3d_density(
                ax_i,
                data[idx],
                project_slices=project_slices,
                crop=crop,
                labels=labels,
                ticks=ticks,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                elev=elev,
                azim=azim,
                edges=edges,
                colorbar=colorbar,
                levels=levels,
            )
            if titles and idx < len(titles):
                ax_i.set_title(titles[idx])

        for ax_i in axes[n_plots:]:
            fig.delaxes(ax_i)

        fig.tight_layout()
        return fig, axes

    def show(
        self,
        *,
        ax: Axes | Sequence[Axes] | None = None,
        cmap: str = "magma",
        figsize: tuple[float, float] | None = None,
        ncols: int = 3,
        titles: str | Sequence[str] | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        colorbar: bool = True,
        project_slices: int = 10,
        crop: tuple[slice, slice, slice] = (slice(None), slice(None), slice(None)),
        labels: tuple[str, str, str] = ("X", "Y", "Z"),
        ticks: tuple[Sequence[float], Sequence[float], Sequence[float]] = ([], [], []),
        elev: float = 40,
        azim: float = -30,
        zoom: float = 0.8,
        edges: bool = True,
        levels: int = 64,
    ) -> None:
        """Display 3D density using :meth:`plot`."""

        import matplotlib.pyplot as plt

        self.plot(
            ax=ax,
            cmap=cmap,
            figsize=figsize,
            ncols=ncols,
            titles=titles,
            vmin=vmin,
            vmax=vmax,
            colorbar=colorbar,
            project_slices=project_slices,
            crop=crop,
            labels=labels,
            ticks=ticks,
            elev=elev,
            azim=azim,
            zoom=zoom,
            edges=edges,
            levels=levels,
        )
        plt.show()

    # -------------------------------------------------------- power-spectrum API
    @partial(
        jax.jit,
        static_argnames=["multipoles", "los", "batch_size", "kmax", "dk", "compensate_order", "shotnoise"],
    )
    def power(
        self,
        mesh2: DensityField | None = None,
        *,
        kedges: Array | jnp.ndarray | None = None,
        dk: float | None = None,
        kmax: float | None = None,
        multipoles: Iterable[int] | None = 0,
        los: Array | Iterable[float] = (0.0, 0.0, 1.0),
        batch_size: int | None = None,
        compensate_order: int | str | None = None,
        shotnoise: tuple[int | str, float] | None = None,
    ) -> PowerSpectrum:
        """Compute the 3D matter power spectrum P(k).

        Parameters mirror :func:`jax_fli.power.power`. Any keyword
        arguments are forwarded verbatim to that helper.

        ``compensate_order`` deconvolves the mass-assignment window of that
        order (NGP=1, CIC=2, TSC=3, PCS=4). ``shotnoise=(order, nbar)`` subtracts
        the aliased Poisson shot noise before deconvolving (auto-spectrum only);
        ``nbar = N / box.prod()`` (for one particle per cell,
        ``nbar = prod(mesh_size) / prod(box_size)``).
        """
        box_shape = tuple(self.box_size)

        data1 = self.array
        data2 = mesh2.array if mesh2 is not None else None

        if data1.ndim == 3:
            data1 = jnp.expand_dims(data1, axis=0)
            data2 = jnp.expand_dims(data2, axis=0) if data2 is not None else None
        elif data1.ndim != 4:
            raise ValueError("DensityField.power expects array shape (X,Y,Z) or (B,X,Y,Z)")

        if data2 is not None and data2.shape != data1.shape:
            raise ValueError("mesh2 must match mesh shape for power")

        def _power_fn(pair):
            arr1, arr2 = pair
            return power_fn(
                arr1,
                arr2,
                box_shape=box_shape,
                kedges=kedges,
                dk=dk,
                kmax=kmax,
                multipoles=multipoles,
                los=los,
                compensate_order=compensate_order,
                shotnoise=shotnoise,
            )

        k, pk = jax.lax.map(_power_fn, (data1, data2), batch_size=batch_size)
        k, pk = k[0], pk.squeeze()
        name = self.name + "_pk" if self.name else "power_spectrum"
        return PowerSpectrum(
            name=name,
            wavenumber=k,
            array=pk,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            scale_factors=self.scale_factors,
            status=FieldStatus.SPECTRA,
            unit=SpectralUnit.POWER_SPECTRA,
        )

    @partial(
        jax.jit,
        static_argnames=["multipoles", "los", "batch_size", "kmax", "dk", "compensate_order", "shotnoise"],
    )
    def cross_power(
        self,
        *,
        kedges: Array | jnp.ndarray | None = None,
        dk: float | None = None,
        kmax: float | None = None,
        multipoles: Iterable[int] | None = 0,
        los: Array | Iterable[float] = (0.0, 0.0, 1.0),
        batch_size: int | None = None,
        compensate_order: int | str | None = None,
        shotnoise: tuple[int | str, float] | None = None,
    ) -> PowerSpectrum:
        """Compute all cross-power spectra for batched density fields.

        For a batched field with B maps, computes K = B*(B+1)/2 cross-spectra
        corresponding to all unique pairs (i,j) where i <= j, in upper triangular order:
        (0,0), (0,1), ..., (0,B-1), (1,1), (1,2), ..., (B-1,B-1)

        Parameters
        ----------
        kedges : array_like, optional
            Edges of k-bins for power spectrum estimation.
        multipoles : int or iterable of int, default=0
            Multipole moments to compute (0 for monopole only).
        los : array_like, default=(0.0, 0.0, 1.0)
            Line-of-sight direction for multipole decomposition.
        batch_size : int, optional
            Batch size for lax.map processing. None means no batching.
        compensate_order : int, str, or None
            Deconvolve the mass-assignment window of this order from every pair.
        shotnoise : (order, nbar) or None
            Subtracted from auto-spectra only; since every pair here is computed
            as a cross-spectrum it does not affect the result (kept for symmetry
            with :meth:`power`).

        Returns
        -------
        PowerSpectrum
            Power spectrum object with array shape (K,) or (K, n_multipoles) where K = B*(B+1)/2

        Raises
        ------
        ValueError
            If array is not 4D or has fewer than 2 maps in batch dimension.

        Examples
        --------
        >>> # Create batched field with 4 density maps
        >>> field = DensityField(array=jnp.ones((4, 64, 64, 64)), ...)
        >>> cross_pk = field.cross_power()
        >>> cross_pk.array.shape[0]  # K = 4*(4+1)/2 = 10
        10
        """
        data = self.array

        # Validate batched 4D input with at least 2 maps
        if data.ndim != 4:
            raise ValueError(
                f"cross_power requires batched array with shape (B, X, Y, Z), "
                f"got array with {data.ndim} dimensions. Use power() for single fields."
            )

        n_maps = data.shape[0]
        if n_maps < 2:
            raise ValueError(
                f"cross_power requires at least 2 maps in batch dimension, got {n_maps}. Use power() for single fields."
            )

        box_shape = tuple(self.box_size)
        multipoles_static = tuple(multipoles) if isinstance(multipoles, (list | tuple)) else (multipoles,)
        los_tuple = None if multipoles_static == 0 else tuple(np.asarray(los, dtype=float))

        def pair_fn(mesh_i, mesh_j):
            """Compute power spectrum for a single (i, j) pair."""
            k, pk = power_fn(
                mesh_i,
                mesh_j,
                box_shape=box_shape,
                kedges=kedges,
                dk=dk,
                kmax=kmax,
                multipoles=multipoles_static,
                los=los_tuple,
                compensate_order=compensate_order,
                shotnoise=shotnoise,
            )
            return k, pk

        # Compute all upper triangular pairs using tri_map
        results = tri_map(data, pair_fn, batch_size=batch_size)
        k_stack, pk_stack = results

        # Extract wavenumber from first result (all pairs share same k-bins)
        wavenumber = k_stack[0]

        name = self.name + "_cross_pk" if self.name else "cross_power_spectrum"
        return PowerSpectrum(
            name=name,
            wavenumber=wavenumber,
            array=pk_stack,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            scale_factors=self.scale_factors,
            status=FieldStatus.SPECTRA,
            unit=SpectralUnit.POWER_SPECTRA,
        )

    @partial(jax.jit, static_argnames=["batch_size", "kmax", "dk", "compensate_order", "shotnoise"])
    def transfer(
        self,
        other: DensityField,
        *,
        kedges: Array | jnp.ndarray | None = None,
        dk: float | None = None,
        kmax: float | None = None,
        batch_size: int | None = None,
        compensate_order: int | str | None = None,
        shotnoise: tuple[int | str, float] | None = None,
    ) -> PowerSpectrum:
        """Monopole transfer function sqrt(P_other / P_self).

        ``compensate_order``/``shotnoise`` are applied to both auto-spectra.
        """

        def _compute(pair):
            arr1, arr2 = pair
            return transfer(
                arr1,
                arr2,
                box_shape=self.box_size,
                kedges=kedges,
                dk=dk,
                kmax=kmax,
                compensate_order=compensate_order,
                shotnoise=shotnoise,
            )

        data1 = self.array
        data2 = other.array

        if data1.ndim == 3:
            data1 = data1[None, ...]
            data2 = data2[None, ...]
        elif data1.ndim != 4:
            raise ValueError("DensityField.transfer expects array shape (X,Y,Z) or (B,X,Y,Z)")

        if data2.shape != data1.shape:
            raise ValueError("other array must match shape for transfer")

        k_stack, spectra_stack = jax.lax.map(_compute, (data1, data2), batch_size=batch_size)
        wavenumber = k_stack[0]
        spectra = spectra_stack if self.array.ndim == 4 else spectra_stack[0]
        name = self.name + "_transfer" if self.name else "transfer_function"
        return PowerSpectrum(
            name=name,
            wavenumber=wavenumber,
            array=spectra,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            scale_factors=self.scale_factors,
            status=FieldStatus.SPECTRA,
            unit=SpectralUnit.POWER_SPECTRA,
        )

    @partial(jax.jit, static_argnames=["batch_size", "kmax", "dk", "compensate_order", "shotnoise"])
    def coherence(
        self,
        other: DensityField,
        *,
        kedges: Array | jnp.ndarray | None = None,
        dk: float | None = None,
        kmax: float | None = None,
        batch_size: int | None = None,
        compensate_order: int | str | None = None,
        shotnoise: tuple[int | str, float] | None = None,
    ) -> PowerSpectrum:
        """Monopole coherence pk01 / sqrt(pk0 pk1).

        ``compensate_order`` deconvolves every spectrum; ``shotnoise`` is
        subtracted from the two auto-spectra only.
        """

        def _compute(pair):
            arr1, arr2 = pair
            return coherence(
                arr1,
                arr2,
                box_shape=self.box_size,
                kedges=kedges,
                dk=dk,
                kmax=kmax,
                compensate_order=compensate_order,
                shotnoise=shotnoise,
            )

        data1 = self.array
        data2 = other.array

        if data1.ndim == 3:
            data1 = data1[None, ...]
            data2 = data2[None, ...]
        elif data1.ndim != 4:
            raise ValueError("DensityField.coherence expects array shape (X,Y,Z) or (B,X,Y,Z)")

        if data2.shape != data1.shape:
            raise ValueError("other array must match shape for coherence")

        k_stack, spectra_stack = jax.lax.map(_compute, (data1, data2), batch_size=batch_size)
        wavenumber = k_stack[0]
        spectra = spectra_stack if self.array.ndim == 4 else spectra_stack[0]
        name = self.name + "_coherence" if self.name else "coherence"
        return PowerSpectrum(
            name=name,
            wavenumber=wavenumber,
            array=spectra,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            scale_factors=self.scale_factors,
            status=FieldStatus.SPECTRA,
            unit=SpectralUnit.POWER_SPECTRA,
        )

    @classmethod
    def full_like(cls, field: AbstractField, fill_value: float = 0.0) -> DensityField:
        """
        Create a new DensityField with the same metadata as `field`
        and an array filled with `fill_value`.
        """
        return cls.FromDensityMetadata(
            array=jnp.full(field.mesh_size, fill_value),
            field=field,
        )

    def is_batched(self) -> bool:
        """Return True if the DensityField has a leading batch dimension (S or N×S)."""
        return self.array.ndim in (4, 5)

    def is_multi_batched(self) -> bool:
        """Return True when the field has both a simulation-batch (N) and snapshot (S) dimension."""
        return self.array.ndim == 5
