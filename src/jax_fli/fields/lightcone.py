from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import jax_healpy as jhp
from jax.image import resize

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from warnings import warn

from .._src.base._core import AbstractField
from .._src.base._enums import FieldStatus, SpectralUnit
from .._src.base._tri_map import tri_map
from .._src.fields._plotting import generate_titles, plot_flat_density, plot_spherical_density, prepare_axes
from ..summary_statistics import (
    PDF,
    PeakCounts,
    PowerSpectrum,
    angular_cl_flat,
    angular_cl_spherical,
    compute_mcm,
    cross_angular_cl_spherical,
    deconvolve_spherical,
    pdf_spherical,
    peak_counts_spherical,
    starlet_coefficients_spherical,
)
from ..summary_statistics.binned import resolve_bin_edges
from ..summary_statistics.decouple import anafast_masked
from .units import DensityUnit, convert_units


class FlatDensity(AbstractField):
    """Flat-sky (2D) density or shear maps derived from volumetric simulations."""

    STATUS_ENUM = FieldStatus
    # Max array rank accepted by the shape check: (ny, nx), (S, ny, nx), (N, S, ny, nx). Spin-2 shear
    # subclasses raise this to 5 to allow a leading (2, ny, nx) component axis.
    _MAX_ARRAY_NDIM = 4

    def __check_init__(self):
        """Validation hook called after Equinox auto-initialization."""
        super().__check_init__()
        # Validate required fields
        if self.flatsky_npix is None:
            raise ValueError("FlatDensity requires `flatsky_npix`.")
        # Validate array rank and extract the trailing 2D spatial shape. Subclasses with an extra
        # component axis (e.g. spin-2 shear) raise ``_MAX_ARRAY_NDIM`` to admit the extra dimension.
        # (Guard for metadata-only fields whose ``array`` is None, matching SphericalDensity.)
        if self.array is not None and getattr(self.array, "shape", ()) != ():
            ndim = self.array.ndim
            if not (2 <= ndim <= self._MAX_ARRAY_NDIM):
                raise ValueError(
                    f"FlatDensity array must have shape (ny, nx), (S, ny, nx), or (N, S, ny, nx), "
                    f"got {ndim}D array with shape {self.array.shape}."
                )
            spatial_shape = self.array.shape if ndim == 2 else self.array.shape[-2:]
            if spatial_shape != tuple(self.flatsky_npix):
                raise ValueError(
                    f"Array spatial shape {spatial_shape} does not match flatsky_npix {self.flatsky_npix}."
                )

    def apply_sharding(self) -> FlatDensity:
        """Shard the flat-sky map: ``P([None,] "x", "y")`` (first spatial dim→M, second→N).

        Uses the canonical layout in ``field_sharding`` directly. Flat-sky convergence/shear inherit
        this image layout — the flat Kaiser-Squires path is a plain ``vmap`` FFT, not bins-sharded.
        """
        return super().apply_sharding()

    def __getitem__(self, key) -> FlatDensity:
        if self.array.ndim < 3:
            warn(
                f"Indexing only supported for batched FlatDensity (3D or 4D array), "
                f"got array with {self.array.ndim} dimensions"
                "Returning self without indexing."
            )
            return self
        return super().__getitem__(key)

    def to(
        self,
        unit: DensityUnit,
        *,
        omega_m: float | None = None,
        h: float | None = None,
        mean_density: float | None = None,
        normalization: str = "global",
    ) -> FlatDensity:
        """
        Convert the flat-sky map to a different density unit.

        Parameters
        ----------
        unit : DensityUnit
            Target unit for conversion.
        omega_m : float, optional
            Matter density parameter. Required for MSUN_H_PER_MPC3 conversions.
        h : float, optional
            Hubble parameter (H0 / 100 km/s/Mpc). Required for MSUN_H_PER_MPC3 conversions.
        mean_density : float, optional
            Mean density ρ̄ in particles per (Mpc/h)^3. Required when converting
            FROM OVERDENSITY to other units.
        normalization : str, optional
            Overdensity normalization: "global" (default) uses the mean over
            the whole array; "per_plane" normalises each shell (axis 0) by its
            own spatial mean over the pixel axes. Ignored unless converting to
            OVERDENSITY and mean_density is not provided explicitly.

        Returns
        -------
        FlatDensity
            New FlatDensity with converted array and updated unit.
        """
        if self.unit == unit:
            return self

        ny_pix, nx_pix = self.flatsky_npix
        Lx, Ly, _ = self.box_size  # last dim is always Z
        volume_element = (Lx / nx_pix) * (Ly / ny_pix) * self.density_width

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
        show_ticks: bool = False,
    ):
        """
        Visualize one or more flat-sky maps using matplotlib.
        """
        if not jax.core.is_concrete(self.array):
            raise ValueError("Cannot plot/show traced arrays. Use outside of jit context.")

        data = jnp.asarray(self.array)
        if data.ndim == 2:
            data = data[None, ...]
        elif data.ndim != 3:
            raise ValueError("FlatDensity.plot expects array shape (ny, nx) or (n, ny, nx).")

        n_maps = data.shape[0]

        if isinstance(titles, str):
            titles = [titles]

        if titles is None:
            titles = generate_titles("Flat Sky Density", self.scale_factors, n_maps)

        fig, axes = prepare_axes(ax, n_maps, ncols, figsize=figsize)

        for idx, ax_i in enumerate(axes):
            if idx < n_maps:
                plot_flat_density(
                    ax_i,
                    data[idx],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    show_colorbar=colorbar,
                    show_ticks=show_ticks,
                    title=titles[idx] if titles and idx < len(titles) else None,
                )
            else:
                ax_i.axis("off")

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
        show_ticks: bool = False,
    ) -> None:
        """Plot and display flat maps using matplotlib."""
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
            show_ticks=show_ticks,
        )
        plt.show()

    def ud_sample(self, new_npix):
        """
        Resample to new resolution using jax.image.resize.
        """
        if self.array.ndim == 3:
            new_shape = (self.array.shape[0], new_npix, new_npix)
        else:
            new_shape = (new_npix, new_npix)

        resampled = resize(self.array, new_shape, method="bilinear")

        return self.replace(array=resampled, flatsky_npix=(new_npix, new_npix))

    def angular_cl(
        self,
        mesh2: FlatDensity | None = None,
        *,
        field_size: float | None = None,
        pixel_size: float | None = None,
        ell_edges: Iterable[float] | None = None,
        batch_size: int | None = None,
    ) -> PowerSpectrum:
        """Compute a flat-sky angular power spectrum C_ell (auto or cross)."""

        effective_field_size = field_size or self.field_size
        data1 = self.array
        data2 = mesh2.array if mesh2 is not None else None

        if data1.ndim == 2:
            data1 = data1[None, ...]
            data2 = data2[None, ...] if data2 is not None else None
        elif data1.ndim != 3:
            raise ValueError("FlatDensity.angular_cl expects array shape (ny,nx) or (B,ny,nx)")

        if data2 is not None and data2.shape != data1.shape:
            raise ValueError("mesh2 must match shape for cross Cl")

        def _compute(pair):
            m1, m2 = pair
            return angular_cl_flat(
                m1,
                m2,
                pixel_size=pixel_size,
                field_size=effective_field_size,
                ell_edges=ell_edges,
            )

        ell_stack, spectra_stack = jax.lax.map(_compute, (data1, data2), batch_size=batch_size)
        wavenumber = ell_stack[0]
        spectra = spectra_stack if self.array.ndim == 3 else spectra_stack[0]
        return PowerSpectrum(
            name=self.name,
            wavenumber=wavenumber,
            array=spectra,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            comoving_centers=self.comoving_centers,
            density_width=self.density_width,
            z_sources=self.z_sources,
            scale_factors=self.scale_factors,
            flatsky_npix=self.flatsky_npix,
            field_size=self.field_size,
            status=FieldStatus.SPECTRA,
            unit=SpectralUnit.ANGULAR_CL,
        )

    def cross_angular_cl(
        self,
        *,
        field_size: float | None = None,
        pixel_size: float | None = None,
        ell_edges: Iterable[float] | None = None,
        batch_size: int | None = None,
    ) -> PowerSpectrum:
        """Compute all cross-angular power spectra for batched flat-sky maps.

        For a batched field with B maps, computes K = B*(B+1)/2 cross-spectra
        corresponding to all unique pairs (i,j) where i <= j, in upper triangular order:
        (0,0), (0,1), ..., (0,B-1), (1,1), (1,2), ..., (B-1,B-1)

        Parameters
        ----------
        field_size : float, optional
            Field size in degrees. Defaults to self.field_size.
        pixel_size : float, optional
            Pixel size for Fourier transforms.
        ell_edges : array_like, optional
            Edges of ell-bins for angular power spectrum.
        batch_size : int, optional
            Batch size for lax.map processing. None means no batching.

        Returns
        -------
        PowerSpectrum
            Angular power spectrum object with array shape (K,) where K = B*(B+1)/2

        Raises
        ------
        ValueError
            If array is not 3D or has fewer than 2 maps in batch dimension.

        Examples
        --------
        >>> # Create batched flat-sky maps with 4 maps
        >>> flat = FlatDensity(array=jnp.ones((4, 128, 128)), ...)
        >>> cross_cl = flat.cross_angular_cl()
        >>> cross_cl.array.shape[0]  # K = 4*(4+1)/2 = 10
        10
        """
        data = self.array

        # Validate batched 3D input with at least 2 maps
        if data.ndim != 3:
            raise ValueError(
                f"cross_angular_cl requires batched array with shape (B, ny, nx), "
                f"got array with {data.ndim} dimensions. Use angular_cl() for single maps."
            )

        n_maps = data.shape[0]
        if n_maps < 2:
            raise ValueError(
                f"cross_angular_cl requires at least 2 maps in batch dimension, got {n_maps}. "
                "Use angular_cl() for single maps."
            )

        effective_field_size = field_size or self.field_size

        def pair_fn(pair):
            """Compute angular Cl for a single (i, j) pair."""
            map_i, map_j = pair
            ell, cl = angular_cl_flat(
                map_i,
                map_j,
                pixel_size=pixel_size,
                field_size=effective_field_size,
                ell_edges=ell_edges,
            )
            return ell, cl

        # Compute all upper triangular pairs using tri_map
        results = tri_map(data, pair_fn, batch_size=batch_size)
        ell_stack, cl_stack = results

        # Extract ell from first result (all pairs share same ell-bins)
        wavenumber = ell_stack[0]

        name = self.name + "_cross_cl" if self.name else "cross_cl"
        return PowerSpectrum(
            name=name,
            wavenumber=wavenumber,
            array=cl_stack,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            comoving_centers=self.comoving_centers,
            density_width=self.density_width,
            z_sources=self.z_sources,
            scale_factors=self.scale_factors,
            flatsky_npix=self.flatsky_npix,
            field_size=self.field_size,
            status=FieldStatus.SPECTRA,
            unit=SpectralUnit.ANGULAR_CL,
        )

    def transfer(
        self,
        other: FlatDensity,
        *,
        field_size: float | None = None,
        pixel_size: float | None = None,
        ell_edges: Iterable[float] | None = None,
        batch_size: int | None = None,
    ) -> PowerSpectrum:
        """Compute angular transfer function sqrt(Cl_other / Cl_self).

        Parameters
        ----------
        other : FlatDensity
            The other flat-sky field to compare against.
        field_size : float, optional
            Field size in degrees. Defaults to self.field_size.
        pixel_size : float, optional
            Pixel size for Fourier transforms.
        ell_edges : array_like, optional
            Edges of ell-bins for angular power spectrum.
        batch_size : int, optional
            Batch size for lax.map processing.

        Returns
        -------
        PowerSpectrum
            Transfer function T(ell) = sqrt(Cl_other / Cl_self).
        """
        cl_self = self.angular_cl(
            field_size=field_size,
            pixel_size=pixel_size,
            ell_edges=ell_edges,
            batch_size=batch_size,
        )
        cl_other = other.angular_cl(
            field_size=field_size,
            pixel_size=pixel_size,
            ell_edges=ell_edges,
            batch_size=batch_size,
        )
        transfer_ratio = (cl_other.array / cl_self.array) ** 0.5
        name = self.name + "_transfer" if self.name else "transfer"
        return PowerSpectrum(
            name=name,
            wavenumber=cl_self.wavenumber,
            array=transfer_ratio,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            comoving_centers=self.comoving_centers,
            density_width=self.density_width,
            z_sources=self.z_sources,
            scale_factors=self.scale_factors,
            flatsky_npix=self.flatsky_npix,
            field_size=self.field_size,
            status=FieldStatus.SPECTRA,
            unit=SpectralUnit.ANGULAR_CL,
        )

    def coherence(
        self,
        other: FlatDensity,
        *,
        field_size: float | None = None,
        pixel_size: float | None = None,
        ell_edges: Iterable[float] | None = None,
        batch_size: int | None = None,
    ) -> PowerSpectrum:
        """Compute angular coherence Cl_cross / sqrt(Cl_self * Cl_other).

        Parameters
        ----------
        other : FlatDensity
            The other flat-sky field to compute coherence with.
        field_size : float, optional
            Field size in degrees. Defaults to self.field_size.
        pixel_size : float, optional
            Pixel size for Fourier transforms.
        ell_edges : array_like, optional
            Edges of ell-bins for angular power spectrum.
        batch_size : int, optional
            Batch size for lax.map processing.

        Returns
        -------
        PowerSpectrum
            Coherence r(ell) = Cl_cross / sqrt(Cl_self * Cl_other).
        """
        cl_self = self.angular_cl(
            field_size=field_size,
            pixel_size=pixel_size,
            ell_edges=ell_edges,
            batch_size=batch_size,
        )
        cl_other = other.angular_cl(
            field_size=field_size,
            pixel_size=pixel_size,
            ell_edges=ell_edges,
            batch_size=batch_size,
        )
        cl_cross = self.angular_cl(
            other,
            field_size=field_size,
            pixel_size=pixel_size,
            ell_edges=ell_edges,
            batch_size=batch_size,
        )
        coherence_ratio = cl_cross.array / (cl_self.array * cl_other.array) ** 0.5
        name = self.name + "_coherence" if self.name else "coherence"
        return PowerSpectrum(
            name=name,
            wavenumber=cl_self.wavenumber,
            array=coherence_ratio,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            comoving_centers=self.comoving_centers,
            density_width=self.density_width,
            z_sources=self.z_sources,
            scale_factors=self.scale_factors,
            flatsky_npix=self.flatsky_npix,
            field_size=self.field_size,
            status=FieldStatus.SPECTRA,
            unit=SpectralUnit.ANGULAR_CL,
        )

    @classmethod
    def full_like(cls, field: AbstractField, fill_value: float = 0.0) -> FlatDensity:
        """
        Create a new DensityField with the same metadata as `field`
        and an array filled with `fill_value`.
        """
        if field.field_size is None:
            raise ValueError("field_size metadata is required to create full_like FlatDensity.")

        return cls.FromDensityMetadata(
            array=jnp.full(field.field_size, fill_value),
            field=field,
        )

    def is_batched(self) -> bool:
        """Return True if the FlatDensity has a leading batch dimension (S or N×S)."""
        return self.array.ndim in (3, 4)

    def is_multi_batched(self) -> bool:
        """Return True when the field has both a simulation-batch (N) and snapshot (S) dimension."""
        return self.array.ndim == 4


class SphericalDensity(AbstractField):
    """Spherical (HEALPix) density or shear maps produced from simulations."""

    STATUS_ENUM = FieldStatus
    # Max array rank accepted by the shape check: (npix), (S, npix), (N, S, npix). Spin-2 shear
    # subclasses raise this to 4 to allow a trailing (2, npix) component axis.
    _MAX_ARRAY_NDIM = 3

    def __check_init__(self):
        """Validation hook called after Equinox auto-initialization."""
        super().__check_init__()
        # Validate nside is provided
        if self.nside is None:
            raise ValueError("SphericalDensity requires `nside`.")
        # Validate array shape matches HEALPix npix
        if self.array is not None:
            array_shape = getattr(self.array, "shape", ())
            npix = jhp.nside2npix(self.nside)
            if array_shape != ():
                if not (1 <= len(array_shape) <= self._MAX_ARRAY_NDIM):
                    raise ValueError(
                        f"SphericalDensity array must have shape (npix,), (S, npix), or (N, S, npix), "
                        f"got {len(array_shape)}D array with shape {array_shape}."
                    )
                if array_shape[-1] != npix:
                    raise ValueError(
                        f"Array last dimension {array_shape[-1]} does not match "
                        f"HEALPix npix {npix} for nside {self.nside}."
                    )

    def apply_sharding(self) -> SphericalDensity:
        """Shard the HEALPix map: ``P([None,] "x")`` (NPIX→M; the N axis is unused for a bare density).

        Uses the canonical layout in ``field_sharding`` directly — ``get_sharding_for_shape`` trims the
        3-D ``P("x","y")`` to ``P("x")`` for the 1-D npix axis. The convergence/shear subclasses
        override this to add the BINS/N axis.
        """
        return super().apply_sharding()

    def __getitem__(self, key) -> SphericalDensity:
        if self.array.ndim < 2:
            warn(
                f"Indexing only supported for batched SphericalDensity (2D or 3D array), "
                f"got array with {self.array.ndim} dimensions"
                "Returning self without indexing."
            )
            return self
        return super().__getitem__(key)

    def to(
        self,
        unit: DensityUnit,
        *,
        omega_m: float | None = None,
        h: float | None = None,
        mean_density: float | None = None,
        normalization: str = "global",
    ) -> SphericalDensity:
        """
        Convert the spherical (HEALPix) map to a different density unit.

        Parameters
        ----------
        unit : DensityUnit
            Target unit for conversion.
        omega_m : float, optional
            Matter density parameter. Required for MSUN_H_PER_MPC3 conversions.
        h : float, optional
            Hubble parameter (H0 / 100 km/s/Mpc). Required for MSUN_H_PER_MPC3 conversions.
        mean_density : float, optional
            Mean density in particles per (Mpc/h)^3, required when converting
            FROM OVERDENSITY to other units.
        normalization : str, optional
            Overdensity normalization: "global" (default) uses the mean over
            the whole array; "per_plane" normalises each shell (axis 0) by its
            own pixel mean. For a (n_shells, npix) array this gives each shell
            zero global mean. Ignored unless converting to OVERDENSITY and
            mean_density is not provided explicitly.

        Returns
        -------
        SphericalDensity
            New SphericalDensity with converted array and updated unit.
        """
        if self.unit == unit:
            return self

        if self.comoving_centers is None or self.density_width is None:
            raise ValueError("comoving_centers and density_width metadata are required for unit conversion.")

        # Calculate volume per pixel in spherical shell (exact shell volume)
        R_max, R_min = self.comoving_centers + self.density_width / 2, self.comoving_centers - self.density_width / 2
        npix = jhp.nside2npix(self.nside)
        pixel_solid_angle = 4 * jnp.pi / npix  # steradians per pixel
        shell_volume_per_pixel = pixel_solid_angle * (R_max**3 - R_min**3) / 3.0

        new_array = convert_units(
            array=self.array,
            origin=self.unit,
            destination=unit,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            h=h,
            omega_m=omega_m,
            mean_density=mean_density,
            volume_element=shell_volume_per_pixel,
            field_sharding=self.field_sharding,
            normalization=normalization,
        )

        return self.replace(array=new_array, unit=unit)

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
        show_ticks: bool = False,
        projection_type: str = "mollweide",
        fontsize: dict | int | None = None,
        border_linewidth: float = 3.0,
    ):
        """
        Visualize one or more spherical maps using ``healpy.projview``.
        """
        if not jax.core.is_concrete(self.array):
            raise ValueError("Cannot plot/show traced arrays. Use outside of jit context.")

        data = jnp.asarray(self.array)
        if data.ndim == 1:
            data = data[None, ...]
        elif data.ndim < 1:
            raise ValueError("SphericalDensity array rank must be >= 1.")
        else:
            data = data.reshape((-1, data.shape[-1]))

        n_maps = data.shape[0]

        if isinstance(titles, str):
            titles = [titles]

        if titles is None:
            titles = generate_titles("Spherical Density", self.scale_factors, n_maps)

        fig, axes = prepare_axes(ax, n_maps, ncols, figsize=figsize)

        for idx, ax_i in enumerate(axes):
            if idx < n_maps:
                title = titles[idx] if titles and idx < len(titles) else ""
                plot_spherical_density(
                    ax_i,
                    data[idx],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    show_colorbar=colorbar,
                    show_ticks=show_ticks,
                    title=title,
                    projection_type=projection_type,
                    fontsize=fontsize,
                    border_linewidth=border_linewidth,
                )
            else:
                ax_i.axis("off")

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
        show_ticks: bool = False,
        projection_type: str = "mollweide",
        fontsize: dict | int | None = None,
        border_linewidth: float = 3.0,
    ) -> None:
        """
        Plot and display spherical maps using healpy.
        """
        import matplotlib.pyplot as plt

        if not jax.core.is_concrete(self.array):
            raise ValueError("Cannot plot/show traced arrays. Use outside of jit context.")

        self.plot(
            ax=ax,
            cmap=cmap,
            figsize=figsize,
            ncols=ncols,
            titles=titles,
            vmin=vmin,
            vmax=vmax,
            colorbar=colorbar,
            show_ticks=show_ticks,
            projection_type=projection_type,
            fontsize=fontsize,
            border_linewidth=border_linewidth,
        )
        plt.show()

    def ud_sample(self, new_nside):
        """
        Change HEALPix resolution using jax_healpy.ud_grade.
        """
        resampled = jhp.ud_grade(self.array, new_nside)
        return self.replace(array=resampled, nside=new_nside)

    def deconvolve(
        self,
        method: str,
        *,
        lmax: int | None = None,
        lcut: int | None = None,
        kernel_width_arcmin: float | None = None,
        kernel_width_pixels: float | None = None,
        smoothing_interpretation: str = "fwhm",
        iter: int = 0,
        w_floor: float = 1e-8,
        batch_size: int | None = None,
    ) -> SphericalDensity:
        """Deconvolve the HEALPix mass-assignment window from the map(s).

        Painting particles onto a HEALPix grid convolves the field with the
        assignment window ``W_l``. This removes one factor of that window at the
        ``a_lm`` level (``map2alm`` -> divide by ``W_l`` -> ``alm2map``), returning
        a window-corrected (sharpened) map. Batched arrays — ``(S, npix)`` or
        ``(N, S, npix)`` — are deconvolved map-by-map via ``jax.lax.map``.

        ``method`` is required: the field does not record which scheme painted it,
        and the painting default (``"bilinear"``) has no closed-form window. Pass
        the scheme you painted with — ``"ngp"`` (HEALPix pixel window) or
        ``"rbf_neighbor"`` (pixel window times a Gaussian beam, in which case pass
        the same ``kernel_width_*`` used at paint time). ``"bilinear"`` raises
        ``NotImplementedError``.

        Parameters
        ----------
        method : {'ngp', 'rbf_neighbor', 'bilinear'}
            Painting scheme whose window to remove (case-insensitive).
        lmax : int, optional
            Maximum multipole. Defaults to ``3*nside - 1``.
        lcut : int, optional
            Zero the inverse window above this multipole (extra high-l safety).
        kernel_width_arcmin, kernel_width_pixels, smoothing_interpretation :
            RBF beam width (``'rbf_neighbor'`` only); must match painting.
        iter : int, default=0
            ``map2alm`` iterations.
        w_floor : float, default=1e-8
            Modes with ``W_l <= w_floor`` are dropped to avoid 1/0 amplification.
        batch_size : int, optional
            Batch size for ``jax.lax.map`` over batched maps.

        Returns
        -------
        SphericalDensity
            New field with the deconvolved array; metadata is unchanged.

        Notes
        -----
        Typically applied to an OVERDENSITY map; the output may dip slightly
        negative near the band limit. Assumes RING ordering (the default produced
        by this repo's painters). Transform accuracy degrades at high ``l`` under
        the project's ``JAX_ENABLE_X64=False``.
        """

        def _fn(m):
            return deconvolve_spherical(
                m,
                method=method,
                nside=self.nside,
                lmax=lmax,
                lcut=lcut,
                kernel_width_arcmin=kernel_width_arcmin,
                kernel_width_pixels=kernel_width_pixels,
                smoothing_interpretation=smoothing_interpretation,
                iter=iter,
                w_floor=w_floor,
            )

        data = self.array
        if data.ndim == 1:
            new_array = _fn(data)
        else:
            flat = data.reshape((-1, data.shape[-1]))
            out = jax.lax.map(_fn, flat, batch_size=batch_size)
            new_array = out.reshape(data.shape)

        return self.replace(array=new_array)

    def angular_cl(
        self,
        mesh2: SphericalDensity | None = None,
        *,
        lmax: int | None = None,
        method: str = "jax",
        batch_size: int | None = None,
        mask=None,
        purify_e: bool = False,
        purify_b: bool = False,
        mcm=None,
        nlb: int = 16,
    ) -> PowerSpectrum:
        """Compute a spherical (HEALPix) angular power spectrum C_ell (auto or cross).

        With ``mask=None`` this is the plain spectrum (unchanged). With an apodized ``mask`` it
        returns the masked pseudo-``C_l`` **decoupled** into bandpowers (``wavenumber`` = effective
        bandpower multipoles), via :func:`jax_fli.summary_statistics.anafast_masked`. The MCM is built once from
        ``mask`` (or pass a precomputed ``mcm``). ``purify_e``/``purify_b`` are spin-2 only and are
        ignored for a scalar field. Masked estimation uses a jax_healpy ``method`` (``"jax"`` /
        ``"jax_cuda"``); ``"healpy"`` falls back to ``"jax"``.

        ``method`` selects the SHT backend and defaults to ``"jax"`` (jittable/differentiable);
        ``"healpy"`` runs healpy's C ``anafast`` and requires concrete (non-traced) arrays.
        """
        if mask is not None:
            _lmax = lmax if lmax is not None else 3 * self.nside - 1
            _method = method  # pass through; "healpy" runs real healpy on the masked decoupled path
            _mcm = mcm if mcm is not None else compute_mcm(mask, lmax=_lmax, nlb=nlb, pol=False, method=_method)
            d1 = self.array
            d2 = mesh2.array if mesh2 is not None else None
            single = d1.ndim == 1
            flat1 = d1.reshape((-1, d1.shape[-1]))
            flat2 = None if d2 is None else d2.reshape((-1, d2.shape[-1]))
            if _method == "healpy":
                # healpy needs concrete arrays -> loop in Python (cannot run under jax.vmap)
                if flat2 is None:
                    cls = jnp.stack(
                        [
                            anafast_masked(m, mask=mask, lmax=_lmax, method=_method, pol=False, mcm=_mcm, nlb=nlb)[1]
                            for m in flat1
                        ]
                    )
                else:
                    cls = jnp.stack(
                        [
                            anafast_masked(m1, m2, mask=mask, lmax=_lmax, method=_method, pol=False, mcm=_mcm, nlb=nlb)[
                                1
                            ]
                            for m1, m2 in zip(flat1, flat2)
                        ]
                    )
            elif flat2 is None:
                cls = jax.vmap(
                    lambda m: anafast_masked(m, mask=mask, lmax=_lmax, method=_method, pol=False, mcm=_mcm, nlb=nlb)[1]
                )(flat1)
            else:
                cls = jax.vmap(
                    lambda m1, m2: anafast_masked(
                        m1, m2, mask=mask, lmax=_lmax, method=_method, pol=False, mcm=_mcm, nlb=nlb
                    )[1]
                )(flat1, flat2)
            return PowerSpectrum(
                name=self.name,
                wavenumber=_mcm.ell_eff,
                array=cls[0] if single else cls,
                n_components=1,
                mesh_size=self.mesh_size,
                box_size=self.box_size,
                comoving_centers=self.comoving_centers,
                density_width=self.density_width,
                z_sources=self.z_sources,
                scale_factors=self.scale_factors,
                nside=self.nside,
                status=FieldStatus.SPECTRA,
                unit=SpectralUnit.ANGULAR_CL,
            )

        def _compute(pair):
            m1, m2 = pair
            return angular_cl_spherical(m1, m2, lmax=lmax, method=method)

        data1 = self.array
        data2 = mesh2.array if mesh2 is not None else None

        if data1.ndim == 1:
            data1 = data1[None, ...]
            data2 = data2[None, ...] if data2 is not None else None
        elif data1.ndim != 2:
            raise ValueError("SphericalDensity.angular_cl expects array shape (npix) or (B,npix)")

        if method == "healpy":
            spectras = []
            ell = None
            for i in range(data1.shape[0]):
                map1 = data1[i]
                map2 = data2[i] if data2 is not None else None
                ell, spectra = angular_cl_spherical(map1, map2, lmax=lmax, method=method)
                spectras.append(spectra)
            assert ell is not None, "ell should have been set in loop"
            spectra = jnp.stack(spectras, axis=0)
            spectra = spectra if self.array.ndim == 2 else spectra[0]
            return PowerSpectrum(
                name=self.name,
                wavenumber=ell,
                array=spectra,
                mesh_size=self.mesh_size,
                box_size=self.box_size,
                comoving_centers=self.comoving_centers,
                density_width=self.density_width,
                z_sources=self.z_sources,
                scale_factors=self.scale_factors,
                nside=self.nside,
                status=FieldStatus.SPECTRA,
                unit=SpectralUnit.ANGULAR_CL,
            )

        if data2 is not None and data2.shape != data1.shape:
            raise ValueError("mesh2 must match shape for cross Cl")

        ell_stack, spectra_stack = jax.lax.map(_compute, (data1, data2), batch_size=batch_size)
        wavenumber = ell_stack[0]
        spectra = spectra_stack if self.array.ndim == 2 else spectra_stack[0]
        return PowerSpectrum(
            name=self.name,
            wavenumber=wavenumber,
            array=spectra,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            comoving_centers=self.comoving_centers,
            density_width=self.density_width,
            z_sources=self.z_sources,
            scale_factors=self.scale_factors,
            nside=self.nside,
            status=FieldStatus.SPECTRA,
            unit=SpectralUnit.ANGULAR_CL,
        )

    def cross_angular_cl(
        self,
        *,
        lmax: int | None = None,
        method: str = "healpy",
        batch_size: int | None = None,
    ) -> PowerSpectrum:
        """Compute all cross-angular power spectra for batched HEALPix maps.

        For a batched field with B maps, computes K = B*(B+1)/2 cross-spectra
        corresponding to all unique pairs (i,j) where i <= j, in upper triangular order:
        (0,0), (0,1), ..., (0,B-1), (1,1), (1,2), ..., (B-1,B-1)

        This method uses healpy's anafast function with pol=False, which efficiently
        computes all cross-spectra in a single call.

        Parameters
        ----------
        lmax : int, optional
            Maximum multipole moment. Defaults to 3*nside-1.
        method : str, default="healpy"
            Must be "healpy". JAX method is not supported for cross-spectra.
        batch_size : int, optional
            Not used for healpy method. Included for API consistency.

        Returns
        -------
        PowerSpectrum
            Angular power spectrum object with array shape (K, n_ell) where K = B*(B+1)/2

        Raises
        ------
        ValueError
            If method is not "healpy", array is not 2D, or has fewer than 2 maps.

        Examples
        --------
        >>> # Create batched HEALPix maps with 4 maps
        >>> sphere = SphericalDensity(array=jnp.ones((4, 12*64**2)), nside=64, ...)
        >>> cross_cl = sphere.cross_angular_cl()
        >>> cross_cl.array.shape[0]  # K = 4*(4+1)/2 = 10
        10

        Notes
        -----
        Unlike FlatDensity.cross_angular_cl which uses tri_map, this implementation
        relies on healpy.anafast which natively computes all cross-spectra when
        given multiple maps. This is more efficient than computing pairs individually.
        """
        data = self.array

        # Validate batched 2D input with at least 2 maps
        if data.ndim != 2:
            raise ValueError(
                f"cross_angular_cl requires batched array with shape (B, npix), "
                f"got array with {data.ndim} dimensions. Use angular_cl() for single maps."
            )

        n_maps = data.shape[0]
        if n_maps < 2:
            raise ValueError(
                f"cross_angular_cl requires at least 2 maps in batch dimension, got {n_maps}. "
                "Use angular_cl() for single maps."
            )

        # Call the power module function
        ell, angular_cls = cross_angular_cl_spherical(data, lmax=lmax, method=method)

        return PowerSpectrum(
            name=self.name,
            wavenumber=ell,
            array=angular_cls,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            comoving_centers=self.comoving_centers,
            density_width=self.density_width,
            z_sources=self.z_sources,
            scale_factors=self.scale_factors,
            nside=self.nside,
            status=FieldStatus.SPECTRA,
            unit=SpectralUnit.ANGULAR_CL,
        )

    def transfer(
        self,
        other: SphericalDensity,
        *,
        lmax: int | None = None,
        method: str = "jax",
        batch_size: int | None = None,
    ) -> PowerSpectrum:
        """Compute angular transfer function sqrt(Cl_other / Cl_self).

        Parameters
        ----------
        other : SphericalDensity
            The other spherical field to compare against.
        lmax : int, optional
            Maximum multipole moment. Defaults to 3*nside-1.
        method : str, default="jax"
            Method for computing power spectrum ("jax" or "healpy").
        batch_size : int, optional
            Batch size for lax.map processing.

        Returns
        -------
        PowerSpectrum
            Transfer function T(ell) = sqrt(Cl_other / Cl_self).
        """
        cl_self = self.angular_cl(
            lmax=lmax,
            method=method,
            batch_size=batch_size,
        )
        cl_other = other.angular_cl(
            lmax=lmax,
            method=method,
            batch_size=batch_size,
        )
        transfer_ratio = (cl_other.array / cl_self.array) ** 0.5
        name = "transfer" if self.name is None else self.name + "_transfer"
        return PowerSpectrum(
            name=name,
            wavenumber=cl_self.wavenumber,
            array=transfer_ratio,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            comoving_centers=self.comoving_centers,
            density_width=self.density_width,
            z_sources=self.z_sources,
            scale_factors=self.scale_factors,
            nside=self.nside,
            status=FieldStatus.SPECTRA,
            unit=SpectralUnit.ANGULAR_CL,
        )

    def coherence(
        self,
        other: SphericalDensity,
        *,
        lmax: int | None = None,
        method: str = "jax",
        batch_size: int | None = None,
    ) -> PowerSpectrum:
        """Compute angular coherence Cl_cross / sqrt(Cl_self * Cl_other).

        Parameters
        ----------
        other : SphericalDensity
            The other spherical field to compute coherence with.
        lmax : int, optional
            Maximum multipole moment. Defaults to 3*nside-1.
        method : str, default="jax"
            Method for computing power spectrum ("jax" or "healpy").
        batch_size : int, optional
            Batch size for lax.map processing.

        Returns
        -------
        PowerSpectrum
            Coherence r(ell) = Cl_cross / sqrt(Cl_self * Cl_other).
        """
        cl_self = self.angular_cl(
            lmax=lmax,
            method=method,
            batch_size=batch_size,
        )
        cl_other = other.angular_cl(
            lmax=lmax,
            method=method,
            batch_size=batch_size,
        )
        cl_cross = self.angular_cl(
            other,
            lmax=lmax,
            method=method,
            batch_size=batch_size,
        )
        coherence_ratio = cl_cross.array / (cl_self.array * cl_other.array) ** 0.5
        name = "coherence" if self.name is None else self.name + "_coherence"
        return PowerSpectrum(
            name=name,
            wavenumber=cl_self.wavenumber,
            array=coherence_ratio,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            comoving_centers=self.comoving_centers,
            density_width=self.density_width,
            z_sources=self.z_sources,
            scale_factors=self.scale_factors,
            nside=self.nside,
            status=FieldStatus.SPECTRA,
            unit=SpectralUnit.ANGULAR_CL,
        )

    def _binned_metadata(self) -> dict:
        """Field metadata copied onto a binned-statistic result (PDF / peak counts)."""
        return dict(
            name=self.name,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            comoving_centers=self.comoving_centers,
            density_width=self.density_width,
            z_sources=self.z_sources,
            scale_factors=self.scale_factors,
            nside=self.nside,
            field_size=self.field_size,
            status=FieldStatus.SPECTRA,
        )

    def compute_pdf(
        self,
        *,
        bins: int | jnp.ndarray = 50,
        range: tuple[float, float] | None = None,
        density: bool = True,
        soft: bool = False,
        bandwidth: float | None = None,
        batch_size: int | None = None,
    ) -> PDF:
        """One-point PDF of the map's pixel values.

        ``bins`` is a bin count (spanned by ``range``, or the concrete data ``[min, max]`` when
        ``range`` is omitted — eager only) or an explicit 1-D edges array. ``density=True``
        normalises the histogram to unit integral. ``soft=True`` evaluates a differentiable
        Gaussian KDE at the bin centers (``bandwidth`` defaults to the mean bin width). Batched
        maps ``(S, npix)`` / ``(N, S, npix)`` are reduced map-by-map via ``jax.lax.map`` into a
        ``(B, n_bins)`` result. Returns a :class:`~jax_fli.summary_statistics.pdf.PDF`.
        """
        data = self.array
        edges = resolve_bin_edges(data, bins, range)
        centers = 0.5 * (edges[:-1] + edges[1:])

        def _fn(m):
            return pdf_spherical(m, bins=edges, density=density, soft=soft, bandwidth=bandwidth)[1]

        if data.ndim == 1:
            arr = _fn(data)
        else:
            flat = data.reshape((-1, data.shape[-1]))
            arr = jax.lax.map(_fn, flat, batch_size=batch_size)

        return PDF(bins=centers, array=arr, **self._binned_metadata())

    def compute_peak_counts(
        self,
        *,
        bins: int | jnp.ndarray = 50,
        range: tuple[float, float] | None = None,
        normalize: bool = False,
        soft: bool = False,
        bandwidth: float | None = None,
        batch_size: int | None = None,
    ) -> PeakCounts:
        """Peak counts: histogram of local-maxima heights.

        A pixel is a peak iff it strictly exceeds all of its HEALPix neighbours. With
        ``normalize=True`` the map is binned in S/N units (``(m - mean) / std`` per map; default
        span ``(-2, 6)`` σ when ``bins`` is a count and ``range`` is omitted). ``soft=True`` uses a
        differentiable Gaussian soft-binning of the peak heights (peak *detection* stays hard).
        Batched maps are reduced map-by-map. Returns a
        :class:`~jax_fli.summary_statistics.peak_counts.PeakCounts`.
        """
        data = self.array
        default_range = (-2.0, 6.0) if normalize else None
        edges = resolve_bin_edges(data, bins, range, default_range=default_range)
        centers = 0.5 * (edges[:-1] + edges[1:])

        def _fn(m):
            return peak_counts_spherical(
                m, nside=self.nside, bins=edges, normalize=normalize, soft=soft, bandwidth=bandwidth
            )[1]

        if data.ndim == 1:
            arr = _fn(data)
        else:
            flat = data.reshape((-1, data.shape[-1]))
            arr = jax.lax.map(_fn, flat, batch_size=batch_size)

        return PeakCounts(bins=centers, array=arr, **self._binned_metadata())

    def starlet_coefficients(self, *, nscales: int = 5, normalize: bool = False) -> StarletCoefficients:
        """Spherical starlet (undecimated wavelet) coefficients — one HEALPix map per scale.

        Requires the optional CosmoStat backend (``pip install jax-fli[starlet]``) and is
        concrete-only (cannot run under ``jit``). Operates on a single map — index a single
        shell/bin first (e.g. ``field[i]``) if batched. ``normalize=True`` divides each scale by
        its ``TabNorm``. Returns a :class:`StarletCoefficients` (a ``SphericalDensity`` batched
        over scales, shape ``(nscales, npix)``).
        """
        if not jax.core.is_concrete(self.array):
            raise ValueError("starlet_coefficients is concrete-only; call outside of a jit context.")
        if self.array.ndim != 1:
            raise ValueError(
                "starlet_coefficients expects a single map (array ndim == 1); "
                "index a single shell/bin first (e.g. field[i])."
            )
        coef, tab_norm = starlet_coefficients_spherical(self.array, nside=self.nside, nscales=nscales)
        coeffs = StarletCoefficients(
            array=jnp.asarray(coef),
            nscales=nscales,
            tab_norm=jnp.asarray(tab_norm),
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            observer_position=self.observer_position,
            field_sharding=self.field_sharding,
            halo_size=self.halo_size,
            nside=self.nside,
            flatsky_npix=self.flatsky_npix,
            field_size=self.field_size,
            z_sources=self.z_sources,
            scale_factors=self.scale_factors,
            comoving_centers=self.comoving_centers,
            density_width=self.density_width,
            status=self.status,
            unit=self.unit,
            name=self.name,
        )
        return coeffs.normalized if normalize else coeffs

    @classmethod
    def full_like(cls, field: AbstractField, fill_value: float = 0.0) -> SphericalDensity:
        """
        Create a new DensityField with the same metadata as `field`
        and an array filled with `fill_value`.
        """
        if field.nside is None:
            raise ValueError("nside metadata is required to create full_like SphericalDensity.")

        return cls.FromDensityMetadata(
            array=jnp.full((jhp.nside2npix(field.nside),), fill_value),
            field=field,
        )

    def is_batched(self) -> bool:
        """Return True if the SphericalDensity has a leading batch dimension (S or N×S)."""
        return self.array.ndim in (2, 3)

    def is_multi_batched(self) -> bool:
        """Return True when the field has both a simulation-batch (N) and snapshot (S) dimension."""
        return self.array.ndim == 3


class StarletCoefficients(SphericalDensity):
    """Spherical starlet wavelet coefficients — one HEALPix map per scale.

    A :class:`SphericalDensity` whose batch axis is the wavelet scale, so the array has shape
    ``(nscales, npix)`` and the inherited ``.plot()`` (healpy mollview), ``.stack()`` and field
    IO/catalog machinery all apply. ``tab_norm`` is the per-scale normalisation returned by the
    CosmoStat transform; :attr:`normalized` rescales each scale by it. Produced by
    :meth:`SphericalDensity.starlet_coefficients`.
    """

    nscales: int = eqx.field(static=True, default=0)
    tab_norm: jax.Array | None = None

    @property
    def normalized(self) -> StarletCoefficients:
        """Each scale divided by its ``tab_norm`` (zeros left unscaled). Identity if no norm."""
        if self.tab_norm is None:
            return self
        norm = jnp.where(self.tab_norm == 0, 1.0, self.tab_norm)
        return self.replace(array=self.array / norm[:, None])

    def plot(self, *, titles: str | Sequence[str] | None = None, **kwargs):
        """Mollview one panel per starlet scale (titles default to ``"Starlet scale i"``)."""
        if titles is None:
            titles = [f"Starlet scale {i + 1}" for i in range(self.array.shape[0])]
        return super().plot(titles=titles, **kwargs)
