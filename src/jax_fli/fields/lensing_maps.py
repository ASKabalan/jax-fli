from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

import jax
import jax.numpy as jnp

from .._src.base._enums import FieldStatus, SpectralUnit
from .lightcone import FlatDensity, SphericalDensity
from .units import ConvergenceUnit, convert_units

if TYPE_CHECKING:
    from ..summary_statistics.power_spec import PowerSpectrum

# NOTE: Kaiser-Squires (.._src.lensing) and the power estimators (..summary_statistics) are imported lazily
# inside the methods below. lensing_maps is imported during ``jax_fli.fields`` initialization, and
# ``.._src.lensing`` pulls in ``_born`` -> ``...fields`` (a cycle); deferring the imports to call
# time avoids it (all modules are loaded by then).

__all__ = [
    "FlatKappaField",
    "SphericalKappaField",
    "FlatShearField",
    "SphericalShearField",
]


# --------------------------------------------------------------------------- #
# Flat-sky convergence                                                        #
# --------------------------------------------------------------------------- #
class FlatKappaField(FlatDensity):
    """
    Convergence map in flat-sky (Cartesian) geometry.

    Inherits from FlatDensity, adding lensing-specific methods for
    shear derivation.

    Attributes
    ----------
    z_sources : float | jnp.ndarray | Any
        Source redshift(s) for the convergence map.
    array : Array
        Convergence values, shape (ny, nx) or (n_planes, ny, nx).
    """

    def to(
        self,
        unit: ConvergenceUnit,
    ) -> FlatKappaField:
        """
        Convert convergence units (currently a no-op, but wired via convert_units).

        Both DIMENSIONLESS and EFFECTIVE_DENSITY are numerically κ = Σ/Σ_crit;
        this is mostly semantic for now.
        """
        if self.unit == unit:
            return self

        new_array = convert_units(
            array=self.array,
            origin=self.unit,
            destination=unit,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            # convergence conversions currently ignore these
            h=None,
            omega_m=None,
            mean_density=None,
            volume_element=None,
            field_sharding=self.field_sharding,
        )
        return self.replace(array=new_array, unit=unit)

    @jax.jit(static_argnames=["reduced_shear"])
    def get_shear(self, *, reduced_shear: bool = False) -> FlatShearField:
        """Compute shear ``(gamma1, gamma2)`` from convergence via flat-sky Kaiser-Squires.

        Maps ``array`` of shape ``(ny, nx)`` / ``(S, ny, nx)`` / ``(N, S, ny, nx)`` to a
        ``FlatShearField`` with a spin-2 axis inserted before the spatial axes: ``(2, ny, nx)`` /
        ``(S, 2, ny, nx)`` / ``(N, S, 2, ny, nx)``. Jittable; the shear is pure E-mode and needs no
        field/pixel size (the flat spin-2 operator is scale-invariant).

        If ``reduced_shear`` is True, returns the reduced shear ``g = gamma / (1 - kappa)`` (the
        quantity actually observed in weak-lensing surveys) instead of ``gamma``; ``kappa`` is the
        convergence in ``self.array``, broadcast over the spin-2 component axis.
        """
        from .._src.lensing import kappa2shear_flat

        shear = kappa2shear_flat(self.array)
        if reduced_shear:
            shear = shear / (1.0 - jnp.expand_dims(self.array, axis=-3))
        return FlatShearField.FromDensityMetadata(
            array=shear,
            field=self,
            status=FieldStatus.GAMMA,
            unit=self.unit,
        )


# --------------------------------------------------------------------------- #
# Spherical convergence                                                       #
# --------------------------------------------------------------------------- #
class SphericalKappaField(SphericalDensity):
    """
    Convergence map in spherical (HEALPix) geometry.

    Inherits from SphericalDensity, adding lensing-specific methods for
    shear derivation.

    Attributes
    ----------
    z_sources : float | jnp.ndarray | Any
        Source redshift(s) for the convergence map.
    array : Array
        Convergence values, shape (npix,) or (n_planes, npix).
    """

    def to(
        self,
        unit: ConvergenceUnit,
    ) -> SphericalKappaField:
        """
        Convert convergence units (currently a no-op, but wired via convert_units).
        """
        if self.unit == unit:
            return self

        new_array = convert_units(
            array=self.array,
            origin=self.unit,
            destination=unit,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            h=None,
            omega_m=None,
            mean_density=None,
            volume_element=None,
            field_sharding=self.field_sharding,
        )
        return self.replace(array=new_array, unit=unit)

    @jax.jit(static_argnames=["reduced_shear", "lmax", "method", "iter", "debug_sharding"])
    def get_shear(
        self,
        *,
        reduced_shear: bool = False,
        lmax: int | None = None,
        method: str = "jax",
        iter: int = 3,
        debug_sharding: bool = False,
    ) -> SphericalShearField:
        """Compute shear ``(gamma1, gamma2)`` from convergence via Kaiser-Squires (pure E-mode).

        Maps ``array`` of shape ``(npix,)`` / ``(S, npix)`` / ``(N, S, npix)`` to a
        ``SphericalShearField`` with a trailing spin-2 axis: ``(2, npix)`` / ``(S, 2, npix)`` /
        ``(N, S, 2, npix)``. Jittable; ``lmax`` defaults to ``3*nside-1``. ``iter`` sets the
        ``map2alm`` Jacobi iterations (default 3).

        When ``field_sharding`` spans multiple devices the per-bin transform runs inside a
        ``jax.shard_map`` and the **shear mirrors the convergence sharding**: convergence ``P(b, x)``
        gives shear ``P(b, None, x)`` (``b`` = bins axis, ``x`` = npix axis). If the convergence is
        sharded on a bins axis (``b`` a real mesh axis, e.g. ``P("y", "x")``) the bins distribute
        across devices; otherwise the SHT is replicated across devices (a warning is emitted) but the
        shear is still returned sharded on ``npix`` like the convergence. The result is identical to
        the single-device path. ``debug_sharding`` prints the shape and sharding at each stage
        (works under ``jax.jit``).

        If ``reduced_shear`` is True, returns the reduced shear ``g = gamma / (1 - kappa)`` (the
        quantity actually observed in weak-lensing surveys) instead of ``gamma``; ``kappa`` is the
        convergence in ``self.array``, broadcast over the spin-2 component axis.
        """
        from .._src.lensing import kappa2shear_spherical

        shear = kappa2shear_spherical(
            self.array,
            lmax=lmax,
            method=method,
            iter=iter,
            sharding=self.field_sharding,
            debug_sharding=debug_sharding,
        )
        if reduced_shear:
            shear = shear / (1.0 - jnp.expand_dims(self.array, axis=-2))
        return SphericalShearField.FromDensityMetadata(
            array=shear,
            field=self,
            status=FieldStatus.GAMMA,
            unit=self.unit,
        )

    def apply_sharding(self) -> SphericalKappaField:
        """Shard the spherical convergence into the lensing layout ``P([None,] "y", "x")`` (BINS/N,
        NPIX/M; batch axes replicated), keeping ``field_sharding`` the canonical 3-D mesh layout.

        Warns when the mesh's second (N) axis has size 1 and there is more than one bin (lensing
        will not distribute over tomographic bins); raises ``ValueError`` for a mis-shaped mesh whose
        N axis does not divide the bin count. No-op on a single device or a metadata-only field.
        """
        from .._src.lensing._sharding import _convergence_spec, _is_multi_device, lensing_axes

        if not _is_multi_device(self.field_sharding) or self.array is None:
            return self
        arr = self.array
        nbins = arr.shape[-2] if arr.ndim >= 2 else 1
        _, _, _, n_size, _ = lensing_axes(self.field_sharding, nbins)  # raises the bad-divisor case
        if n_size == 1 and nbins > 1:  # nbins == 1 has nothing to distribute regardless of the mesh
            warn(
                "convergence sharding: the mesh's second (N) axis has size 1, so the lensing "
                "convergence and shear will not be distributed over tomographic bins. Use a "
                "(devices // n_bins, n_bins) mesh to distribute lensing over bins.",
                stacklevel=2,
            )
        return self.replace(
            array=jax.lax.with_sharding_constraint(arr, _convergence_spec(self.field_sharding, nbins, arr.ndim))
        )


# --------------------------------------------------------------------------- #
# Flat-sky shear                                                              #
# --------------------------------------------------------------------------- #
class FlatShearField(FlatDensity):
    """
    Shear map (γ1, γ2) in flat-sky (Cartesian) geometry.

    The spin-2 component axis sits **immediately before the two spatial axes**: valid array shapes
    are ``(2, ny, nx)`` (single), ``(S, 2, ny, nx)`` (per source bin), or ``(N, S, 2, ny, nx)``
    (realisations × source bins). Leading axes are batch.
    """

    # Allow the extra component axis: (2, ny, nx) / (S, 2, ny, nx) / (N, S, 2, ny, nx).
    _MAX_ARRAY_NDIM = 5

    def __check_init__(self):
        if self.array is not None and getattr(self.array, "shape", ()) != ():
            shape = self.array.shape
            if len(shape) not in (3, 4, 5):
                raise ValueError(
                    f"FlatShearField array must be (2, ny, nx), (S, 2, ny, nx), or (N, S, 2, ny, nx), "
                    f"got {len(shape)}D array with shape {shape}."
                )
            if shape[-3] != 2:
                raise ValueError(
                    f"FlatShearField requires a spin-2 component axis of size 2 at position -3, got shape {shape}."
                )

    def is_batched(self) -> bool:
        """True if there is a leading batch axis beyond the ``(2, ny, nx)`` spin-2 map."""
        return self.array.ndim in (4, 5)

    def is_multi_batched(self) -> bool:
        """True for a two-level batch ``(N, S, 2, ny, nx)``."""
        return self.array.ndim == 5

    def __getitem__(self, key) -> FlatShearField:
        # Indexing must not slice the spin-2 component axis: only a leading batch axis is indexable.
        if self.array.ndim < 4:
            warn(
                f"Indexing only supported for batched FlatShearField (4D or 5D array), got array with "
                f"{self.array.ndim} dimensions. Returning self without indexing."
            )
            return self
        return super().__getitem__(key)

    # Spin-2 maps are not scalar fields: the inherited FlatDensity spectral/resample helpers treat the
    # ``(..., 2, ny, nx)`` component axis as a batch and would silently return wrong results, so they
    # are disabled. (Spin-2 angular power is implemented for the HEALPix ``SphericalShearField`` only.)
    def angular_cl(self, *args, **kwargs):
        raise NotImplementedError(
            "FlatShearField has no spin-2 angular_cl; flat-sky shear E/B power is not implemented. "
            "Use SphericalShearField for spin-2 angular power spectra."
        )

    def cross_angular_cl(self, *args, **kwargs):
        raise NotImplementedError("FlatShearField has no spin-2 cross_angular_cl (not implemented).")

    def transfer(self, *args, **kwargs):
        raise NotImplementedError("FlatShearField has no spin-2 transfer (not implemented).")

    def coherence(self, *args, **kwargs):
        raise NotImplementedError("FlatShearField has no spin-2 coherence (not implemented).")

    def ud_sample(self, *args, **kwargs):
        raise NotImplementedError(
            "FlatShearField.ud_sample is not implemented (the inherited resampler mishandles the "
            "spin-2 component axis)."
        )

    def to(
        self,
        unit: ConvergenceUnit,
    ) -> FlatShearField:
        """
        Convert shear units (currently just semantic – same as convergence units).
        """
        if self.unit == unit:
            return self

        new_array = convert_units(
            array=self.array,
            origin=self.unit,
            destination=unit,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            h=None,
            omega_m=None,
            mean_density=None,
            volume_element=None,
            field_sharding=self.field_sharding,
        )
        return self.replace(array=new_array, unit=unit)

    def get_convergence(self) -> FlatKappaField:
        """Compute convergence from shear via inverse flat-sky Kaiser-Squires.

        Maps ``(2, ny, nx)`` / ``(S, 2, ny, nx)`` / ``(N, S, 2, ny, nx)`` to a ``FlatKappaField`` of
        shape ``(ny, nx)`` / ``(S, ny, nx)`` / ``(N, S, ny, nx)``. Jittable. The recovered map has
        zero mean (the DC mode is unconstrained), so for band-limited convergence it reproduces
        ``kappa - mean(kappa)`` (the even-grid Nyquist row/column is not reconstructed).
        """
        from .._src.lensing import shear2kappa_flat

        kappa = shear2kappa_flat(self.array)
        return FlatKappaField.FromDensityMetadata(
            array=kappa,
            field=self,
            status=FieldStatus.KAPPA,
            unit=self.unit,
        )


# --------------------------------------------------------------------------- #
# Spherical shear                                                             #
# --------------------------------------------------------------------------- #
class SphericalShearField(SphericalDensity):
    """
    Shear map (γ1, γ2) in spherical (HEALPix) geometry.

    The **trailing two axes** are the spin-2 component and the pixels: valid array shapes are
    ``(2, npix)`` (single), ``(S, 2, npix)`` (per source bin), or ``(N, S, 2, npix)`` (realisations
    × source bins). Leading axes are batch.
    """

    # Allow the extra trailing spin-2 axis: (2, npix) / (S, 2, npix) / (N, S, 2, npix).
    _MAX_ARRAY_NDIM = 4

    def __check_init__(self):
        if self.array is not None and getattr(self.array, "shape", ()) != ():
            shape = self.array.shape
            if len(shape) not in (2, 3, 4):
                raise ValueError(
                    f"SphericalShearField array must be (2, npix), (S, 2, npix), or (N, S, 2, npix), "
                    f"got {len(shape)}D array with shape {shape}."
                )
            if shape[-2] != 2:
                raise ValueError(
                    f"SphericalShearField requires a spin-2 component axis of size 2 at position -2, got shape {shape}."
                )

    def is_batched(self) -> bool:
        """True if there is a leading batch axis beyond the ``(2, npix)`` spin-2 map."""
        return self.array.ndim in (3, 4)

    def is_multi_batched(self) -> bool:
        """True for a two-level batch ``(N, S, 2, npix)``."""
        return self.array.ndim == 4

    def __getitem__(self, key) -> SphericalShearField:
        # Indexing must not slice the spin-2 component axis: only a leading batch axis is indexable.
        if self.array.ndim < 3:
            warn(
                f"Indexing only supported for batched SphericalShearField (3D or 4D array), got array "
                f"with {self.array.ndim} dimensions. Returning self without indexing."
            )
            return self
        return super().__getitem__(key)

    def to(
        self,
        unit: ConvergenceUnit,
    ) -> SphericalShearField:
        """
        Convert shear units (currently just semantic – same as convergence units).
        """
        if self.unit == unit:
            return self

        new_array = convert_units(
            array=self.array,
            origin=self.unit,
            destination=unit,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            h=None,
            omega_m=None,
            mean_density=None,
            volume_element=None,
            field_sharding=self.field_sharding,
        )
        return self.replace(array=new_array, unit=unit)

    def get_convergence(self, *, lmax: int | None = None, method: str = "jax", iter: int = 3) -> SphericalKappaField:
        """Compute convergence from shear via inverse Kaiser-Squires (uses the E-mode).

        Maps ``(2, npix)`` / ``(S, 2, npix)`` / ``(N, S, 2, npix)`` to a ``SphericalKappaField`` of
        shape ``(npix,)`` / ``(S, npix)`` / ``(N, S, npix)``. Jittable. ``iter`` sets the
        ``map2alm_spin`` Jacobi iterations (default 3).
        """
        from .._src.lensing import shear2kappa_spherical

        kappa = shear2kappa_spherical(self.array, lmax=lmax, method=method, iter=iter)
        return SphericalKappaField.FromDensityMetadata(
            array=kappa,
            field=self,
            status=FieldStatus.KAPPA,
            unit=self.unit,
        )

    def apply_sharding(self) -> SphericalShearField:
        """Shard the spherical shear into the lensing layout ``P([None,] "y", None, "x")`` (BINS/N,
        component replicated, NPIX/M), keeping ``field_sharding`` the canonical 3-D mesh layout.

        Mirrors the convergence layout with the spin-2 component axis replicated. Silent — only raises
        ``ValueError`` for a mis-shaped mesh whose N axis does not divide the bin count (the N=1
        *warning* belongs to the convergence/producer, not the shear). No-op on a single device or a
        metadata-only field. (``get_shear`` already returns sharded shear; this serves the catalog
        load path, where a shear field is rebuilt and re-distributed.)
        """
        from .._src.lensing._sharding import _is_multi_device, _shear_spec

        if not _is_multi_device(self.field_sharding) or self.array is None:
            return self
        arr = self.array
        nbins = arr.shape[-3] if arr.ndim >= 3 else 1
        return self.replace(
            array=jax.lax.with_sharding_constraint(arr, _shear_spec(self.field_sharding, nbins, arr.ndim - 1))
        )

    def angular_cl(
        self,
        *,
        mask=None,
        lmax: int | None = None,
        method: str = "jax",
        purify_e: bool = False,
        purify_b: bool = False,
        mcm=None,
        nlb: int = 16,
    ) -> PowerSpectrum:
        """Spin-2 angular power spectrum (EE, EB, BB) — ``pol=True`` automatically.

        Returns a 3-component :class:`PowerSpectrum` (``n_components=3``): ``(3, n_ell)`` for a
        single ``(2, npix)`` map, ``(B, 3, n_ell)`` for batched (leading dims flattened to ``B``).

        * ``mask=None`` → plain / coupled pseudo-``C_l`` (pre-multiply the map by the apodized mask
          for the forward-model coupled-pseudo path); ``ell`` runs ``0..lmax``.
        * ``mask`` given → masked pseudo-``C_l`` **decoupled** into bandpowers (``ell`` = effective
          bandpower multipoles); ``purify_e``/``purify_b`` applied. EE/BB are decoupled; EB is the
          binned (coupled) pseudo. The MCM is built once from ``mask`` and reused across the batch
          (or pass a precomputed ``mcm``).
        """
        from ..summary_statistics.decouple import anafast_masked, compute_mcm
        from ..summary_statistics.power_spec import PowerSpectrum

        data = self.array
        npix = data.shape[-1]
        _lmax = lmax if lmax is not None else 3 * self.nside - 1
        single = data.ndim == 2
        flat = data.reshape((-1, 2, npix))

        _mcm = mcm
        if mask is not None and _mcm is None:
            _mcm = compute_mcm(mask, lmax=_lmax, nlb=nlb, pol=True, method=method)

        def _one(g):
            return anafast_masked(
                g,
                mask=mask,
                lmax=_lmax,
                method=method,
                pol=True,
                purify_e=purify_e,
                purify_b=purify_b,
                mcm=_mcm,
                nlb=nlb,
            )[1]

        cls = jax.vmap(_one)(flat)  # (B, 3, n)
        if mask is not None:
            assert _mcm is not None
            ell = _mcm.ell_eff
        else:
            ell = jnp.arange(_lmax + 1) * 1.0
        array = cls[0] if single else cls
        return PowerSpectrum(
            name=self.name,
            wavenumber=ell,
            array=array,
            n_components=3,
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
