from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from .._src.base._enums import FieldStatus, SpectralUnit
from .lightcone import FlatDensity, SphericalDensity
from .units import ConvergenceUnit, convert_units

if TYPE_CHECKING:
    from ..power.power_spec import PowerSpectrum

# NOTE: Kaiser-Squires (.._src.lensing) and the power estimators (..power) are imported lazily
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

    def get_shear(self, cosmo: Any | None = None):
        """
        Compute shear (γ1, γ2) from convergence via Kaiser-Squires inversion.
        """
        raise NotImplementedError("Shear computation from kappa (flat-sky) not implemented yet.")


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

    def get_shear(self, *, lmax: int | None = None, method: str = "jax") -> SphericalShearField:
        """Compute shear ``(gamma1, gamma2)`` from convergence via Kaiser-Squires (pure E-mode).

        Maps ``array`` of shape ``(npix,)`` / ``(S, npix)`` / ``(N, S, npix)`` to a
        ``SphericalShearField`` with a trailing spin-2 axis: ``(2, npix)`` / ``(S, 2, npix)`` /
        ``(N, S, 2, npix)``. Jittable; ``lmax`` defaults to ``3*nside-1``.
        """
        from .._src.lensing import kappa2shear

        shear = kappa2shear(self.array, lmax=lmax, method=method)
        return SphericalShearField.FromDensityMetadata(
            array=shear,
            field=self,
            status=FieldStatus.GAMMA,
            unit=self.unit,
        )


# --------------------------------------------------------------------------- #
# Flat-sky shear                                                              #
# --------------------------------------------------------------------------- #
class FlatShearField(FlatDensity):
    """
    Shear map (γ1, γ2) in flat-sky (Cartesian) geometry.

    By convention you can store γ1, γ2 either as:
      - separate FlatShearField instances, or
      - an extra leading / trailing dimension in `array`.
    """

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

    def get_convergence(self, cosmo: Any | None = None):
        """
        Compute convergence from shear via Kaiser-Squires inversion.
        """
        raise NotImplementedError("Convergence from shear (flat-sky) not implemented yet.")


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
                    f"SphericalShearField requires a spin-2 component axis of size 2 at position -2, "
                    f"got shape {shape}."
                )

    def is_batched(self) -> bool:
        """True if there is a leading batch axis beyond the ``(2, npix)`` spin-2 map."""
        return self.array.ndim in (3, 4)

    def is_multi_batched(self) -> bool:
        """True for a two-level batch ``(N, S, 2, npix)``."""
        return self.array.ndim == 4

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

    def get_convergence(self, *, lmax: int | None = None, method: str = "jax") -> SphericalKappaField:
        """Compute convergence from shear via inverse Kaiser-Squires (uses the E-mode).

        Maps ``(2, npix)`` / ``(S, 2, npix)`` / ``(N, S, 2, npix)`` to a ``SphericalKappaField`` of
        shape ``(npix,)`` / ``(S, npix)`` / ``(N, S, npix)``. Jittable.
        """
        from .._src.lensing import shear2kappa

        kappa = shear2kappa(self.array, lmax=lmax, method=method)
        return SphericalKappaField.FromDensityMetadata(
            array=kappa,
            field=self,
            status=FieldStatus.KAPPA,
            unit=self.unit,
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
        from ..power.decouple import anafast_masked, compute_mcm
        from ..power.power_spec import PowerSpectrum

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
        ell = _mcm.ell_eff if mask is not None else (jnp.arange(_lmax + 1) * 1.0)
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
