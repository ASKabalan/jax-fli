"""Utilities for converting MCMC sample dicts to Catalog format."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from ..fields import DensityField
from ..fields.lensing_maps import FlatKappaField, FlatShearField, SphericalKappaField, SphericalShearField
from ..io import Catalog

if TYPE_CHECKING:
    from ..probabilistic_models.config import Configurations

__all__ = ["sample2catalog", "colour_ic"]


def _append_metrics_row(metrics: dict, batch_id: int, base_path: str) -> None:
    """Append one row to {base_path}/metrics.md, creating headers on first write."""
    if metrics is None:
        metrics = {
            "mean_num_steps": None,
            "total_divergences": None,
            "mean_accept_prob": None,
        }
    md_path = os.path.join(base_path, "metrics.md")
    headers = (
        "| Batch | Avg Steps | Divergences | Mean Accept |",
        "|-------|-----------|-------------|-------------|",
    )
    row = "| {batch} | {steps} | {div} | {acc} |".format(
        batch=batch_id,
        steps=f"{metrics['mean_num_steps']:.1f}" if metrics["mean_num_steps"] is not None else "N/A",
        div=str(metrics["total_divergences"]) if metrics["total_divergences"] is not None else "N/A",
        acc=f"{metrics['mean_accept_prob']:.3f}" if metrics["mean_accept_prob"] is not None else "N/A",
    )
    write_headers = not os.path.exists(md_path)
    with open(md_path, "a") as f:
        if write_headers:
            f.write("\n".join(headers) + "\n")
        f.write(row + "\n")


def default_save(samples, path, batch_id, metrics=None):
    """Default save callback that just saves the samples as an orbax checkpoint."""
    save_path = os.path.join(path, "samples")
    os.makedirs(save_path, exist_ok=True)
    base_path = os.path.dirname(path)
    _append_metrics_row(metrics, batch_id, base_path)
    np.savez(os.path.join(path, f"samples_batch_{batch_id}.npz"), **samples)


def colour_ic(config: Configurations):
    """Build a per-sample ``post_process`` that colours the white ``initial_conditions`` field.

    Mirrors :func:`sample2catalog`: a ``config``-closing factory. The returned transform is passed as
    the ``post_process`` argument of :func:`~jax_fli.infer.batched_sampling` (applied once per sample
    inside the sampling scan); it recolours the white ``initial_conditions`` to the physical density
    field with the sample's own ``cosmo`` via :func:`~jax_fli.initial.interpolate_initial_conditions`.
    """
    from ..initial import interpolate_initial_conditions

    def recolour(sample: dict) -> dict:
        return {
            **sample,
            "initial_conditions": interpolate_initial_conditions(
                sample["initial_conditions"],
                config.mesh_size,
                config.box_size,
                cosmo=sample["cosmo"],
                field_sharding=config.field_sharding,
            ).array,
        }

    return recolour


#
def sample2catalog(config: Configurations):
    """Build a save callback that writes both orbax checkpoints and parquet Catalogs.

    The returned callback is suitable for the ``save_callback`` parameter of
    :func:`~jax_fli.sampling.batched_sampling`.

    Parameters
    ----------
    config : Configurations
        The same configuration object used to build the probabilistic model.

    Returns
    -------
    cb : Callable[[dict, dict, str, int], None]
        Callback with signature ``(samples, metrics, path, batch_id)``.
    """

    is_spherical = config.geometry == "spherical"
    # The observable can be convergence (kappa) or shear / reduced shear; pick the matching field class.
    if config.lensing_output == "convergence":
        ObservableFieldCls = SphericalKappaField if is_spherical else FlatKappaField
    else:
        ObservableFieldCls = SphericalShearField if is_spherical else FlatShearField

    def cb(samples, path, batch_id, metrics=None):
        """Save orbax checkpoint and parquet Catalogs for one batch.

        Parameters
        ----------
        samples : dict
            Sample dictionary from the MCMC run.
        metrics : dict
            Aggregated sampler diagnostics (mean_num_steps, total_divergences, mean_accept_prob).
        path : str
            Full orbax checkpoint path (e.g. ``"output/run/samples"``).
        batch_id : int
            Integer batch index.
        """
        base_path = os.path.dirname(path)
        _append_metrics_row(metrics, batch_id, base_path)

        cosmo = samples["cosmo"]
        initial_conditions = samples.get("initial_conditions")

        if initial_conditions is None:
            # Power-spectrum model: no IC field, save cosmo params as npz instead
            print("No initial conditions found, saving cosmo parameters to npz.")
            cosmo_dir = os.path.join(path, "samples")
            os.makedirs(cosmo_dir, exist_ok=True)
            cosmo_dict = {
                "Omega_c": cosmo.Omega_c,
                "Omega_b": cosmo.Omega_b,
                "h": cosmo.h,
                "n_s": cosmo.n_s,
                "sigma8": cosmo.sigma8,
                "w0": cosmo.w0,
                "wa": cosmo.wa,
                "Omega_k": cosmo.Omega_k,
                "Omega_nu": cosmo.Omega_nu,
            }
            np.savez(os.path.join(cosmo_dir, f"cosmo_{batch_id}.npz"), **cosmo_dict)
        else:
            # Save the IC samples as a parquet Catalog
            ic_dir = os.path.join(path, "samples")
            os.makedirs(ic_dir, exist_ok=True)

            intitial_condition_meta_data = samples["initial_conditions_meta_data"]
            if not isinstance(initial_conditions, DensityField):
                initial_conditions = DensityField.FromDensityMetadata(
                    array=initial_conditions,
                    field=intitial_condition_meta_data,
                )

            initial_conditions = initial_conditions.replace(
                name=f"initial_conditions_batch_{batch_id}",
                z_sources=jnp.zeros(initial_conditions.shape[0]),
                comoving_centers=jnp.zeros(initial_conditions.shape[0]),
                scale_factors=jnp.zeros(initial_conditions.shape[0]),
                density_width=jnp.zeros(initial_conditions.shape[0]),
            )

            sample_catalog = Catalog(field=initial_conditions, cosmology=cosmo)
            sample_catalog.to_parquet(os.path.join(ic_dir, f"samples_{batch_id}.parquet"))

        # Check if samples has lensing observables (convergence or shear)
        if "observable_0" not in samples:
            print("No observable samples found, skipping observable catalog saving.")
            return
        fields_dir = os.path.join(path, "observable_fields")
        os.makedirs(fields_dir, exist_ok=True)
        # find out how many tomographic bins there are by counting keys
        observable_keys = [k for k in samples if k.startswith("observable_") and k.split("_")[-1].isdigit()]
        n_bins = len(observable_keys)
        # Create the observable fields class
        observable_meta_data = samples["observable_meta_data"]
        cosmo_arr = np.asarray(cosmo.Omega_c)
        if cosmo_arr.ndim > 0:
            # Batched cosmology (Predictive mode): build one catalog entry per sample
            n_samples = int(cosmo_arr.size)
            fields_list = []
            cosmo_list = []
            for s_idx in range(n_samples):
                cosmo_s = jax.tree.map(lambda p: p[s_idx], cosmo)
                meta_s = observable_meta_data[s_idx]
                observable_s = jnp.stack([samples[f"observable_{i}"][s_idx] for i in range(n_bins)], axis=0)
                fields_list.append(
                    ObservableFieldCls.FromDensityMetadata(
                        array=observable_s, field=meta_s, name=f"observable_fields_batch_{batch_id}_sample_{s_idx}"
                    )
                )
                cosmo_list.append(cosmo_s)
            observable_catalog = Catalog(field=fields_list, cosmology=cosmo_list)
        else:
            observable_array = jnp.stack([samples[f"observable_{i}"] for i in range(n_bins)], axis=0)
            observable_field = ObservableFieldCls.FromDensityMetadata(
                array=observable_array, field=observable_meta_data, name=f"observable_fields_batch_{batch_id}"
            )
            observable_catalog = Catalog(field=observable_field, cosmology=cosmo)
        observable_catalog.to_parquet(os.path.join(fields_dir, f"fields_{batch_id}.parquet"))

        if "lightcone" in samples:
            lightcone_dir = os.path.join(path, "lightcones")
            os.makedirs(lightcone_dir, exist_ok=True)
            lightcone = samples["lightcone"]
            lightcone = lightcone.replace(name=f"lightcone_batch_{batch_id}")
            lightcone_catalog = Catalog(field=lightcone, cosmology=cosmo)
            lightcone_catalog.to_parquet(os.path.join(lightcone_dir, f"lightcone_{batch_id}.parquet"))

    return cb
