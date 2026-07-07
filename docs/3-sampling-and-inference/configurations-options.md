# `Configurations` — forward & probabilistic model options

Every option consumed by the full-field forward model
(`make_full_field_model`) and the probabilistic model
(`full_field_probmodel`) lives on the `Configurations` dataclass
(`jax_fli.probabilistic_models.config`). This page lists them by section
with their types and defaults.

```python
import jax_fli as jfli
config = jfli.ppl.Configurations(
    mesh_size=(128, 128, 128),
    box_size=(1000.0, 1000.0, 1000.0),
    fiducial_cosmology=jax_cosmo.Planck18,
    nz_shear=jfli.data.get_des_y3_nz_shear(),
    priors={"Omega_c": jfli.infer.PreconditionnedUniform(0.1, 0.5)},
    sigma_e=0.26,
    nside=512,
)
model = jfli.ppl.full_field_probmodel(config)
```

## Mandatory parameters

| Option | Type | Description |
|--------|------|-------------|
| `mesh_size` | `tuple[int, int, int]` | Initial-conditions grid resolution. |
| `box_size` | `tuple[float, float, float]` | Simulation box in Mpc/h. Must be large enough to reach the source redshifts. |
| `fiducial_cosmology` | `Callable` | Factory called as `fiducial_cosmology(**sampled_priors) -> jax_cosmo.Cosmology` (e.g. `jax_cosmo.Planck18`). |
| `nz_shear` | `list` | Source redshift distribution per tomographic bin (floats are wrapped as `delta_nz`). |
| `priors` | `dict[str, dist]` | `{parameter_name: prior}` sampled into `fiducial_cosmology`. |
| `sigma_e` | `float` | Intrinsic ellipticity dispersion (sets the per-pixel shape-noise σ). |

## Geometry & initial conditions

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `geometry` | `str` | `"spherical"` | `"spherical"` (HEALPix) or `"flat"` (Cartesian flat-sky). |
| `nside` | `int \| None` | `None` | HEALPix resolution (spherical). |
| `flatsky_npix` | `tuple[int, int] \| None` | `None` | Flat-sky grid `(ny, nx)`. |
| `field_size` | `tuple[float, float] \| None` | `None` | Flat-sky field of view in degrees. |
| `halo_size` | `tuple[float, float]` | `(0, 0)` | Halo padding for periodic boundaries. |
| `observer_position` | `tuple[float, float, float]` | `(0.5, 0.5, 0.5)` | Observer in normalized box coords. The center sees the whole sky; **any other position triggers an apodized visibility mask** on the observable (spherical only). See [Masking & likelihood](#masking--likelihood). |
| `field_sharding` | `Any` | `None` | JAX sharding for distributed arrays. |
| `log_lightcone` | `bool` | `False` | Register the lightcone as a deterministic site. |

## LPT

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `lpt_order` | `int` | `2` | LPT order (1 or 2). |
| `t0` | `float` | `0.01` | Initial scale factor (LPT snapshot; also the solver start). |
| `paint_order` | `str` | `"cic"` | Mass assignment: NGP / CIC / TSC / PCS. Shared with the solver. |
| `gradient_order` | `int` | `1` | Force gradient order (0 = exact `ik`, 4 = 4th-order FD). Shared with the solver. |
| `laplace_fd` | `bool` | `False` | Finite-difference Laplacian for the force. Shared with the solver. |
| `dealiased` | `bool` | `False` | Dealias the painted density in Fourier space (LPT only). |
| `exact_growth` | `bool` | `False` | Use exact (ODE) growth factors (LPT only). |

## N-body solver

`DoubleKickDrift` (reversible, uniform a-stepping) is the default; the
correction kernel is fixed to `NoCorrection` (no PGD/Sharpening), the
interpolation is `DriftInterp` when `drift_on_lightcone=True` else
`NoInterp` (OnionTiler / TelephotoInterp are not exposed here).

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `sim_mode` | `str` | `"pm"` | Pipeline depth: `"pm"` (LPT → N-body lightcone) or `"lpt"` (LPT-only lightcone, no N-body). With `"lpt"` the N-body options below are inert. |
| `nbody_solver` | `str` | `"DoubleKickDrift"` | `"DoubleKickDrift"` / `"DriftKickDrift"` / `"BullFrog"`. |
| `nb_steps` | `int` | `100` | Number of integration steps (must be ≥ `number_of_shells`). |
| `t1` | `float` | `1.0` | Final scale factor. |
| `number_of_shells` | `int` | `8` | Lightcone shells. |
| `shell_spacing` | `str` | `"comoving"` | Shell distribution: `"comoving"` or `"a"`. |
| `time_stepping` | `str` | `"D"` | Integrator time-stepping variable: `"D"` (growth-factor stepping) or `"a"` (uniform a-stepping). |
| `min_width` | `float` | `50.0` | Minimum shell width (Mpc/h comoving). |
| `drift_on_lightcone` | `bool` | `False` | Drift particles to their lightcone-crossing epoch (`DriftInterp`). Set `True` for lightcone experiments. |
| `deconvolution` | `bool` | `False` | Deconvolve the mass-assignment window (solver). |
| `adjoint` | `str` | `"checkpointed"` | `"checkpointed"` or `"reverse"` (reverse needs a reversible solver + `time_stepping='a'`). |
| `checkpoints` | `int` | `10` | Checkpoints for the checkpointed adjoint. |

`paint_order`, `gradient_order`, `laplace_fd` (see LPT) are also applied to the solver's force/painting for consistency.

## Painting

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `scheme` | `str` | `"bilinear"` | Lightcone painting interpolation scheme. |
| `paint_nside` | `int \| None` | `None` | HEALPix resolution for painting (defaults to `nside`). |
| `kernel_width_arcmin` | `float \| None` | `None` | Painting smoothing kernel width (arcmin). |
| `kernel_width_pixels` | `float \| None` | `None` | Painting smoothing kernel width (pixels). |

## Lensing & output

Lensing is **born only** (`raytrace` raises). The output observable is
selected by `lensing_output`.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `lensing` | `str` | `"born"` | Must be `"born"`. |
| `lensing_output` | `str` | `"convergence"` | `"convergence"` (κ, `(S, npix)`), `"shear"` (γ, `(S, 2, npix)`), or `"reduced_shear"` (g = γ/(1−κ)). |
| `min_redshift` | `float` | `0.01` | Lower bound of the born integration. |
| `max_redshift` | `float` | `1.5` | Upper bound of the born integration. |

## Masking & likelihood

Two independent masks:

- **Visibility mask** — built from `observer_position` (when off-center) via
  `jaxpm.spherical.spherical_visibility_mask`, apodized with
  `apodization_scale_deg`, and multiplied into the observable
  (KS-then-×apodize). Spherical geometry only.
- **Survey mask `mask`** (e.g. DES Y3) — used **only in the likelihood**: the
  per-pixel σ is inflated to `sigma_unobserved` where `mask == 0`, so
  unobserved pixels do not constrain the fit (map shapes stay fixed).

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `apodization_scale_deg` | `float` | `1.0` | C2 apodization scale (deg) for the observer visibility mask. |
| `mask` | `Array \| None` | `None` | Survey footprint, a `(npix,)` HEALPix mask at the model nside (or `None`). |
| `sigma_unobserved` | `float` | `1e3` | Likelihood σ on pixels with `mask == 0`. |

Observed per-pixel σ inside the footprint is `sigma_e / sqrt(n_gal · pixel_area_arcmin²)`.

## Power-spectrum model (not used by the full-field model)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `ells` | `Array` | `jnp.arange(2, 2048)` | Multipoles for the C(ℓ) model. |
| `f_sky` | `float` | `1.0` | Sky fraction for the Gaussian C(ℓ) covariance. |

## Sampler settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `sampler` | `str` | `"NUTS"` | Sampler type: `"NUTS"` or `"MCLMC"`. |
| `nuts_max_num_doublings` | `int` | `10` | NUTS leapfrog trajectory doubling depth (BlackJAX `max_tree_depth` analogue). |
| `nuts_target_accept` | `float` | `0.8` | NUTS window-adaptation target acceptance rate. |
| `mclmc_desired_energy_var` | `float` | `1e-3` | MCLMC desired energy variance for `L`/`step_size` tuning. |
| `mclmc_num_tune` | `int \| None` | `None` | MCLMC tuning steps; defaults to `num_warmup`. |
| `mclmc_init_step_size` | `float` | `1e-4` | MCLMC initial step scale factor. |
| `mclmc_diagonal_preconditioning` | `bool` | `False` | Enable diagonal preconditioning for MCLMC. |
