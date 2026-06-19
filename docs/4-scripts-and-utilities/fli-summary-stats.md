# fli-summary-stats

Compute summary statistics from Parquet catalogs produced by `fli-simulate` (or the loaders in
[notebook 10](../2-advanced-usage/10-External-Catalog.ipynb)). It scans a folder, detects each
field type, converts to overdensity where needed, and applies the matching estimator:

| Field type | Statistic |
|------------|-----------|
| `DensityField` | 3-D $P(k)$ |
| `FlatDensity` / `FlatKappaField` | flat-sky angular $C_\ell$ |
| `SphericalDensity` / `SphericalKappaField` | HEALPix angular $C_\ell$ (optionally footprint-masked) |

Outputs are written next to each input with a `summary_stats_` prefix and round-trip through
`jfli.io.Catalog.from_parquet`.

> Higher-order, map-based statistics (one-point **PDF**, **peak counts**, spherical **starlet**
> wavelets) on spherical maps are planned — see the `TODO(summary-stats)` in
> `scripts/entry/fli_summary_stats.py`; they land with the `jax_fli.summary_statistics` module.

## Usage

```bash
fli-summary-stats out/                       # all .parquet under out/
fli-summary-stats out/ --mask des_y3         # spherical maps masked to the DES Y3 footprint
```

## Spherical footprint mask

For spherical (HEALPix) fields, `--mask` restricts the observed footprint before computing the
$C_\ell$ (returned as MCM-decoupled bandpowers when masked):

- `infer_from_observer_position` (default) — the apodized observer-visibility mask built from the
  field's stored observer position; a **no-op for a centered observer** (whole sky).
- `none` — no masking.
- `des_y3` — the DES Y3 footprint.
- a path to a `.npy` / `.npz` / `.fits` HEALPix map.

The mask is apodized with a C2 window of `--apodization-scale-deg` (default 1°). Use
`--observer-position OX OY OZ` to override the position read from the field metadata.

## Key arguments

The parser composes one group per concern (`add_summary_stats_scan_args`, `_flat_`, `_spherical_`,
`_density_`, `_mask_`, `_common_`, and the shared `add_common_args`): a folder/regex to scan, the
estimator method (`healpy` vs `jax`), `--lmax`, $\ell$/$k$ binning, deconvolution / shot-noise
options (`--compensate-order` / `--shotnoise-order`), and the mask/apodization above. Run
`fli-summary-stats --help` for the complete set.

The 3D corrections are demonstrated interactively in
[notebook 2 (deconvolution + shot noise)](../1-introduction-and-basics/02-LPT-Simulation.ipynb)
and [notebook 3](../1-introduction-and-basics/03-PM-Simulation.ipynb).
