# fli-spectra

Compute power spectra from Parquet catalogs produced by `fli-simulate` (or the loaders in
[notebook 8](../2-advanced-usage/08-External-Catalog.ipynb)). It scans a folder, detects each
field type, converts to overdensity where needed, and applies the matching estimator:

| Field type | Spectrum |
|------------|----------|
| `DensityField` | 3-D $P(k)$ |
| `FlatDensity` / `FlatKappaField` | flat-sky angular $C_\ell$ |
| `SphericalDensity` / `SphericalKappaField` | HEALPix angular $C_\ell$ |

## Usage

```bash
fli-spectra --path out/ --output spectra/
```

It writes a spectra `Catalog` that round-trips through `jfli.io.Catalog.from_parquet`.

## Key arguments

The parser composes several groups (`add_spectra_scan_args`, `_flat_`, `_spherical_`,
`_density_`, `_common_`): a folder/regex to scan, the estimator method (e.g. `healpy` vs `jax`),
`--lmax`, $\ell$/$k$ binning, deconvolution / shot-noise options, and `--compensate-order`.
Run `fli-spectra --help` for the complete set.

The same corrections are demonstrated interactively in
[notebook 2 (deconvolution + shot noise)](../1-introduction-and-basics/02-LPT-Simulation.ipynb)
and [notebook 3](../1-introduction-and-basics/03-PM-Simulation.ipynb).
