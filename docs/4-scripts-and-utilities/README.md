# Scripts & Utilities

`jax-fli` installs a set of console scripts (declared in `pyproject.toml`) that wrap the library
for batch and HPC use. Each one runs `jax.distributed.initialize()` *before* importing the
package, so the same command works on a single GPU or across many nodes. Every script accepts
`--help` for the complete option list.

| Script | Purpose |
|--------|---------|
| [`fli-simulate`](fli-simulate.md) | Run the IC → LPT / PM / lensing pipeline |
| [`fli-spectra`](fli-spectra.md) | Compute power spectra from Parquet catalogs |
| [`fli-born-rt`](fli-born-rt.md) | Add Born convergence to existing lightcones |
| [`fli-dorian-rt`](fli-dorian-rt.md) | Add dorian ray-traced convergence (MPI) |
| [`fli-samples`](fli-samples.md) | Draw prior-predictive samples from a model |
| [`fli-infer`](fli-infer.md) | Full-field MCMC inference on κ maps |
| [`fli-2pcf`](fli-2pcf.md) | Power-spectrum-level MCMC inference |
| [`fli-extract`](fli-extract.md) | Stream MCMC catalogs → per-chain statistics |
| [`fli-launcher`](fli-launcher.md) | SLURM dispatcher for grids of the above |

## Shared argument groups

Most scripts reuse the argument groups defined in `jax_fli.scripts.parser`:

- **cosmology** — `--Omega-c --Omega-b --h --n-s --sigma8 --w0 --wa --Omega-k --Omega-nu`
- **integration** — `--lpt-order {1,2} --t0 --t1 --solver {kdk,dkd,bf} --n-steps`
- **simulation** — `--mesh-size --box-size --halo-size --observer --seed`
- **lensing** — `--min-z --max-z --n-integrate`
- **distributed** — `--nodes`

The defaults are deliberately small (`--mesh-size 64 64 64`, `--box-size 200 200 200`) so a bare
invocation runs anywhere.
