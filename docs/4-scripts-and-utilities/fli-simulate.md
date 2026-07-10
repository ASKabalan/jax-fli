# fli-simulate

Run the core `jax_fli` pipeline — initial conditions → LPT / PM N-body → lightcone painting (optionally lensing) — and write the result as a Parquet `Catalog`.

## Usage

```bash
fli-simulate --sim-mode pm \
    --mesh-size 512 512 512 --box-size 1000 1000 1000 \
    --solver bf --nb-steps 20 --nside 512 \
    --output out.parquet --name fiducial
```

`--sim-mode` selects the pipeline:

| Mode | What it runs |
|------|--------------|
| `lpt` | Lagrangian Perturbation Theory only |
| `pm` | full Particle-Mesh integration |
| `lensing` | PM lightcone + Born convergence (requires `--nside` or `--flatsky-npix`) |

(`--sim-mode` is required; the valid values are exactly `lpt`, `pm`, `lensing`.)

## Key arguments

| Group | Flags |
|-------|-------|
| simulation | `--mesh-size`, `--box-size`, `--halo-multiplier`, `--observer-position`, `--seed` |
| output target | `--nside` (HEALPix), `--flatsky-npix`, `--density` (3-D box), `--scheme`, `--paint-nside`, `--kernel-width-pixels` |
| integration | `--solver {kdk,dkd,bf}`, `--lpt-order {1,2}`, `--t0`, `--t1`, `--nb-steps`, `--nb-shells`, `--paint-order`, `--shell-spacing`, `--drift-on-lightcone` |
| lensing (mode `lensing`) | `--nz-shear`, `--min-z`, `--max-z`, `--n-integrate`, `--quadrature {midpoint,gauss_legendre}` |
| gradient | `--grad {none,reverse,checkpoint,checkpointed_<N>}` (differentiate the forward model w.r.t. the IC) |
| cosmology | `--Omega-c --Omega-b --h --sigma8 --n-s --w0 --wa` |
| output | `--output <file.parquet>`, `--shells-per-file <N>`, `--name <label>` |
| benchmarking | `--perf`, `--iterations` |

Run `fli-simulate --help` for the full list and defaults.

## Distributed

Launch one process per GPU; field sharding and halo exchange are automatic. See [multi-host PM](../2-advanced-usage/11-multi-host-pm.md) and the [Distributed PM notebook](../1-introduction-and-basics/05-Distributed-PM.ipynb).
