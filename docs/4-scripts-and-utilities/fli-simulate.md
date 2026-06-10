# fli-simulate

Run the core `jax_fli` pipeline — initial conditions → LPT / PM N-body → lightcone painting
(optionally lensing) — and write the result as a Parquet `Catalog`.

## Usage

```bash
fli-simulate --sim-mode pm \
    --mesh-size 512 512 512 --box-size 1000 1000 1000 \
    --solver bf --n-steps 20 --nside 512 \
    --output out/ --name fiducial
```

`--sim-mode` selects the pipeline:

| Mode | What it runs |
|------|--------------|
| `lpt` | Lagrangian Perturbation Theory only |
| `pm` / `nbody` | full Particle-Mesh integration |
| `lensing` | PM lightcone + Born convergence (requires `--nside` or `--flatsky-npix`) |

## Key arguments

| Group | Flags |
|-------|-------|
| simulation | `--mesh-size`, `--box-size`, `--halo-size`, `--observer`, `--seed` |
| output target | `--nside` (HEALPix), `--flatsky-npix`, `--density` (3-D box) |
| integration | `--solver {kdk,dkd,bf}`, `--lpt-order {1,2}`, `--t0`, `--t1`, `--n-steps`, `--nb-shells` |
| cosmology | `--Omega-c --Omega-b --h --sigma8 --n-s --w0 --wa` |
| output | `--output <dir>`, `--name <label>` |
| benchmarking | `--perf`, `--iterations` |

Run `fli-simulate --help` for the full list and defaults.

## Distributed

Launch one process per GPU; field sharding and halo exchange are automatic. See
[multi-host PM](../2-advanced-usage/10-multi-host-pm.md) and the
[Distributed PM notebook](../1-introduction-and-basics/04-Distributed-PM.ipynb).
