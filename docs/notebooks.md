# jax-fli documentation

Executable tutorials and reference for **jax-fli** — differentiable cosmological forward modeling
on JAX. Each notebook is committed with its outputs; to keep the docs runnable on a laptop **most**
committed outputs are from small-scale **CPU** runs, with production parameters left in the code
(a few — e.g. *Drift on the Lightcone* — instead commit a full **GPU / cluster** run). Rerun any
notebook on a GPU / cluster to reproduce the full-resolution figures.

## Introduction & basics

1. [Library Fundamentals](1-introduction-and-basics/01-basics.ipynb) — the four field types,
   painting to 3-D / flat-sky / HEALPix, power spectra, and saving catalogs.
2. [LPT Lightcone](1-introduction-and-basics/02-LPT-Simulation.ipynb) — a Zel'dovich lightcone,
   comparison to theory, and **window deconvolution + shot-noise subtraction**.
3. [PM Simulation](1-introduction-and-basics/03-PM-Simulation.ipynb) — the BullFrog N-body solver
   and 3-D $P(k)$ vs Halofit with CIC deconvolution + shot noise.
4. [Distributed PM](1-introduction-and-basics/04-Distributed-PM.ipynb) — multi-device sharding,
   halo exchange, and distributed painting.

## Advanced usage

5. [PM Interpolation](2-advanced-usage/05-PM-Interpolation.ipynb) — `TelephotoInterp` /
   `OnionTiler` for shells that reach beyond the box.
6. [Drift on the Lightcone](2-advanced-usage/06-Drift-on-Lightcone.ipynb) — drifting particles to
   their lightcone-crossing epoch so the density field needs fewer, thicker shells.
7. [Advanced PM](2-advanced-usage/07-Advanced-PM.ipynb) — PGD small-scale correction and shell
   partitioning (equal width vs equal volume).
8. [Weak Lensing](2-advanced-usage/08-Lensing.ipynb) — Born + ray-traced convergence, shear maps,
   and saving convergence/shear catalogs.
9. [External Catalogs](2-advanced-usage/09-External-Catalog.ipynb) — loading CosmoGrid and
   GowerStreet lightcones.
10. [Multi-host PM](2-advanced-usage/10-multi-host-pm.md) — launching across nodes and validating
    convergence against a CosmoGrid reference.

## Sampling & inference

_In preparation_ — see [Sampling & inference](3-sampling-and-inference/). The command-line
inference tools already exist; see below.

## Scripts & utilities

Command-line entry points for batch / HPC runs — one page per script under
[Scripts & utilities](4-scripts-and-utilities/).

## Experiments

_In preparation_ — see [Experiments](5-experiments/).
