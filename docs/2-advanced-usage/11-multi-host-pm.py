#!/usr/bin/env python
"""Distributed (multi-host) PM lightcone -> Born convergence -> Parquet catalog.

This is the script you launch across nodes. The whole pipeline is identical to the
single-device notebooks; only the device mesh changes. Run it either way:

  # Multi-host on SLURM (one process per GPU, jax.distributed coordinates them):
  srun -n $SLURM_NTASKS python 11-multi-host-pm.py --multihost \
       --mesh 1024 --box 3000 --nside 1024 --nb-shells 16 --out kappa_sim.parquet

  # Single host with fake CPU devices (laptop / CI smoke test):
  XLA_FLAGS="--xla_force_host_platform_device_count=4" JAX_PLATFORMS=cpu \
       python 11-multi-host-pm.py --mesh 64 --nside 64 --out kappa_sim.parquet

The output Parquet holds a ``SphericalKappaField`` (one map per source bin) plus the
cosmology, ready to load with ``jax_fli.io.Catalog.from_parquet`` (see 11-multi-host-pm.md).
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=int, default=64, help="cells per axis")
    parser.add_argument("--box", type=float, default=3000.0, help="box size [Mpc/h]")
    parser.add_argument("--nside", type=int, default=64, help="HEALPix nside")
    parser.add_argument("--nb-shells", type=int, default=8, help="lightcone shells")
    parser.add_argument("--out", type=str, default="kappa_sim.parquet")
    parser.add_argument(
        "--multihost",
        action="store_true",
        help="call jax.distributed.initialize() (one process per device across nodes)",
    )
    args = parser.parse_args()

    # Multi-host coordination must be set up *before* the JAX backend is touched.
    import jax

    if args.multihost:
        jax.distributed.initialize()

    import jax_cosmo as jc
    import jax_fli as jfli
    from jax.sharding import AxisType, NamedSharding
    from jax.sharding import PartitionSpec as P

    n_dev = jax.device_count()
    if jax.process_index() == 0:
        print(f"processes={jax.process_count()}  devices={n_dev}")

    # 2-D device mesh: partition the first two spatial axes (jaxpm convention).
    mesh = jax.make_mesh((n_dev, 1), ("x", "y"), axis_types=(AxisType.Auto, AxisType.Auto))
    sharding = NamedSharding(mesh, P("x", "y"))

    cosmo = jc.Planck18()
    key = jax.random.PRNGKey(0)

    # Ghost-cell width for the halo exchange at shard boundaries. Without it
    # (halo_size=0) the PM forces are wrong at the x-slab edges and the lightcone
    # shells show repeated patterns that worsen with more devices. The halo must
    # exceed the largest particle displacement (in cells) that crosses a boundary;
    # mesh//4 is a safe, generous default. Harmless on a single device.
    halo = args.mesh // 4
    initial_field = jfli.gaussian_initial_conditions(
        key,
        (args.mesh,) * 3,
        (args.box,) * 3,
        cosmo=cosmo,
        nside=args.nside,
        field_sharding=sharding,
        halo_size=(halo, halo),
    )
    dx, p = jfli.lpt(cosmo, initial_field, ts=0.1, order=1)

    solver = jfli.DoubleKickDrift(
        interp_kernel=jfli.OnionTiler(
            painting=jfli.PaintingOptions(target="spherical", scheme="rbf_neighbor"),
            drift_on_lightcone=True,
        ),
        t0=0.1,
        t1=1.0,
        n_steps=max(10, args.nb_shells + 2),
    )
    lightcone = jfli.nbody(cosmo, dx, p, nb_shells=args.nb_shells, solver=solver)

    nz_sources = jfli.io.get_stage3_nz_shear()
    kappa = jfli.born(cosmo, lightcone, nz_shear=nz_sources)
    shear = kappa.get_shear()

    # Only the lead process writes the gathered result.
    if jax.process_index() == 0:
        jfli.io.Catalog(field=shear, cosmology=cosmo).to_parquet(args.out.replace(".parquet", "_shear.parquet"))
        print(f"wrote {args.out}: {type(shear).__name__} {shear.shape}")
        jfli.io.Catalog(field=kappa, cosmology=cosmo).to_parquet(args.out.replace(".parquet", "_kappa.parquet"))


if __name__ == "__main__":
    main()
