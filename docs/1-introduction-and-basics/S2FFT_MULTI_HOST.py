import jax

jax.distributed.initialize()

from functools import partial

import jax.numpy as jnp
import jax.random as jr
import jax_healpy as jhp
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P

nside = 32
key = jr.key(0)

mesh = jax.make_mesh((4, 1), ("x", "y"), axis_types=(AxisType.Auto, AxisType.Auto))
sharding = NamedSharding(mesh, P("x", "y"))
print(f"Mesh shape     : {mesh.shape}")
print(f"Partition spec : {sharding.spec}")


npix = jhp.nside2npix(nside)
hp_map = jax.random.normal(
    key,
    (
        4,
        npix,
    ),
)
hp_map = jax.lax.with_sharding_constraint(hp_map, sharding)


def sharded_map2alm(hp_map, method):
    partialled = partial(jhp.map2alm, method=method)
    return jax.shard_map(partialled, mesh=mesh, in_specs=P("x", None), out_specs=P("x", None))(hp_map)


jax.debug.visualize_array_sharding(hp_map)
jax.clear_caches()


jax_alms = sharded_map2alm(hp_map, method="jax").block_until_ready()
print("JAX COMPUTED")
cuda_alms = sharded_map2alm(hp_map, method="jax_cuda").block_until_ready()
print("CUDA COMPUTED")

rel_error = jnp.linalg.norm(cuda_alms - jax_alms) / jnp.linalg.norm(jax_alms)
print(f"Relative error between cuda and jax implementations: {rel_error:.2e}")
