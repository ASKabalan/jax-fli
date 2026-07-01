"""Multi-process (mpirun) sibling of ``14-LPTLensingInference.ipynb``.

The notebook's single-process *fake-CPU multi-device* path deadlocks: jaxdecomp's distributed
collectives (``pfft3d`` all-to-all + the painting halo ``ppermute``) are scheduled inside BlackJAX's
data-dependent NUTS tree-building ``while_loop``, and on XLA:CPU's single-process *in-process*
communicator a divergent collective order hangs. The robust route is genuine multi-process: each
process drives ONE device and collectives go cross-process (gloo), which has no such ordering issue.
This is also exactly how the cluster runs ``fli-infer``.

Run (local fake 4-CPU):
    mpirun -n 4 python docs/3-sampling-and-inference/14_lpt_inference_mp.py

On a real GPU cluster, drop ``JAX_PLATFORMS=cpu`` and launch one rank per GPU (srun/mpirun).
"""

import os
from datetime import datetime

# One CPU device per process => N ranks give an N-device global mesh. (Do NOT set
# jax_num_cpu_devices: that is the single-process fake-multi-device path, the one that deadlocks.)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "True"
# float64 is required (float32 => NaN posterior gradient => numpyro init rejects the start point);
# deactivating the jax_cosmo host-side cache makes the background ODE in-graph, removing the
# device-0 host constant that otherwise fails to convert under the distributed partitioner.
os.environ["JAX_COSMO_DEACTIVATE_CACHE"] = "1"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.97"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def maybe_initialize_distributed():
    """Initialize JAX distributed under a real multi-process launch (mpirun / srun).

    Mirrors ``bin/fli-infer:_maybe_init_distributed`` but passes an explicit coordinator so it works
    with a plain local ``mpirun`` (no SLURM). Proxy vars are stripped first so the coordinator's gRPC
    channel connects directly to the sibling ranks instead of an unreachable HTTP/VSCode-remote proxy.
    """
    size = int(
        os.environ.get("OMPI_COMM_WORLD_SIZE") or os.environ.get("PMI_SIZE") or os.environ.get("SLURM_NTASKS") or 1
    )
    if size <= 1:
        return None, None
    rank = int(
        os.environ.get("OMPI_COMM_WORLD_RANK") or os.environ.get("PMI_RANK") or os.environ.get("SLURM_PROCID") or 0
    )
    for key in ("VSCODE_PROXY_URI", "no_proxy", "NO_PROXY"):
        os.environ.pop(key, None)

    import jax

    coordinator = os.environ.get("JAX_COORDINATOR_ADDRESS", "localhost:12399")
    jax.distributed.initialize(coordinator_address=coordinator, num_processes=size, process_id=rank)
    print(
        f"[{datetime.now():%H:%M:%S}] JAX distributed up: {jax.process_count()} processes, "
        f"rank {jax.process_index()}, devices={jax.device_count()}",
        flush=True,
    )
    return rank, size


RANK, SIZE = maybe_initialize_distributed()

import jax
import jax_cosmo as jc
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P
from numpyro.handlers import condition, seed, trace

import jax_fli as jfli

jax.config.update("jax_enable_x64", True)
# The jax_cosmo host-constant that broke Shardy is gone (cache deactivated), but keep the GSPMD
# partitioner explicitly for the distributed run — it is the configuration the cluster validates.
jax.config.update("jax_use_shardy_partitioner", False)

IS_LEAD = (RANK or 0) == 0


def log(*a):
    if IS_LEAD:
        print(*a, flush=True)


log(f"Devices: {jax.device_count()}  ({jax.devices()})")

# ── Mesh: shard the npix/mesh axis over all ranks. (4,1) P('x','y') matches the notebook. ──
ndev = jax.device_count()
if ndev > 1:
    mesh = jax.make_mesh((ndev, 1), ("x", "y"), axis_types=(AxisType.Auto, AxisType.Auto))
    sharding = NamedSharding(mesh, P("x", "y"))
else:
    mesh, sharding = None, None

# ── Config (identical physics to the notebook) ──
priors = {
    "Omega_c": jfli.infer.dist.PreconditionnedUniform(0.1, 0.5),
    "sigma8": jfli.infer.dist.PreconditionnedUniform(0.6, 1.0),
}
mesh_size = (64, 64, 64)
nside = 64
cosmo = jc.Planck18()
box_size = tuple(float(x) for x in jfli.utils.compute_box_size_from_redshift(cosmo, 0.25, (0.5, 0.5, 0.5)))

config = jfli.ppl.Configurations(
    mesh_size=mesh_size,
    box_size=box_size,
    halo_size=(4, 4),
    nb_steps=30,
    field_sharding=sharding,
    sim_mode="lpt",
    nbody_solver="BullFrog",
    t0=0.001,
    t1=1.0,
    lpt_order=1,
    number_of_shells=5,
    paint_order="cic",
    gradient_order=4,
    laplace_fd=True,
    shell_spacing="a",
    time_stepping="D",
    min_width=10.0,
    lensing_output="convergence",
    map2alm_method="jax",
    min_redshift=0.001,
    max_redshift=0.25,
    n_integrate=32,
    nside=nside,
    geometry="spherical",
    scheme="rbf_neighbor",
    observer_position=(0.5, 0.5, 0.5),
    paint_nside=nside,
    kernel_width_pixels=0.8,
    fiducial_cosmology=jc.Planck18,
    nz_shear=[0.25],
    log_lightcone=True,
    priors=priors,
    sigma_e=0.3,
    adjoint="checkpointed",
    checkpoints=5,
    sampler="NUTS",
    nuts_max_num_doublings=2,
    nuts_target_accept=0.65,
)

prob_model = jfli.ppl.full_field_probmodel(config)

# ── Mock observation + truth warm-start (sharded normal_field => globally consistent across ranks) ──
tr = trace(seed(prob_model, 0)).get_trace()
omega_c, sigma8 = tr["Omega_c"]["value"], tr["sigma8"]["value"]
init_field = tr["initial_conditions"]["value"]
data = {f"observable_{i}": tr[f"observable_{i}"]["value"] for i in range(len(config.nz_shear))}

init_params = {"Omega_c": omega_c, "sigma8": sigma8, "initial_conditions": init_field.array}
cond_model = condition(prob_model, data=data)

log(f"Truth: Omega_c={float(omega_c):.4f} sigma8={float(sigma8):.4f}")
log("Starting batched_sampling (NUTS, warmup=10, samples=50, batch=1) ...")

if mesh is not None:
    jax.set_mesh(mesh)

jfli.infer.batched_sampling(
    cond_model,
    init_params=init_params,
    path="output/demo_inference_mp",
    rng_key=jax.random.PRNGKey(0),
    num_warmup=10,
    num_samples=50,
    batch_count=1,
    sampler="NUTS",
    max_num_doublings=2,
    target_accept=0.65,
    progress_bar=False,
    save_callback=jfli.infer.sample2catalog(config),
)

if mesh is not None:
    # All ranks must reach the FFT collective in lockstep before exit.
    jax.block_until_ready(jax.numpy.zeros(1))
log("DONE — samples written to output/demo_inference_mp")
