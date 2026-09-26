#!/usr/bin/env python
"""Multi-GPU MAP mass mapping: notebook 14 (Part I / Part II) as a script, one process per GPU.

The model, the truth, the data and the Adam MAP are those of ``14-LPT-MassMapping.ipynb``:
- 2LPT on a spherical lightcone of capped equal-volume shells from ``--r-min``;
- a per-shell resolution cut;
- a kappa likelihood band-limited to ``--ell-min <= ell <= ell_max`` with shape noise ``--sigma-e`` for the DES Y3
  bins (``--gals-per-arcmin2 15`` gives the notebook's Part II, Stage IV).

The white IC is sharded in x-slabs over all devices (``(n_dev, 1)`` mesh); kappa is replicated before every spherical
harmonic transform, because a sharded SHT hangs. Rank 0 writes the notebook's parquet layout:
- ``true_ic``, ``truth_density_lightcone``, ``lattice_density_lightcone``, ``truth_kappa`` and ``observed_kappa``;
- ``*_evolution/`` frames;
- ``metrics.json``, ``loss_history.npz`` and ``summary.json``.

So the notebook's ``load_run`` / ``plot_*`` functions and ``animation/animate_map_prep.py`` read it unchanged.

Memory (float64 value-and-gradient, the dominant cost):
- One A100 80 GB, measured: 745 bytes per voxel, i.e. 53.4 GB at 416^3 (the largest that fits) and 66.7 GB at 448^3
  (out of memory).
- Split over n devices in x-slabs with ``--halo-cells h``, the sharded part scales as N^2 * (N / n + 2 h).
- On top of that, every device holds a replicated spherical part (lightcone maps and the SHTs of the cuts), which
  grows with NSIDE.
- XLA's compiled memory on CPU, 1 vs 8 devices with h = 8 (``--memory-only``):
  - 128^3: 2.23 -> 0.53 GB, i.e. 0.24x (the slab formula gives 0.25x);
  - 192^3: 8.82 -> 3.03 GB, of which about 1.5 GB is the replicated part at NSIDE 128.
- On 8 x A100 80 GB (about 72 GB usable per GPU with MEM_FRACTION 0.95 and the CUDA SHT outside XLA's pool), that
  allows N = 768 with NSIDE 512 (ell_max 399, PAINT_NSIDE 256, the notebook's rule): 745 B * 768^2 * (96 + 32) = 56 GB
  sharded with h = 16, plus the replicated part.
- N = 832 (70 GB sharded) does not fit reliably, and N = 1024 (125 GB) does not fit.
- Run ``--memory-only`` on the cluster first: it compiles the value-and-gradient, prints XLA's per-device memory and
  exits.

  # 8 GPUs on one node (SLURM), memory check then the run:
  srun -n 8 --gpus-per-task 1 python 15-LPT-Multihost-MassMapping.py --mesh 768 --memory-only
  srun -n 8 --gpus-per-task 1 python 15-LPT-Multihost-MassMapping.py --mesh 768 --out-dir MESH768_DESY3
  srun -n 8 --gpus-per-task 1 python 15-LPT-Multihost-MassMapping.py --mesh 768 --gals-per-arcmin2 15 \
      --map-max-iter 400 --out-dir MESH768_STAGE4

  # the same with mpirun on one 8-GPU machine:
  mpirun -np 8 python 15-LPT-Multihost-MassMapping.py --mesh 768 --out-dir MESH768_DESY3

  # CPU check, 4 processes x 2 fake devices (8 devices, as on 8 GPUs):
  XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu mpirun -np 4 \
      python 15-LPT-Multihost-MassMapping.py --mesh 64 --halo-cells 4 --map-max-iter 4 --n-snapshots 2 --out-dir cpu_check
"""

import os

os.environ["JAX_ENABLE_X64"] = "True"  # float32 IC gradients diverge through the LPT + painting chain
# setdefault: an exported JAX_PLATFORMS (the CPU check) wins
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
# The CUDA SHT keeps its cuFFT plans outside XLA's pool: no preallocation, XLA capped at MEM_FRACTION.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
# XLA's Eigen intra-op threads deadlock inside large checkpointed-adjoint compiles when CPU affinity is low
# (CI flaky-timeout precedent, commit 72c318b). Appended so an exported XLA_FLAGS (fake devices) survives.
os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} --xla_cpu_multi_thread_eigen=false"

from datetime import datetime


def _maybe_init_distributed() -> None:
    """Initialize JAX distributed only under a real multi-process launch (srun / mpirun).

    As in ``bin/fli-simulate``: the proxy variables are stripped *before* ``jax.distributed.initialize()`` so the
    coordinator's gRPC channel connects directly to the sibling tasks.
    """
    multi = (
        int(os.environ.get("SLURM_NTASKS", 0)) > 1
        or int(os.environ.get("OMPI_COMM_WORLD_SIZE", 0)) > 1
        or int(os.environ.get("PMI_SIZE", 0)) > 1
    )
    if not multi:
        return
    for key in ("VSCODE_PROXY_URI", "no_proxy", "NO_PROXY"):
        os.environ.pop(key, None)
    import jax

    jax.distributed.initialize()
    print(f"[{datetime.now():%H:%M:%S}] rank {jax.process_index()}/{jax.process_count()} initialized", flush=True)


_maybe_init_distributed()

import argparse
import json
import time
from pathlib import Path

import datasets

datasets.disable_progress_bar()

import healpy as hp
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
import optax
from jax.experimental.mesh_utils import create_hybrid_device_mesh
from jax.experimental.multihost_utils import process_allgather, sync_global_devices
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from numpyro.handlers import condition
from numpyro.infer.util import potential_energy
from scipy.special import ndtri

import jax_fli as jfli
from jax_fli.data.nz import get_des_y3_nz_shear
from jax_fli.initial import interpolate_initial_conditions

jax.config.update("jax_enable_x64", True)
RANK, N_PROC = jax.process_index(), jax.process_count()
T_START = time.time()


def log(msg: str) -> None:
    """Rank-0, flushed progress line (srun buffers stdout otherwise)."""
    if RANK == 0:
        print(f"[{datetime.now():%H:%M:%S} +{time.time() - T_START:6.0f}s] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # resolution: everything below --mesh defaults to notebook 14's rule derived from it
    p.add_argument("--mesh", type=int, default=416, help="cells per axis")
    p.add_argument("--nside", type=int, default=None, help="HEALPix nside of the kappa maps (default 2 * paint nside)")
    p.add_argument(
        "--paint-nside", type=int, default=None, help="painting nside (default: smallest 2^k >= ell_max / 2)"
    )
    p.add_argument(
        "--ell-max", type=int, default=None, help="likelihood scale cut (default: k_Nyq * chi at the lensing peak)"
    )
    p.add_argument("--ell-taper", type=int, default=None, help="cosine taper width (default 12.5%% of ell_max)")
    p.add_argument("--ell-min", type=int, default=2, help="lowest multipole in the likelihood (shear has no ell < 2)")
    p.add_argument("--halo-cells", type=int, default=16, help="ghost cells on the sharded axis")
    # lightcone
    p.add_argument("--n-shells", type=int, default=22)
    p.add_argument("--min-width", type=float, default=50.0, help="Mpc/h, floor of the outer equal-volume shells")
    p.add_argument("--max-width", type=float, default=150.0, help="Mpc/h, cap of the inner equal-volume shells")
    p.add_argument("--r-min", type=float, default=300.0, help="Mpc/h, inner edge of the lightcone")
    # survey
    p.add_argument("--des-bins", type=int, nargs="+", default=(1, 2), help="DES Y3 bins (0-based)")
    p.add_argument("--sigma-e", type=float, default=0.26)
    p.add_argument(
        "--gals-per-arcmin2", type=float, default=None, help="galaxies/arcmin^2 per bin (default: DES Y3's own)"
    )
    p.add_argument("--max-z", type=float, default=1.0)
    # run
    p.add_argument("--map-max-iter", type=int, default=300, help="Adam steps")
    p.add_argument("--n-snapshots", type=int, default=10, help="frames saved after the first guess")
    p.add_argument("--adam-lr", type=float, default=0.05)
    p.add_argument("--init-scale", type=float, default=0.3, help="first guess amplitude, in prior standard deviations")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--memory-only", action="store_true", help="compile the value-and-gradient, print its memory, exit")
    # backend / cluster
    p.add_argument(
        "--map2alm-method",
        choices=("jax", "jax_cuda"),
        default="jax_cuda",
        help="jax_cuda needs one process per GPU; it falls back to jax on CPU",
    )
    p.add_argument("--gpus-per-node", type=int, default=int(os.environ.get("SLURM_GPUS_ON_NODE", 8)))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mesh_n, n_bins = args.mesh, len(args.des_bins)

    # ---- device mesh: x-slabs over all devices; hybrid (NVLink inside a node) when the devices span several nodes
    n_dev = jax.device_count()
    pdims = (n_dev, 1)
    if mesh_n % n_dev:
        raise ValueError(f"--mesh {mesh_n} must be divisible by the {n_dev} devices")
    n_slices = len({getattr(d, "slice_index", None) for d in jax.devices()})
    if n_slices <= 1 or N_PROC == 1 or n_dev <= args.gpus_per_node:
        dmesh = jax.make_mesh(pdims, ("x", "y"), axis_types=(AxisType.Auto, AxisType.Auto))
    else:
        dmesh = Mesh(create_hybrid_device_mesh((args.gpus_per_node, 1), (n_dev // args.gpus_per_node, 1)), ("x", "y"))
    sharding = NamedSharding(dmesh, P("x", "y"))
    replicated = NamedSharding(dmesh, P())
    halo_size = tuple(args.halo_cells if p > 1 else 0 for p in pdims)
    map2alm = args.map2alm_method if jax.default_backend() == "gpu" else "jax"  # the CUDA SHT has no CPU rule

    # ---- resolution (notebook 14, section 3)
    cosmo = jc.Planck18()
    box = tuple(float(x) for x in jfli.utils.compute_box_size_from_redshift(cosmo, args.max_z, (0.5, 0.5, 0.5)))
    L, chi_max = box[0], box[0] / 2
    kw = {} if args.gals_per_arcmin2 is None else {"gals_per_arcmin2": [args.gals_per_arcmin2] * 4}
    nz_shear = [get_des_y3_nz_shear(zmax=args.max_z, **kw)[i] for i in args.des_bins]
    chi = np.linspace(1.0, chi_max, 2000)
    z_s = np.linspace(1e-3, args.max_z, 1000)
    chi_s = np.asarray(jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(jnp.asarray(z_s))))
    q = np.array(
        [
            np.trapezoid(np.array(nz(jnp.asarray(z_s)))[None] * np.clip(1 - chi[:, None] / chi_s, 0, None), z_s, axis=1)
            * chi
            / np.asarray(jc.background.a_of_chi(cosmo, jnp.asarray(chi)))
            for nz in nz_shear
        ]
    )
    k_nyq = np.pi * mesh_n / L
    ell_max = args.ell_max or int(np.floor(k_nyq * chi[q.argmax(axis=1)].min()))
    ell_taper = args.ell_taper or int(round(0.125 * ell_max))
    paint_nside = args.paint_nside or int(2 ** np.ceil(np.log2(ell_max / 2)))
    nside = args.nside or 2 * paint_nside
    n_gal = np.array([float(nz.gals_per_arcmin2) for nz in nz_shear])
    sigma_b = args.sigma_e / np.sqrt(n_gal * hp.nside2pixarea(nside, degrees=True) * 3600)
    est_gb = 745 * mesh_n**2 * (mesh_n / n_dev + 2 * halo_size[0]) / 1e9
    log(
        f"jax {jax.__version__}  backend {jax.default_backend()}  processes {N_PROC}  devices {n_dev}  mesh {pdims}  "
        f"halo {halo_size}  MEM_FRACTION {os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']}"
    )
    log(
        f"MESH {mesh_n}^3  cell {L / mesh_n:.2f} Mpc/h  ell {args.ell_min}-{ell_max} (taper {ell_taper})  PAINT_NSIDE {paint_nside}  "
        f"NSIDE {nside}  galaxies/arcmin^2 {n_gal}  sigma_b {np.round(sigma_b, 5)}  estimated {est_gb:.1f} GB per device"
    )

    # ---- model (notebook 14, section 4)
    priors = {
        "Omega_c": jfli.infer.PreconditionnedUniform(0.1, 0.5),
        "sigma8": jfli.infer.PreconditionnedUniform(0.6, 1.0),
    }
    config = jfli.ppl.Configurations(
        mesh_size=(mesh_n,) * 3,
        box_size=box,
        halo_size=halo_size,
        field_sharding=sharding,
        sim_mode="lpt",
        lpt_order=2,
        paint_order="cic",
        gradient_order=4,
        laplace_fd=True,
        number_of_shells=args.n_shells,
        shell_spacing="equal_vol",
        min_width=args.min_width,
        max_width=args.max_width,
        r_min=args.r_min,
        resolution_cut=True,
        geometry="spherical",
        scheme="bilinear",
        nside=nside,
        paint_nside=paint_nside,
        kernel_width_pixels=0.8,
        observer_position=(0.5, 0.5, 0.5),
        lensing_output="convergence",
        normalization="global",
        map2alm_method=map2alm,
        min_redshift=0.001,
        max_redshift=args.max_z,
        n_integrate=8,
        fiducial_cosmology=jc.Planck18,
        nz_shear=nz_shear,
        priors=priors,
        sigma_e=args.sigma_e,
        ell_max=ell_max,
        ell_taper_width=ell_taper,
        ell_min=args.ell_min,
    )
    model, forward = jfli.ppl.full_field_probmodel(config), jfli.ppl.make_full_field_model(config)
    shape = (mesh_n,) * 3
    normal = jax.jit(lambda key, scale: scale * jax.random.normal(key, shape), out_shardings=sharding)
    w_truth, w_init = normal(jax.random.PRNGKey(args.seed), 1.0), normal(jax.random.PRNGKey(30), args.init_scale)
    # the noise draw is a host constant, identical on every process
    noise = (
        np.asarray(jax.random.normal(jax.random.PRNGKey(args.seed + 1000), (n_bins, 12 * nside**2))) * sigma_b[:, None]
    )
    sig = sigma_b[:, None]

    def band(kappa):
        # kappa band-limited to ell_min <= ell <= ell_max, replicated first (a sharded SHT hangs)
        k = kappa.replace(array=jax.lax.with_sharding_constraint(kappa.array, replicated), field_sharding=None)
        return jnp.stack(
            [k[b].scale_cut(ell_max, ell_taper, l_min=args.ell_min, method=map2alm).array for b in range(n_bins)]
        )

    @jax.jit
    def predict(white):
        # white IC -> (physical IC, lightcone after the resolution cut, band-limited kappa)
        ic = interpolate_initial_conditions(
            white,
            config.mesh_size,
            config.box_size,
            cosmo=cosmo,
            observer_position=config.observer_position,
            halo_size=config.halo_size,
            nside=config.nside,
            field_sharding=sharding,
        )
        kappa, lightcone = forward(cosmo, ic)
        return ic, lightcone, kappa.replace(array=band(kappa), field_sharding=None)

    base_fid = {
        f"{k}_base": float(ndtri((float(getattr(cosmo, k)) - float(p.low)) / (float(p.high) - float(p.low))))
        for k, p in priors.items()
    }

    def make_potential(x_obs):
        obs = {f"observable_{i}": x_obs[i] for i in range(n_bins)} | {k: jnp.asarray(v) for k, v in base_fid.items()}
        conditioned = condition(model, data=obs)
        return jax.jit(lambda w: potential_energy(conditioned, (), {}, {"initial_conditions": w}))

    if args.memory_only:
        vg = jax.jit(jax.value_and_grad(make_potential(jnp.zeros((n_bins, 12 * nside**2)))))
        log("compiling value_and_grad ...")
        m = vg.lower(w_init).compile().memory_analysis()
        log(
            f"per device: temporaries {m.temp_size_in_bytes / 1e9:.2f} GB  arguments {m.argument_size_in_bytes / 1e9:.2f} GB  "
            f"outputs {m.output_size_in_bytes / 1e9:.2f} GB  (estimate {est_gb:.1f} GB)"
        )
        sync_global_devices("memory-only")
        return

    out = Path(args.out_dir)
    if RANK == 0:
        out.mkdir(parents=True, exist_ok=True)
    sync_global_devices("out-dir")

    def save(fld, name):
        # every rank enters to_parquet (it starts with a process_allgather); only rank 0 writes the file
        jfli.io.Catalog(field=[fld], cosmology=[cosmo]).to_parquet(str(out / f"{name}.parquet"))

    # ---- truth and data (notebook 14, section 5)
    log("truth forward (compiles predict) ...")
    zeros = jnp.zeros(mesh_n)
    as_catalog = lambda ic, name: ic.replace(
        name=name, z_sources=zeros, comoving_centers=zeros, scale_factors=zeros, density_width=zeros
    )
    ic_truth, lightcone_truth, kappa_truth = predict(w_truth)
    lightcone_lattice = predict(normal(jax.random.PRNGKey(0), 0.0))[1]  # IC = 0: the unperturbed lattice
    x_obs = jax.jit(lambda k: k + noise, out_shardings=replicated)(kappa_truth.array)
    save(as_catalog(ic_truth, "true_ic"), "true_ic")
    save(lightcone_truth, "truth_density_lightcone")
    save(lightcone_lattice, "lattice_density_lightcone")
    save(kappa_truth, "truth_kappa")
    save(kappa_truth.replace(array=x_obs), "observed_kappa")
    kappa_truth_np = np.asarray(process_allgather(kappa_truth.array, tiled=True))
    kappa_truth_host = kappa_truth.replace(array=kappa_truth_np)
    log(f"truth written; kappa rms {np.round(kappa_truth_np.std(axis=1), 5)}  noise per pixel {np.round(sigma_b, 5)}")

    # ---- potential and its checks (notebook 14, section 5)
    potential = make_potential(x_obs)
    vg = jax.jit(jax.value_and_grad(potential))
    log("compiling value_and_grad ...")
    m = vg.lower(w_init).compile().memory_analysis()
    log(f"per device: temporaries {m.temp_size_in_bytes / 1e9:.2f} GB (estimate {est_gb:.1f} GB)")
    f_init, g_init = jax.block_until_ready(vg(w_init))
    t0 = time.time()
    jax.block_until_ready(vg(w_init))
    log(f"value_and_grad {time.time() - t0:.2f} s per call")
    v = normal(jax.random.PRNGKey(32), 1.0)
    v = v / jnp.linalg.norm(v)
    fd = (float(potential(w_init + 1e-4 * v)) - float(potential(w_init - 1e-4 * v))) / 2e-4
    log(f"FD check: autodiff {float(jnp.vdot(g_init, v)):.8e}  finite difference {fd:.8e}")
    potential_truth = float(potential(w_truth))
    chi2_truth = float(jnp.sum(((x_obs - kappa_truth.array) / sig) ** 2)) / x_obs.size
    log(f"at the truth: potential {potential_truth:.6e}  chi2/N_pix {chi2_truth:.4f}")

    # ---- MAP with frames (notebook 14, section 5)
    prefix = {
        "ic": "ic",
        "density": "lc",
        "kappa": "kappa",
        "coh_ic": "coh_ic",
        "trans_ic": "trans_ic",
        "coh_kappa": "coh_kappa",
        "trans_kappa": "trans_kappa",
    }
    paths = {name: out / f"{name}_evolution" for name in prefix}
    if RANK == 0:
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
            for stale in path.glob("*.parquet"):  # a rerun must not mix frames from an older run
                stale.unlink()
    sync_global_devices("frame-dirs")
    frames = np.linspace(0, args.map_max_iter, args.n_snapshots + 1).astype(int)
    metrics_rows = []
    f32 = lambda fld: fld.replace(array=fld.array.astype(jnp.float32))

    def snapshot(position, frame, step, loss):
        # every rank: IC, lightcone, kappa and their spectra vs the truth; rank 0 writes the files and the metrics row
        ic, lc, kappa = predict(position)
        kappa_host = kappa.replace(array=np.asarray(process_allgather(kappa.array, tiled=True)))
        fields = {
            "ic": f32(as_catalog(ic, f"ic_{frame}")),
            "density": f32(lc),
            "kappa": kappa,
            "coh_ic": ic_truth.coherence(ic).replace(name=f"coh_ic_{frame}"),
            "trans_ic": ic_truth.transfer(ic).replace(name=f"trans_ic_{frame}"),
            "coh_kappa": kappa_truth_host.coherence(kappa_host, method="healpy").replace(name=f"coh_kappa_{frame}"),
            "trans_kappa": kappa_truth_host.transfer(kappa_host, method="healpy").replace(name=f"trans_kappa_{frame}"),
        }
        for name, fld in fields.items():
            jfli.io.Catalog(field=[fld], cosmology=[cosmo]).to_parquet(
                str(paths[name] / f"{prefix[name]}_{frame}.parquet")
            )
        coh_ic = np.asarray(process_allgather(fields["coh_ic"].array, tiled=True))
        coh_k = np.asarray(fields["coh_kappa"].array)[..., args.ell_min :]
        row = {
            "frame": int(frame),
            "step": int(step),
            "loss": float(loss),
            "MSE_IC": float(jnp.mean((ic.array - ic_truth.array) ** 2)),
            "MSE_kappa": float(np.mean((kappa_host.array - kappa_truth_np) ** 2)),
            "coh_IC": float(np.nanmean(coh_ic)),
            "coh_IC_lowk": float(np.nanmean(coh_ic[..., :5])),
            "coh_kappa": float(np.nanmean(coh_k)),
            "coh_kappa_lowl": float(np.nanmean(coh_k[..., :5])),
            "trans_IC": float(np.nanmean(np.asarray(process_allgather(fields["trans_ic"].array, tiled=True)))),
            "trans_kappa": float(np.nanmean(np.asarray(fields["trans_kappa"].array)[..., args.ell_min :])),
        }
        metrics_rows.append(row)
        if RANK == 0:
            (out / "metrics.json").write_text(json.dumps(metrics_rows, indent=2))
        log(
            f"frame {frame}/{args.n_snapshots} (step {step}): loss {loss:.6e}  r(IC, low k) {row['coh_IC_lowk']:.3f}  r(kappa, low ell) {row['coh_kappa_lowl']:.3f}"
        )
        sync_global_devices(f"frame-{frame}")

    solver = optax.adam(optax.cosine_decay_schedule(init_value=args.adam_lr, decay_steps=args.map_max_iter, alpha=0.05))

    @jax.jit
    def opt_step(position, opt_state):
        val, grad = vg(position)
        updates, opt_state = solver.update(grad, opt_state, position)
        return optax.apply_updates(position, updates), opt_state, val, jnp.linalg.norm(grad)

    position, opt_state, losses, gnorms = w_init, solver.init(w_init), [], []
    snapshot(position, 0, 0, float(f_init))
    log(f"Adam {args.map_max_iter} steps, frames at {frames.tolist()}; compiling the step ...")
    for step in range(1, args.map_max_iter + 1):
        t0 = time.time()
        position, opt_state, val, gn = opt_step(position, opt_state)
        losses.append(float(val))
        gnorms.append(float(gn))
        dt = time.time() - t0
        if dt >= 1.0 or step % 10 == 0 or step <= 2:
            drop = losses[-26] - losses[-1] if len(losses) > 25 else float("nan")
            log(
                f"step {step}/{args.map_max_iter}  loss {losses[-1]:.6e}  |grad| {gn:.3e}  drop over 25 steps {drop:.1f}  {dt:.2f} s/step"
            )
        if step in frames[1:]:
            snapshot(position, int(np.searchsorted(frames, step)), step, losses[-1])

    chi2_map = float(jnp.sum(((x_obs - predict(position)[2].array) / sig) ** 2)) / x_obs.size
    summary = dict(
        potential_truth=potential_truth,
        potential_map=float(potential(position)),
        chi2_truth=chi2_truth,
        chi2_map=chi2_map,
    )
    if RANK == 0:
        np.savez(
            out / "loss_history.npz",
            steps=np.arange(1, len(losses) + 1),
            loss=np.asarray(losses),
            grad_norm=np.asarray(gnorms),
        )
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log(
        f"objective: MAP {summary['potential_map']:.6e}  truth {potential_truth:.6e}   chi2/N_pix: MAP {chi2_map:.4f}  truth {chi2_truth:.4f}"
    )
    sync_global_devices("done")


if __name__ == "__main__":
    main()
    if jax.process_count() > 1:
        jax.distributed.shutdown()
